import click
import torch
torch.autograd.set_detect_anomaly(True)

from datasets import load_from_disk
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.utils import logging
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)


@click.command()
@click.option('--model_name', default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit', help='Model name')
@click.option('--max_seq_length', default=512, help='Maximum sequence length')
@click.option('--quant_bits', default=4, help='Use 4-bit or 8-bit quantization')
@click.option('--use_nested_quant', default=False, help='Use double quantization')
@click.option('--max_steps', default=10, help='Maximum training steps')
@click.option('--batch_size', default=1, help='Batch size')
@click.option('--grad_size', default=2, help='Grad size')
@click.option('--epochs', default=1, help='number of epochs')
@click.option('--out_dir', default='outputs', help='Output directory')
@click.option('--save_steps', default=5, help='Save steps')
@click.option('--train_set_dir', default='/export/home/mohamedbayan/finetune_parallel/prepared_dataset',
              help='Training dataset directory')
@click.option('--dev_set_dir', default='/export/home/mohamedbayan/datasets/balanced_shuffled_dataset',
              help='Development dataset directory')
@click.option('--model_path', default='/export/home/mohamedbayan/models/model_hf', help='Model path')
@click.option('--start_from_last_checkpoint', default=False, help='Start from last checkpoint')
@click.option('--lora_adapter_dir', default='lora_adapter', help='LoRA adapter directory')
@click.option('--merged_model_dir', default='merged_model', help='Merged model directory')
def main(model_name, max_seq_length, quant_bits, use_nested_quant, max_steps, batch_size, out_dir, save_steps,
         train_set_dir, dev_set_dir, model_path, start_from_last_checkpoint, grad_size, lora_adapter_dir,
         merged_model_dir, epochs):
    device_map = "DDP"  # Change as needed
    #device_map = "auto"

    if device_map == "DDP":
        device_string = PartialState().process_index
        device_map = {'': device_string}
        print(f"{device_map=}")

    HAS_BFLOAT16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if HAS_BFLOAT16 else torch.float16

    load_in_8bit = quant_bits == 8
    load_in_4bit = quant_bits == 4
    bnb_config = BitsAndBytesConfig(
    load_in_8bit=load_in_8bit,  # to enable 8-bit quantization rather than 4
    load_in_4bit=load_in_4bit,
    #bnb_4bit_use_double_quant=True, # this should be a configuration, double quantization reduces precision
    bnb_4bit_use_double_quant=use_nested_quant,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config= bnb_config if (load_in_4bit or load_in_8bit) else None # use bnb if we load in 4 or 8 bits
    )
    #print("Dtype of model to train..." + str(model.dtype))


    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_seq_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = max_seq_length
    tokenizer.padding_side = "right"  
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if load_in_4bit or load_in_8bit:
        print("Loading model in " + str(dtype) + " bits")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else: # was set to always true so we will never run the if above
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)

    #print("Dtype of model to train..." + str(model.dtype))

    dataset = load_from_disk(train_set_dir)
    #dev_set = load_from_disk(dev_set_dir)
    from multiprocessing import cpu_count
    def apply_chat_template(example, tokenizer):
        task = example["task"].lower()
        output_ = ""
        if "summ" in task:
            output_ = "Summary: "
        else:
            output_ = "Label: "

        system_prompt = "You are a social media expert providing accurate analysis and insights."
        instruction = example["native_instruction"]
        input = example["input"]
        output = example["output"]
        message = []
        message.append({"role":"system", "content": system_prompt})
        message.append({"role":"user", "content": instruction + "\nInput: "+ input})
        message.append({"role":"assistant", "content": output_+output})

        example["Text"] = tokenizer.apply_chat_template(message, tokenize=False, max_length=4050, truncation=True)
        return example
    dataset = dataset.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                #remove_columns=column_names,
                                desc="Applying chat template",)
    
    #dataset = dataset.shuffle(seed=42).select(range(1000)) ### TO BE REMOVED, FOR TESTING




    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="Text",
        max_seq_length=max_seq_length,
        packing=True,
        tokenizer=tokenizer,

        #eval_dataset=dev_set,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_size,
            warmup_steps=10,
            # max_steps=max_steps,
            num_train_epochs=epochs,
            learning_rate=2e-4,
            fp16=not HAS_BFLOAT16,
            bf16=HAS_BFLOAT16,
            logging_steps=1,
            output_dir=out_dir,
            optim="adamw_hf",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            #evaluation_strategy="steps",
            #eval_steps=600,
            report_to="wandb",
            save_steps=save_steps,
            save_total_limit=3,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Ignore warnings
    logging.set_verbosity(logging.CRITICAL)

    trainer_stats = trainer.train(resume_from_checkpoint=start_from_last_checkpoint)
    #print("Dtype of checkpoint to start from ..." + str(trainer.model.dtype))

    # if False:
    print(f"{PartialState().process_index=}")
    trainer.model.save_pretrained(out_dir+f"/lora-epc{epochs}", save_adapter=True, save_config=True)

    trainer.model.save_pretrained(lora_adapter_dir, save_adapter=True, save_config=True)

    # Load the base model and LoRA adapter, and merge them
    #  torch_dtype=dtype to allow for loading in half precision (16) to avoid doubling size
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    #print("Dtype of model to start from to save ..." + str(base_model.dtype))

    # autocast_adapter_dtype is default to True so it casts the model to float32 causing its size to be double
    # we need the model to stay in float16 like we loaded and trained it with, so set to False
    model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter_dir, torch_dtype=dtype, autocast_adapter_dtype=False)
    #print("Dtype of model to merge with lora adapter ..." + str(model_to_merge.dtype))


    merged_model = model_to_merge.merge_and_unload()
    #print("Dtype of the final merged model..." + str(merged_model.dtype))

    #merged_model.half()
    # Save the merged model
    merged_model.save_pretrained(merged_model_dir)
    trainer.tokenizer.save_pretrained(merged_model_dir)

    # trainer.save_model("/workspace/model_and_tokenizer")
    # tokenizer.save_pretrained("/workspace/model_and_tokenizer")

    # trainer.model.save_pretrained(model_path)
    # model.save_pretrained(out_dir)
    # tokenizer.save_pretrained(out_dir)

    file_path = 'trainer_stats.json'
    with open(file_path, 'w') as file:
        json.dump(trainer_stats, file, indent=4)
        print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")


if __name__ == "__main__":
    main()




# nohup accelerate launch /workspace/llamalens/scripts/train.py \
#   --model_name "/workspace/llamalens/base_models/Meta-Llama-3.1-8B-Instruct" \
#   --max_seq_length 4096 \
#   --quant_bits -1 \
#   --use_nested_quant False \
#   --batch_size 4 \
#   --grad_size 2 \
#   --epochs 2 \
#   --out_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/outputs" \
#   --save_steps 500 \
#   --train_set_dir "/workspace/llamalens/data/finetuning_datasets/native_tuning_datasets/native_train_20k" \
#   --dev_set_dir "/workspace/llamalens/data/finetuning_datasets/native_tuning_datasets/native_dev_full" \
#   --start_from_last_checkpoint True \
#   --lora_adapter_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/lora_adapter" \
#   --merged_model_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/merged_model" \
#   > /workspace/llamalens/output.log 2>&1 &

