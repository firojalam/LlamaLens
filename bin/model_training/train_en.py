import click
import torch
import json
from datasets import load_from_disk
from accelerate import PartialState
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, set_seed as transformers_set_seed, logging
)
from peft import (
    prepare_model_for_kbit_training, LoraConfig, TaskType,
    get_peft_model, PeftModel
)
from trl import SFTTrainer
from multiprocessing import cpu_count


torch.autograd.set_detect_anomaly(True)

@click.command()
@click.option('--model_name', default='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit')
@click.option('--max_seq_length', default=512)
@click.option('--quant_bits', default=4)
@click.option('--use_nested_quant', is_flag=True)
@click.option('--max_steps', default=10)
@click.option('--batch_size', default=1)
@click.option('--grad_size', default=2)
@click.option('--epochs', default=1)
@click.option('--out_dir', default='outputs')
@click.option('--save_steps', default=5)
@click.option('--train_set_dir', required=True)
@click.option('--dev_set_dir', required=True)
@click.option('--model_path', default='/export/home/mohamedbayan/models/model_hf')
@click.option('--start_from_last_checkpoint', is_flag=True)
@click.option('--lora_adapter_dir', default='lora_adapter')
@click.option('--merged_model_dir', default='merged_model')
def main(model_name, max_seq_length, quant_bits, use_nested_quant, max_steps, batch_size, out_dir, save_steps,
         train_set_dir, dev_set_dir, model_path, start_from_last_checkpoint, grad_size, lora_adapter_dir,
         merged_model_dir, epochs):

    device_map = {'': PartialState().process_index}
    print(f"{device_map=}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=(quant_bits == 8),
        load_in_4bit=(quant_bits == 4),
        bnb_4bit_use_double_quant=use_nested_quant,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config=bnb_config if quant_bits in [4, 8] else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_seq_length)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = max_seq_length

    lora_config = LoraConfig(
        r=128, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM,
    )

    if quant_bits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)

    def apply_chat_template(example, tokenizer):
        prefix = "Summary: " if "summ" in example["task"].lower() else "Label: "
        system_prompt = "You are a social media expert providing accurate analysis and insights."
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{example['instruction']}\nInput: {example['input']}"},
            {"role": "assistant", "content": f"{prefix}{example['output_en']}"}
        ]
        example["Text"] = tokenizer.apply_chat_template(message, tokenize=False, max_length=4050, truncation=True)
        return example

    train_data = load_from_disk(train_set_dir).map(apply_chat_template, num_proc=cpu_count(), fn_kwargs={"tokenizer": tokenizer})
    dev_data = load_from_disk(dev_set_dir).map(apply_chat_template, num_proc=cpu_count(), fn_kwargs={"tokenizer": tokenizer})

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,
        dataset_text_field="Text",
        max_seq_length=max_seq_length,
        packing=True,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_size,
            num_train_epochs=epochs,
            warmup_steps=10,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=out_dir,
            optim="adamw_hf",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            evaluation_strategy="steps",
            eval_steps=save_steps,
            report_to="wandb",
            save_steps=save_steps,
            save_total_limit=3,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu_stats.name}. Max memory = {round(gpu_stats.total_memory / 1e9, 3)} GB.")
    print(f"{round(torch.cuda.max_memory_reserved() / 1e9, 3)} GB of memory reserved.")

    logging.set_verbosity(logging.CRITICAL)
    trainer_stats = trainer.train(resume_from_checkpoint=start_from_last_checkpoint)

    trainer.model.save_pretrained(f"{out_dir}/lora-epc{epochs}", save_adapter=True, save_config=True)
    trainer.model.save_pretrained(lora_adapter_dir, save_adapter=True, save_config=True)

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter_dir, torch_dtype=dtype, autocast_adapter_dtype=False)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)

    with open('trainer_stats.json', 'w') as f:
        json.dump(trainer_stats, f, indent=4)

    print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds ({round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes)")

if __name__ == "__main__":
    main()
