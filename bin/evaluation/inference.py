import os
import json
import pandas as pd
import torch
import click
from transformers import AutoModelForCausalLM, AutoTokenizer

@click.command()
@click.option('--dataset-path', required=True, help='Path to the dataset JSON file')
@click.option('--results-folder-path', required=True, help='Path to the results directory')
@click.option('--model-path', required=True, help='Path to the model directory')
def main(dataset_path, results_folder_path, model_path):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the JSON file into a Pandas DataFrame
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        lang, dataset, file_id = row["lang"], row["dataset"], row["id"]
        
        # Construct file path
        dir_path = os.path.join(results_folder_path, lang, dataset)
        file_path = os.path.join(dir_path, f"{file_id}.json")
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists!")
            continue
        
        os.makedirs(dir_path, exist_ok=True)
        
        # Prepare input for model inference
        task = row["task"].lower()
        output_prefix = "Summary: " if "sum" in task else "Label: "
        messages = [
            {"role": "system", "content": "You are a social media expert providing accurate analysis and insights."},
            {"role": "user", "content": f"{row['instruction']}\nInput: {row['input']}"},
            {"role": "assistant", "content": output_prefix}
        ]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            continue_final_message=True,
            tokenize=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,  
            pad_token_id=tokenizer.eos_token_id, 
            temperature=0.001
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        row["response"] = tokenizer.decode(response, skip_special_tokens=True)
        
        # Save results
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(row.to_dict(), f, ensure_ascii=False, indent=4)

    print("Data points have been saved successfully.")

if __name__ == "__main__":
    main()
