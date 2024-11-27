import os
import click
import json
import pandas as pd
from combine_datasets import *
from sklearn.model_selection import train_test_split
from transformers import pipeline
import torch  # Import torch as it's used in model_kwargs
def split_dataframe(df, max_samples=100):
    """Splits a DataFrame into a list of DataFrames with a maximum number of samples."""
    return [df.iloc[i:i + max_samples] for i in range(0, df.shape[0], max_samples)]

def get_response(pipe,  df, instruction):

    max_tokens = 512
    
    temperature = 0.0001
    top_p = 1
    dataset_name = df.iloc[0]["dataset"]
    task_name = df.iloc[0]["task"]
    if "ummariz" not in task_name:
        max_tokens = 20
    texts = df["input"].tolist()
    prompts = [format_prompt(instruction, texts[i], dataset_name ,task_name ) for i in range(len(df))]
    messages = [
        [{"role": "system", "content": "You are a social media expert providing accurate analysis and insights."},{
            "role": "user",
            "content": prompt
        },
        ]
        for prompt in prompts
    ]

    # Process batches using KeyDataset
    results = []
    for out in pipe(
        (messages),
        max_new_tokens=max_tokens,
        temperature=temperature,
        batch_size=min(4,len(df)),
        top_p = 1

    ):
        print(out)
        print("\n\n")
        results.append(out[0]["generated_text"][-1]["content"])
    df["responses"] = results
    df["prompt"] = prompts
    return df

def format_prompt(instruction, text, dataset_name ,task_name ):
    if "summarization" in dataset_name or "summarization" in task_name:
        formated_prompt = instruction + "\n" + "text: " + text + "\n" + "summary: "
    else:
        formated_prompt = instruction + "\n" + "text: " + text + "\n" + "label: "
    return formated_prompt


def infer_datasets(instructions_path, intermediate_base_path, results_foler_path, samples = -1, pipe=None):
    dfs = []
    instructions = load_instructions(instructions_path)
    print(len(os.listdir(intermediate_base_path)), "folders detected")
    for name in os.listdir(intermediate_base_path):
        try:
            instruction = instructions[name][0]  # get the first instruction
            file_path = os.path.join(intermediate_base_path, name, "test.json")
            
            # read the JSON file
            df = pd.read_json(file_path, lines=True)
            #print(df.head())
            print(f"Langth of {file_name} is {len(df)}")
            if samples != -1:
                df = df.sample(n=samples, random_state=42)  #Just for testing, to be removed late
            for dataframe in split_dataframe(df, max_samples=20):

                output_path = os.path.join(results_foler_path, name)
                if not os.path.exists(output_path):
                    # Create the folder
                    os.makedirs(output_path)
                instruction = instructions[name][0]  
                df = get_response(pipe = pipe , df=dataframe, instruction = instruction)
                for index, row in df.iterrows():
                    data_row = row.to_dict()
                    dataset_name = data_row["dataset"].lower()
                    response = data_row["responses"].lower()
                    task_name = data_row["task"].lower()
                    id = data_row["id"]
                    file_name = id + ".json"
                    json_file_path = os.path.join(output_path, file_name)
                    processed_output = response.lower().strip()
                    if "\n" in processed_output:
                        processed_output = processed_output.split("\n")[0]
                    if " " in processed_output and  "ummariz" not in df.task.iloc[0]:
                        processed_output = processed_output.split(" ")[0]
                    



                    json_data = {
                    "Id": id,
                    "dataset": dataset_name,
                    "task": task_name,
                    "text": data_row["input"],
                    "input": data_row["prompt"],
                    "label": data_row["output"],
                    "output":response,
                    "processed_output": processed_output
                    }
                    if not os.path.exists(json_file_path):
                        # Create and write the initial data to the JSON file
                        with open(json_file_path, 'w', encoding='utf-8') as file:
                            json.dump(json_data, file, indent=4, ensure_ascii=False)
                        print(f"File created: {json_file_path}")
                    else:
                        print(f"File already exists: {json_file_path}")
                
            
        except Exception as e:
            print(f"An error occurred while processing the file: {file_path}")
            print(f"Error: {e}")

@click.command()
@click.option('--instructions-path', required=True, type=click.Path(exists=True), help="Path to the instructions JSON file.")
@click.option('--intermediate-base-path', required=True, type=click.Path(exists=True), help="Path to the intermediate datasets base folder.")
@click.option('--results-folder-path', required=True, type=click.Path(), help="Path to store the results.")
@click.option('--model_path', required=True, type=click.Path(), help="Path of model.")
@click.option('--samples', default=-1, type=int, help="Number of samples to process.")
@click.option('--device', default=0, type=int, help="which gpu.")

def main(instructions_path, intermediate_base_path, results_folder_path, samples, model_path, device):
    """CLI entry point for inferring datasets."""
        # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={
            "torch_dtype": torch.float16,
            #"quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
        device= device
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'





    infer_datasets(instructions_path, intermediate_base_path, results_folder_path, samples, pipe)

if __name__ == '__main__':
    main()
