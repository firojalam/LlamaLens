import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from datasets import Dataset

def prepare_dataset_llama3(dataset=None):
    # Define the prompt templates
    default_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}\ninput: {}\nlabel: <|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"""

    arabic_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}: {}\nsummary in Arabic: <|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"""

    hindi_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}: {}\nsummary in Hindi: <|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"""

    english_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}: {}\nsummary in English: <|eot_id|><|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>"""

    dataset = Dataset.from_pandas(dataset)
    EOS_TOKEN = '<|end_of_text|>'

    def formatting_prompts_func(examples):
        sys_prompt = "You are a social media expert providing accurate analysis and insights."

        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        languages = examples.get("lang", ["en"] * len(instructions))  # Default to "en" if "lang" is not present
        datasets = examples.get("dataset", [""] * len(instructions))  # Default to empty string if "dataset" is not present
        tasks = examples.get("task", [""] * len(instructions))  # Default to empty string if "task" is not present

        texts = []
        for instruction, input, output, lang, dataset, task in zip(instructions, inputs, outputs, languages, datasets, tasks):
            # Select prompt template based on the language
            if "Summarization" in task or "Summarization" in dataset:
                if lang == "ar":
                    prompt = arabic_prompt
                elif lang == "hi":
                    prompt = hindi_prompt
                elif lang == "en":
                    prompt = english_prompt
                else:
                    print("Set to default", dataset)
                    prompt = default_prompt  # Fallback to default if language is not recognized
            else:
                prompt = default_prompt

            text = prompt.format(sys_prompt, instruction, input, output) + EOS_TOKEN
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset





def load_instructions(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def list_files(directory: str) -> List[str]:
    return os.listdir(directory)


def process_datasets(
    base_path: str,samples = -1, split = "train") -> pd.DataFrame:
    intermediate_datasets = list_files(base_path)
    train_dfs = []

    for dataset_folder in intermediate_datasets:
        dataset_path = os.path.join(base_path, dataset_folder)
        print(f"Processing folder: {dataset_folder}")

        train_df_path = os.path.join(dataset_path,split+".jsonl")
        df = pd.read_json(train_df_path, lines=True)


        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if samples>0 and samples<len(df) and "ummariz" not in df.task.iloc[0]: #To be changed
                df, _ = train_test_split(df, train_size=int(samples), stratify=df['output'], random_state=42)
        if "ummariz" in df.task.iloc[0] and samples>0:
            df = df.sample(n=samples, random_state=42)  
        #print(df.dataset.iloc[0], df.isnull().sum())


        #print(df.columns)
        #print(df["instruction"].value_counts())

        train_dfs.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Total number of records: {len(combined_df)}")
    return train_dfs, combined_df


def main():
    """Main function to execute the script."""
    pass


if __name__ == "__main__":
    main()
