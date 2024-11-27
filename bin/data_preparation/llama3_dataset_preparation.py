import argparse
import os
import pandas as pd
from combine_datasets import *
from sklearn.model_selection import train_test_split

# Set up argument parsing
parser = argparse.ArgumentParser(description="Prepare LLaMA 3 instruction dataset.")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples to process (-1 for all samples).")
parser.add_argument("--split", type=str, default="train", help="Dataset split to process (e.g., 'train', 'test').")
parser.add_argument("--intermediate_datasets_base", type=str, required=True, help="Base path for intermediate datasets.")
parser.add_argument("--shuffling", type=str, default="none", choices=["none", "fully", "by_language", "by_task"], help="Shuffling strategy for the dataset.")
parser.add_argument("--dataset_directory", type=str, required=True, help="Directory to save the prepared dataset.")

args = parser.parse_args()

# Assign parsed arguments to variables
samples = args.samples
split = args.split
intermediate_datasets_base = args.intermediate_datasets_base
shuffling = args.shuffling
dataset_directory = args.dataset_directory

# Process datasets
languages = ["ar", "en", "hi"]
dfs = []
for lang in languages:
    intermediate_datasets = os.path.join(intermediate_datasets_base, lang)
    _, dfs_combined = process_datasets(intermediate_datasets, samples=samples, split=split)
    dfs.append(dfs_combined)

# Shuffling logic
if shuffling == "none":
    df = pd.concat(dfs, ignore_index=True)
elif shuffling == "fully":
    df = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
elif shuffling == "by_language":
    for i in range(len(dfs)):
        dfs[i] = dfs[i].sample(frac=1).reset_index(drop=True)
    df = pd.concat(dfs, ignore_index=True)
elif shuffling == "by_task":
    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby('task', group_keys=False).apply(lambda x: x.sample(frac=1)).sort_values(by='task').reset_index(drop=True)

# Prepare and save dataset
df['original_id'] = df['original_id'].astype(str)
prepared_dataset = prepare_dataset_llama3(df)
prepared_dataset.save_to_disk(dataset_directory)
