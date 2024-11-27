#!/bin/bash

# Run the Llama3 dataset preparation script
python3 bin/data_preparation/llama3_dataset_preparation.py \
    --samples -1 \
    --split "train" \
    --intermediate_datasets_base "data/instruction_datasets" \
    --shuffling "none" \
    --dataset_directory "data/finetuning_datasets/testing_dataset"
