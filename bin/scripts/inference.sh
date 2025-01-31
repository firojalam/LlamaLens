#!/bin/bash

# Define paths
DATASET_PATH="/workspace/llamalens/data/test_full_native_english.json"
RESULTS_FOLDER="/workspace/llamalens/results/non-native"
MODEL_PATH="/workspace/llamalens/trained_models/Llamalens_EN_28_1_25/merged_model"

# Run the Python script
python clean_inference_script.py --dataset-path "$DATASET_PATH" --results-folder-path "$RESULTS_FOLDER" --model-path "$MODEL_PATH"
