#!/bin/bash

nohup  accelerate launch bin/model_training/parallel_fine_tuning_llama3_quantized.py \
  --model_name "base_models/Meta-Llama-3.1-8B-Instruct" \
  --max_seq_length 512 \
  --quant_bits 4 \
  --use_nested_quant False \
  --batch_size 16 \
  --grad_size 2 \
  --epochs 1\
  --quant_bits 4 \
  --use_nested_quant False \
  --out_dir "trained_models/Meta-Llama-3.1-8B-Instruct-shuffled_by_language_20k_4bit/outputs" \
  --save_steps 500 \
  --train_set_dir "data/finetuning_datasets/shuffled_by_language_20k" \
  --dev_set_dir "data/validation_data_500" \
  --start_from_last_checkpoint False \
  --lora_adapter_dir "trained_models/Meta-Llama-3.1-8B-Instruct-shuffled_by_language_20k_4bit/lora_adapter" \
  --merged_model_dir "trained_models/Meta-Llama-3.1-8B-Instruct-shuffled_by_language_20k_4bit/merged_model" \
