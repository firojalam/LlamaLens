#!/bin/bash

nohup accelerate launch /workspace/llamalens/scripts/train.py \
  --model_name "/workspace/llamalens/base_models/Meta-Llama-3.1-8B-Instruct" \
  --max_seq_length 4096 \
  --quant_bits -1 \
  --use_nested_quant False \
  --batch_size 4 \
  --grad_size 2 \
  --epochs 2 \
  --out_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/outputs" \
  --save_steps 500 \
  --train_set_dir "/workspace/llamalens/data/finetuning_datasets/native_tuning_datasets/native_train_20k" \
  --dev_set_dir "/workspace/llamalens/data/finetuning_datasets/native_tuning_datasets/native_dev_full" \
  --start_from_last_checkpoint True \
  --lora_adapter_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/lora_adapter" \
  --merged_model_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/merged_model" \
  > /workspace/llamalens/output.log 2>&1 &
