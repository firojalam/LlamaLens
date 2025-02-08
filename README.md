# [LlamaLens: Specialized Multilingual LLM forAnalyzing News and Social Media Content](https://arxiv.org/pdf/2410.15308)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-CC--BY--NC--SA-blue) [![Paper](https://img.shields.io/badge/Paper-Download%20PDF-green)](https://arxiv.org/pdf/2410.15308)



## Overview
LlamaLens is a specialized multilingual LLM designed for analyzing news and social media content. It focuses on 18 NLP tasks, leveraging 52 datasets across Arabic, English, and Hindi.

<p align="center">
<picture>
<img width="352" alt="capablities_tasks_datasets" src="https://github.com/user-attachments/assets/23bbb62b-0983-4df7-9d6b-64bef777e11c">
</picture>
</p>

## LlamaLens
This repo includes scripts needed to run our full pipeline, including data preprocessing and sampling, instruction dataset creation, model fine-tuning, inference and evaluation.

## Features

- **Multilingual Support**: Arabic, English, and Hindi.
- **Comprehensive NLP Tasks**: 18 tasks utilizing 52 datasets.
- **Domain Optimization**: Tailored for news and social media content analysis.

## Dataset

The model was trained on the [LlamaLens dataset](https://huggingface.co/collections/QCRI/llamalens-672f7e0604a0498c6a2f0fe9).

## Model

Access the LlamaLens model on [Hugging Face](https://huggingface.co/QCRI/LlamaLens).


## Setup

### Installation

1. Ensure you have Python and `pip` installed on your system.
2. Clone the repository (if applicable):
   ```sh
   git clone https://github.com/firojalam/LlamaLens.git
   cd LlamaLens
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. You may need to update transformers library
    ```sh
    pip install --upgrade transformers
    ```

## Instruction-following Dataset
This repository includes a script to prepare the Llama3.1 instruction dataset. You can customize the dataset preparation process using various parameters, including sample size, dataset split, shuffling strategy, and the output directory for the processed dataset. All datasets are available on [hugginface](https://huggingface.co/collections/QCRI/llamalens-672f7e0604a0498c6a2f0fe9)

## Parameters

### `samples`
- **Description**: Defines the number of samples to be used in the dataset.
- **Usage**:
  - Set to `-1` to use the full dataset.
  - Set to any positive integer to specify the maximum number of samples (using stratified sampling).

### `split`
- **Description**: Specifies the dataset split to generate.
- **Choices**:
  - `"train"`: Training set
  - `"test"`: Testing set
  - `"dev"`: Development set

### `intermediate_datasets_base`
- **Description**: Path to the directory containing subdirectories for different languages (e.g., `ar`, `en`, `hi`).
- **Usage**: The base directory should contain the following subdirectories:
  - `ar`: Arabic language dataset
  - `en`: English language dataset
  - `hi`: Hindi language dataset
- Example: `/path/to/intermediate_datasets/ar`, `/path/to/intermediate_datasets/en`, `/path/to/intermediate_datasets/hi`.

### `shuffling`
- **Description**: Defines how the dataset is shuffled.
- **Choices**:
  - `"none"`: No shuffling.
  - `"by_task"`: Shuffle the dataset within each task.
  - `"by_language"`: Shuffle the dataset within each language.
  - `"fully"`: Shuffle the entire dataset.
- **Usage**: Choose the option that best matches the configuration in the original paper.

### `dataset_directory`
- **Description**: Path to the directory where the prepared dataset will be saved.
- **Usage**: Specify the directory where you want to save the final dataset after it is processed.

## Running the Script

To run the dataset preparation script, use the following command. Adjust the parameters as needed:
```sh
python3 bin/data_preparation/llama3_dataset_preparation.py \
    --samples -1 \
    --split "train" \
    --intermediate_datasets_base "data/instruction_datasets" \
    --shuffling "none" \
    --dataset_directory "finetuning_datasets/testing_dataset"
```


## Model Training

This is an example of how run the training script on full precision mode:
```sh
accelerate launch bin/model_training/train-native.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
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
  --merged_model_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/merged_model"
```
This is an example of how run the training script on quantized mode:

```sh

accelerate launch bin/model_training/train-native.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --max_seq_length 4096 \
  --quant_bits 4 \
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
  --merged_model_dir "/workspace/llamalens/trained_models/Llamalens_native_28_1_25/merged_model"
```

## Model Inference

To run inference for a specific language, you have to specify the intermediate folder that contains multiple datasets.

```sh
python bin/evaluation/inference.py --dataset-path "$DATASET_PATH" --results-folder-path "$RESULTS_FOLDER" --model-path "$MODEL_PATH"
```

## Results Evaluation
To score results, run the follwing script:
```sh
python bin/evaluation/evaluate.py \
--experiment_dir results/Meta-Llama-3.1-8B-Instruct-shuffled_by_language_20k_4bit/ar
--output_dir scores/Meta-Llama-3.1-8B-Instruct-shuffled_by_language_20k_4bit/ar
```



## File Format

Each JSONL file in the dataset follows a structured format with the following fields:

- `id`: Unique identifier for each data entry.
- `original_id`: Identifier from the original dataset, if available.
- `input`: The original text that needs to be analyzed.
- `output`: The label assigned to the text after analysis.
- `dataset`: Name of the dataset the entry belongs.
- `task`: The specific task type.
- `lang`: The language of the input text.
- `instruction`: A brief set of instructions describing how the text should be labeled.
- `text`: A formatted structure including instructions and response for the task in a conversation format between the system, user, and assistant, showing the decision process.



**Example entry in JSONL file:**

```
{
    "id": "d1662e29-11cf-45cb-bf89-fa5cd993bc78",
    "original_id": "nan",
    "input": "الدفاع الجوي السوري يتصدى لهجوم صاروخي على قاعدة جوية في حمص",
    "output": "not_claim",
    "dataset": "ans-claim",
    "task": "Claim detection",
    "lang": "ar",
    "instruction": "Analyze the given text and label it as 'claim' if it includes a factual statement that can be verified, or 'not_claim' if it's not a checkable assertion. Return only the label without any explanation, justification or additional text.",
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a social media expert providing accurate analysis and insights.<|eot_id|><|start_header_id|>user<|end_header_id|>Analyze the given text and label it as 'claim' if it includes a factual statement that can be verified, or 'not_claim' if it's not a checkable assertion. Return only the label without any explanation, justification or additional text.\ninput: الدفاع الجوي السوري يتصدى لهجوم صاروخي على قاعدة جوية في حمص\nlabel: <|eot_id|><|start_header_id|>assistant<|end_header_id|>not_claim<|eot_id|><|end_of_text|>"
}
```



# License
This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).


# Citation
Please cite [our paper](https://arxiv.org/pdf/2410.15308) when using this model:

```
   @article{kmainasi2024llamalensspecializedmultilingualllm,
     title={LlamaLens: Specialized Multilingual LLM for Analyzing News and Social Media Content},
     author={Mohamed Bayan Kmainasi and Ali Ezzat Shahroor and Maram Hasanain and Sahinur Rahman Laskar and Naeemul Hassan and Firoj Alam},
     year={2024},
     journal={arXiv preprint arXiv:2410.15308},
     volume={},
     number={},
     pages={},
     url={https://arxiv.org/abs/2410.15308},
     eprint={2410.15308},
     archivePrefix={arXiv},
     primaryClass={cs.CL}
   }
```
