python bin/evaluation/inference.py \
    --instructions-path support_data/instructions/instructions_gpt-4o_claude-3-5-sonnet_ar.json \
    --intermediate-base-path data/intermediate_datasets_ar \
    --results-folder-path "results/Test_results" \
    --model-path "base_models/Meta-Llama-3.1-8B-Instruct" \
    --samples -1 \
    --device 0
