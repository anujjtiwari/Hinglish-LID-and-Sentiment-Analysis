#!/bin/bash
#SBATCH --job-name=mistral
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log

source ~/research/sentiment_analysis/venv/bin/activate

cd ~/research/sentiment_analysis

python3 main.py --model_id "mistralai/Mistral-7B-Instruct-v0.3" \
    --output_path "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/results" \
    --shot_value 0 \
    --test_data  "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/comi_lingua_dataset/test.csv" \
    --max_seq_length 512 \
    --temperature 0.0 \
    --zero_shot_prompt_path "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/prompts/zero_shot_lid_prompt.txt" \
    --one_shot_prompt_path "/nlsasfs/home/aidrive/dassuv/research/sentiment_analysis/prompts/one_shot_lid_prompt.txt" 