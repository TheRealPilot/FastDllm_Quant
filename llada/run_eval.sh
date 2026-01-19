#!/bin/bash
#SBATCH --job-name=llada_eval
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# Load modules
module load cuda/12.6.0

# Activate conda environment
source ~/.bashrc
conda activate fastdllm

# Navigate to directory
cd /scratch/gautschi/dlimpus/Fast-dLLM/llada

# Run evaluation
accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 1 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,show_speed=True
