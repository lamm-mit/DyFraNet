#!/bin/bash
#SBATCH --job-name="DyFraNet"
#SBATCH --partition=sched_mit_buehler_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:0:0
#SBATCH --output=cout_numf1_small.txt
#SBATCH --error=cerr_numf1_small.txt

source ~/ml.sh
source activate ML

python3 main.py --numframe 0
