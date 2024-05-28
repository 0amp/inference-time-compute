#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=llama3_best_of_128
#SBATCH --output=llama3_best_of_128.out

python best_of_n.py