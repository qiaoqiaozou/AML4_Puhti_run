#!/bin/bash
#SBATCH --job-name=food101_ai
#SBATCH --account=project_2014146
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1,nvme:10

module purge
module load pytorch

set -x

srun python3 assignment4_food101.py