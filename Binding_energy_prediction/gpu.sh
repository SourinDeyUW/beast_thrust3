#!/bin/sh
#SBATCH --job-name=testcase
#SBATCH -N 1
#SBATCH -n 14    ##14 cores(of28) so you get 1/2 of machine RAM (64 GB of 128GB)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
#SBATCH --output job%j.out
#SBATCH --error jerr
#SBATCH -p v100-16gb-hiprio


##Load your modules and run code here

module load python3/anaconda/2021.11
module load cuda/11.1
#conda activate pdos_new
conda run -n pdos_new python adcopy.py
#conda activate /home/sourin/.conda/envs/pdos
#python3 ad_e3nn.py
