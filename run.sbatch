#!/bin/bash
#Here is a comment
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=finetune_chemberta_random_train_smiles_temp_old_vocab
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/finetune_chemberta_random_train_smiles_temp_old_vocab-%j.out
#SBATCH --error=./sbatch_files/finetune_chemberta_random_train_smiles_temp_old_vocab-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=64000

module load python/anaconda-2022.05

source activate TransPolymer

python3 Downstream.py

