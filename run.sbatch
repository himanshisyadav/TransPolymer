#!/bin/bash
#Here is a comment
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=midway3-0294
#SBATCH --job-name=freqI_temp_first_salt_next_pooling_all_scaled
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/freqI_all_scaled_pooling_temp_first_token_salt_nex-%j.out
#SBATCH --error=./sbatch_files/freqI_all_scaled_pooling_temp_first_token_salt_nex-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=32000

module load python/anaconda-2022.05

source activate TransPolymer

python3 Downstream.py

