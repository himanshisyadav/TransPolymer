#!/bin/bash
#Here is a comment
#SBATCH --time=03:00:00
#SBATCH --nodelist=midway3-0294
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=random_fusion
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/random/random_no_mol_scaled_linear_huber_5_temp_mean_768-%j.out
#SBATCH --error=./sbatch_files/random/random_no_mol_scaled_linear_huber_5_temp_mean_768-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=64000

module load python/anaconda-2022.05

source activate TransPolymer

python3 Downstream_learned_feature.py