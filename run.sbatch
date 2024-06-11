#!/bin/bash
#Here is a comment
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=midway3-0294
#SBATCH --job-name=strat
#SBATCH --account=pi-chibueze
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/strat/strat_exact_all_scaled_nonfusion_no_mol_new_huber_5-%j.out
#SBATCH --error=./sbatch_files/strat/strat_exact_all_scaled_nonfusion_no_mol_new_huber_5-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=32000

module load python/anaconda-2022.05

source activate TransPolymer

python3 Downstream.py

