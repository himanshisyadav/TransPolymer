#!/bin/bash
#Here is a comment
#SBATCH --time=03:00:00
#SBATCH --nodelist=midway3-0294
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=no_pretrain
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/project/rcc/hyadav/TransPolymer_2/sbatch_files/no_pretrain/strat_exact_all_scaled_fusion_no_mol_huber_5_no_pretrain-%j.out
#SBATCH --error=/project/rcc/hyadav/TransPolymer_2/sbatch_files/no_pretrain/strat_exact_all_scaled_fusion_no_mol_huber_5_no_pretrain-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=64000

module load python/anaconda-2022.05

source activate /project/rcc/hyadav/TransPolymer

python3 Downstream.py