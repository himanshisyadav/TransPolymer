#!/bin/bash
#Here is a comment
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=tsne
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/tsne-%j.out
#SBATCH --error=./sbatch_files/tsne-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=32000

module load python/anaconda-2022.05

source activate TransPolymer

python3 tSNE.py