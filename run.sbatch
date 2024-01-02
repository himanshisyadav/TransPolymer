#!/bin/bash
#Here is a comment
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=inference_chemberta_random_train_all_dollar_old_vocab_best_model
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/inference_chemberta_random_train_all_dollar_old_vocab_best_model_data_freqI-%j.out
#SBATCH --error=./sbatch_files/inference_chemberta_random_train_all_dollar_old_vocab_best_model_data_freqI-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=64000

module load python/anaconda-2022.05

source activate TransPolymer

python3 Inference.py

