#!/bin/bash
#Here is a comment
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp_4
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/infer_random_train_exact_paper_seq_vocab_sup_PE_I-%j.out
#SBATCH --error=./sbatch_files/infer_random_train_exact_paper_seq_vocab_sup_PE_I-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=1000

module load python/anaconda-2022.05

source activate TransPolymer

python3 Inference.py

