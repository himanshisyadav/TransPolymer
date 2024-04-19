#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp_4
#SBATCH --account=rcc-staff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./sbatch_files/inference/random_train_scaled_temp_concat_linear_huber_5_multihead_8_best_model-%j.out
#SBATCH --error=./sbatch_files/inference/random_train_scaled_temp_concat_linear_huber_5_multihead_8_best_model-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=END
#SBATCH --mem-per-gpu=1000

module load python/anaconda-2022.05
source activate TransPolymer

yaml_file="./inference.yaml"

# Use Python to parse the YAML file
parsed_data=$(python - <<END
import yaml

# Load the YAML file
with open("$yaml_file", "r") as file:
    data = yaml.safe_load(file)

# Extract paths of all datasets and models
dataset_paths = [dataset['path'] for dataset in data.get('datasets', [])]
model_paths = [model['path'] for model in data.get('models', [])]

# Print the paths
print("Dataset paths:")
print("\n".join(dataset_paths))
print("\nModel paths:")
print("\n".join(model_paths))
END
)

echo "$parse_data.datasets.0.path"
echo "$parse_data.datasets.2.path"
echo "$parse_data.model.path"

