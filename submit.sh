#!/bin/bash -l
#PBS -N random
#PBS -l select=10:system=polaris
#PBS -l place=scatter:shared
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:grand
#PBS -q prod
#PBS -A electrolyte-chibueze
#PBS -e logs/
#PBS -o logs/

module load conda

conda activate /grand/electrolyte-chibueze/hyadav/conda/TransPolymer_env/

python /grand/electrolyte-chibueze/hyadav/TransPolymer_fusion/Downstream_huber.py