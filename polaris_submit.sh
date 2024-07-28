#!/bin/bash -l
#PBS -N random
#PBS -l select=1:system=polaris
#PBS -l place=scatter:shared
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A electrolyte-chibueze
#PBS -e logs/
#PBS -o logs/

module load conda

conda activate /grand/electrolyte-chibueze/hyadav/conda/TransPolymer_env/

python /grand/electrolyte-chibueze/hyadav/TransPolymer/Downstream_huber.py