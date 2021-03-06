#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 4                    # two cores
#SBATCH --mem=12G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL

# update
export PYTHONPATH=.
export WANDB_CONFIG_DIR=/om2/user/rylansch
export WANDB_API_KEY=51a0a43a1b4ba9981701d60c5f6887cd5bf9e03e

source dpmeanspp_venv/bin/activate

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

python 02_sklearn_assumptions/assumptions.py
