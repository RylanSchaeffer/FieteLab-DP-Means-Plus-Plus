#!/bin/bash
#SBATCH -p use-everything
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL


# Run this, then pipe ID to each individual run
# export WANDB_CONFIG_DIR=/om2/user/rylansch
# export WANDB_API_KEY=51a0a43a1b4ba9981701d60c5f6887cd5bf9e03e
# source dpmeanspp_venv/bin/activate
# wandb sweep 04_tabular_datasets/sweep.yaml

for i in {1..5}
do
  sbatch 04_tabular_datasets/run_one.sh cpi4ynfd
  sleep 10
done
