#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=4G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-user=rylansch
#SBATCH --mail-type=FAIL


# Run this, then pipe ID to each individual run
# wandb sweep 01_mixture_of_gaussians/sweep.yaml

for i in {1..20}
do
  sbatch 01_mixture_of_gaussians/run_one.sh cm8fwjsi
done
