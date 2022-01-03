import numpy as np
import os
import pandas as pd
# import wandb

from src.plot import plot_all

exp_dir = '01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir, 'results')
sweep_results_df_path = os.path.join(results_dir, 'sweep_results.csv')

# api = wandb.Api()
#
# # Project is specified by <entity/project-name>
# runs = api.runs("rylan/dp-means++-mixture-of-gaussians")
#
# summary_list, config_list, name_list = [], [], []
# for run in runs:
#     # .summary contains the output keys/values for metrics like accuracy.
#     #  We call ._json_dict to omit large files
#     summary_list.append(run.summary._json_dict)
#
#     # .config contains the hyperparameters.
#     #  We remove special values that start with _.
#     config_list.append(
#         {k: v for k, v in run.config.items()
#          if not k.startswith('_')})
#
#     # .name is the human-readable name of the run.
#     name_list.append(run.name)
#
# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
# })
#
# runs_df.to_csv(sweep_results_df_path)

sweep_results_df = pd.read_csv(sweep_results_df_path, index_col=None)

# Keep only finished runs
sweep_results_df = sweep_results_df[sweep_results_df['State'] == 'finished']

# Compute rho / sigma aka SNR
sweep_results_df['cov_prefactor_ratio'] = sweep_results_df['centroids_prior_cov_prefactor'] \
                                          / sweep_results_df['likelihood_cov_prefactor']

plot_all(sweep_results_df=sweep_results_df,
         plot_dir=results_dir)

print('Finished.')
