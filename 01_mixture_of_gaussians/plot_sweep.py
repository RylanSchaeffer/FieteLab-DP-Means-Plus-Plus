import numpy as np
import os
import pandas as pd

from src.plot import plot_all


exp_dir = '01_mixture_of_gaussians'
results_dir = os.path.join(exp_dir, 'results')
sweep_results_df_path = os.path.join(results_dir, 'sweep_results.csv')

sweep_results_df = pd.read_csv(sweep_results_df_path, index_col=None)

sweep_results_df['cov_prefactor_ratio'] = sweep_results_df['centroids_prior_cov_prefactor']\
                                 / sweep_results_df['likelihood_cov_prefactor']

plot_all(sweep_results_df=sweep_results_df,
         plot_dir=results_dir)
