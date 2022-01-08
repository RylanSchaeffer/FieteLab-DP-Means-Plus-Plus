"""
Fit DP-Means to a synthetic mixture of isotropic Gaussians.

Example usage:

python 01_mixture_of_gaussians/run_one.py
"""

from itertools import product
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer
import wandb

from src.data.synthetic import sample_mixture_model
from src.inference.dpmeans_quick import DPMeans
from src.metrics import compute_predicted_clusters_scores


config_defaults = {
    'n_samples': 1000,
    'n_features': 2,
    'n_clusters': 10,
    'max_distance_param': 1.,
    'centroids_prior_cov_prefactor': 5.,
    'likelihood_cov_prefactor': 1.,
    'init_method': 'dp-means++',  # either 'dp-means' or 'dp-means++'
    'repeat_idx': 0,
}
wandb.init(project='dp-means++-mixture-of-gaussians',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)


exp_dir = '01_mixture_of_gaussians'
results_dir_path = os.path.join(exp_dir, 'results')

# Set seed
np.random.seed(config['repeat_idx'])

mixture_model_results = sample_mixture_model(
    num_obs=config['n_samples'],
    obs_dim=config['n_features'],
    mixing_probabilities=np.ones(config['n_clusters']) / config['n_clusters'],
    centroids_prior_cov_prefactor=config['centroids_prior_cov_prefactor'],
    likelihood_cov_prefactor=config['likelihood_cov_prefactor'])


dpmeans = DPMeans(
    max_distance_param=config['max_distance_param'],
    init=config['init_method'],
    random_state=config['repeat_idx'],
    verbose=True)

# time using timer because https://stackoverflow.com/a/25823885/4570472
start_time = timer()
dpmeans.fit(X=mixture_model_results['obs'])
stop_time = timer()
runtime = stop_time - start_time

results = {
    'Runtime': runtime,
    'Num Iter Till Convergence': dpmeans.n_iter_,
    'Num Initial Clusters': dpmeans.num_init_clusters_,
    'Num Inferred Clusters': dpmeans.num_clusters_,
    'Loss': dpmeans.loss_,
}

scores, pred_cluster_assignments = compute_predicted_clusters_scores(
    cluster_assignment_posteriors=dpmeans.labels_,
    true_cluster_assignments=mixture_model_results['cluster_assignments'],
)
results.update(scores)

wandb.log(results, step=0)

print('Finished run.')
