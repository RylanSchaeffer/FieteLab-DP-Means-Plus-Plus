"""
Fit DP-Means to a synthetic mixture of isotropic Gaussians.

Example usage:

python 01_mixture_of_gaussians/run_one.py
"""

from itertools import product
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from timeit import default_timer as timer
import wandb

from src.data.real import load_dataset
from src.inference.dpmeans_quick import DPMeans
from src.metrics import compute_predicted_clusters_scores


config_defaults = {
    'max_distance_param': 1.,
    'init_method': 'dp-means',  # either 'dp-means' or 'dp-means++'
    'dataset_name': 'cancer_gene_expression_2016',
    'repeat_idx': 0,
    'scaler': 'standard'
}
wandb.init(project='dp-means++-uci_datasets',
           config=config_defaults)
config = wandb.config

print(f'Running:')
for key, value in config.items():
    print(key, ' : ', value)


exp_dir = '03_uci_datasets'
results_dir_path = os.path.join(exp_dir, 'results')

# Set seed
np.random.seed(config['repeat_idx'])

dataset = load_dataset(dataset_name=config['dataset_name'])
obs = dataset['observations']
labels = dataset['labels']

if config['scaler'] == 'standard':
    scaler = StandardScaler()
    scaled_obs = scaler.fit_transform(X=obs)
elif config['scaler'] == 'robust':
    scaler = RobustScaler()
    scaled_obs = scaler.fit_transform(X=obs)
elif config['scaler'] == 'none':
    scaled_obs = obs
else:
    raise ValueError(f"Impermissible scaler: {config['scaler']}")

dpmeans = DPMeans(
    max_distance_param=config['max_distance_param'],
    init=config['init_method'],
    random_state=config['repeat_idx'],
    verbose=True)

# time using timer because https://stackoverflow.com/a/25823885/4570472
start_time = timer()
dpmeans.fit(X=scaled_obs)
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
    true_cluster_assignments=labels,
)
results.update(scores)

wandb.log(results, step=0)

print('Finished run.')
