"""
Fit DP-Means to Scikit-Learn's demo datasets showing K-Means assumptions.

Modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html

Example usage:

python 02_sklearn_examples/assumptions.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.datasets import make_blobs
from timeit import default_timer as timer

from src.inference.dpmeans_quick import DPMeans
from src.metrics import compute_predicted_clusters_scores
from src.plot import plot_all

np.random.seed(0)
exp_dir = '02_sklearn_assumptions'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

n_samples = 1500
random_state = 170
n_clusters = 3

# Construct datasets
# 1. Isotropic Gaussians
X_3blobs, y_3blobs = make_blobs(
    n_samples=n_samples,
    random_state=random_state,
    centers=n_clusters,
    shuffle=True,)

# 2. Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X_3blobs, transformation)
y_aniso = y_3blobs.copy()

# 3. Different variance
X_uneq_var, y_uneq_var = make_blobs(
    n_samples=n_samples,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=random_state
)

# 4. Unevenly sized blobs
X_uneq_proportions = np.vstack((X_3blobs[y_3blobs == 0][:500],
                                X_3blobs[y_3blobs == 1][:100],
                                X_3blobs[y_3blobs == 2][:10]))
y_uneq_proportions = np.concatenate((y_3blobs[y_3blobs == 0][:500],
                                     y_3blobs[y_3blobs == 1][:100],
                                     y_3blobs[y_3blobs == 2][:10]))

datasets = [
    ('Ideal', X_3blobs, y_3blobs),  # 5.
    ('Anisotropic Variance', X_aniso, y_aniso),
    ('Unequal Variance', X_uneq_var, y_uneq_var),
    ('Unequal Proportions', X_uneq_proportions, y_uneq_proportions),
]

max_distance_params = np.logspace(-2, 2, 30)
# max_distance_params = [1.]
scores_df_rows = []

for max_distance_param in max_distance_params:

    fig, axes = plt.subplots(nrows=2,
                             ncols=len(datasets),
                             figsize=(3 * len(datasets), 6))

    for col_idx, (dataset_name, X, y) in enumerate(datasets):

        for row_idx, init_method in enumerate(['dp-means', 'dp-means++']):

            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ax.set_ylabel(f'{init_method}')
            if row_idx == 0:
                ax.set_title(dataset_name)

            dpmeans = DPMeans(
                max_distance_param=max_distance_param,
                init=init_method,
                random_state=0,
                verbose=True)

            start_time = timer()
            dpmeans.fit(X)
            stop_time = timer()
            runtime = stop_time - start_time

            y_pred = dpmeans.predict(X)

            sweep_results_df = dict(dataset_name=dataset_name,
                                    init_method=init_method,
                                    max_distance_param=max_distance_param)
            sweep_results_df.update({
                'Runtime': runtime,
                'Num Iter Till Convergence': dpmeans.n_iter_,
                'Num Initial Clusters': dpmeans.num_init_clusters_,
                'Num Inferred Clusters': dpmeans.num_clusters_,
                'Loss': dpmeans.loss_,
            })

            scores, _ = compute_predicted_clusters_scores(
                cluster_assignment_posteriors=y_pred,
                true_cluster_assignments=y)
            sweep_results_df.update(scores)
            scores_df_rows.append(sweep_results_df)

            ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=2)

    plt.savefig(os.path.join(results_dir,
                             f'assumptions_lambda={max_distance_param}.png'),
                bbox_inches='tight',
                dpi=300)
    # plt.show()
    plt.close()

# Plot scores vs max distance for each dataset
scores_df = pd.DataFrame(scores_df_rows)
scores_df.to_csv(os.path.join(results_dir, 'scores_df.csv'),
                 index=False)
for dataset_name, dataset_scores_df in scores_df.groupby('dataset_name'):
    plot_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(plot_dir, exist_ok=True)
    plot_all(
        sweep_results_df=scores_df,
        plot_dir=plot_dir)

print('Finished 02_sklearn_assumptions/assumptions.py.')
