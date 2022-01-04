"""
Fit DP-Means to Scikit-Learn's datasets showing K-Means assumptions. Modified
from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html

Example usage:

python 02_sklearn_examples/assumptions.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.datasets import make_blobs

from src.inference.dpmeans_quick import DPMeans

np.random.seed(0)
exp_dir = '02_sklearn_examples'
results_dir = os.path.join(exp_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

n_samples = 1500
random_state = 170
n_clusters = 3

# Construct datasets
# 1. Isotropic Gaussians
X_3blobs, y = make_blobs(
    n_samples=n_samples,
    random_state=random_state,
    centers=n_clusters)

# 2. Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X_3blobs, transformation)

# 3. Different variance
X_uneq_var, _ = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# 4. Unevenly sized blobs
X_uneq_proportions = np.vstack((X_3blobs[y == 0][:500], X_3blobs[y == 1][:100], X_3blobs[y == 2][:10]))

datasets = [
    ('Normal', X_3blobs),
    ('Anisotropic Variance', X_aniso),
    ('Unequal Variance', X_uneq_var),
    ('Unequal Proportions', X_uneq_proportions),
]

fig, axes = plt.subplots(nrows=2,
                         ncols=len(datasets),
                         figsize=(3*len(datasets), 6))

for col_idx, (dataset_name, X) in enumerate(datasets):

    for row_idx, init_method in enumerate(['dp-means', 'dp-means++']):

        ax = axes[row_idx, col_idx]
        if col_idx == 0:
            ax.set_ylabel(f'{init_method}')
        if row_idx == 0:
            ax.set_title(dataset_name)

        dpmeans = DPMeans(
            max_distance_param=5.,
            init=init_method,
            random_state=0,
            verbose=True)

        dpmeans.fit(X)
        y_pred = dpmeans.predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=y_pred, s=2)


plt.savefig(os.path.join(results_dir,
                         f'assumptions.png'),
            bbox_inches='tight',
            dpi=300)
# plt.show()
plt.close()
