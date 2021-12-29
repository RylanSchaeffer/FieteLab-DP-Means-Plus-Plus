from itertools import product
import numpy as np
import pandas as pd

from src.data.synthetic import sample_mixture_model
from src.metrics import compute_predicted_clusters_scores
from src.inference.dpmeans_quick import DPMeans

# Some random data
# X = np.eye(57)[np.random.choice(np.arange(57), replace=True, size=100), :]

obs_dim = 13
mixture_model_results = sample_mixture_model(
    num_obs=1000,
    mixing_probabilities=np.ones(10) / 10.,
    likelihood_params_prior=dict(mean=np.zeros(obs_dim),
                                 cov=10 * np.eye(obs_dim)))

max_distance_params = np.logspace(-4, 4, 31)
init_methods = ['dp-means']  # , 'dp-means++']

df_rows = []

for init_method, max_distance_param in product(init_methods, max_distance_params):

    dpmeans = DPMeans(
        max_distance_param=max_distance_param,
        init=init_method)

    dpmeans.fit(X=mixture_model_results['obs'])

    row = {
        'Initialization': init_method,
        'lambda': max_distance_param,
        'Num Iter To Convergence': dpmeans.n_iter_,
        'Num Clusters': dpmeans.cluster_centers_.shape[0],
    }

    scores, pred_cluster_assignments = compute_predicted_clusters_scores(
        cluster_assignment_posteriors=dpmeans.labels_,
        true_cluster_assignments=mixture_model_results['cluster_assignments'],
    )
    row.update(scores)

    df_rows.append(row)

df = pd.DataFrame(df_rows)

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(data=df, x='lambda', y='Adjusted Mutual Info Score',
             hue='Initialization')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.show()

print(10)
