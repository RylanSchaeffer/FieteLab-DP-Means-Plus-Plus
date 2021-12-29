from itertools import product
import numpy as np
import pandas as pd

from src.data.synthetic import sample_mixture_model
from src.inference.dpmeans_quick import DPMeans
from src.metrics import compute_predicted_clusters_scores
from src.plot import plot_all

obs_dim = 13
mixture_model_results = sample_mixture_model(
    num_obs=1000,
    mixing_probabilities=np.ones(10) / 10.,
    likelihood_params_prior=dict(mean=np.zeros(obs_dim),
                                 cov=10 * np.eye(obs_dim)))

max_distance_params = np.logspace(-3, 3, 61)
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
        'Num Clusters': dpmeans.num_clusters_,
    }

    scores, pred_cluster_assignments = compute_predicted_clusters_scores(
        cluster_assignment_posteriors=dpmeans.labels_,
        true_cluster_assignments=mixture_model_results['cluster_assignments'],
    )
    row.update(scores)

    df_rows.append(row)

results_df = pd.DataFrame(df_rows)

plot_all(
    results_df=results_df,
    mixture_model_results=mixture_model_results)
