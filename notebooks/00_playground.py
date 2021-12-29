from itertools import product
import numpy as np
import pandas as pd

from src.data.synthetic import sample_mixture_model
from src.inference.dpmeans_quick import DPMeans
from src.metrics import compute_predicted_clusters_scores
from src.plot import plot_all

obs_dim = 13

max_distance_params = np.logspace(-3, 3, 41)
init_methods = ['dp-means']  # , 'dp-means++']
repeats = np.arange(5)
df_rows = []

for repeat_idx in repeats:

    mixture_model_results = sample_mixture_model(
        num_obs=1000,
        obs_dim=13,
        mixing_probabilities=np.ones(10) / 10.,
        centroids_prior_cov_prefactor=10.,
        likelihood_cov_prefactor=1.,)

    for init_method, max_distance_param in product(init_methods, max_distance_params):

        dpmeans = DPMeans(
            max_distance_param=max_distance_param,
            init=init_method,
            random_state=repeat_idx)

        dpmeans.fit(X=mixture_model_results['obs'])

        row = {
            'Initialization': init_method,
            'lambda': max_distance_param,
            'Num Iter Till Convergence': dpmeans.n_iter_,
            'Num Inferred Clusters': dpmeans.num_clusters_,
            'Repeat': repeat_idx,
            'Num Obs': mixture_model_results['obs'].shape[0],
            'Obs Dim': mixture_model_results['obs'].shape[1],
            'centroids_prior_cov_prefactor': mixture_model_results['centroids_prior_cov_prefactor'],
            'likelihood_cov_prefactor': mixture_model_results['likelihood_cov_prefactor'],
            'Num True Clusters': len(np.unique(mixture_model_results['cluster_assignments'])),
        }

        scores, pred_cluster_assignments = compute_predicted_clusters_scores(
            cluster_assignment_posteriors=dpmeans.labels_,
            true_cluster_assignments=mixture_model_results['cluster_assignments'],
        )
        row.update(scores)

        df_rows.append(row)

results_df = pd.DataFrame(df_rows)

plot_all(results_df=results_df)
