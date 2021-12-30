from itertools import product
import numpy as np
import os
import pandas as pd
from timeit import default_timer as timer

from src.data.synthetic import sample_mixture_model
from src.inference.dpmeans_quick import DPMeans
from src.metrics import compute_predicted_clusters_scores
from src.plot import plot_all

obs_dim = 13

exp_dir = '01_mixture_of_gaussians'
results_dir_path = os.path.join(exp_dir, 'results')

max_distance_params = np.logspace(-3, 3, 7)
init_methods = ['dp-means', 'dp-means++']
# init_methods = ['dp-means++']
repeats = np.arange(5)
df_rows = []

np.random.seed(0)

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
            random_state=repeat_idx,
            verbose=True)

        # time using timer because https://stackoverflow.com/a/25823885/4570472
        start_time = timer()
        dpmeans.fit(X=mixture_model_results['obs'])
        stop_time = timer()
        runtime = stop_time - start_time

        row = {
            'Initialization': init_method,
            'lambda': max_distance_param,
            'Repeat': repeat_idx,
            'Num Obs': mixture_model_results['obs'].shape[0],
            'Obs Dim': mixture_model_results['obs'].shape[1],
            'centroids_prior_cov_prefactor': mixture_model_results['centroids_prior_cov_prefactor'],
            'likelihood_cov_prefactor': mixture_model_results['likelihood_cov_prefactor'],
            'Num True Clusters': len(np.unique(mixture_model_results['cluster_assignments'])),
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
        row.update(scores)

        df_rows.append(row)

results_df = pd.DataFrame(df_rows)
results_df.to_csv(os.path.join(results_dir_path, 'results.csv'))

plot_all(results_df=results_df,
         plot_dir=results_dir_path)
