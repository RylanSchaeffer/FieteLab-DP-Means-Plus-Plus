import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info
from typing import Dict, List, Tuple
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, stable_cumsum
# from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_is_fitted, _check_sample_weight



# def _init_centroids_dpmeans_plusplus(X: np.ndarray,
#                                      max_distance_param: float,
#                                      x_squared_norms: np.ndarray,
#                                      random_state: np.random.RandomState,
#                                      n_local_trials: int = None):
#     """Init clusters according to DP-means++
#
#     Parameters
#     ----------
#     X : array or sparse matrix, shape (n_samples, n_features)
#         The data to pick seeds for. To avoid memory copy, the input data
#         should be double precision (dtype=np.float64).
#
#     n_clusters : integer
#         The number of seeds to choose
#
#     x_squared_norms : array, shape (n_samples,)
#         Squared Euclidean norm of each data point.
#
#     random_state : int, RandomState instance
#         The generator used to initialize the centers. Use an int to make the
#         randomness deterministic.
#         See :term:`Glossary <random_state>`.
#
#     n_local_trials : integer, optional
#         The number of seeding trials for each center (except the first),
#         of which the one reducing inertia the most is greedily chosen.
#         Set to None to make the number of trials depend logarithmically
#         on the number of seeds (2+log(k)); this is the default.
#
#     Notes
#     -----
#     Selects initial cluster centers for k-mean clustering in a smart way
#     to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
#     "k-means++: the advantages of careful seeding". ACM-SIAM symposium
#     on Discrete algorithms. 2007
#     """
#
#     assert max_distance_param > 0.
#
#     n_samples, n_features = X.shape
#
#     # How many clusters are appropriate? We know that the prior on the
#     # number of clusters is alpha * log ( 1 + N \alpha). But what is alpha?
#     # For DP-Means, alpha = (1 + rho/sigma)^{d/2} exp(-lambda/2 sigma).
#     # alpha = np.exp(-max_distance_param / 2.)
#     # n_clusters = int(alpha * np.log(1. + n_samples / alpha))
#     # n_clusters = max(n_clusters, 1)
#     n_clusters = int(np.round(np.log(1. + n_samples)))
#
#     # We may have many centers as data
#     centers = np.empty((n_samples, n_features), dtype=X.dtype)
#
#     assert x_squared_norms is not None, 'x_squared_norms None in _init_centroids_dpmeans_plusplus'
#
#     # Set the number of local seeding trials if none is given
#     if n_local_trials is None:
#         # This is what Arthur/Vassilvitskii tried, but did not report
#         # specific results for other than mentioning in the conclusion
#         # that it helped.
#         n_local_trials = 2 + int(np.log(n_clusters))
#
#     # Pick first center randomly
#     center_id = random_state.randint(n_samples)
#     if sp.issparse(X):
#         centers[0] = X[center_id].toarray()
#     else:
#         centers[0] = X[center_id]
#
#     # Initialize list of closest distances and calculate current potential
#     closest_dist_sq = euclidean_distances(
#         centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
#         squared=True)
#     current_pot = closest_dist_sq.sum()
#
#     # Pick the remaining n_clusters-1 points
#     for c in range(1, n_clusters):
#         # Choose center candidates by sampling with probability proportional
#         # to the squared distance to the closest existing center
#         rand_vals = random_state.random_sample(n_local_trials) * current_pot
#         candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
#                                         rand_vals)
#         # XXX: numerical imprecision can result in a candidate_id out of range
#         np.clip(candidate_ids, None, closest_dist_sq.size - 1,
#                 out=candidate_ids)
#
#         # Compute distances to center candidates
#         distance_to_candidates = euclidean_distances(
#             X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
#
#         # Decide which candidate is the best
#         best_candidate = None
#         best_pot = None
#         best_dist_sq = None
#         for trial in range(n_local_trials):
#             # Compute potential when including center candidate
#             new_dist_sq = np.minimum(closest_dist_sq,
#                                      distance_to_candidates[trial])
#             new_pot = new_dist_sq.sum()
#
#             # Store result if it is the best local trial so far
#             if (best_candidate is None) or (new_pot < best_pot):
#                 best_candidate = candidate_ids[trial]
#                 best_pot = new_pot
#                 best_dist_sq = new_dist_sq
#
#         # Permanently add best center candidate found in local tries
#         if sp.issparse(X):
#             centers[c] = X[best_candidate].toarray()
#         else:
#             centers[c] = X[best_candidate]
#         current_pot = best_pot
#         closest_dist_sq = best_dist_sq
#
#     return centers


def _init_centroids_dpmeans_plusplus(X: np.ndarray,
                                     max_distance_param: float,
                                     x_squared_norms: np.ndarray,
                                     random_state: np.random.RandomState,
                                     n_local_trials: int = None):
    """
    Init clusters according to DP-means++
    """

    assert max_distance_param > 0.

    n_samples, n_features = X.shape

    assert x_squared_norms is not None, 'x_squared_norms None in _init_centroids_dpmeans_plusplus'

    chosen_center_indices = np.full(shape=n_samples,
                                    fill_value=False,
                                    dtype=np.bool)
    # Take the first observation as a center, since no centroids exist
    # to compare against.
    chosen_center_indices[0] = True
    unique_center_ids = {0}
    # if sp.issparse(X):
    #     centers[center_id] = X[center_id].toarray()
    # else:
    #     centers[center_id] = X[center_id]

    # We may have as many centers as data
    max_n_clusters = n_samples
    for cluster_idx in range(1, max_n_clusters):

        # Initialize list of closest distances and calculate current potential
        # Shape: (n_current_clusters, n_samples)
        distances_to_existing_centers = cdist(
            XA=X[chosen_center_indices],
            XB=X,
        )
        # Shape: (n_samples,)
        distances_to_nearest_center = np.min(
            distances_to_existing_centers,
            axis=0)

        # Zero out contenders closer than max allowable distance
        distances_to_nearest_center[distances_to_nearest_center < max_distance_param] = 0.

        # Terminate when every sample is within the maximum allowable distance
        if np.all(distances_to_nearest_center == 0.):
            break

        # If we haven't terminated, continue adding clusters.
        squared_distances_to_nearest_cluster = np.square(distances_to_nearest_center)

        sampling_distribution = squared_distances_to_nearest_cluster / \
                                np.sum(squared_distances_to_nearest_cluster)

        # from matplotlib.colors import LogNorm
        # import matplotlib.pyplot as plt
        # plt.close('all')
        # plt.scatter(X[chosen_center_indices, 0],
        #             X[chosen_center_indices, 1],
        #             color='r')
        # plt.scatter(X[sampling_distribution > 1e-5, 0],
        #             X[sampling_distribution > 1e-5, 1],
        #             c=sampling_distribution[sampling_distribution > 1e-5],
        #             norm=LogNorm(vmin=1e-5, vmax=1.),
        #             s=2)
        # plt.title(f'Cluster Idx: {cluster_idx}')
        # plt.show()
        # plt.close()
        center_id = random_state.choice(
            a=n_samples,
            p=sampling_distribution)

        # Center IDs should not be able to be selected again
        if center_id in unique_center_ids:
            raise ValueError(f'Center IDs should not be selectable more than '
                             f'once, but {center_id} was selected twice.')

        # Permanently add best center candidate found in local tries
        unique_center_ids.add(center_id)
        chosen_center_indices[center_id] = True

    # Keep only the selected observations as clusters.
    chosen_centers = X[chosen_center_indices].copy()
    return chosen_centers


class DPMeans:

    def __init__(self,
                 max_distance_param: float = 10.,
                 init: str = 'dp-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: int = None,
                 copy_x: bool = True,
                 algorithm: str = 'auto'):

        self.max_distance_param = max_distance_param
        self.init = init  # expected, dp-means++
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

        self.cluster_centers_ = None
        self.num_clusters_ = None
        self.num_init_clusters_ = None
        self.labels_ = None
        self.n_iter_ = None
        self.loss_ = None

    def _init_centroids(self, X, x_squared_norms, init, random_state,
                        init_size=None):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        max_distance_param = self.max_distance_param

        # Randomly permute data
        X = X[random_state.permutation(n_samples)]

        if isinstance(init, str) and init == 'dp-means':
            centers = _init_centroids_dpmeans(X,
                                              max_distance_param=max_distance_param,
                                              x_squared_norms=x_squared_norms,
                                              random_state=random_state, )
        elif isinstance(init, str) and init == 'dp-means++':
            # centers = _init_centroids_dpmeans_plusplus_old(X, max_distance_param=max_distance_param,
            #                                            random_state=random_state,
            #                                            x_squared_norms=x_squared_norms)
            centers = _init_centroids_dpmeans_plusplus(X=X,
                                                       max_distance_param=max_distance_param,
                                                       x_squared_norms=x_squared_norms,
                                                       random_state=random_state)
        elif isinstance(init, str) and init == 'random':
            # seeds = random_state.permutation(n_samples)[:n_clusters]
            # centers = X[seeds]
            raise NotImplementedError
        else:
            raise ValueError(f"Init {init} must be one of: dp-means, dp-means++, random.")

        return centers

    def fit(self, X, y=None, sample_weight=None):
        random_state = check_random_state(self.random_state)
        # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = self.init

        # subtract of mean of x for more accurate distance computations
        X = X.copy()
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self.algorithm in {"auto", "full"}:
            dpmeans_single = dp_means
            # self._check_mkl_vcomp(X, X.shape[0])
        else:
            raise NotImplementedError

        # Initialize centers
        centers_init = self._init_centroids(
            X, x_squared_norms=x_squared_norms, init=init,
            random_state=random_state)
        self.num_init_clusters_ = centers_init.shape[0]

        if self.verbose:
            print(f"Init method: {self.init} selected {self.num_init_clusters_}"
                  f" inital clusters with lambda={self.max_distance_param}.")

        # run DP-means
        labels, centers, n_iter_ = dpmeans_single(
            X=X, max_distance_param=self.max_distance_param,
            centers_init=centers_init, max_iter=self.max_iter,
        )

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            centers += X_mean

        # Shape: (num centers, num data)
        squared_distances_to_centers = euclidean_distances(
            X=centers,
            Y=X,
            squared=True)
        # Shape: (num data,)
        squared_distances_to_nearest_center = np.min(
            squared_distances_to_centers,
            axis=0)
        loss = np.sum(squared_distances_to_nearest_center)

        self.cluster_centers_ = centers
        self.num_clusters_ = centers.shape[0]
        self.labels_ = labels
        self.n_iter_ = n_iter_
        self.loss_ = loss
        return self

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return -_labels_inertia(X, sample_weight, x_squared_norms,
                                self.cluster_centers_)[1]

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        distances_x_to_centers = cdist(X, self.cluster_centers_)

        # sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        distances_to_cluster_centers = euclidean_distances(
            X=X,
            Y=self.cluster_centers_,
            squared=False)

        labels = np.argmin(distances_to_cluster_centers, axis=1)

        return labels


def dp_means(X: np.ndarray,
             centers_init: np.ndarray,
             max_distance_param: float,
             max_iter: int):
    # max_iter = 1 corresponds to "online."
    # max_iter > 1 corresponds to "offline"
    assert max_distance_param > 0.
    assert isinstance(max_iter, int)
    assert max_iter > 0

    num_obs, obs_dim = X.shape

    centers = np.zeros_like(X)
    num_centers = centers_init.shape[0]
    centers[:num_centers, :] = centers_init

    cluster_assignments = np.full(num_obs, fill_value=-1, dtype=np.int)

    iter_idx = 0
    for iter_idx in range(max_iter):

        print(f'Num centers: {num_centers}')

        no_datum_reassigned = True

        # Assign data to centers.
        for obs_idx in range(num_obs):

            # Compute distance from datum to each centroids.
            distances = cdist(
                XA=X[obs_idx, np.newaxis, :],
                XB=centers[:num_centers, :])

            # If smallest distance greater than cutoff max_distance_param,
            # then we create a new cluster.
            if np.min(distances) > max_distance_param:

                # centroid of new cluster = new sample
                centers[num_centers, :] = X[obs_idx, :]
                new_assigned_cluster = num_centers

                # increment number of clusters by 1
                num_centers += 1

            else:
                # If the smallest distance is less than the cutoff max_distance_param, assign point
                # to one of the older clusters
                new_assigned_cluster = np.argmin(distances)

            # Check whether this datum is being assigned to a new center.
            if cluster_assignments[obs_idx] != new_assigned_cluster:
                no_datum_reassigned = False

            # Record the observation's assignment
            cluster_assignments[obs_idx] = new_assigned_cluster

        # If no data was assigned to a different cluster, then we've converged.
        if iter_idx > 0 and no_datum_reassigned:
            break

        # Update centers based on assigned data.
        centers_to_keep = np.full(centers.shape[0],
                                  fill_value=False,
                                  dtype=np.bool)
        for center_idx in range(num_centers):

            # Get indices of all observations assigned to that cluster.
            indices_of_points_in_assigned_cluster = cluster_assignments == center_idx

            # Get observations assigned to that cluster.
            points_in_assigned_cluster = X[indices_of_points_in_assigned_cluster, :]

            # If this cluster has no assigned points, skip.
            if points_in_assigned_cluster.shape[0] == 0:
                continue

            # If this cluster has assigned points, mark to not delete.
            centers_to_keep[center_idx] = True

            # Recompute centroid from assigned observations.
            centers[center_idx, :] = np.mean(points_in_assigned_cluster,
                                             axis=0)

        # Delete old clusters and shift remaining clusters up.
        centers_to_keep = centers[centers_to_keep, :]
        num_centers = centers_to_keep.shape[0]
        centers[:num_centers, :] = centers_to_keep
        centers[num_centers:, :] = 0.

    # Increment by 1 since range starts at 0 but humans start counting at 1
    iter_idx += 1

    # Clean up centers by removing any center with no data assigned.
    cluster_ids, num_assigned_points = np.unique(cluster_assignments,
                                                 return_counts=True)
    nonempty_clusters = cluster_ids[num_assigned_points > 0]
    centers = centers[nonempty_clusters]

    return cluster_assignments, centers, iter_idx
