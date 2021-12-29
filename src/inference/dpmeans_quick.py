import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info
from typing import Dict, List, Tuple
import warnings

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_is_fitted, _check_sample_weight


def _init_centroids_dpmeans(X: np.ndarray,
                            max_distance_param: float,
                            x_squared_norms,
                            random_state,
                            **kwargs):
    num_obs, obs_dim = X.shape
    centers = np.zeros_like(X)

    # Always take the first observation as a center, since no centroids exist
    # to compare against.
    centers[0] = X[0]
    num_centers = 1

    # Consequently, we start with the first observation
    for obs_idx in range(1, num_obs):
        x = X[obs_idx, np.newaxis, :]  # Shape: (1, sample dim)
        distances_x_to_centers = cdist(x, centers[:num_centers, :])
        if np.min(distances_x_to_centers) > max_distance_param:
            centers[num_centers] = x
            num_centers += 1

    return centers[:num_centers]


def _init_centroids_dpmeans_plusplus(X: np.ndarray,
                                     max_distance_param: float,
                                     x_squared_norms,
                                     random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The inital centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    assert concentration > 0.

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


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
        self.labels_ = None
        self.n_iter_ = None

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
            centers = _init_centroids_dpmeans(X, max_distance_param=max_distance_param,
                                              random_state=random_state,
                                              x_squared_norms=x_squared_norms)
        elif isinstance(init, str) and init == 'dp-means++':
            centers = _init_centroids_dpmeans_plusplus(X, max_distance_param=max_distance_param,
                                                       random_state=random_state,
                                                       x_squared_norms=x_squared_norms)
        elif isinstance(init, str) and init == 'random':
            # seeds = random_state.permutation(n_samples)[:n_clusters]
            # centers = X[seeds]
            raise NotImplementedError
        else:
            raise ValueError(f"Init {init} must be one of: dp-means, dp-means++, random.")

        return centers

    def fit(self, X, y=None, sample_weight=None):
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

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
        if self.verbose:
            print("Initialization complete")

        # run a k-means once
        labels, centers, n_iter_ = dpmeans_single(
            X=X, max_distance_param=self.max_distance_param,
            centers_init=centers_init, max_iter=self.max_iter,
        )

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            centers += X_mean

        self.cluster_centers_ = centers
        self.num_clusters_ = centers.shape[0]
        self.labels_ = labels
        self.n_iter_ = n_iter_
        return self


def dp_means(X: np.ndarray,
             centers_init: np.ndarray,
             max_distance_param: float,
             max_iter: int):
    # if num_passes = 1, then this is "online."
    # if num_passes > 1, then this if "offline"
    assert max_distance_param > 0.
    assert isinstance(max_iter, int)
    assert max_iter > 0

    num_obs, obs_dim = X.shape

    # Each datum might be its own cluster.
    max_num_clusters = num_obs

    centers = np.zeros_like(X)
    num_centers = centers_init.shape[0]
    centers[:num_centers, :] = centers_init

    cluster_assignments = np.zeros((max_num_clusters, max_num_clusters),
                                   dtype=np.int8)  # Only need to store 0/1

    iter_idx = 0
    for iter_idx in range(max_iter):

        no_datum_reassigned = True

        # Assign data to centers.
        for obs_idx in range(num_obs):

            # compute distance of new sample from previous centroids:
            distances = np.linalg.norm(X[obs_idx, :] - centers[:num_centers, :],
                                       axis=1)

            # if smallest distance greater than cutoff max_distance_param, create new cluster.
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
            if iter_idx > 0 and cluster_assignments[obs_idx, new_assigned_cluster] == 0:
                no_datum_reassigned = False

            # Record the observation's assignment
            cluster_assignments[obs_idx, new_assigned_cluster] = 1

        # If no data was assigned to a different cluster, then, we've converged.
        if iter_idx > 0 and no_datum_reassigned:
            break

        # Update centers based on assigned data.
        for center_idx in range(num_centers):

            # Get indices of all observations assigned to that cluster.
            indices_of_points_in_assigned_cluster = cluster_assignments[:, center_idx] == 1

            # Get observations assigned to that cluster.
            points_in_assigned_cluster = X[indices_of_points_in_assigned_cluster, :]

            if points_in_assigned_cluster.shape[0] >= 1:

                # Recompute centroid from assigned observations.
                centers[center_idx, :] = np.mean(points_in_assigned_cluster,
                                                 axis=0)

    # Increment by 1 since range starts at 0 but humans start at 1
    iter_idx += 1

    # Clean up centers by removing any center with no data assigned
    points_per_cluster = np.sum(cluster_assignments, axis=0)
    nonempty_clusters = points_per_cluster > 0
    centers = centers[nonempty_clusters]

    return cluster_assignments, centers, iter_idx
