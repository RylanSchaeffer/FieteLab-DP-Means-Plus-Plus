import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info
from typing import Dict, List, Tuple
import warnings

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_is_fitted, _check_sample_weight


def _dpmeans_plusplus(X: np.ndarray,
                      concentration: float,
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


def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


def _dpmeans_single_kulis(X, sample_weight, centers_init, max_iter=300,
                          verbose=False, x_squared_norms=None, tol=1e-4,
                          n_threads=1):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    x_squared_norms : ndarray of shape (n_samples,), default=None
        Precomputed x_squared_norms.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    # if sp.issparse(X):
    #     lloyd_iter = lloyd_iter_chunked_sparse
    #     _inertia = _inertia_sparse
    # else:
    lloyd_iter = lloyd_iter_chunked_dense
    _inertia = _inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers_new,
                       weight_in_clusters, labels, center_shift, n_threads)

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                if verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift**2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(f"Converged at iteration {i}: center shift "
                              f"{center_shift_tot} within tolerance {tol}.")
                    break

            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers,
                       weight_in_clusters, labels, center_shift, n_threads,
                       update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels)

    return labels, inertia, centers, i + 1


class DPMeans(TransformerMixin, ClusterMixin, BaseEstimator):

    def __init__(self,
                 max_distance: float = 10.,
                 init: str = 'dp-means++',
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 verbose: int = 0,
                 random_state: int = None,
                 copy_x: bool = True,
                 algorithm: str = 'auto', ):

        self.max_distance = max_distance
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params(self,
                      X: np.ndarray):

        # n_init
        if self.n_init <= 0:
            raise ValueError(
                f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= "
                             f"n_clusters={self.n_clusters}.")

        # tol
        self._tol = _tolerance(X, self.tol)

        # algorithm
        if self.algorithm not in ("auto", "full"):
            raise ValueError(f"Algorithm must be 'auto', 'full' or 'elkan', "
                             f"got {self.algorithm} instead.")

        self._algorithm = self.algorithm
        if self._algorithm == "auto":
            self._algorithm = "full" if self.n_clusters == 1 else "elkan"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn("algorithm='elkan' doesn't make sense for a single "
                          "cluster. Using 'full' instead.", RuntimeWarning)
            self._algorithm = "full"

        # init
        if not (hasattr(self.init, '__array__') or callable(self.init)
                or (isinstance(self.init, str)
                    and self.init in ["dp-means++", "random"])):
            raise ValueError(
                f"init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead.")

        if hasattr(self.init, '__array__') and self._n_init != 1:
            warnings.warn(
                f"Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.", RuntimeWarning, stacklevel=2)
            self._n_init = 1

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}.")
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}.")

    def _check_test_data(self, X):
        X = self._validate_data(X, accept_sparse='csr', reset=False,
                                dtype=[np.float64, np.float32],
                                order='C', accept_large_sparse=False)
        return X

    def _check_mkl_vcomp(self, X, n_samples):
        """Warns when vcomp and mkl are both present"""
        # The BLAS call inside a prange in lloyd_iter_chunked_dense is known to
        # cause a small memory leak when there are less chunks than the number
        # of available threads. It only happens when the OpenMP library is
        # vcomp (microsoft OpenMP) and the BLAS library is MKL. see #18653
        if sp.issparse(X):
            return

        active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if active_threads < self._n_threads:
            modules = threadpool_info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules]
            if has_vcomp and has_mkl:
                if not hasattr(self, "batch_size"):  # KMeans
                    warnings.warn(
                        f"KMeans is known to have a memory leak on Windows "
                        f"with MKL, when there are less chunks than available "
                        f"threads. You can avoid it by setting the environment"
                        f" variable OMP_NUM_THREADS={active_threads}.")
                else:  # MiniBatchKMeans
                    warnings.warn(
                        f"MiniBatchKMeans is known to have a memory leak on "
                        f"Windows with MKL, when there are less chunks than "
                        f"available threads. You can prevent it by setting "
                        f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
                        f"setting the environment variable "
                        f"OMP_NUM_THREADS={active_threads}")

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
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == 'dp-means++':
            centers, _ = _dpmeans_plusplus(X, n_clusters,
                                           random_state=random_state,
                                           x_squared_norms=x_squared_norms)
        elif isinstance(init, str) and init == 'random':
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, '__array__'):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(
                centers, dtype=X.dtype, copy=False, order='C')
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = _dpmeans_single_kulis
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            raise NotImplementedError

        best_inertia = None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init,
                random_state=random_state)
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, centers_init, max_iter=self.max_iter,
                verbose=self.verbose, tol=self._tol,
                x_squared_norms=x_squared_norms, n_threads=self._n_threads)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """Guts of transform method; no input validation."""
        return euclidean_distances(X, self.cluster_centers_)

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

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return _labels_inertia(X, sample_weight, x_squared_norms,
                               self.cluster_centers_, self._n_threads)[0]

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

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_sample_weights_invariance':
                    'zero sample_weight is not equivalent to removing samples',
            },
        }


def dp_means(observations: np.ndarray,
             concentration_param: float,
             likelihood_model: str,
             learning_rate: float,
             num_passes: int):
    # if num_passes = 1, then this is "online."
    # if num_passes > 1, then this if "offline"
    assert concentration_param > 0
    assert isinstance(num_passes, int)
    assert num_passes > 0

    # set learning rate to 0; unused
    learning_rate = 0

    # dimensionality of points
    num_obs, obs_dim = observations.shape
    max_num_clusters = num_obs
    num_clusters = 1

    # centroids of clusters
    means = np.zeros(shape=(max_num_clusters, obs_dim))

    # initial cluster = first data point
    means[0, :] = observations[0, :]

    # empirical online classification labels
    table_assignments = np.zeros((max_num_clusters, max_num_clusters))
    table_assignments[0, 0] = 1

    for pass_idx in range(num_passes):
        for obs_idx in range(1, len(observations)):

            # compute distance of new sample from previous centroids:
            distances = np.linalg.norm(observations[obs_idx, :] - means[:num_clusters, :],
                                       axis=1)
            assert len(distances) == num_clusters

            # if smallest distance greater than cutoff concentration_param, create new cluster:
            if np.min(distances) > concentration_param:

                # increment number of clusters by 1:
                num_clusters += 1

                # centroid of new cluster = new sample
                means[num_clusters - 1, :] = observations[obs_idx, :]
                table_assignments[obs_idx, num_clusters - 1] = 1.

            else:

                # If the smallest distance is less than the cutoff concentration_param, assign point
                # to one of the older clusters
                assigned_cluster = np.argmin(distances)
                table_assignments[obs_idx, assigned_cluster] = 1.

        for cluster_idx in range(num_clusters):
            # get indices of all observations assigned to that cluster:
            indices_of_points_in_assigned_cluster = table_assignments[:, cluster_idx] == 1

            # get observations assigned to that cluster
            points_in_assigned_cluster = observations[indices_of_points_in_assigned_cluster, :]

            assert points_in_assigned_cluster.shape[0] >= 1

            # recompute centroid incorporating this new sample
            means[cluster_idx, :] = np.mean(points_in_assigned_cluster,
                                            axis=0)

    table_assignment_posteriors_running_sum = np.cumsum(np.copy(table_assignments), axis=0)

    # returns classes assigned and centroids of corresponding classes
    dp_means_offline_results = dict(
        table_assignment_posteriors=table_assignments,
        table_assignment_posteriors_running_sum=table_assignment_posteriors_running_sum,
        parameters=dict(means=means),
    )
    return dp_means_offline_results
