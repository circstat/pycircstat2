from typing import Optional, Union

import numpy as np

from .descriptive import circ_dist, circ_kappa, circ_mean_and_r
from .distributions import vonmises
from .utils import data2rad


class MovM:
    """
    Mixture of von Mises (MovM) Clustering.

    This class implements the Expectation-Maximization (EM) algorithm for clustering 
    circular data using a mixture of von Mises distributions. It is analogous to 
    Gaussian Mixture Models (GMM) but adapted for directional statistics.

    Parameters
    ----------
    burnin : int, default=30
        Number of initial iterations before checking for convergence.
    n_clusters : int, default=5
        The number of von Mises distributions (clusters) to fit.
    n_iters : int, default=100
        Maximum number of iterations for the EM algorithm.
    full_cycle : int, default=360
        Used for converting degree-based data into radians.
    unit : {"degree", "radian"}, default="degree"
        Specifies whether input data is in degrees or radians.
    random_seed : int, default=2046
        Random seed for reproducibility.
    threshold : float, default=1e-16
        Convergence threshold based on the negative log-likelihood difference.

    Attributes
    ----------
    converged : bool
        Whether the algorithm has converged.
    nLL : np.ndarray
        Array of negative log-likelihood values over iterations.
    m : np.ndarray
        Cluster means (circular means).
    r : np.ndarray
        Cluster mean resultant vectors.
    p : np.ndarray
        Cluster probabilities.
    kappa : np.ndarray
        Concentration parameters for each von Mises component.
    gamma : np.ndarray
        Responsibility matrix (posterior probabilities of clusters for each data point).
    labels : np.ndarray
        The most probable cluster assignment for each data point.

    Examples
    --------
        import numpy as np
        from pycircstat2.clustering import MovM
        np.random.seed(42)
        x1 = np.random.vonmises(mu=0, kappa=5, size=100)
        x2 = np.random.vonmises(mu=np.pi, kappa=10, size=100)
        x = np.concatenate([x1, x2])
        np.random.shuffle(x)
        movm = MovM(n_clusters=2, n_iters=200, unit="radian", random_seed=42)
        movm.fit(x, verbose=False)
    """

    def __init__(
        self,
        burnin: int = 30,
        n_clusters: int = 5,
        n_iters: int = 100,
        full_cycle: Union[int, float] = 360,
        unit: str = "degree",
        random_seed: int = 2046,
        threshold: float = 1e-16,
    ):
        self.burnin = (
            burnin  # wait untill burinin step of iterations for convergence
        )
        self.threshold = threshold  # convergence threshold
        self.n_clusters = n_clusters  # number of clusters to estimate
        self.n_iters = n_iters  # maximum number of iterations for EM
        self.full_cycle = full_cycle  # for data conversion
        self.unit = unit  # for data conversion
        self.random_seed = random_seed
        self.converged = False  # place holder

        self.m_ = None  # cluster means
        self.r_ = None  # cluster mean resultant vectors
        self.p_ = None  # cluster probabilities
        self.kappa_ = None  # cluster kappas
        self.gamma_ = None  # update gamma one last time
        self.labels_ = None # final cluster assignments

    def _initialize(
        self,
        x: np.ndarray,
        n_clusters_init: int,
    ) -> tuple:
        """
        Initializes cluster parameters before running the EM algorithm.

        Parameters
        ----------
        x : np.ndarray
            Input circular data in radians.
        n_clusters_init : int
            Number of initial clusters.

        Returns
        -------
        tuple
            - m (np.ndarray): Initial cluster means.
            - kappa (np.ndarray): Initial concentration parameters.
            - p (np.ndarray): Initial cluster probabilities.
        """
        # number of samples
        n = len(x)  

        # initial cluster probability
        p = np.ones(n_clusters_init) / n_clusters_init 

        # initial labels
        z = np.random.choice(np.arange(n_clusters_init), size=n) 

        # initial means and resultant vector lengths
        m, r = map(
            np.array,
            zip(*[circ_mean_and_r(x[z == i]) for i in range(n_clusters_init)]),
        )  

        # initial kappa (without correction by hard-coding a larger enough n)
        kappa = np.array([circ_kappa(r=r[i]) for i in range(n_clusters_init)])  

        return m, kappa, p

    def fit(self, X: np.ndarray, verbose: Union[bool, int] = 0):
        """
        Fits the mixture of von Mises model to the given data using the EM algorithm.

        Parameters
        ----------
        X : np.ndarray
            Input data points in degrees or radians.
        verbose : bool or int, default=0
            If True, prints progress every iteration. If an integer, prints every `verbose` iterations.

        Updates
        -------
        - self.m : Fitted cluster means.
        - self.kappa : Fitted concentration parameters.
        - self.p : Fitted cluster probabilities.
        - self.labels : Final cluster assignments.
        """
        # seed
        np.random.seed(self.random_seed)

        # meta
        self.data = X
        self.alpha = alpha = (
            X if self.unit == "radian" else data2rad(X, self.full_cycle)
        )
        self.n = n = len(X)

        # init
        m, kappa, p = self._initialize(alpha, self.n_clusters)

        # EM
        if verbose:
            print("Iter".ljust(10) + "nLL")
        self.nLL = np.ones(self.n_iters) * np.nan
        for i in range(self.n_iters):
            # E step
            gamma = self.compute_gamma(alpha=self.alpha, p=p, m=m, kappa=kappa)
            gamma_normed = gamma / np.sum(gamma, axis=0)

            # M step
            p = (
                np.sum(gamma_normed, axis=1)
                / np.sum(gamma_normed, axis=1).sum()
            )
            
            m, r = map(
                np.array,
                zip(
                    *[
                        circ_mean_and_r(alpha=alpha, w=gamma_normed[i])
                        for i in range(self.n_clusters)
                    ]
                ),
            )
            kappa = np.array(
                [circ_kappa(r=r[i]) for i in range(self.n_clusters)]
            )

            nLL = self.compute_nLL(gamma)
            self.nLL[i] = nLL

            if verbose:
                if i % int(verbose) == 0:
                    print(f"{i}".ljust(10) + f"{nLL:.03f}")

            if (
                i > self.burnin
                and np.abs(self.nLL[i] - self.nLL[i - 1]) < self.threshold
            ):
                self.nLL = self.nLL[~np.isnan(self.nLL)]
                self.converged = True
                self.converged_iters = len(self.nLL)

                if verbose:
                    print(f"Converged at iter {i}. Final nLL = {nLL:.3f}\n")
                break
        else:
            if verbose:
                print(
                    f"Reached max iter {self.n_iters}. Final nLL = {nLL:.3f}\n"
                )

        # save results
        self.m_ = m  # cluster means
        self.r_ = r  # cluster mean resultant vectors
        self.p_ = p  # cluster probabilities
        self.kappa_ = kappa  # cluster kappas
        self.gamma_ = self.compute_gamma(
            alpha=self.alpha, p=p, m=m, kappa=kappa
        )  # update gamma one last time
        self.labels_ = self.gamma_.argmax(axis=0)

    def compute_gamma(
        self,
        alpha: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
        kappa: np.ndarray,
    )-> np.ndarray:
        """
        Computes posterior probabilities (responsibilities) for each cluster.

        Returns
        -------
        np.ndarray
            Cluster assignment probabilities for each data point.
        """
        gamma = np.vstack(
            [
                p[i] * vonmises.pdf(alpha, kappa=kappa[i], mu=m[i])
                for i in range(self.n_clusters)
            ]
        )
        return gamma

    def compute_nLL(self, gamma: np.ndarray)-> float:
        """
        Computes the negative log-likelihood.

        Parameters
        ----------
        gamma : np.ndarray
            The responsibility matrix (posterior probabilities of clusters for each data point).

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        nLL = -np.sum(np.log(np.sum(gamma, axis=0) + 1e-16))
        return nLL

    def compute_BIC(self)-> float:
        """
        Computes the Bayesian Information Criterion (BIC) for model selection.

        Returns
        -------
        float
            The computed BIC value.
        """
        nLL = self.compute_nLL(self.gamma_)
        nparams = self.n_clusters * 3 - 1  # n_means + n_kappas + (n_ps - 1)
        bic = 2 * nLL + np.log(self.n) * nparams

        return bic

    def predict_density(
        self,
        x: Optional[np.ndarray] = None,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    )-> np.ndarray:
        """
        Predicts density estimates for given points.

        Parameters
        ----------
        x : np.ndarray, optional
            Points at which to estimate the density.
        unit : {"degree", "radian"}, optional
            Specifies whether input data is in degrees or radians.
        full_cycle : int, optional
            Number of intervals for data conversion.

        Returns
        -------
        np.ndarray
            Estimated density at the provided points.
        """
        unit = self.unit if unit is None else unit
        full_cycle = self.full_cycle if full_cycle is None else full_cycle

        if x is None:
            x = np.linspace(0, 2 * np.pi, 100)

        alpha = x if unit == "radian" else data2rad(x, full_cycle)

        d = [
            self.p_[i] * vonmises.pdf(alpha, kappa=self.kappa_[i], mu=self.m_[i])
            for i in range(self.n_clusters)
        ]
        return np.sum(d, axis=0)

    def predict(self, x: np.ndarray)-> np.ndarray:
        """
        Predicts cluster assignments for new data.

        Parameters
        ----------
        x : np.ndarray
            New data points in degrees or radians.

        Returns
        -------
        np.ndarray
            Predicted cluster labels.
        """
        alpha = x if self.unit == "radian" else data2rad(x, self.full_cycle)

        gamma = self.compute_gamma(
            alpha=alpha, p=self.p_, m=self.m_, kappa=self.kappa_
        )

        return gamma.argmax(axis=0)


class CircHAC:
    """
    Hierarchical agglomerative clustering for circular (1D) data,
    with optional dendrogram tracking.

    Each merge is recorded: (clusterA, clusterB, distance, new_cluster_size).

    This is a "center-merge" approach: each cluster is represented by its
    circular mean, and we merge the two clusters with the smallest
    *absolute* circular difference in means (using circ_dist).
    The merges form a dendrogram we can plot or output.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters desired.
    n_init_clusters : int or None, default=None
        If None, every point starts as its own cluster (default HAC).
        If a number, `CircKMeans` is used to pre-cluster data before HAC.
    unit : {"radian", "degree"}, default="degree"
        If "degree", data is converted to radians internally.
    full_cycle : int, default=360
        For data conversion if unit="degree".
    metric : {"center", "geodesic", "angularseparation", "chord"}, default="center"
        The distance metric used to measure the difference between cluster centers.
        We'll take its absolute value so that it's a nonnegative distance.
    random_seed : int, optional
        Not used by default, but if you add any random steps, you may set it here.

    Attributes
    ----------
    centers_ : np.ndarray, shape (k,)
        Final cluster center angles (in radians).
    r_ : np.ndarray, shape (k,)
        Resultant vector length for each cluster.
    labels_ : np.ndarray, shape (n_samples,)
        Cluster assignment for each data point, in {0, ..., k-1}.
    merges_ : np.ndarray, shape (m, 4)
        Dendrogram merge history:
        - merges_[step, 0] = ID of cluster A
        - merges_[step, 1] = ID of cluster B
        - merges_[step, 2] = distance used to merge
        - merges_[step, 3] = new cluster size after merge
        Note: these cluster IDs are the "old" ones, not necessarily 0..(k-1) at each step.
    """

    def __init__(
        self,
        n_clusters=2,
        n_init_clusters=None, 
        unit="degree",
        full_cycle=360,
        metric="center",
        random_seed=None
    ):
        self.n_clusters = n_clusters
        self.n_init_clusters = n_init_clusters
        self.unit = unit
        self.full_cycle = full_cycle
        self.metric = metric
        self.random_seed = random_seed

        self.centers_ = None
        self.r_ = None
        self.labels_ = None
        self.merges_ = None

    def _initialize_clusters(self, X):
        """Initializes clusters using CircKMeans or default HAC."""
        n_samples = len(X)

        # Default HAC: every point is its own cluster
        if self.n_init_clusters is None or self.n_init_clusters >= n_samples:
            return np.arange(n_samples), X  # Standard HAC

        # Use CircKMeans for pre-clustering
        kmeans = CircKMeans(n_clusters=self.n_init_clusters, unit="radian", metric=self.metric, random_seed=self.random_seed)
        kmeans.fit(X)
        
        init_labels = kmeans.labels_
        init_centers = kmeans.centers_
        
        return init_labels, init_centers

    def fit(self, X):
        """
        Perform agglomerative clustering on `X`.

        Parameters
        ----------
        X : np.ndarray
            Input angles in degrees or radians.

        Returns
        -------
        self : CircHAC
        """
        self.data = X = np.asarray(X)
        if self.unit == "degree":
            self.alpha = alpha = data2rad(X, k=self.full_cycle)
        else:
            self.alpha = alpha = X

        n = len(alpha)
        if n <= self.n_clusters:
            self.labels_ = np.arange(n)
            self.centers_ = alpha.copy()
            self.r_ = np.ones(n)
            self.merges_ = np.empty((0, 4))
            return self

        # Step 1: Initialize with pre-clustering or start from scratch
        cluster_ids, cluster_means = self._initialize_clusters(alpha)
        cluster_sizes = np.ones(len(cluster_means), dtype=int)

        merges = []  # Track merge history

        while len(np.unique(cluster_ids)) > self.n_clusters:
            # Compute cluster means
            unique_clusters = np.unique(cluster_ids)
            cluster_means_dict = {c: cluster_means[c] for c in unique_clusters}

            # Find best pair to merge
            best_dist = np.inf
            best_i, best_j = None, None
            for i in unique_clusters:
                for j in unique_clusters:
                    if j <= i:
                        continue
                    dist_ij = circ_dist(cluster_means_dict[i], cluster_means_dict[j], metric=self.metric)
                    if dist_ij < best_dist:
                        best_dist = dist_ij
                        best_i, best_j = i, j

            if best_i is None or best_j is None:
                break  # No valid merge found

            # Record merge
            new_size = cluster_sizes[best_i] + cluster_sizes[best_j]
            merges.append([best_i, best_j, best_dist, new_size])

            # Merge clusters
            cluster_ids[cluster_ids == best_j] = best_i
            cluster_sizes[best_i] = new_size
            cluster_means[best_i] = circ_mean_and_r(alpha[cluster_ids == best_i])[0]

        # Assign final cluster labels
        unique_ids = np.unique(cluster_ids)
        label_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
        self.labels_ = np.array([label_map[c] for c in cluster_ids], dtype=int)

        # Compute final cluster centers and resultant lengths
        k = len(unique_ids)
        self.centers_ = np.zeros(k, dtype=float)
        self.r_ = np.zeros(k, dtype=float)
        for i in range(k):
            subset = alpha[self.labels_ == i]
            mean_i, r_i = circ_mean_and_r(subset)
            self.centers_[i] = mean_i
            self.r_[i] = r_i

        # Store merges
        self.merges_ = np.array(merges, dtype=object)

    def predict(self, alpha):
        """
        Assign new angles to the closest cluster center.

        Parameters
        ----------
        alpha : array-like of shape (n_samples,)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
        """
        alpha = np.asarray(alpha)
        if self.unit == "degree":
            alpha = data2rad(alpha, k=self.full_cycle)
        else:
            alpha = alpha

        n_samples = len(alpha)
        k = len(self.centers_)
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            a_i = alpha[i]
            # measure distance to each center
            best_c, best_d = None, np.inf
            for c in range(k):
                dist_ic = circ_dist(a_i, self.centers_[c], metric=self.metric)
                dval = float(abs(dist_ic))
                if dval < best_d:
                    best_d = dval
                    best_c = c
            labels[i] = best_c
        return labels

    def plot_dendrogram(self, ax=None, **kwargs):
        """
        Plot a rudimentary dendrogram from merges_.

        This is a basic approach that uses cluster IDs directly as "labels"
        on the x-axis. Because cluster IDs might not be contiguous or in ascending
        order, the result can look jumbled. A more sophisticated approach
        would re-compute a consistent labeling for each step.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            If None, create a new figure/axes.
        **kwargs : dict
            Passed along to ax.plot(), e.g. color, linewidth, etc.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        merges = self.merges_
        if merges.size == 0:
            ax.set_title("No merges recorded (maybe n <= n_clusters?).")
            return ax

        # merges_ is (step, 4): [clusterA, clusterB, dist, new_size]
        # We want to plot something like a dendrogram:
        #  - each row is a merge event
        #  - x-axis might show cluster A and cluster B, y the 'distance'
        # But cluster IDs might keep re-labelling, so a quick hack is we show them as is.

        for step, (ca, cb, distval, new_size) in enumerate(merges):
            # We'll draw a "u" connecting ca and cb at height distval
            # Then the newly formed cluster could get ID=cb or something
            # This is a naive approach that won't produce a fancy SciPy-like dendrogram
            # but enough to illustrate what's happening.

            x1, x2 = ca, cb
            y = distval
            # a line from (x1, 0) to (x1, y), from (x2, 0) to (x2, y),
            # then a horizontal line across at y
            # we can color them or style them with kwargs

            ax.plot([x1, x1], [0, y], **kwargs)
            ax.plot([x2, x2], [0, y], **kwargs)
            ax.plot([x1, x2], [y, y], **kwargs)

        ax.set_title("Rudimentary Dendrogram")
        ax.set_xlabel("Cluster ID (raw internal IDs)")
        ax.set_ylabel("Distance")
        return ax


    def silhouette_score(self):
        """
        Compute the average silhouette for a cluster assignment on circular data.
        
        angles: np.ndarray shape (n,) in radians
        labels: np.ndarray shape (n,) in {0,1,...,K-1}
        metric: "chord", "geodesic", "center", etc.

        Returns
        -------
        float
            The mean silhouette over all points.
        """
        angles = self.alpha
        labels = self.labels_
        metric = self.metric
        n = len(angles)
        if n < 2:
            return 0.0

        silhouette_values = np.zeros(n, dtype=float)

        # Precompute all pairwise distances
        # shape => (n,n)
        pairwise = circ_dist(angles[:,None], angles[None,:], metric=metric)
        pairwise = np.abs(pairwise)  # ensure nonnegative

        for i in range(n):
            c_i = labels[i]
            # points in cluster c_i
            in_cluster_i = (labels == c_i)
            # average distance to own cluster
            # excluding the point itself
            a_i = pairwise[i, in_cluster_i].mean() if in_cluster_i.sum() > 1 else 0.0

            # find min average distance to another cluster
            b_i = np.inf
            for c_other in np.unique(labels):
                if c_other == c_i:
                    continue
                in_other = (labels == c_other)
                dist_i_other = pairwise[i, in_other].mean()
                if dist_i_other < b_i:
                    b_i = dist_i_other

            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

        return silhouette_values.mean()
    

class CircKMeans:
    """
    K-Means clustering for circular (1D) data.

    This is analogous to standard K-Means, but uses circular
    distance and circular means. The algorithm is:

    1) Initialize cluster centers (angles in radians).
    2) Assignment step:
       Assign each data point to the cluster with the minimal
       circular distance.
    3) Update step:
       Recompute each cluster center as the circular mean of
       the assigned points.
    4) Repeat until convergence or max_iters.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to form.
    max_iter : int, default=100
        Maximum number of iterations.
    metric : {"center", "chord", "geodesic", "angularseparation"}, default="chord"
        The distance measure used for assignment.
    unit : {"degree","radian"}, default="degree"
        Whether input data is in degrees or radians.
        If "degree", we convert to radians internally.
    full_cycle : int, default=360
        For data conversion if unit="degree".
    tol : float, default=1e-6
        Convergence threshold. If centers move less than `tol` in total,
        the algorithm stops.
    random_seed : int, default=None
        For reproducible initialization.

    Attributes
    ----------
    centers_ : np.ndarray, shape (n_clusters,)
        The final cluster center angles (in radians).
    labels_ : np.ndarray, shape (n_samples,)
        The assigned cluster indices for each data point.
    inertia_ : float
        The final sum of distances (or sum of squared distances) if you prefer,
        from each point to its cluster center. By default, we store
        sum of chosen distance measure.
    """

    def __init__(
        self,
        n_clusters=2,
        max_iter=100,
        metric="center",
        unit="degree",
        full_cycle=360,
        tol=1e-6,
        random_seed=None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.unit = unit
        self.full_cycle = full_cycle
        self.tol = tol
        self.random_seed = random_seed

        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Fit the K-means on 1D circular data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Angles in degrees (if self.unit=="degree") or radians.

        Returns
        -------
        self
        """
        self.data = X = np.asarray(X)
        if self.unit == "degree":
            self.alpha = alpha = data2rad(X, k=self.full_cycle)
        else:
            self.alpha = alpha = X

        rng = np.random.default_rng(self.random_seed)

        n_samples = len(alpha)
        if n_samples < self.n_clusters:
            # trivial: each point is its own cluster
            self.labels_ = np.arange(n_samples)
            self.centers_ = alpha.copy()
            self.inertia_ = 0.0
            return self

        # 1) initialize cluster centers by picking random points from data
        init_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = alpha[init_indices]

        labels = np.zeros(n_samples, dtype=int)
        for iteration in range(self.max_iter):
            # 2) assignment step
            dist_mat = np.zeros((self.n_clusters, n_samples))
            for c in range(self.n_clusters):
                # measure distance from alpha to center[c]
                dist_mat[c] = np.abs(circ_dist(alpha, centers[c], metric=self.metric))

            labels_new = dist_mat.argmin(axis=0)

            # 3) update step
            new_centers = np.zeros_like(centers)
            for c in range(self.n_clusters):
                mask = (labels_new == c)
                if np.any(mask):
                    # circular mean of assigned points
                    m, _ = circ_mean_and_r(alpha[mask])
                    new_centers[c] = m
                else:
                    # if no points assigned, keep old center or random re-init
                    new_centers[c] = centers[c]

            # check for shift
            shift = np.sum(np.abs(np.angle(np.exp(1j*centers) / np.exp(1j*new_centers))))
            # or a simpler approach: sum of circ_dist(centers, new_centers)
            # shift = float(np.sum(np.abs(circ_dist(centers, new_centers, metric=self.metric))))

            labels = labels_new
            centers = new_centers

            if shift < self.tol:
                break

        # final
        self.centers_ = centers
        self.labels_ = labels

        # compute final inertia => sum of distances from points to assigned center
        total_dist = 0.0
        for c in range(self.n_clusters):
            mask = (labels == c)
            if np.any(mask):
                dvals = np.abs(circ_dist(alpha[mask], centers[c], metric=self.metric))
                total_dist += dvals.sum()
        self.inertia_ = total_dist

    def predict(self, X):
        """
        Predict cluster assignment for new data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,)
        """
        X = np.asarray(X)
        if self.unit == "degree":
            alpha = data2rad(X, k=self.full_cycle)
        else:
            alpha = X

        n_samples = len(alpha)
        labels = np.zeros(n_samples, dtype=int)
        if self.centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        dist_mat = np.zeros((self.n_clusters, n_samples))
        for c in range(self.n_clusters):
            dist_mat[c] = np.abs(circ_dist(alpha, self.centers_[c], metric=self.metric))
        labels = dist_mat.argmin(axis=0)
        return labels
