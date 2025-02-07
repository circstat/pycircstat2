from typing import Optional, Union

import numpy as np

from .descriptive import circ_kappa, circ_mean_and_r
from .distributions import vonmises
from .utils import data2rad


class MoVM:
    """
    Mixture of von Mises (MoVM) Clustering.

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
    n_intervals : int, default=360
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
        from pycircstat2.clustering import MoVM
        np.random.seed(42)
        x1 = np.random.vonmises(mu=0, kappa=5, size=100)
        x2 = np.random.vonmises(mu=np.pi, kappa=10, size=100)
        x = np.concatenate([x1, x2])
        np.random.shuffle(x)
        movm = MoVM(n_clusters=2, n_iters=200, unit="radian", random_seed=42)
        movm.fit(x, verbose=False)


    """

    def __init__(
        self,
        burnin: int = 30,
        n_clusters: int = 5,
        n_iters: int = 100,
        n_intervals: int = 360,
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
        self.n_intervals = n_intervals  # for data conversion
        self.unit = unit  # for data conversion
        self.random_seed = random_seed
        self.converged = False  # place holder

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

    def fit(self, x: np.ndarray, verbose: Union[bool, int] = 0):
        """
        Fits the mixture of von Mises model to the given data using the EM algorithm.

        Parameters
        ----------
        x : np.ndarray
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
        self.x = x
        self.x_rad = x_rad = (
            x if self.unit == "radian" else data2rad(x, self.n_intervals)
        )
        self.n = n = len(x)

        # init
        m, kappa, p = self._initialize(x_rad, self.n_clusters)

        # EM
        if verbose:
            print(f"Iter".ljust(10) + f"nLL")
        self.nLL = np.ones(self.n_iters) * np.nan
        for i in range(self.n_iters):
            # E step
            gamma = self.compute_gamma(x_rad=self.x_rad, p=p, m=m, kappa=kappa)
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
                        circ_mean_and_r(alpha=x_rad, w=gamma_normed[i])
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
        self.m = m  # cluster means
        self.r = r  # cluster mean resultant vectors
        self.p = p  # cluster probabilities
        self.kappa = kappa  # cluster kappas
        self.gamma = self.compute_gamma(
            x_rad=self.x_rad, p=p, m=m, kappa=kappa
        )  # update gamma one last time
        self.labels = self.gamma.argmax(axis=0)

    def compute_gamma(
        self,
        x_rad: np.ndarray,
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
                p[i] * vonmises.pdf(x_rad, kappa=kappa[i], mu=m[i])
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
        nLL = self.compute_nLL(self.gamma)
        nparams = self.n_clusters * 3 - 1  # n_means + n_kappas + (n_ps - 1)
        bic = 2 * nLL + np.log(self.n) * nparams

        return bic

    def predict_density(
        self,
        x: Optional[np.ndarray] = None,
        unit: Union[str, None] = None,
        n_intervals: Union[float, int, None] = None,
    )-> np.ndarray:
        """
        Predicts density estimates for given points.

        Parameters
        ----------
        x : np.ndarray, optional
            Points at which to estimate the density.
        unit : {"degree", "radian"}, optional
            Specifies whether input data is in degrees or radians.
        n_intervals : int, optional
            Number of intervals for data conversion.

        Returns
        -------
        np.ndarray
            Estimated density at the provided points.
        """
        unit = self.unit if unit is None else unit
        n_intervals = self.n_intervals if n_intervals is None else n_intervals

        if x is None:
            x = np.linspace(0, 2 * np.pi, 100)

        x_rad = x if unit == "radian" else data2rad(x, n_intervals)

        d = [
            self.p[i] * vonmises.pdf(x_rad, kappa=self.kappa[i], mu=self.m[i])
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
        x_rad = x if self.unit == "radian" else data2rad(x, self.n_intervals)

        gamma = self.compute_gamma(
            x_rad=x_rad, p=self.p, m=self.m, kappa=self.kappa
        )

        return gamma.argmax(axis=0)
