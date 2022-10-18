import numpy as np
from scipy.stats import vonmises

from .descriptive import circ_kappa, circ_mean
from .utils import data2rad


class MoVM:

    """
    Mixture of von Mises Clustering
    """

    def __init__(self):
        pass

    def _initialize(
        self,
        x,
        n_clusters_init: int = 5,
    ):

        n = len(x)
        p = np.ones(n_clusters_init) / n_clusters_init
        l = np.random.choice(np.arange(n_clusters_init), size=n)
        m, r = map(
            np.array, zip(*[circ_mean(x[l == i]) for i in range(n_clusters_init)])
        )
        kappa = np.array([circ_kappa(r=r[i], n=n) for i in range(n_clusters_init)])

        return m, kappa, p

    def fit(
        self,
        x: np.ndarray,
        n_clusters: int = 5,
        n_intervals: int = 360,
        unit: str = "degree",
        max_iters: int = 100,
        random_seed: int = 2046,
    ):

        # seed
        np.random.seed(random_seed)

        # meta
        self.x = x
        if unit != "radian":
            self.x_rad = x_rad = data2rad(x, n_intervals)
        else:
            self.x_rad = x_rad = x

        self.n = n = len(x)
        self.n_clusters = k = n_clusters
        self.max_iters = max_iters

        m, kappa, p = self._initialize(x_rad, n_clusters)

        for i in range(max_iters):

            # E step
            gamma = np.vstack(
                [p[i] * vonmises.pdf(x_rad, kappa=kappa[i], loc=m[i]) for i in range(k)]
            )
            gamma = gamma / np.sum(gamma, axis=0)

            # M step
            p = np.sum(gamma, axis=1) / np.sum(gamma, axis=1).sum()
            m, r = map(
                np.array, zip(*[circ_mean(alpha=x_rad, w=gamma[i]) for i in range(k)])
            )
            kappa = np.array([circ_kappa(r=r[i], n=n) for i in range(k)])

        # save results
        self.m = m  # cluster means
        self.r = r  # cluster mean resultant vectors
        self.p = p  # cluster probabilities
        self.kappa = kappa  # cluster kappa
        self.gamma = gamma
        self.labels = gamma.argmax(axis=0)

    def BIC(self):

        gamma = np.vstack(
            [
                self.p[i] * vonmises.pdf(self.x_rad, kappa=self.kappa[i], loc=self.m[i])
                for i in range(self.n_clusters)
            ]
        )
        LL = np.sum(np.log(np.sum(gamma, axis=0)))

        nparams = self.n_clusters * 3 - 1  # n_means + n_kappas + (n_ps - 1)
        bic = -2 * LL + np.log(self.n) * nparams

        return bic, LL
