from typing import Union

import numpy as np
from scipy.stats import vonmises

from .descriptive import circ_kappa, circ_mean
from .utils import data2rad


class MoVM:

    """
    Mixture of von Mises Clustering
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
        self.burnin = burnin  # wait untill burinin step of iterations for convergence
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
    ):
        n = len(x)  # number of samples
        p = np.ones(n_clusters_init) / n_clusters_init  # initial cluster probability
        z = np.random.choice(np.arange(n_clusters_init), size=n)  # initial labels
        m, r = map(
            np.array,
            zip(*[circ_mean(x[z == i], return_r=True) for i in range(n_clusters_init)]),
        )  # initial means and resultant vector lengths
        kappa = np.array(
            [circ_kappa(r=r[i]) for i in range(n_clusters_init)]
        )  # initial kappa (without correction by hard-coding a larger enough n)

        return m, kappa, p

    def fit(self, x: np.ndarray, verbose: Union[bool, int] = 0):
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
            p = np.sum(gamma_normed, axis=1) / np.sum(gamma_normed, axis=1).sum()
            m, r = map(
                np.array,
                zip(
                    *[
                        circ_mean(alpha=x_rad, w=gamma_normed[i], return_r=True)
                        for i in range(self.n_clusters)
                    ]
                ),
            )
            kappa = np.array([circ_kappa(r=r[i]) for i in range(self.n_clusters)])

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
                print(f"Reached max iter {self.n_iters}. Final nLL = {nLL:.3f}\n")

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
    ):
        gamma = np.vstack(
            [
                p[i] * vonmises.pdf(x_rad, kappa=kappa[i], loc=m[i])
                for i in range(self.n_clusters)
            ]
        )
        return gamma

    def compute_nLL(self, gamma: np.ndarray):
        nLL = -np.sum(np.log(np.sum(gamma, axis=0) + 1e-16))
        return nLL

    def compute_BIC(self):
        nLL = self.compute_nLL(self.gamma)
        nparams = self.n_clusters * 3 - 1  # n_means + n_kappas + (n_ps - 1)
        bic = 2 * nLL + np.log(self.n) * nparams

        return bic

    def predict_density(
        self,
        x: np.ndarray = None,
        unit: Union[str, None] = None,
        n_intervals: Union[float, int, None] = None,
    ):
        unit = self.unit if unit is None else unit
        n_intervals = self.n_intervals if n_intervals is None else n_intervals

        if x is None:
            x = np.linspace(0, 2 * np.pi, 100)

        x_rad = x if unit == "radian" else data2rad(x, n_intervals)

        d = [
            self.p[i] * vonmises.pdf(x_rad, kappa=self.kappa[i], loc=self.m[i])
            for i in range(self.n_clusters)
        ]
        return np.sum(d, axis=0)

    def predict(self, x: np.ndarray):
        x_rad = x if self.unit == "radian" else data2rad(x, self.n_intervals)

        gamma = self.compute_gamma(x_rad=x_rad, p=self.p, m=self.m, kappa=self.kappa)

        return gamma.argmax(axis=0)
