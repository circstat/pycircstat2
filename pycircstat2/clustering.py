from __future__ import annotations

import copy
import inspect
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
from hea.R.rng import RMersenneTwister
from scipy.special import logsumexp

from .descriptive import circ_dist, circ_kappa, circ_mean_and_r
from .distributions import CircularContinuous, CircularLL, katojones, vmlss, vonmises
from .regression import (
    _SMOOTH_RE,
    _center_ref,
    _circ_sd_quad,
    _lines_circular,
    _resolve_gam_family,
    _surface_maps,
    _tile_k,
    _to_02pi,
    _to_polars,
    _unwrap_runs,
    _wrap,
    circ_gam,
)
from .utils import data2rad

ALLOWED_MOCD_DISTRIBUTIONS = {
    "cardioid",
    "cartwright",
    "wrapnorm",
    "wrapcauchy",
    "vonmises",
}


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
    params_ : list of dict or None
        Per-component parameter dictionaries ({"mu", "kappa"}) populated after :meth:`fit`.

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
        random_seed: Optional[int] = 2046,
        threshold: float = 1e-16,
    ):
        if burnin < 0:
            raise ValueError("`burnin` must be non-negative.")
        if n_clusters <= 0:
            raise ValueError("`n_clusters` must be a positive integer.")
        if n_iters <= 0:
            raise ValueError("`n_iters` must be a positive integer.")
        if threshold <= 0:
            raise ValueError("`threshold` must be positive.")
        if unit not in {"degree", "radian"}:
            raise ValueError("`unit` must be either 'degree' or 'radian'.")

        self.burnin = burnin
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.full_cycle = full_cycle
        self.unit = unit
        self._rng = np.random.default_rng(random_seed)

        self.converged = False
        self.converged_iters: Optional[int] = None

        # Attributes populated after fitting (scikit-learn style trailing underscore)
        self.m_: Optional[np.ndarray] = None
        self.r_: Optional[np.ndarray] = None
        self.p_: Optional[np.ndarray] = None
        self.kappa_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.nLL: Optional[np.ndarray] = None
        self.data: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.n: Optional[int] = None
        self.params_: Optional[List[Dict[str, float]]] = None

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
        n = len(x)
        if n_clusters_init > n:
            raise ValueError(
                "Number of clusters cannot exceed number of observations during initialisation."
            )

        # Randomly assign each observation to a cluster ensuring no cluster is empty
        for _ in range(100):
            labels = self._rng.integers(n_clusters_init, size=n)
            if all(np.any(labels == c) for c in range(n_clusters_init)):
                break
        else:
            raise RuntimeError(
                "Failed to initialise clusters without empty components."
            )

        means = np.zeros(n_clusters_init, dtype=float)
        resultants = np.zeros(n_clusters_init, dtype=float)
        kappas = np.zeros(n_clusters_init, dtype=float)

        for c in range(n_clusters_init):
            subset = x[labels == c]
            m_c, r_c = circ_mean_and_r(subset)
            means[c] = m_c
            resultants[c] = r_c
            kappa_c = circ_kappa(r=r_c)
            if not np.isfinite(kappa_c):
                kappa_c = 1e-3
            kappas[c] = max(kappa_c, 1e-3)

        p = np.full(n_clusters_init, 1.0 / n_clusters_init, dtype=float)
        return means, kappas, p

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
        X = np.asarray(X, dtype=float).reshape(-1)
        if X.size == 0:
            raise ValueError("Input data must contain at least one observation.")

        alpha = X if self.unit == "radian" else data2rad(X, k=self.full_cycle)
        self.data = X
        self.alpha = alpha
        self.n = alpha.size

        means, kappa, p = self._initialize(alpha, self.n_clusters)

        if verbose:
            header = "Iter".ljust(10) + "nLL"
            print(header)

        nLL_history = np.full(self.n_iters, np.nan)

        for iteration in range(self.n_iters):
            log_responsibilities = self._log_gamma(alpha, p, means, kappa)
            log_norm = np.logaddexp.reduce(log_responsibilities, axis=0)
            gamma_normed = np.exp(log_responsibilities - log_norm)

            # M-step updates
            p = gamma_normed.sum(axis=1)
            p /= p.sum()

            means_updated = np.zeros_like(means)
            resultants = np.zeros_like(means)
            for c in range(self.n_clusters):
                weights = gamma_normed[c]
                if np.allclose(weights.sum(), 0.0):
                    means_updated[c] = means[c]
                    resultants[c] = 0.0
                else:
                    mc, rc = circ_mean_and_r(alpha, w=weights)
                    means_updated[c] = mc
                    resultants[c] = rc

            kappas = np.array([max(circ_kappa(r=rc), 1e-3) for rc in resultants])

            means, kappa = means_updated, kappas

            nLL = -np.sum(log_norm)
            nLL_history[iteration] = nLL

            if verbose and (iteration % int(verbose or 1) == 0):
                print(f"{iteration}".ljust(10) + f"{nLL:.3f}")

            if (
                iteration > self.burnin
                and np.abs(nLL_history[iteration] - nLL_history[iteration - 1])
                < self.threshold
            ):
                self.converged = True
                self.converged_iters = iteration + 1
                if verbose:
                    print(f"Converged at iter {iteration}. Final nLL = {nLL:.3f}\n")
                break
        else:
            if verbose:
                print(f"Reached max iter {self.n_iters}. Final nLL = {nLL:.3f}\n")

        self.nLL = nLL_history[~np.isnan(nLL_history)]

        self.m_ = means
        self.r_ = resultants
        self.p_ = p
        self.kappa_ = kappa
        self.params_ = [
            {"mu": float(self.m_[i]), "kappa": float(self.kappa_[i])}
            for i in range(self.n_clusters)
        ]
        log_gamma_final = self._log_gamma(alpha, p, means, kappa)
        log_norm_final = np.logaddexp.reduce(log_gamma_final, axis=0, keepdims=True)
        gamma_final = np.exp(log_gamma_final - log_norm_final)
        self.gamma_ = gamma_final
        self.labels_ = gamma_final.argmax(axis=0)
        return self

    def compute_gamma(
        self,
        alpha: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
        kappa: np.ndarray,
    ) -> np.ndarray:
        """
        Computes posterior probabilities (responsibilities) for each cluster.

        Returns
        -------
        np.ndarray
            Cluster assignment probabilities for each data point.
        """
        log_gamma = self._log_gamma(alpha, p, m, kappa)
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=0, keepdims=True)
        return gamma

    def _log_gamma(
        self,
        alpha: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
        kappa: np.ndarray,
    ) -> np.ndarray:
        log_prob = np.vstack(
            [
                np.log(p[i] + 1e-32) + vonmises.logpdf(alpha, m[i], kappa[i])
                for i in range(self.n_clusters)
            ]
        )
        return log_prob

    def compute_nLL(
        self,
        alpha: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
        kappa: np.ndarray,
    ) -> float:
        """
        Computes the negative log-likelihood.

        Parameters
        ----------
        alpha : np.ndarray
            Input data in radians.
        p : np.ndarray
            Component probabilities.
        m : np.ndarray
            Component means.
        kappa : np.ndarray
            Component concentrations.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        log_gamma = self._log_gamma(alpha, p, m, kappa)
        log_norm = np.logaddexp.reduce(log_gamma, axis=0)
        return -float(np.sum(log_norm))

    def compute_BIC(self) -> float:
        """
        Computes the Bayesian Information Criterion (BIC) for model selection.

        Returns
        -------
        float
            The computed BIC value.
        """
        if self.gamma_ is None:
            raise ValueError("Model must be fitted before computing BIC.")
        nLL = self.compute_nLL(self.alpha, self.p_, self.m_, self.kappa_)
        nparams = self.n_clusters * 3 - 1  # n_means + n_kappas + (n_ps - 1)
        bic = 2 * nLL + np.log(self.n) * nparams

        return bic

    def predict_density(
        self,
        x: Optional[np.ndarray] = None,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    ) -> np.ndarray:
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
            x = np.linspace(0, 2 * np.pi, 400, endpoint=False)
            if unit == "degree":
                x = np.rad2deg(x)
        x = np.asarray(x, dtype=float).reshape(-1)
        alpha = x if unit == "radian" else data2rad(x, k=full_cycle)

        density_components = np.array(
            [
                p_c * vonmises.pdf(alpha, mu=m_c, kappa=k_c)
                for p_c, m_c, k_c in zip(self.p_, self.m_, self.kappa_)
            ]
        )
        return density_components.sum(axis=0)

    def predict_proba(
        self,
        x: np.ndarray,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    ) -> np.ndarray:
        """
        Returns component posterior probabilities for new observations.
        """
        if self.p_ is None or self.kappa_ is None or self.m_ is None:
            raise ValueError("Model must be fitted before calling predict_proba().")

        unit = self.unit if unit is None else unit
        full_cycle = self.full_cycle if full_cycle is None else full_cycle
        x = np.asarray(x, dtype=float).reshape(-1)
        alpha = x if unit == "radian" else data2rad(x, k=full_cycle)

        log_gamma = self._log_gamma(alpha, self.p_, self.m_, self.kappa_)
        log_norm = np.logaddexp.reduce(log_gamma, axis=0, keepdims=True)
        return np.exp(log_gamma - log_norm)

    def predict(
        self,
        x: np.ndarray,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    ) -> np.ndarray:
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
        proba = self.predict_proba(x, unit=unit, full_cycle=full_cycle)
        return proba.argmax(axis=0)


class MoKJ:
    """
    Mixture of Kato–Jones (MoKJ) Clustering.

    EM algorithm for clustering circular data with a mixture of Kato–Jones
    components (Kato & Jones, 2015). Each component controls mean direction (mu),
    mean resultant length (gamma), and second-order moment magnitude/phase (rho, lam),
    thus flexibly capturing skewness and peakedness per mode.

    References
    ----------
    - Kato, S., & Jones, M.C. (2015). A tractable and interpretable four-parameter
      family of unimodal distributions on the circle. *Biometrika*, 102(1), 181–190.
    - Nagasaki, K., Kato, S., Nakanishi, W., & Jones, M.C. (2024/2025).
      Traffic count data analysis using mixtures of Kato–Jones distributions.
      *JRSS C (Applied Statistics)*. (EM for KJ mixtures; reparametrization details.)

    Parameters
    ----------
    burnin : int, default=30
        Number of initial EM iterations before checking convergence.
    n_clusters : int, default=5
        Number of Kato–Jones mixture components.
    n_iters : int, default=100
        Maximum EM iterations.
    full_cycle : int or float, default=360
        Used to convert degrees to radians when unit="degree".
    unit : {"degree", "radian"}, default="degree"
        Input unit of X.
    random_seed : int or None, default=2046
        RNG seed for initialization.
    threshold : float, default=1e-16
        Convergence threshold on |nLL_t - nLL_{t-1}|.
    mle_maxiter : int, default=500
        Max iterations for per-component weighted MLE in M-step.
    mle_ftol : float, default=1e-9
        Function tolerance for per-component weighted MLE.
    min_comp_weight : float, default=1e-6
        Minimum mixture weight; components below may be reinitialized/frozen.

    Attributes (after fit)
    ----------------------
    converged : bool
    converged_iters : Optional[int]
    nLL : np.ndarray
        Negative log-likelihood history (finite prefix).
    mu_ : np.ndarray  shape (K,)
    gamma_ : np.ndarray  shape (K,)
    rho_ : np.ndarray  shape (K,)
    lam_ : np.ndarray  shape (K,)
    p_ : np.ndarray  shape (K,)
        Mixture weights.
    gamma_resp_ : np.ndarray  shape (K, n)
        Responsibilities.
    labels_ : np.ndarray  shape (n,)
        MAP component labels.
    data : np.ndarray
        Original X as provided.
    alpha : np.ndarray
        Data in radians.
    n : int
    params_ : list of dict or None
        Per-component parameter dictionaries ({"mu", "gamma", "rho", "lam"}) after fit.
    """

    def __init__(
        self,
        burnin: int = 30,
        n_clusters: int = 5,
        n_iters: int = 100,
        full_cycle: Union[int, float] = 360,
        unit: str = "degree",
        random_seed: Optional[int] = 2046,
        threshold: float = 1e-16,
        mle_maxiter: int = 500,
        mle_ftol: float = 1e-9,
        min_comp_weight: float = 1e-6,
    ):
        if burnin < 0:
            raise ValueError("`burnin` must be non-negative.")
        if n_clusters <= 0:
            raise ValueError("`n_clusters` must be a positive integer.")
        if n_iters <= 0:
            raise ValueError("`n_iters` must be a positive integer.")
        if threshold <= 0:
            raise ValueError("`threshold` must be positive.")
        if unit not in {"degree", "radian"}:
            raise ValueError("`unit` must be either 'degree' or 'radian'.")

        self.burnin = burnin
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.full_cycle = full_cycle
        self.unit = unit
        self._rng = np.random.default_rng(random_seed)

        self.mle_maxiter = int(mle_maxiter)
        self.mle_ftol = float(mle_ftol)
        self.min_comp_weight = float(min_comp_weight)
        self._gamma_floor = 1e-4
        self._gamma_margin = 5e-4
        self._rho_margin = 5e-4
        self._constraint_margin = 5e-4
        self._s_shrink = 5e-3

        self.converged = False
        self.converged_iters: Optional[int] = None

        self.mu_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.lam_: Optional[np.ndarray] = None
        self.p_: Optional[np.ndarray] = None
        self.gamma_resp_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.nLL: Optional[np.ndarray] = None
        self.data: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.n: Optional[int] = None
        self.params_: Optional[List[Dict[str, float]]] = None

    # ---------- initialization ----------

    def _initialize(
        self,
        x_rad: np.ndarray,
        n_clusters_init: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Random-assign points to K clusters (no empty clusters), then per-cluster
        initialize KJ params via method-of-moments."""
        n = len(x_rad)
        if n_clusters_init > n:
            raise ValueError(
                "Number of clusters exceeds sample size during initialization."
            )

        labels = None
        if "CircKMeans" in globals():
            try:
                seed = int(self._rng.integers(0, 2**32 - 1))
                kmeans = CircKMeans(
                    n_clusters=n_clusters_init,
                    unit="radian",
                    metric="center",
                    random_seed=seed,
                )
                kmeans.fit(x_rad)
                labels = kmeans.labels_.astype(int, copy=True)
                if len({int(c) for c in labels}) < n_clusters_init:
                    labels = None
            except Exception:
                labels = None

        if labels is None:
            for _ in range(100):
                candidate = self._rng.integers(n_clusters_init, size=n)
                if all(np.any(candidate == c) for c in range(n_clusters_init)):
                    labels = candidate
                    break
            else:
                raise RuntimeError(
                    "Failed to initialize clusters without empty components."
                )

        mu0 = np.zeros(n_clusters_init, float)
        gamma0 = np.zeros(n_clusters_init, float)
        rho0 = np.zeros(n_clusters_init, float)
        lam0 = np.zeros(n_clusters_init, float)

        for c in range(n_clusters_init):
            subset = x_rad[labels == c]
            # Moments init (fast, robust). Your katojones.fit already wraps moments logic.
            est = katojones.fit(subset, method="moments", return_info=False)
            mu0[c], gamma0[c], rho0[c], lam0[c] = self._regularise_params(est)

        p0 = np.full(n_clusters_init, 1.0 / n_clusters_init, dtype=float)
        return mu0, gamma0, rho0, lam0, p0

    # ---------- regularisation helpers ----------

    def _constraint_value(self, gamma: float, rho: float, lam: float) -> float:
        cos_lam = np.cos(lam)
        sin_lam = np.sin(lam)
        return (rho * cos_lam - gamma) ** 2 + (rho * sin_lam) ** 2

    def _regularise_params(
        self, params: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        mu, gamma, rho, lam = params
        mu = float(np.mod(mu, 2.0 * np.pi))
        gamma = float(np.clip(gamma, self._gamma_floor, 1.0 - self._gamma_margin))
        rho = float(np.clip(rho, 0.0, 1.0 - self._rho_margin))
        lam = float(np.mod(lam, 2.0 * np.pi))

        limit = (1.0 - gamma) ** 2
        if limit <= 0.0:
            gamma = 1.0 - self._gamma_margin
            limit = (1.0 - gamma) ** 2

        if self._constraint_value(gamma, rho, lam) >= limit - self._constraint_margin:
            # steer back inside feasible disk
            s, phi = katojones._aux_from_rho_lam(gamma, rho, lam)
            s = float(np.clip(s, 0.0, 1.0 - self._s_shrink))
            s *= 1.0 - self._s_shrink
            rho, lam = katojones._rho_lam_from_aux(gamma, s, phi)
            rho = float(np.clip(rho, 0.0, 1.0 - self._rho_margin))
            lam = float(np.mod(lam, 2.0 * np.pi))

        return mu, gamma, rho, lam

    def _violates_or_degenerate(
        self, params: Tuple[float, float, float, float]
    ) -> bool:
        mu, gamma, rho, lam = params
        if not np.all(np.isfinite([mu, gamma, rho, lam])):
            return True
        if gamma <= self._gamma_floor or rho >= 1.0 - self._rho_margin:
            return True
        limit = (1.0 - gamma) ** 2
        if limit <= 0.0:
            return True
        return (
            self._constraint_value(gamma, rho, lam)
            >= limit - self._constraint_margin / 2.0
        )

    # ---------- core likelihood pieces ----------

    def _component_logpdf(
        self,
        alpha: np.ndarray,
        mu: np.ndarray,
        gamma: np.ndarray,
        rho: np.ndarray,
        lam: np.ndarray,
    ) -> np.ndarray:
        """Return array shape (K, n) of component log-densities."""
        K = mu.size
        logs = np.vstack(
            [
                katojones.logpdf(
                    alpha, mu=mu[k], gamma=gamma[k], rho=rho[k], lam=lam[k]
                )
                for k in range(K)
            ]
        )
        return logs

    def _log_gamma(self, alpha, p, mu, gamma, rho, lam) -> np.ndarray:
        """Unnormalized log-responsibilities, shape (K, n)."""
        log_mix = np.log(np.clip(p, 1e-300, None))[:, None]
        log_comp = self._component_logpdf(alpha, mu, gamma, rho, lam)
        return log_mix + log_comp

    def _nll(self, alpha, p, mu, gamma, rho, lam) -> float:
        log_gamma = self._log_gamma(alpha, p, mu, gamma, rho, lam)
        ll = np.sum(logsumexp(log_gamma, axis=0))
        return float(-ll)

    # ---------- public API ----------

    def fit(self, X: np.ndarray, verbose: Union[bool, int] = 0):
        """
        Fit the MoKJ model by EM.

        Parameters
        ----------
        X : array-like, shape (n,)
            Circular data in degrees or radians (see `unit`).
        verbose : bool or int
            If True, print progress each iteration; if int > 0, print every `verbose` iters.
        """
        X = np.asarray(X, dtype=float).reshape(-1)
        if X.size == 0:
            raise ValueError("Input data must contain at least one observation.")
        alpha = X if self.unit == "radian" else data2rad(X, k=self.full_cycle)

        self.data = X
        self.alpha = alpha
        self.n = n = alpha.size

        mu, gamma, rho, lam, p = self._initialize(alpha, self.n_clusters)

        if verbose:
            print("Iter".ljust(10) + "nLL")

        nLL_hist = np.full(self.n_iters, np.nan)
        last_nll = np.inf

        for it in range(self.n_iters):
            # E-step
            log_resp = self._log_gamma(alpha, p, mu, gamma, rho, lam)
            log_norm = logsumexp(log_resp, axis=0, keepdims=True)
            resp = np.exp(log_resp - log_norm)  # (K, n)

            # M-step: weights
            p = resp.sum(axis=1)
            p = np.clip(p, self.min_comp_weight, None)
            p /= p.sum()

            # M-step: per-component params via weighted MLE, with fallback to moments
            mu_new = np.empty_like(mu)
            gamma_new = np.empty_like(gamma)
            rho_new = np.empty_like(rho)
            lam_new = np.empty_like(lam)

            for k in range(self.n_clusters):
                w = resp[k]
                wsum = float(w.sum())

                moment_est = self._regularise_params(
                    katojones.fit(alpha, method="moments", weights=w, return_info=False)
                )

                if not np.isfinite(wsum) or wsum <= self.min_comp_weight * n:
                    # too small / degenerate: keep previous or reinit via moments
                    mu_new[k], gamma_new[k], rho_new[k], lam_new[k] = moment_est
                    continue

                # Start from current params; do weighted MLE as in the EM literature
                mle_params = None
                initial_params = self._regularise_params(
                    (mu[k], gamma[k], rho[k], lam[k])
                )
                for start_params in (initial_params, moment_est):
                    try:
                        est, _info = katojones.fit(
                            alpha,
                            method="mle",
                            weights=w,
                            initial=start_params,
                            optimizer="L-BFGS-B",
                            options={
                                "maxiter": self.mle_maxiter,
                                "ftol": self.mle_ftol,
                            },
                            return_info=True,
                        )
                        est = self._regularise_params(est)
                        if not self._violates_or_degenerate(est):
                            mle_params = est
                            break
                    except Exception:
                        continue

                if mle_params is None:
                    mle_params = moment_est

                mu_new[k], gamma_new[k], rho_new[k], lam_new[k] = mle_params

            mu, gamma, rho, lam = mu_new, gamma_new, rho_new, lam_new

            # bookkeeping
            nLL = self._nll(alpha, p, mu, gamma, rho, lam)
            nLL_hist[it] = nLL
            if verbose and (it % int(verbose or 1) == 0):
                print(f"{it}".ljust(10) + f"{nLL:.6f}")

            # convergence check
            if it > self.burnin and abs(last_nll - nLL) < self.threshold:
                self.converged = True
                self.converged_iters = it + 1
                if verbose:
                    print(f"Converged at iter {it}. Final nLL = {nLL:.6f}\n")
                break
            last_nll = nLL
        else:
            if verbose:
                print(f"Reached max iter {self.n_iters}. Final nLL = {nLL:.6f}\n")

        # Save final state
        self.nLL = nLL_hist[~np.isnan(nLL_hist)]
        self.mu_, self.gamma_, self.rho_, self.lam_ = mu, gamma, rho, lam
        self.p_ = p
        self.params_ = [
            {
                "mu": float(mu[i]),
                "gamma": float(gamma[i]),
                "rho": float(rho[i]),
                "lam": float(lam[i]),
            }
            for i in range(self.n_clusters)
        ]
        # final responsibilities & labels
        log_resp = self._log_gamma(alpha, p, mu, gamma, rho, lam)
        log_norm = logsumexp(log_resp, axis=0, keepdims=True)
        self.gamma_resp_ = np.exp(log_resp - log_norm)
        self.labels_ = self.gamma_resp_.argmax(axis=0)
        return self

    # ---------- utilities ----------

    def compute_BIC(self) -> float:
        """
        Bayesian Information Criterion for the original KJ mixture.
        Uses p = 4*K + (K-1) = 5K - 1 parameters.
        """
        if self.gamma_resp_ is None:
            raise ValueError("Model must be fitted before computing BIC.")
        nLL = self._nll(
            self.alpha, self.p_, self.mu_, self.gamma_, self.rho_, self.lam_
        )
        nparams = 5 * self.n_clusters - 1
        return 2 * nLL + np.log(self.n) * nparams

    def predict_proba(
        self,
        x: np.ndarray,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    ) -> np.ndarray:
        """
        Posterior component probabilities for new points.
        """
        if self.p_ is None:
            raise ValueError("Model must be fitted before calling predict_proba().")
        unit = self.unit if unit is None else unit
        full_cycle = self.full_cycle if full_cycle is None else full_cycle
        x = np.asarray(x, dtype=float).reshape(-1)
        alpha = x if unit == "radian" else data2rad(x, k=full_cycle)
        log_resp = self._log_gamma(
            alpha, self.p_, self.mu_, self.gamma_, self.rho_, self.lam_
        )
        log_norm = logsumexp(log_resp, axis=0, keepdims=True)
        return np.exp(log_resp - log_norm)

    def predict(
        self,
        x: np.ndarray,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    ) -> np.ndarray:
        """MAP assignments for new data."""
        return self.predict_proba(x, unit=unit, full_cycle=full_cycle).argmax(axis=0)

    def predict_density(
        self,
        x: Optional[np.ndarray] = None,
        unit: Union[str, None] = None,
        full_cycle: Union[float, int, None] = None,
    ) -> np.ndarray:
        """
        Mixture density at points x.
        """
        if self.p_ is None:
            raise ValueError("Model must be fitted before calling predict_density().")
        unit = self.unit if unit is None else unit
        full_cycle = self.full_cycle if full_cycle is None else full_cycle

        if x is None:
            x = np.linspace(0, 2 * np.pi, 400, endpoint=False)
            if unit == "degree":
                x = np.rad2deg(x)
        x = np.asarray(x, dtype=float).reshape(-1)
        alpha = x if unit == "radian" else data2rad(x, k=full_cycle)

        dens = np.zeros_like(alpha, dtype=float)
        for pc, muc, gc, rhoc, lamc in zip(
            self.p_, self.mu_, self.gamma_, self.rho_, self.lam_
        ):
            dens += pc * katojones.pdf(alpha, mu=muc, gamma=gc, rho=rhoc, lam=lamc)
        return dens


class MoCD:
    """
    Mixture of Circular Distributions (MoCD).

    This class generalises `MovM` to any circular distribution that exposes
    ``logpdf`` and ``fit`` methods accepting weighted observations.  All mixture
    components share the same distribution family (e.g. von Mises, wrapped Cauchy,
    wrapped normal, inverse Batschelet).  Users choose the underlying family and
    the EM algorithm re-estimates the component parameters and mixing weights.

    Notes
    -----
    * The current implementation assumes each component uses **the same**
      distribution.  Extending EM to support heterogeneous components
      (different families per cluster) is feasible – responsibilities are still
      well-defined – but requires bookkeeping for a potentially different set of
      parameters and optimisation routines per component.  That design is left
      for future work.
    * The supplied distribution must expose ``logpdf`` and a ``fit`` method with
      a ``weights`` keyword argument.  Most distributions in `pycircstat2`
      follow that convention.
    * Parameter order is inferred from ``distribution.shapes`` where available;
      otherwise ``param_names`` must be provided.
    * The current implementation restricts distributions to cardioid, Cartwright,
      wrapped normal (``wrapnorm``), wrapped Cauchy (``wrapcauchy``), or von Mises
      while other families are under investigation.
    """

    def __init__(
        self,
        distribution: CircularContinuous = vonmises,
        *,
        param_names: Optional[List[str]] = None,
        fit_method: Optional[Union[str, List[str], Tuple[str, ...]]] = "auto",
        fit_kwargs: Optional[Dict[str, object]] = None,
        n_clusters: int = 3,
        n_iters: int = 100,
        burnin: int = 20,
        threshold: float = 1e-6,
        unit: str = "degree",
        full_cycle: Union[int, float] = 360,
        random_seed: Optional[int] = None,
    ) -> None:
        if not isinstance(distribution, CircularContinuous):
            raise TypeError(
                "`distribution` must be an instance of CircularContinuous (e.g. vonmises)."
            )
        if n_clusters <= 0:
            raise ValueError("`n_clusters` must be positive.")
        if n_iters <= 0:
            raise ValueError("`n_iters` must be positive.")
        if burnin < 0:
            raise ValueError("`burnin` must be non-negative.")
        if threshold <= 0:
            raise ValueError("`threshold` must be positive.")
        if unit not in {"degree", "radian"}:
            raise ValueError("`unit` must be either 'degree' or 'radian'.")

        self.distribution = distribution
        distribution_name = getattr(self.distribution, "name", None)
        if not distribution_name:
            distribution_name = self.distribution.__class__.__name__
        distribution_name_key = distribution_name.lower()
        if distribution_name_key not in ALLOWED_MOCD_DISTRIBUTIONS:
            allowed = ", ".join(sorted(ALLOWED_MOCD_DISTRIBUTIONS))
            raise ValueError(
                f"`distribution` '{distribution_name}' is not currently supported by MoCD. "
                f"Allowed options: {allowed}."
            )

        self.n_clusters = int(n_clusters)
        self.n_iters = int(n_iters)
        self.burnin = int(burnin)
        self.threshold = float(threshold)
        self.unit = unit
        self.full_cycle = full_cycle
        self.fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
        self._rng = np.random.default_rng(random_seed)

        fit_signature = inspect.signature(self.distribution.fit)
        if "weights" not in fit_signature.parameters:
            raise ValueError(
                "The selected distribution does not expose a `weights=` keyword in its fit method. "
                "MoCD requires weighted fitting to perform the EM M-step."
            )

        inferred_names: List[str] = []
        if param_names is not None:
            inferred_names = list(param_names)
        else:
            shapes = getattr(self.distribution, "shapes", None)
            if shapes:
                inferred_names = [
                    name.strip() for name in shapes.split(",") if name.strip()
                ]

        if not inferred_names:
            raise ValueError(
                "`param_names` could not be inferred. Please provide the parameter order explicitly."
            )

        self.param_names = inferred_names
        if "method" in self.fit_kwargs:
            method_value = self.fit_kwargs.pop("method")
            self._method_candidates = [str(method_value).lower()]
        else:
            self._method_candidates = self._normalise_fit_method(fit_method)

        distribution_name = distribution_name_key
        if (
            fit_method is None
            or (isinstance(fit_method, str) and fit_method.lower() == "auto")
        ) and distribution_name in {"vonmises_flattopped", "inverse_batschelet"}:
            self._method_candidates = ["mle"]

        # Model attributes populated after fitting
        self.converged: bool = False
        self.converged_iters: Optional[int] = None
        self.nLL: Optional[np.ndarray] = None
        self.p_: Optional[np.ndarray] = None
        self.params_: Optional[List[Dict[str, float]]] = None
        self.param_matrix_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.data: Optional[np.ndarray] = None
        self.n: Optional[int] = None

    def _normalise_fit_method(
        self, fit_method: Optional[Union[str, List[str], Tuple[str, ...]]]
    ) -> List[Optional[str]]:
        if fit_method is None:
            return [None]

        if isinstance(fit_method, (list, tuple)):
            if not fit_method:
                return [None]
            return [None if m is None else str(m).lower() for m in fit_method]

        method_str = str(fit_method).lower()
        if method_str == "auto":
            return ["moments", "mle"]
        return [method_str]

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([float(params[name]) for name in self.param_names], dtype=float)

    def _array_to_params(
        self, values: Union[Dict[str, float], Tuple[float, ...], List[float]]
    ) -> Dict[str, float]:
        if isinstance(values, dict):
            return {name: float(values[name]) for name in self.param_names}
        arr = np.atleast_1d(values).astype(float)
        if arr.size != len(self.param_names):
            raise ValueError(
                f"Expected {len(self.param_names)} parameters, but got {arr.size}. "
                "Please supply `param_names` matching the distribution."
            )
        return {name: float(arr[i]) for i, name in enumerate(self.param_names)}

    def _fit_component(
        self,
        alpha: np.ndarray,
        weights: np.ndarray,
        current_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        weights = np.asarray(weights, dtype=float)
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 1e-12:
            if current_params is not None:
                return current_params
            weights = np.ones_like(weights, dtype=float)
            total_weight = float(np.sum(weights))

        last_error: Optional[Exception] = None
        for method in self._method_candidates:
            fit_options = dict(self.fit_kwargs)
            if method is not None:
                fit_options.setdefault("method", method)
            fit_options["weights"] = weights

            try:
                params_est, _info = self.distribution.fit(
                    alpha, return_info=True, **fit_options
                )
            except TypeError:
                fit_options.pop("return_info", None)
                try:
                    params_est = self.distribution.fit(alpha, **fit_options)
                except Exception as exc:  # pragma: no cover
                    last_error = exc
                    continue
            except Exception as exc:
                last_error = exc
                continue

            try:
                return self._array_to_params(params_est)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                continue

        raise RuntimeError(
            "Failed to fit mixture component; attempted methods "
            f"{self._method_candidates} with last error: {last_error}"
        )

    def _initialize(
        self, alpha: np.ndarray
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        n = alpha.size
        if self.n_clusters > n:
            raise ValueError(
                "Number of clusters cannot exceed number of observations during initialisation."
            )

        for _ in range(128):
            labels = self._rng.integers(self.n_clusters, size=n)
            if all(np.any(labels == c) for c in range(self.n_clusters)):
                break
        else:
            raise RuntimeError(
                "Failed to initialise mixture components without empty clusters."
            )

        params_list: List[Dict[str, float]] = []
        for c in range(self.n_clusters):
            mask = labels == c
            count = int(mask.sum())
            params = self._fit_component(alpha[mask], np.ones(count, dtype=float))
            params_list.append(params)

        p = np.full(self.n_clusters, 1.0 / self.n_clusters, dtype=float)
        return params_list, p

    def _log_gamma(
        self,
        alpha: np.ndarray,
        p: np.ndarray,
        params_list: List[Dict[str, float]],
    ) -> np.ndarray:
        log_prob = np.vstack(
            [
                np.log(p[i] + 1e-32) + self.distribution.logpdf(alpha, **params_list[i])
                for i in range(self.n_clusters)
            ]
        )
        return log_prob

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, verbose: Union[bool, int] = 0) -> "MoCD":
        X = np.asarray(X, dtype=float).reshape(-1)
        if X.size == 0:
            raise ValueError("Input data must contain at least one observation.")

        alpha = X if self.unit == "radian" else data2rad(X, k=self.full_cycle)

        self.data = X
        self.alpha = alpha
        self.n = alpha.size

        params_list, p = self._initialize(alpha)

        if verbose:
            header = "Iter".ljust(10) + "nLL"
            print(header)

        nLL_history = np.full(self.n_iters, np.nan)

        for iteration in range(self.n_iters):
            log_resp = self._log_gamma(alpha, p, params_list)
            log_norm = logsumexp(log_resp, axis=0, keepdims=True)
            gamma_normed = np.exp(log_resp - log_norm)

            p = gamma_normed.sum(axis=1)
            p /= p.sum()

            params_updated: List[Dict[str, float]] = []
            for c in range(self.n_clusters):
                weights = gamma_normed[c]
                if np.allclose(weights.sum(), 0.0):
                    params_updated.append(params_list[c])
                    continue
                params_updated.append(
                    self._fit_component(alpha, weights, current_params=params_list[c])
                )
            params_list = params_updated

            nLL = -float(np.sum(log_norm))
            nLL_history[iteration] = nLL

            if verbose and (iteration % int(verbose or 1) == 0):
                print(f"{iteration}".ljust(10) + f"{nLL:.3f}")

            if (
                iteration > self.burnin
                and np.abs(nLL_history[iteration] - nLL_history[iteration - 1])
                < self.threshold
            ):
                self.converged = True
                self.converged_iters = iteration + 1
                if verbose:
                    print(f"Converged at iter {iteration}. Final nLL = {nLL:.3f}\n")
                break
        else:
            if verbose:
                print(
                    f"Reached max iter {self.n_iters}. Final nLL = {nLL_history[self.n_iters - 1]:.3f}\n"
                )

        self.nLL = nLL_history[~np.isnan(nLL_history)]
        self.p_ = p
        self.params_ = params_list
        self.param_matrix_ = np.vstack(
            [self._params_to_array(params) for params in params_list]
        )

        final_log = self._log_gamma(alpha, p, params_list)
        final_norm = logsumexp(final_log, axis=0, keepdims=True)
        gamma_final = np.exp(final_log - final_norm)
        self.gamma_ = gamma_final
        self.labels_ = gamma_final.argmax(axis=0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.gamma_ is None or self.p_ is None or self.params_ is None:
            raise ValueError("Model must be fitted before calling `predict_proba`.")

        X = np.asarray(X, dtype=float).reshape(-1)
        alpha = X if self.unit == "radian" else data2rad(X, k=self.full_cycle)

        log_resp = self._log_gamma(alpha, self.p_, self.params_)
        log_norm = logsumexp(log_resp, axis=0, keepdims=True)
        return np.exp(log_resp - log_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=0)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.p_ is None or self.params_ is None:
            raise ValueError("Model must be fitted before calling `score_samples`.")

        X = np.asarray(X, dtype=float).reshape(-1)
        alpha = X if self.unit == "radian" else data2rad(X, k=self.full_cycle)
        log_resp = self._log_gamma(alpha, self.p_, self.params_)
        return logsumexp(log_resp, axis=0)

    def score(self, X: np.ndarray) -> float:
        log_likelihood = self.score_samples(X)
        return float(np.mean(log_likelihood))

    def predict_density(
        self,
        X: Optional[np.ndarray] = None,
        *,
        unit: Optional[str] = None,
        full_cycle: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        if self.p_ is None or self.params_ is None:
            raise ValueError("Model must be fitted before calling `predict_density`.")

        unit = self.unit if unit is None else unit
        full_cycle = self.full_cycle if full_cycle is None else full_cycle

        if X is None:
            X = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
            if unit == "degree":
                X = np.rad2deg(X)

        X = np.asarray(X, dtype=float).reshape(-1)
        alpha = X if unit == "radian" else data2rad(X, k=full_cycle)

        pdf_components = np.vstack(
            [self.distribution.pdf(alpha, **params) for params in self.params_]
        )
        density = np.sum(self.p_[:, None] * pdf_components, axis=0)
        return density

    def bic(self) -> float:
        if self.alpha is None or self.p_ is None or self.params_ is None:
            raise ValueError("Model must be fitted before computing BIC.")
        log_likelihood = self.score_samples(self.alpha)
        nLL = -float(np.sum(log_likelihood))
        n_params_component = len(self.param_names)
        n_params_total = self.n_clusters * n_params_component + (self.n_clusters - 1)
        return 2.0 * nLL + np.log(self.n) * n_params_total

    # Aliases for compatibility with the MovM API
    def predict_density_grid(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        return self.predict_density(X)

    def compute_BIC(self) -> float:
        return self.bic()


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
        n_clusters: int = 2,
        n_init_clusters: Optional[int] = None,
        unit: str = "degree",
        full_cycle: Union[int, float] = 360,
        metric: str = "center",
        random_seed: Optional[int] = None,
    ):
        if n_clusters <= 0:
            raise ValueError("`n_clusters` must be a positive integer.")
        if n_init_clusters is not None and n_init_clusters <= 0:
            raise ValueError("`n_init_clusters` must be positive when provided.")
        if unit not in {"degree", "radian"}:
            raise ValueError("`unit` must be either 'degree' or 'radian'.")
        metric = metric.lower()
        valid_metrics = {"center", "geodesic", "angularseparation", "chord"}
        if metric not in valid_metrics:
            raise ValueError(f"`metric` must be one of {valid_metrics}.")

        self.n_clusters = n_clusters
        self.n_init_clusters = n_init_clusters
        self.unit = unit
        self.full_cycle = full_cycle
        self.metric = metric
        self._rng = np.random.default_rng(random_seed)

        self.centers_: Optional[np.ndarray] = None
        self.r_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.merges_: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.data: Optional[np.ndarray] = None

    def _initialize_clusters(self, alpha: np.ndarray) -> Dict[int, List[int]]:
        n_samples = alpha.size
        if self.n_init_clusters is None or self.n_init_clusters >= n_samples:
            return {i: [i] for i in range(n_samples)}

        # Pre-cluster using CircKMeans to obtain a manageable starting point
        seed = int(self._rng.integers(0, 2**32 - 1))
        kmeans = CircKMeans(
            n_clusters=self.n_init_clusters,
            unit="radian",
            metric=self.metric,
            random_seed=seed,
        )
        kmeans.fit(alpha)

        clusters: Dict[int, List[int]] = {}
        for cid in range(self.n_init_clusters):
            indices = np.where(kmeans.labels_ == cid)[0]
            if indices.size:
                clusters[cid] = indices.tolist()

        if not clusters:
            return {i: [i] for i in range(n_samples)}
        return clusters

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
        self.data = X = np.asarray(X, dtype=float).reshape(-1)
        if X.size == 0:
            raise ValueError("Input data must contain at least one observation.")

        alpha = X if self.unit == "radian" else data2rad(X, k=self.full_cycle)
        self.alpha = alpha

        n = alpha.size
        if n <= self.n_clusters:
            self.labels_ = np.arange(n, dtype=int)
            self.centers_ = alpha.copy()
            self.r_ = np.ones(n, dtype=float)
            self.merges_ = np.empty((0, 4), dtype=float)
            return self

        clusters = self._initialize_clusters(alpha)
        next_cluster_id = max(clusters.keys()) + 1 if clusters else 0
        merges: List[List[float]] = []

        while len(clusters) > self.n_clusters:
            means = {
                cid: circ_mean_and_r(alpha[indices])[0]
                for cid, indices in clusters.items()
            }
            cluster_ids = list(clusters.keys())

            best_dist = np.inf
            best_pair: Optional[Tuple[int, int]] = None
            for idx, cid_i in enumerate(cluster_ids):
                for cid_j in cluster_ids[idx + 1 :]:
                    dist_ij = circ_dist(means[cid_i], means[cid_j], metric=self.metric)
                    if dist_ij < best_dist:
                        best_dist = dist_ij
                        best_pair = (cid_i, cid_j)

            if best_pair is None:
                break

            cid_i, cid_j = best_pair
            merged_indices = clusters[cid_i] + clusters[cid_j]
            merges.append(
                [cid_i, cid_j, float(abs(best_dist)), float(len(merged_indices))]
            )

            del clusters[cid_i]
            del clusters[cid_j]
            clusters[next_cluster_id] = merged_indices
            next_cluster_id += 1

        final_ids = list(clusters.keys())
        labels = np.empty(n, dtype=int)
        centers = np.zeros(len(final_ids), dtype=float)
        resultants = np.zeros(len(final_ids), dtype=float)
        for new_label, cid in enumerate(final_ids):
            indices = clusters[cid]
            labels[indices] = new_label
            mean_i, r_i = circ_mean_and_r(alpha[indices])
            centers[new_label] = mean_i
            resultants[new_label] = r_i

        self.labels_ = labels
        self.centers_ = centers
        self.r_ = resultants
        self.merges_ = (
            np.array(merges, dtype=float) if merges else np.empty((0, 4), dtype=float)
        )
        return self

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
        if self.centers_ is None:
            raise ValueError("Model must be fitted before calling predict().")

        alpha = np.asarray(alpha, dtype=float)
        alpha = alpha if self.unit == "radian" else data2rad(alpha, k=self.full_cycle)

        labels = np.zeros(alpha.size, dtype=int)
        for i, angle in enumerate(alpha):
            distances = [
                abs(circ_dist(angle, center, metric=self.metric))
                for center in self.centers_
            ]
            labels[i] = int(np.argmin(distances))
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
            ca = int(ca)
            cb = int(cb)
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
        pairwise = circ_dist(angles[:, None], angles[None, :], metric=metric)
        pairwise = np.abs(pairwise)  # ensure nonnegative

        for i in range(n):
            c_i = labels[i]
            # points in cluster c_i
            in_cluster_i = labels == c_i
            # average distance to own cluster
            # excluding the point itself
            a_i = pairwise[i, in_cluster_i].mean() if in_cluster_i.sum() > 1 else 0.0

            # find min average distance to another cluster
            b_i = np.inf
            for c_other in np.unique(labels):
                if c_other == c_i:
                    continue
                in_other = labels == c_other
                dist_i_other = pairwise[i, in_other].mean()
                if dist_i_other < b_i:
                    b_i = dist_i_other

            silhouette_values[i] = (
                (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            )

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
        random_seed=None,
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
        self.data = X = np.asarray(X, dtype=float)
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
                mask = labels_new == c
                if np.any(mask):
                    # circular mean of assigned points
                    m, _ = circ_mean_and_r(alpha[mask])
                    new_centers[c] = m
                else:
                    # if no points assigned, keep old center or random re-init
                    new_centers[c] = centers[c]

            # check for shift
            shift = np.sum(
                np.abs(np.angle(np.exp(1j * centers) / np.exp(1j * new_centers)))
            )
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
            mask = labels == c
            if np.any(mask):
                dvals = np.abs(circ_dist(alpha[mask], centers[c], metric=self.metric))
                total_dist += dvals.sum()
        self.inertia_ = total_dist
        return self

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
        if self.centers_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        if self.unit == "degree":
            alpha = data2rad(X, k=self.full_cycle)
        else:
            alpha = X

        n_samples = len(alpha)
        dist_mat = np.zeros((self.n_clusters, n_samples))
        for c in range(self.n_clusters):
            dist_mat[c] = np.abs(circ_dist(alpha, self.centers_[c], metric=self.metric))
        return dist_mat.argmin(axis=0)


# =========================================================================== #
#  circ_kmeans — k-means on the circle / torus
# =========================================================================== #
# Distinct from the :class:`CircKMeans` class
# above, which is 1-D, seeds from random data points and minimises a
# ``circ_dist`` metric: this one is the *torus* Lloyd iteration whose
# dissimilarity is the summed cosine distance Σ_j {1 − cos(θ_j − μ_j)} and whose
# centres are the per-coordinate circular means, seeded by k-means++. That is
# the hard-assignment limit of a von Mises mixture with common concentration
# (Banerjee et al. 2005), which is exactly why ``circ_mix`` seeds and splits its
# components with it and not with the class.
def _mix_rng(seed: Optional[int]) -> RMersenneTwister:
    """The R-bit-exact Mersenne Twister every random draw here runs on.

    ``seed=None`` draws a fresh nondeterministic seed, so an unseeded call
    simply works and varies run to run, the way an unseeded fit rides the
    ambient global RNG stream. Pass an integer for a reproducible fit."""
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return RMersenneTwister(int(seed))


def _mix_seeds(rng: RMersenneTwister, R: int) -> np.ndarray:
    """``R`` independent restart seeds drawn from ``rng``.

    Sampled with replacement: the seeds only have to start *different* streams,
    and the collision probability over R draws is ~R²/2^32 (≈1e-5 at R = 10),
    where a collision merely repeats one random restart. Sampling without
    replacement over the 2^31 seed space is orders of magnitude slower.
    """
    return rng.sample_int(_MIX_INT_MAX, int(R), True)


def _circ_costs(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``n × K`` summed cosine distances Σ_j {1 − cos(x_ij − μ_kj)} — the torus
    dissimilarity, additive over the ``d`` angular coordinates."""
    out = np.zeros((x.shape[0], mu.shape[0]), dtype=float)
    for j in range(x.shape[1]):
        out += 1.0 - np.cos(x[:, j][:, None] - mu[:, j][None, :])
    return out


def _circ_assign(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Nearest-centre labels under the circular (cosine) distance."""
    return np.argmin(_circ_costs(x, mu), axis=1)


def _circ_centers(x: np.ndarray, cl: np.ndarray, K: int, mu: np.ndarray) -> np.ndarray:
    """Per-cluster circular-mean centres — the circular mean is the exact
    minimiser of a cluster's summed cosine distance (it maximises
    Σ_i cos(θ_i − μ)), so this update is the circular analogue of recentring on
    the arithmetic mean. An empty cluster is reseeded to the currently
    worst-explained point, so a cluster that momentarily empties does not
    collapse for good."""
    mu = np.array(mu, dtype=float, copy=True)
    for k in range(K):
        ix = np.flatnonzero(cl == k)
        if ix.size:
            for j in range(x.shape[1]):
                mu[k, j] = np.arctan2(
                    float(np.mean(np.sin(x[ix, j]))), float(np.mean(np.cos(x[ix, j])))
                )
        else:
            mu[k, :] = x[int(np.argmax(_circ_costs(x, mu).sum(axis=1)))]
    return mu


def _circ_kmeans_seed(x: np.ndarray, K: int, rng: RMersenneTwister) -> np.ndarray:
    """k-means++ seed indices (Arthur & Vassilvitskii 2007): the first centre is
    uniform, each next is drawn with probability proportional to D², the squared
    Euclidean distance to the nearest centre already chosen. In the (cos, sin)
    embedding D² = ‖e^{iθ} − e^{iμ}‖² = 2(1 − cos(θ − μ)) = 2 × the cosine
    distance :func:`_circ_costs` returns — so the correct D² weight **is** that
    cosine distance (``dmin``), NOT its square (squaring would give a D⁴
    weighting)."""
    n = x.shape[0]
    idx = np.zeros(K, dtype=int)
    idx[0] = rng.unif_index(n)
    if K == 1:
        return idx
    dmin = _circ_costs(x, x[idx[0]][None, :])[:, 0]
    for k in range(1, K):
        if float(dmin.sum()) > 0:
            idx[k] = int(rng.sample_prob(dmin, 1, False)[0])
        else:
            idx[k] = rng.unif_index(n)
        dmin = np.minimum(dmin, _circ_costs(x, x[idx[k]][None, :])[:, 0])
    return idx


def _circ_kmeans_run(
    x: np.ndarray, K: int, nstart: int, iter_max: int, rng: RMersenneTwister
) -> Dict[str, np.ndarray]:
    """One ``circ_kmeans`` fit: best of ``nstart`` k-means++ starts."""

    def one() -> Dict[str, Any]:
        mu = x[_circ_kmeans_seed(x, K, rng)]
        cl = _circ_assign(x, mu)
        for _ in range(iter_max):
            mu = _circ_centers(x, cl, K, mu)
            new = _circ_assign(x, mu)
            if np.array_equal(new, cl):
                break
            cl = new
        mu = _circ_centers(x, cl, K, mu)
        D = _circ_costs(x, mu)
        within = np.array([float(D[cl == k, k].sum()) for k in range(K)])
        return {
            "cluster": cl,
            "centers": mu,
            "withinss": within,
            "tot_withinss": float(within.sum()),
        }

    best = one()
    for _ in range(max(1, int(nstart)) - 1):
        cand = one()
        if cand["tot_withinss"] < best["tot_withinss"]:
            best = cand
    best["size"] = np.bincount(best["cluster"], minlength=K)
    return best


def circ_kmeans(
    x,
    centers: int,
    nstart: int = 5,
    iter_max: int = 50,
    random_seed: Optional[int] = None,
):
    """k-means clustering on the circle / torus.

    A Lloyd iteration whose dissimilarity is the summed cosine distance
    :math:`\\sum_j \\{1 - \\cos(\\theta_j - \\mu_j)\\}` and whose centres are the
    per-coordinate circular means
    :math:`\\mu_j = \\mathrm{atan2}(\\overline{\\sin\\theta_j},
    \\overline{\\cos\\theta_j})`. On angular data this is the right analogue of
    ordinary k-means: it respects wrap-around (θ and θ + 2π are the same point),
    which Euclidean k-means on the raw radians does not, and its centres stay
    *on* the circle rather than at the radially shrunk arithmetic mean of the
    (cos θ, sin θ) embedding.

    Each Lloyd update sets a centre to the circular mean of its members, which
    is exactly the minimiser of that cluster's summed cosine distance, so the
    alternation is coordinate descent on one objective (monotone, convergent).
    Since :math:`1 - \\cos(\\theta - \\mu) = \\tfrac12 \\|e^{i\\theta} -
    e^{i\\mu}\\|^2`, this is spherical k-means on the unit circle with centres
    projected back onto it — the hard-assignment limit of a von Mises mixture
    with common concentration (Banerjee et al., 2005), which is why it is the
    right seed for :func:`circ_mix`'s von Mises-family EM. For ``d > 1`` columns
    the distance sums over coordinates, clustering on the product of circles —
    the same torus factorisation :func:`circ_mix` uses for a joint angular
    response.

    Starts are chosen by k-means++ on the circular distance (Arthur and
    Vassilvitskii, 2007), which spreads the initial centres and makes empty
    clusters rare; the lowest total-within-cluster-distance partition over
    ``nstart`` starts is returned. The result depends on the random seed.

    Parameters
    ----------
    x : array-like
        Angles in **radians**, one row per observation and one column per
        circular coordinate (a 1-D input is one column).
    centers : int
        The number of clusters ``K``.
    nstart : int, default=5
        Number of k-means++ starts; the lowest-distance partition is kept.
    iter_max : int, default=50
        Maximum Lloyd iterations per start.
    random_seed : int, optional
        Seed for the R-bit-exact Mersenne Twister driving the k-means++ draws.

    Returns
    -------
    dict
        ``cluster`` (0-based labels), ``centers`` (``K × d``, radians on
        (−π, π]), ``withinss``, ``tot_withinss``, ``size``.

    References
    ----------
    - Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Trans. Inf.
      Theory*, 28, 129–137.
    - Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages of
      careful seeding. *SODA*, 1027–1035.
    - Banerjee, A., Dhillon, I. S., Ghosh, J. & Sra, S. (2005). Clustering on
      the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6,
      1345–1382.

    See Also
    --------
    circ_mix : uses this to seed and split its components.
    CircKMeans : the 1-D ``circ_dist``-metric class.

    Examples
    --------
        import numpy as np
        from pycircstat2.clustering import circ_kmeans
        rng = np.random.default_rng(1)
        theta = np.r_[rng.normal(0, 0.3, 50), rng.normal(np.pi, 0.3, 50)] % (2 * np.pi)
        km = circ_kmeans(theta, 2, random_seed=1)
        km["centers"]        # two mean directions, near 0 and pi
    """
    arr = np.asarray(x, dtype=float)
    # a 1-D input is ONE column (n observations), not one row
    x = arr.reshape(-1, 1) if arr.ndim == 1 else np.atleast_2d(arr)
    K = int(centers)
    n = x.shape[0]
    if K < 1:
        raise ValueError("circ_kmeans: `centers` must be a single integer >= 1.")
    if K > n:
        raise ValueError(
            f"circ_kmeans: more cluster centres ({K}) than data points ({n})."
        )
    return _circ_kmeans_run(x, K, nstart, iter_max, _mix_rng(random_seed))


def _euclid_kmeans(
    x: np.ndarray, K: int, nstart: int, iter_max: int, rng: RMersenneTwister
) -> Dict[str, Any]:
    """Ordinary (squared-Euclidean) k-means for the linear ``l~c`` leg.

    Same k-means++/Lloyd shape as :func:`_circ_kmeans_run`. Only used to
    *seed* the EM."""

    def costs(a: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return ((a[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)

    def one() -> Dict[str, Any]:
        idx = np.zeros(K, dtype=int)
        idx[0] = rng.unif_index(x.shape[0])
        dmin = costs(x, x[idx[0]][None, :])[:, 0]
        for k in range(1, K):
            idx[k] = (
                int(rng.sample_prob(dmin, 1, False)[0])
                if float(dmin.sum()) > 0
                else rng.unif_index(x.shape[0])
            )
            dmin = np.minimum(dmin, costs(x, x[idx[k]][None, :])[:, 0])
        mu = x[idx].astype(float)
        cl = np.argmin(costs(x, mu), axis=1)
        for _ in range(iter_max):
            for k in range(K):
                m = cl == k
                if m.any():
                    mu[k] = x[m].mean(axis=0)
                else:
                    mu[k] = x[int(np.argmax(costs(x, mu).min(axis=1)))]
            new = np.argmin(costs(x, mu), axis=1)
            if np.array_equal(new, cl):
                break
            cl = new
        D = costs(x, mu)
        within = np.array([float(D[cl == k, k].sum()) for k in range(K)])
        return {
            "cluster": cl,
            "centers": mu,
            "withinss": within,
            "tot_withinss": float(within.sum()),
        }

    best = one()
    for _ in range(max(1, int(nstart)) - 1):
        cand = one()
        if cand["tot_withinss"] < best["tot_withinss"]:
            best = cand
    best["size"] = np.bincount(best["cluster"], minlength=K)
    return best


def _mix_kmeans(
    x: np.ndarray, K: int, circular: bool, rng: RMersenneTwister, nstart: int = 5
) -> Dict[str, Any]:
    """Cluster a per-unit feature on the response's geometry: circular k-means
    on the circle / torus for an angular response, ordinary k-means for the
    linear ``l~c`` leg. Both return ``cluster``/``centers``, so the two call
    sites (init and split) read the same."""
    if circular:
        return _circ_kmeans_run(x, K, nstart, 50, rng)
    return _euclid_kmeans(x, K, nstart, 50, rng)


def _mix_assign(x: np.ndarray, centers: np.ndarray, circular: bool) -> np.ndarray:
    """Nearest-centre labels for ``x`` given ``centers``, on the response's
    geometry — circular cosine distance for angles, squared Euclidean for the
    linear leg. The split move uses it to label every unit by the two
    sub-centres found on the split component's members."""
    if circular:
        return _circ_assign(x, centers)
    D = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(D, axis=1)


# =========================================================================== #
#  circ_mix — formula grammar
# =========================================================================== #
# A component spec is a **string**, a list of strings (circ_gam's own
# location-scale grammar: further linear predictors of ONE response), or a list
# mixing strings and nested lists (a joint / torus density: one chain-rule
# factor per distinct left-hand side). These five helpers parse that grammar.
_MIX_VAR_RE = re.compile(r"[^\W\d_]\w*")


def _mix_formula_list(formula) -> List[Any]:
    """A spec as a list of elements (a bare string is a one-element spec)."""
    return [formula] if isinstance(formula, str) else list(formula)


def _mix_factor_response(f) -> Optional[str]:
    """The response owning one chain-rule FACTOR — a formula string, or (the
    nested case) a list of location-scale formulas whose first carries the
    left-hand side. ``None`` when the factor has no LHS (an LHS-less
    location-scale predictor of the preceding factor)."""
    if not isinstance(f, str):
        f = f[0] if len(f) else ""
    if "~" not in f:
        return None
    lhs = f.split("~", 1)[0].strip()
    m = _MIX_VAR_RE.search(lhs) if lhs else None
    return m.group(0) if m else None


def _mix_response(formula) -> Optional[str]:
    """Response name — the left-hand side of the first formula."""
    fl = _mix_formula_list(formula)
    return _mix_factor_response(fl[0]) if fl else None


def _mix_responses(formula) -> List[str]:
    """All DISTINCT responses across a spec, in chain-rule order. One element
    ⇒ a single-response cell; two or more ⇒ a joint product component."""
    out: List[str] = []
    for f in _mix_formula_list(formula):
        r = _mix_factor_response(f)
        if r is not None and r not in out:
            out.append(r)
    return out


def _mix_n_responses(formula) -> int:
    """Number of distinct responses: 1 = a single-response cell (circ_gam's own
    location-scale grammar applies); ≥ 2 = a joint product component. Elements
    after the first with no LHS share the preceding response."""
    return len(_mix_responses(formula))


def _mix_factor_specs(formula) -> List[Any]:
    """Split a spec into chain-rule FACTORS, each an ordinary ``circ_gam`` spec.

    - a single formula string        → one factor (single-response cell)
    - a list with ≤ 1 response       → one factor (circ_gam's own location-scale
                                       grammar: LSS predictors of one response)
    - a list with ≥ 2 responses      → one factor per distinct response; an
                                       element with an LHS starts a factor, any
                                       LHS-less element after it (or a nested
                                       list) supplies that factor's further
                                       location-scale predictors
    """
    if isinstance(formula, str):
        return [formula]
    if _mix_n_responses(formula) <= 1:
        return [formula]
    factors: List[Any] = []
    cur: Optional[List[str]] = None
    for f in formula:
        if not isinstance(f, str):  # a nested per-factor location-scale list
            if cur is not None:
                factors.append(cur)
                cur = None
            factors.append(list(f))
        elif _mix_factor_response(f) is not None:  # has an LHS → a new factor
            if cur is not None:
                factors.append(cur)
            cur = [f]
        else:  # LHS-less → a location-scale predictor of the current factor
            if cur is None:
                cur = []
            cur.append(f)
    if cur is not None:
        factors.append(cur)
    # unwrap a single-formula factor to a bare string (circ_gam's plain entry)
    return [fl[0] if len(fl) == 1 else fl for fl in factors]


def _mix_formula_smooth(formula) -> bool:
    """Does the spec contain a penalised smooth term — ``s()``/``te()``/``ti()``
    /``t2()``? A textual heuristic across all factors, sharing ``circ_lm``'s
    word-boundary regex (which keeps the trailing ``s(`` of ``cos(``/``sin(``
    from matching). Drives the smoothing-parameter handling and the
    monotonicity expectation."""
    txt = " ".join(
        f if isinstance(f, str) else (f[0] if len(f) else "")
        for f in _mix_formula_list(formula)
    )
    return bool(_SMOOTH_RE.search(txt))


def _mix_resp_circular(family) -> bool:
    """Response geometry is a family fact — circular iff the family is one of
    the ``CircularLL`` laws. The linear ``l~c`` leg rides hea's ``gaulss`` /
    ``gammals``, which are not ``CircularLL``."""
    return isinstance(family, CircularLL)


# =========================================================================== #
#  circ_mix — the clustering unit
# =========================================================================== #
def _mix_group_index(group, data, n: int) -> Dict[str, Any]:
    """``group=None`` ⇒ each ROW is its own unit; ``group="id"`` ⇒ the SUBJECT
    index, so a whole trajectory seats at one cluster. The integer ``grp`` maps
    each row to its unit; a rowsum over it collapses rows to units and
    ``g[grp]`` broadcasts a unit's weight back to its rows."""
    if group is None:
        return {
            "grp": np.arange(n, dtype=int),
            "n_units": n,
            "kind": "row",
            "labels": None,
        }
    if not isinstance(group, str):
        raise ValueError("`group` must name a single column, e.g. group='id'.")
    if group not in data.columns:
        raise ValueError(f"the group variable '{group}' is not a column of `data`.")
    vals = data[group].to_numpy()
    labels, grp = np.unique(vals, return_inverse=True)
    return {
        "grp": np.asarray(grp, dtype=int).ravel(),
        "n_units": int(labels.size),
        "kind": "subject",
        "labels": [str(v) for v in labels],
    }


def _mix_rowsum(x: np.ndarray, grp: np.ndarray, n_units: int) -> np.ndarray:
    """Sum a per-ROW matrix within unit → an ``n_units × K`` per-unit matrix.
    Identity (a view) when every row is its own unit."""
    if (
        x.shape[0] == n_units
        and grp.size == n_units
        and np.array_equal(grp, np.arange(n_units))
    ):
        return x
    return np.column_stack(
        [
            np.bincount(grp, weights=x[:, k], minlength=n_units)
            for k in range(x.shape[1])
        ]
    )


def _mix_unit_feature(
    row_feat: np.ndarray, grp: np.ndarray, n_units: int
) -> np.ndarray:
    """Aggregate a per-ROW feature matrix to per-UNIT means (identity when each
    row is its own unit). The unit average of a circular response's (cos, sin)
    is its mean resultant — a sensible curve summary for subject init."""
    tot = _mix_rowsum(np.asarray(row_feat, dtype=float), grp, n_units)
    cnt = np.bincount(grp, minlength=n_units).astype(float)
    return tot / cnt[:, None]


def _mix_unit_angle(theta: np.ndarray, grp: np.ndarray, n_units: int) -> np.ndarray:
    """Aggregate per-ROW angle(s) to the per-UNIT circular mean (identity for
    rows). A vector → a per-unit vector; a matrix, one column per response →
    a per-unit matrix, column-wise. The angular form of
    :func:`_mix_unit_feature`, used to seed ``circ_kmeans`` under a group."""
    th = np.asarray(theta, dtype=float)
    vec = th.ndim == 1
    if vec:
        th = th[:, None]
    s = _mix_rowsum(np.sin(th), grp, n_units)
    c = _mix_rowsum(np.cos(th), grp, n_units)
    out = np.arctan2(s, c)
    return out[:, 0] if vec else out


# =========================================================================== #
#  circ_mix — the component interface
# =========================================================================== #
# The EM loop never touches a CircGAM directly; it goes through the five
# ``_cmp_*`` accessors below. A single-response component wraps ONE weighted
# circ_gam fit. The joint component is a second shape — a PRODUCT of d weighted
# fits, one per chain-rule factor f(y) = Π_j f(y_j | parents_j) — whose joint
# log-density is the SUM of the factor log-densities. Keeping every
# per-component access behind these five is exactly what makes the joint case
# additive: the inner EM loop never learns which shape it holds.
class _MixComponent:
    """One mixture component: a single weighted :func:`circ_gam` fit."""

    __slots__ = ("fit",)

    def __init__(self, fit):
        self.fit = fit

    @property
    def fits(self) -> List[Any]:
        """The component's factor fits — one, for a single-response cell."""
        return [self.fit]


class _MixProduct(_MixComponent):
    """A joint (product) component: ``d`` weighted :func:`circ_gam` fits in
    chain-rule order, named by their response."""

    __slots__ = ("_fits", "names")

    def __init__(self, fits, names):
        super().__init__(fits[0])
        self._fits = list(fits)
        self.names = list(names)

    @property
    def fits(self) -> List[Any]:
        return self._fits


def _circ_logpdf(fit, newdata=None) -> np.ndarray:
    """The per-fit E-step density — the primitive :func:`_cmp_logpdf` calls.

    The unweighted per-observation log-density log f(yᵢ | xᵢ) of a fitted
    :func:`circ_gam`, recovered by feeding the prediction ``lpmatrix`` back
    through the family's ``ll()`` at ``deriv=0``. A family's ``l0`` is the
    unweighted per-observation log-density by construction — the
    weighted-likelihood contract scales the objective and derivative blocks by
    the prior weights but leaves ``l0`` untouched — so this is exactly the
    quantity a finite-mixture E-step needs, and (summed over factors) the
    primitive a joint product component is built from.

    Response-agnostic: it reads only ``l0``, so it works for any location-scale
    family — the circular ones (``vmlss``, ``pnlss``, …) and the
    linear-response ``gaulss``/``gammals`` carrying the ``l~c`` leg — without
    knowing which.
    """
    fam = fit.family
    if not hasattr(fam, "ll"):
        raise ValueError(
            "_circ_logpdf() needs a location-scale family carrying an ll() "
            f"method; family {getattr(fam, 'name', fam)!r} has none."
        )
    resp = _mix_factor_response(fit.formula)
    # `newdata` is in the ORIGINAL (user) frame and must be rotated into the
    # component's fit frame below. `fit.data` is the frame the model was FITTED
    # on, which circ_gam(center=) already rotated -- rotating it again would
    # double-count the centring, so the default path skips the alignment.
    supplied = newdata is not None
    newdata = _to_polars(newdata if supplied else fit.data)
    if resp not in newdata.columns:
        raise ValueError(
            f"newdata must carry the response {resp!r} for the density to be evaluated."
        )
    Xlp = fit.predict(newdata, type="lpmatrix")  # carries the per-LP column split
    y = np.asarray(newdata[resp].to_numpy(), dtype=float)
    # coef/Xlp live in the component's (possibly centred) fit frame; align the
    # original-frame response to it so the density is evaluated consistently.
    # The value is frame-independent (cos(y − μ) is unchanged), so
    # responsibilities, loglik and BIC are identical to an uncentred fit —
    # centring only changes which basin the M-step reaches. This alignment is
    # what lets a mixture's components each centre on their own weighted mode
    # and still be compared on one scale.
    ctr = float(getattr(fit, "circ_center", 0.0) or 0.0)
    if supplied and ctr and np.isfinite(ctr):
        y = _to_02pi(y - ctr)
    out = fam.ll(
        y,
        Xlp,
        np.asarray(fit.coef, dtype=float),
        np.ones(y.size),
        lpi=fit.lpi,
        deriv=0,
    )
    return np.asarray(out["l0"], dtype=float).ravel()


def _cmp_logpdf(cp: _MixComponent, newdata) -> np.ndarray:
    """Per-observation log-density log f_k(yᵢ | xᵢ) on ``newdata``. For a
    product component this is the JOINT log-density: the sum of the factor
    log-densities (chain rule)."""
    if isinstance(cp, _MixProduct):
        return np.sum([_circ_logpdf(f, newdata) for f in cp.fits], axis=0)
    return _circ_logpdf(cp.fit, newdata)


def _cmp_edf(cp: _MixComponent) -> float:
    """Effective degrees of freedom (= #coef for a parametric fit); for a
    product, summed over factors — so df = (K−1) + Σ_k Σ_j edf_kj falls out."""
    return float(sum(float(np.sum(f.edf)) for f in cp.fits))


def _cmp_coef(cp: _MixComponent):
    """The coefficient vector; for a product, the per-factor list — which is
    also the warm-start structure :func:`_mix_fit_component` consumes."""
    if isinstance(cp, _MixProduct):
        return [np.asarray(f.coef, dtype=float) for f in cp.fits]
    return np.asarray(cp.fit.coef, dtype=float)


def _cmp_predict(cp: _MixComponent, newdata, type: str = "response"):
    """Response/link prediction on ``newdata`` as an ``n × n_lp`` array; for a
    product, a per-factor list of them."""

    def one(f):
        out = f.predict(_to_polars(newdata), type=type)
        return out.to_numpy() if isinstance(out, pl.DataFrame) else np.asarray(out)

    if isinstance(cp, _MixProduct):
        return [one(f) for f in cp.fits]
    return one(cp.fit)


def _cmp_sp(cp: _MixComponent):
    """The fitted smoothing parameters — an array for a single component, a
    per-factor list for a product. The shape :func:`_mix_fit_component`'s ``sp``
    consumes: ``penalty="fixed"`` reads it from a pooled pilot fit and holds it,
    ``penalty="scheduled"`` captures it on each re-selection to hold between. A
    parametric factor has none (length 0), which fit_component passes as None."""
    if isinstance(cp, _MixProduct):
        return [np.asarray(f.sp, dtype=float).ravel() for f in cp.fits]
    return np.asarray(cp.fit.sp, dtype=float).ravel()


def _fit_obj_at(fit, coefs, w, lam=None) -> np.ndarray:
    """One fit's weighted M-step OBJECTIVE at arbitrary coefficient vectors,
    scored on its own design: Σᵢ wᵢ log f(yᵢ) minus the degeneracy guard's
    penalty at ``lam``. Lets candidate starts be priced without paying for a
    fit; the candidates share one ``lpmatrix`` build, which is the whole cost.

    The guard term is not optional book-keeping. The M-step maximises the
    PENALISED objective, so on the unpenalised scale a *different* point
    routinely outscores the fit by a few thousandths of a nat — measured at the
    default guard strength, an unpenalised comparison called the fit a failure
    on 50% of M-steps against 1.6% with the guard off.
    """
    fam = fit.family
    data = _to_polars(fit.data)
    y = np.asarray(data[_mix_factor_response(fit.formula)].to_numpy(), dtype=float)
    X = fit.predict(data, type="lpmatrix")
    jj = [np.asarray(ix, dtype=int) for ix in fit.lpi]
    w = np.asarray(w, dtype=float).ravel()
    kernels = (getattr(fam, "degen", ()) or ()) if lam else ()
    out = np.empty(len(coefs))
    for i, coef in enumerate(coefs):
        params = fam._param_values(
            fam._etas(X, np.asarray(coef, dtype=float).ravel(), jj, None)
        )
        obj = float(np.sum(w * np.asarray(fam._loglik_values(y, params)).ravel()))
        for kern in kernels:
            v = np.broadcast_to(np.asarray(params[kern.param], dtype=float), w.shape)
            obj -= float(lam) * float(np.sum(w * kern.rho0(v)))
        out[i] = obj if np.isfinite(obj) else -np.inf
    return out


def _cmp_obj_at(cp: _MixComponent, coefs, w, lam=None) -> np.ndarray:
    """A component's weighted M-step objective at each of several candidate
    coefficient sets, summed over a product component's chain-rule factors. Each
    candidate carries the layout :func:`_cmp_coef` and :func:`_mix_moment_start`
    produce."""
    prod = isinstance(cp, _MixProduct)
    total = np.zeros(len(coefs))
    for j, f in enumerate(cp.fits):
        total += _fit_obj_at(f, [c[j] if prod else c for c in coefs], w, lam)
    return total


def _mix_has_smooth(cp: _MixComponent) -> bool:
    """Does a component carry any penalised smooth (a product is "smooth" if ANY
    factor is)? Drives the EM's monotonicity expectation — parametric /
    fixed-sp EM is monotone, a live REML penalty need not be.

    Keyed on ``sp``, which is empty exactly for a penalty-free fit.
    """
    return any(np.asarray(f.sp, dtype=float).size > 0 for f in cp.fits)


def _fit_degen_penalty(fit, w, lam) -> Dict[str, float]:
    """One fit's MAP penalty λ Σᵢ wᵢ ρ(vᵢ) at its own fitted parameters, per
    guarded parameter. Mirrors ``_lss_map_penalty``'s ``l0`` term exactly — same
    ``_param_values`` seam, same kernels — so it reports the penalty the M-step
    really paid rather than a re-derivation of it."""
    fam = fit.family
    kernels = getattr(fam, "degen", ()) or ()
    if not kernels:
        return {}
    X = fit.predict(_to_polars(fit.data), type="lpmatrix")
    params = fam._param_values(fam._etas(X, np.asarray(fit.coef, dtype=float),
                                         [np.asarray(ix, dtype=int) for ix in fit.lpi],
                                         None))
    out = {}
    for kern in kernels:
        v = np.broadcast_to(np.asarray(params[kern.param], dtype=float), w.shape)
        out[kern.param] = float(lam * np.sum(w * kern.rho0(v)))
    return out


def _mix_degen_report(components, gamma, grp, control) -> Optional[dict]:
    """How hard the degeneracy guard pulled at the fitted optimum.

    ``None`` when the guard is off or the family declares no guarded parameter.
    Otherwise ``{"strength", "lambda_", "penalty", "by_param", "binding"}``:
    each component's λ_k = c/N_k, the penalty it paid in nats, the split of that
    penalty by guarded parameter, and whether it exceeds ``_MIX_DEGEN_LOUD``.

    The penalty is on the log-likelihood scale and upper-bounds nothing, but it
    tracks the unpenalised log-likelihood the M-step traded away for that
    component — which is the number a user needs, because ``loglik``/``bic`` are
    reported *without* it.
    """
    cc = float(control.degen_strength or 0.0)
    if cc <= 0 or not any(
        getattr(f.family, "degen", ()) for cp in components for f in cp.fits
    ):
        return None
    lam, pen, by_param = [], [], []
    for k, cp in enumerate(components):
        w = np.maximum(gamma[grp, k], control.wfloor)
        Nk = float(w.sum())
        lk = (cc / Nk) if Nk > 0 else 0.0
        parts: Dict[str, float] = {}
        for f in cp.fits:
            for name, val in _fit_degen_penalty(f, w, lk).items():
                parts[name] = parts.get(name, 0.0) + val
        lam.append(lk)
        by_param.append(parts)
        pen.append(float(sum(parts.values())))
    pen_arr = np.asarray(pen, dtype=float)
    return {
        "strength": cc,
        "lambda_": np.asarray(lam, dtype=float),
        "penalty": pen_arr,
        "by_param": by_param,
        "binding": pen_arr > _MIX_DEGEN_LOUD,
    }


def _mix_fit_component(
    formula,
    data,
    family,
    weights,
    start=None,
    sp=None,
    optimizer=None,
    warm=None,
    **gam_kwargs,
) -> _MixComponent:
    """Fit one component (the M-step).

    A single-response spec is one weighted :func:`circ_gam`; a multi-response
    (joint) spec is ``d`` weighted fits sharing the SAME responsibilities
    ``weights``, returned as a product component. ``start`` warm-starts each
    factor. ``sp`` FIXES the smoothing parameters (``penalty="fixed"``
    /``"scheduled"``); ``None`` (the default) lets REML re-select them, as for
    parametric components, which have none. ``optimizer`` and ``warm`` speed up
    the REML SEARCH without changing what it selects: they are injected only for
    a factor that is actually searching (its fixed ``sp`` is None) and carries
    smooths, and only when the caller did not already pass them. ``warm``
    becomes hea's ``in_out`` (a search START, not a fixed value), so the
    selected smoothness is identical — just reached in a step or two.
    """

    def norm_sp(s):
        """An empty-length sp (a factor with no smooths) is passed as None —
        nothing to hold."""
        if s is None:
            return None
        arr = np.asarray(s, dtype=float).ravel()
        return None if arr.size == 0 else arr

    # per-component centring off the tan-half wall: default on, weight-aware
    # (each component centres on its own responsibility-weighted mode),
    # overridable via circ_mix(..., center=False). The E-step (_circ_logpdf)
    # frame-aligns, so responsibilities/loglik are unchanged; only the M-step
    # basin improves.
    center = gam_kwargs.pop("center", True)

    def search_args(fixed_j, warm_j) -> Dict[str, Any]:
        """Extra gam args that warm-start / re-route the REML search for one
        factor whose fixed sp is ``fixed_j`` (None ⇒ searching)."""
        if fixed_j is not None:
            return {}  # sp fixed: no search
        a: Dict[str, Any] = {}
        if optimizer is not None and "optimizer" not in gam_kwargs:
            a["optimizer"] = optimizer
        if warm_j is not None and "in_out" not in gam_kwargs:
            w = np.asarray(warm_j, dtype=float).ravel()
            if w.size:
                a["in_out"] = {"sp": w, "scale": 1}
        return a

    def nth(seq, j):
        return seq[j] if isinstance(seq, (list, tuple)) and len(seq) > j else None

    def fit_one(spec, start_j, fx, warm_j):
        """One factor's weighted fit."""
        return circ_gam(
            spec, data, family=family, weights=weights, start=start_j,
            sp=fx, center=center, **search_args(fx, warm_j), **gam_kwargs,
        )

    specs = _mix_factor_specs(formula)
    if len(specs) == 1:
        return _MixComponent(fit_one(specs[0], start, norm_sp(sp), warm))
    fits, names = [], []
    for j, spec in enumerate(specs):
        fits.append(
            fit_one(spec, nth(start, j), norm_sp(nth(sp, j)), nth(warm, j))
        )
        names.append(_mix_factor_response(spec))
    return _MixProduct(fits, names)


def _mix_tally_fit(health, fn, *args, **kwargs) -> _MixComponent:
    """Run one weighted M-step fit, tallying it into ``health`` =
    ``[fits, stalls]``.

    The engine re-raises a convergence warning per component per EM iteration,
    and those warnings stay on stderr where the user can see them: they are the
    only live signal that the M-step is stalling, and the remaining fragility
    behind them belongs upstream, not behind a filter here. What this adds is
    the *denominator* the stderr stream cannot give — how many fits ran, so a
    handful of warnings out of hundreds of fits reads differently from a
    warning on every one. :attr:`CircMix.fit_health` reports the pair.

    ``fit.converged`` is the same signal the warning carries (the engine sets
    the flag exactly when it warns), so the tally is counted off the flag and
    needs no message matching. A throw propagates untouched, and is not tallied:
    the caller's own retry path owns that case."""
    cp = fn(*args, **kwargs)
    health[0] += 1
    health[1] += not all(bool(getattr(f, "converged", True)) for f in cp.fits)
    return cp


def _mix_moment_start(ref_cp: _MixComponent, weights, center=True):
    """A start shaped like ``ref_cp``'s coefficients but carrying THIS
    component's weighted moments, for retrying an M-step that would not start.

    Every component shares one design, so any sibling that fitted supplies the
    coefficient layout. The values come from the component's own
    responsibilities: the concentration from its weighted mean resultant, the
    location intercept from its weighted circular mean, and shape parameters
    from the symmetric reduction member.

    An all-zero start would instead mean concentration = 1 at direction 0 — the
    diffuse model, pointing at the link origin. For a *concentrated* component
    that is the worst available start, and since it is exactly the concentrated
    components whose M-step fails, a zero retry collapses them and drives EM
    downhill.

    **Frames.** The start must be expressed in the frame the retry will FIT in,
    and that frame is not the reference component's. ``ref_cp`` supplies only
    the layout; the retry re-derives its own ``_center_ref`` rotation from
    *these* weights, and that rotation is 0 whenever the weighted mean clears
    the tan-half wall — the common case. The location
    is therefore built in the original frame (undoing ``ref_cp``'s own
    ``circ_center``) and rotated into the retry's frame here. The concentration
    needs none of this: the mean resultant is rotation-invariant, so any frame
    reads it correctly.

    ``center`` mirrors :func:`_mix_fit_component`'s own argument, so a caller
    that turned centring off (or pinned it to a fixed angle) is started in the
    frame it will actually fit in.
    """
    w = np.asarray(weights, dtype=float).ravel()
    outs = []
    for f in ref_cp.fits:
        n = np.asarray(f.coef, dtype=float).size
        coef = np.zeros(n)
        fam = f.family
        dist = getattr(fam, "dist", None)
        roles = getattr(dist, "param_roles", None)
        conc = getattr(dist, "_concentration_start", None)
        lpi = getattr(f, "lpi", None)
        links = getattr(fam, "links", None)
        ok = (
            roles is not None and conc is not None and lpi is not None
            and links is not None
            and sum(1 for p in fam.params if roles[p] == "location") == 1
            and all(hasattr(lk, "link") for lk in links[: len(fam.params)])
        )
        if ok:
            resp = _mix_factor_response(f.formula)
            y = np.asarray(f.data[resp].to_numpy(), dtype=float)
            if w.size == y.size and float(w.sum()) > 0:
                sw = float(w.sum())
                cbar = float(np.sum(w * np.cos(y)) / sw)
                sbar = float(np.sum(w * np.sin(y)) / sw)
                rbar = float(np.hypot(cbar, sbar))
                # the weighted mean direction, carried back to the ORIGINAL
                # frame and then into the frame the retry will fit in (see the
                # docstring); 0 for a location link that carries no rotation.
                loc = 0.0
                j_loc = next(
                    j for j, pn in enumerate(fam.params) if roles[pn] == "location"
                )
                if links[j_loc].name == "tanhalf":
                    yo = _to_02pi(y + float(getattr(f, "circ_center", 0.0) or 0.0))
                    ref = (
                        _center_ref(yo, w)
                        if center is True
                        else (0.0 if center is False else float(center))
                    )
                    loc = float(
                        _wrap(
                            np.arctan2(sbar, cbar)
                            + (float(getattr(f, "circ_center", 0.0) or 0.0) - ref)
                        )
                    )
                vals = [
                    loc
                    if roles[p] == "location"
                    else (float(conc(rbar)) if roles[p] == "concentration" else 0.0)
                    for p in fam.params
                ]
                for j, idx in enumerate(lpi):
                    if j < len(vals) and len(idx):
                        coef[int(idx[0])] = float(links[j].link(vals[j]))
                if not np.all(np.isfinite(coef)):
                    coef = np.zeros(n)
        outs.append(coef)
    return outs if isinstance(ref_cp, _MixProduct) else outs[0]


def _mix_pooled_sp(formula, data, family, **gam_kwargs):
    """The pooled smoothing parameters for ``penalty="fixed"``: fit the
    component spec ONCE on all the data with unit weights and read its
    REML-selected ``sp``. That value is shared across the K components and held
    fixed for the whole EM, so the penalty never moves — the deterministic,
    monotone "select λ once" choice."""
    cp = _mix_fit_component(
        formula, data, family, weights=np.ones(data.height), **gam_kwargs
    )
    return _cmp_sp(cp)


# =========================================================================== #
#  circ_mix — initial responsibilities
# =========================================================================== #
# "kmeans" clusters the (per-unit) response so the components START separated —
# it breaks the symmetric saddle a weak-signal mixture would otherwise stall on.
# "random" assigns labels uniformly (the diversity seed for restarts beyond the
# first). The clustering feature is the per-unit value: the response ANGLE for a
# circular family (clustered by circ_kmeans on the circle / torus — one
# coordinate per response, so a joint spec seeds on both angles jointly) or the
# scaled value for the linear l~c leg. Under a group the per-unit value is the
# within-subject circular (or arithmetic) mean, so subjects — not rows — are
# clustered.
def _mix_init_features(formula, data, family, grp, n_units) -> Dict[str, Any]:
    circ = _mix_resp_circular(family)
    cols = []
    for r in _mix_responses(formula):
        v = np.asarray(data[r].to_numpy(), dtype=float)
        if circ:
            cols.append(_mix_unit_angle(v, grp, n_units))  # per-unit angle
        else:
            sd = float(np.std(v, ddof=1))
            z = (v - float(np.mean(v))) / (sd if sd > 0 else 1.0)
            cols.append(_mix_unit_feature(z[:, None], grp, n_units)[:, 0])
    return {"x": np.column_stack(cols), "circular": circ}  # one row per unit


def _mix_init_gamma(method, feat, K, n, rng) -> np.ndarray:
    lab = None
    if method == "kmeans" and np.unique(feat["x"], axis=0).shape[0] > K:
        try:
            lab = _mix_kmeans(feat["x"], K, feat["circular"], rng)["cluster"]
        except Exception:
            lab = None
    if lab is None:
        lab = rng.sample_replace(K, n)
    g = np.zeros((n, K), dtype=float)
    g[np.arange(n), np.asarray(lab, dtype=int)] = 1.0
    return g


# =========================================================================== #
#  circ_mix — mixture density / responsibilities
# =========================================================================== #
def _mix_loglik_rows(logmix: np.ndarray) -> np.ndarray:
    """Per-unit mixture log-density log Σ_k π_k f_k — row log-sum-exp."""
    return logsumexp(logmix, axis=1)


def _mix_responsibilities(logmix: np.ndarray) -> np.ndarray:
    """Softmax responsibilities: γ_ik ∝ π_k f_k(y_i), row-normalised."""
    m = logmix.max(axis=1, keepdims=True)
    g = np.exp(logmix - m)
    return g / g.sum(axis=1, keepdims=True)


def _mix_classify(logmix: np.ndarray) -> Dict[str, Any]:
    """Hard (CEM / DP-means) assignment of a per-unit ``logmix`` matrix: each
    unit seats wholly at its argmax component. Returns the labels ``z``, the 0/1
    responsibility matrix, and the classification log-likelihood
    Σ_u max_k (log π_k + log f_k) — the criterion hard EM ascends."""
    z = np.argmax(logmix, axis=1)
    g = np.zeros_like(logmix)
    rows = np.arange(z.size)
    g[rows, z] = 1.0
    return {"z": z, "gamma": g, "loglik": float(logmix[rows, z].sum())}


def _mix_gate(gating: Dict[str, Any], n_rows: int, K: int) -> np.ndarray:
    """Constant mixing weights, broadcast over rows."""
    if gating.get("type") == "constant":
        return np.tile(np.asarray(gating["pi"], dtype=float), (n_rows, 1))
    raise ValueError(f"unknown gating type {gating.get('type')!r}.")


def _mix_objective(state: Dict[str, Any], lam: float) -> float:
    """The penalised objective J = −2 logLik + λ·df, df = (K−1) + Σ edf.
    λ = log(#units) makes J the BIC, so "let the data decide K" == "greedily
    minimise BIC by local moves". This is the single comparator the K-search
    uses."""
    df = (state["K"] - 1) + sum(_cmp_edf(cp) for cp in state["components"])
    return -2.0 * state["loglik"] + lam * df


# =========================================================================== #
#  circ_mix — tuning parameters
# =========================================================================== #
_MIX_INT_MAX = 2**31 - 1
_MIX_PENALTIES = ("auto", "fixed", "scheduled")
_MIX_MOVES = ("split", "merge", "death", "birth")
# A component's degeneracy penalty (nats) above which the guard is called out in
# repr(): one nat is roughly the log-likelihood a single parameter buys, so past
# it the guard is shaping the reported fit rather than merely bounding it.
_MIX_DEGEN_LOUD = 1.0


@dataclass
class CircMixControl:
    """Tuning parameters for :func:`circ_mix` — the ``circ_mix.control()`` twin.

    Parameters
    ----------
    lambda_ : float, optional
        The penalty multiplier on the degrees of freedom in the model objective
        J; ``None`` uses ``log(n_units)``, making J the BIC. (Named with a
        trailing underscore because ``lambda`` is a Python keyword.)
    kmin, kmax : int
        Lower and upper bounds on K for automatic-K search (both ``"greedy"``
        and ``"grid"``).
    penalty : {"auto", "fixed", "scheduled"}
        How the per-component M-step handles the smoothing parameters of
        penalised (smooth) terms. ``"auto"`` lets REML select them every M-step,
        so each component gets its own automatically-chosen smoothness — the
        usual GAM behaviour; the trade-off is that the moving penalty makes the
        EM non-monotone for smooth components. ``"fixed"`` selects a single
        smoothness *once*, from a pooled single-component pilot fit on all the
        data, and holds it for every component and iteration — an opt-in for
        speed, a monotone EM, or robustness when ``"auto"``'s per-component
        adaptivity lets a component over-flex and absorb a neighbouring cluster.
        ``"scheduled"`` starts from that same pooled value and re-selects every
        ``sp_every`` iterations *after* the first. For parametric
        (penalty-free) components the three modes coincide.
    sp : array-like or list, optional
        Smoothing parameters to hold fixed (an array for a single-response
        component, a per-factor list for a joint one). When supplied it
        overrides the pilot fit under ``penalty="fixed"``.
    optimizer : str or tuple, default=("efs",)
        The outer optimiser for the REML smoothing-parameter search (ignored by
        ``penalty="fixed"`` and by parametric components, which run no search).
        Extended Fellner–Schall is markedly faster on these families and selects
        the same smoothness.
    start : optional
        Accepted but not yet used.
    sp_every : int, default=5
        Under ``penalty="scheduled"``, the number of EM iterations between REML
        re-selections of the smoothing parameters (held fixed in between).
    restarts : int, default=10
        Number of random-restart EM runs; the largest-log-likelihood run is
        kept.
    init : {"kmeans", "random"}
        Initialisation of the responsibilities. ``"kmeans"`` seeds restart 1 by
        clustering the response — :func:`circ_kmeans` on the circle / torus for
        an angular response, ordinary k-means for the linear ``l~c`` leg — and
        later restarts are random.
    moves : sequence of str
        The structure moves the greedy search may attempt, any of ``"split"``
        (grow), ``"merge"`` and ``"death"`` (shrink); ``"birth"`` (grow,
        redundant with split) is also accepted.
    tol, max_iter : float, int
        EM convergence tolerance (relative change in the log-likelihood) and
        the maximum number of EM iterations per run.
    min_size : int, default=5
        The soft-size floor n_k = Σ_i γ_ik below which a component is dropped by
        a death move, and the minimum members a component must have to be split.
    degen_strength : float, default=0.05
        Strength ``c`` of the size-aware degeneracy guard (``0`` turns the guard
        off, recovering the unguarded EM). A finite mixture's likelihood is
        unbounded — a component can raise it without limit by concentrating onto
        a responsibility-weighted subset (its concentration κ → ∞, or a bounded
        shape parameter driven to its singular boundary, where the Hessian blows
        up and the M-step crashes or grinds). The guard adds to each component's
        weighted M-step a MAP penalty pulling its concentration / shape toward
        the family's diffuse model with strength λ_k = c / N_k, where
        N_k = Σ_i γ_ik is the effective component size.

        **Read it as a soft cap on the concentration.** For a κ-type parameter
        the M-step solves ``A₁(κ̂) = R̄ − c/N_k`` instead of ``A₁(κ̂) = R̄``, and
        since ``A₁⁻¹(1 − x) ≈ 1/(2x)`` no component can exceed

            κ_max ≈ N_k / (2c)

        — at the default ``c = 0.05``, about ``10·N_k``, so a 12-observation
        component is capped near κ ≈ 120 and an honestly tight cluster is left
        alone, while a collapsing one is still bounded. The induced shrinkage is
        ``Δκ ≈ 2κ²c/N_k``: **quadratic in κ**, so raising ``c`` biases exactly
        the best-determined components hardest. ``c = 1`` caps at κ ≈ N_k/2,
        which clips ordinary data — prefer the default and reach for a larger
        ``c`` only against a specific, diagnosed collapse.

        The guard is *not* a component-selection device: dropping spurious
        components is the job of ``min_size``, the ``death`` move, and BIC.

        It never alters the per-observation density used by the E-step, so the
        reported mixture log-likelihood and BIC stay on the *unpenalised* data
        scale — a guarded fit therefore looks like a slightly failed MLE. See
        :attr:`CircMix.degen` for how hard the guard actually pulled. Inert for
        the linear-response legs.
    time_budget : float, optional
        Per-restart wall-clock budget in seconds (``None`` turns it off). A
        backstop behind ``degen_strength``: an EM run exceeding it is aborted
        and treated exactly like a failed restart, so no single fit can hang.
    wfloor : float, default=1e-8
        Lower bound applied to the responsibilities used as M-step prior
        weights, keeping them strictly positive without perturbing the fit.
    seed : int, optional
        Seed set once before the restarts, for reproducibility. Each restart
        then seeds itself from a stream drawn here.
    verbose : bool
        Report per-iteration progress.
    assign : {"soft", "hard"}
        Set by :func:`circ_mix` from its own ``assign=`` argument; it rides the
        control so the EM core reads it without a further threaded argument.
    """

    lambda_: Optional[float] = None
    kmin: int = 1
    kmax: int = 20
    penalty: str = "auto"
    sp: Any = None
    optimizer: Any = ("efs",)
    start: Any = None
    sp_every: int = 5
    restarts: int = 10
    init: str = "kmeans"
    moves: Sequence[str] = ("split", "merge", "death")
    tol: float = 1e-6
    max_iter: int = 200
    min_size: int = 5
    degen_strength: float = 0.05
    time_budget: Optional[float] = None
    wfloor: float = 1e-8
    seed: Optional[int] = None
    verbose: bool = False
    assign: str = "soft"

    def __post_init__(self):
        if self.init == "emEM":
            raise ValueError('init="emEM" is not supported; use "kmeans" or "random".')
        if self.init not in ("kmeans", "random"):
            raise ValueError('`init` should be one of "kmeans", "random".')
        if self.penalty not in _MIX_PENALTIES:
            raise ValueError(f"`penalty` should be one of {_MIX_PENALTIES}.")
        if self.assign not in ("soft", "hard"):
            raise ValueError('`assign` should be one of "soft", "hard".')
        if not self.optimizer or not isinstance(self.optimizer, (str, tuple, list)):
            raise ValueError(
                '`optimizer` must be an outer-optimiser name, e.g. ("efs",).'
            )
        moves = (self.moves,) if isinstance(self.moves, str) else tuple(self.moves)
        bad = [m for m in moves if m not in _MIX_MOVES]
        if bad or not moves:
            raise ValueError(f"`moves` should be a non-empty subset of {_MIX_MOVES}.")
        self.moves = moves
        d = self.degen_strength
        if not np.isscalar(d) or not np.isfinite(d) or d < 0:
            raise ValueError(
                "`degen_strength` must be a single non-negative number "
                "(0 turns the guard off)."
            )
        self.degen_strength = float(d)
        if self.time_budget is not None:
            t = self.time_budget
            if not np.isscalar(t) or not np.isfinite(t) or t <= 0:
                raise ValueError(
                    "`time_budget` must be None (off) or a single positive "
                    "number of seconds."
                )
            self.time_budget = float(t)
        self.kmin = int(self.kmin)
        self.kmax = int(self.kmax)
        self.sp_every = max(1, int(self.sp_every))
        self.restarts = int(self.restarts)
        self.max_iter = int(self.max_iter)
        self.min_size = int(self.min_size)
        self.verbose = bool(self.verbose)


def _mix_control(control) -> CircMixControl:
    """Coerce ``control=`` — a :class:`CircMixControl`, a dict of overrides, or
    ``None`` — to a validated control object."""
    if control is None:
        return CircMixControl()
    if isinstance(control, CircMixControl):
        return control
    if isinstance(control, dict):
        return CircMixControl(**control)
    raise TypeError(
        "`control` must be a CircMixControl, a dict of overrides, or None; got "
        f"{type(control).__name__}."
    )


# =========================================================================== #
#  circ_mix — the EM core
# =========================================================================== #
def _mix_em_core(
    formula, data, family, control, resp, g, grp, n_units, **gam_kwargs
) -> Dict[str, Any]:
    """One EM run from a GIVEN responsibility matrix.

    The shared inner loop — used by a fresh init (:func:`_mix_em_once`) AND by
    every structure move (which warm-starts it from the post-move gamma).
    Ordering is M-then-E so the returned state is mutually consistent:
    ``loglik`` is the observed-data log-likelihood of the returned
    (components, pi), and ``gamma`` are that state's E-step responsibilities.
    Each full E→M cycle is the standard EM ascent step, so the recorded
    log-likelihoods are monotone for parametric (penalty-free) components.
    """
    n = data.height
    K = g.shape[1]
    nu = g.shape[0]
    penalty = control.penalty or "auto"
    hard = control.assign == "hard"
    sp_every = control.sp_every
    # the outer optimiser for the REML smoothing-parameter SEARCH (penalty
    # "auto"/"scheduled"); None for a parametric spec (no search) so those fits
    # stay bit-identical to hea's default.
    opt = control.optimizer if _mix_formula_smooth(formula) else None
    # the M-step's centring setting, needed by _mix_moment_start to build a
    # retry start in the frame the retry will actually fit in.
    ctr_arg = gam_kwargs.get("center", True)
    # [fits attempted, fits the engine reported as not converged] over this run
    # -- the denominator for the engine's per-fit convergence warnings, which
    # still reach stderr unfiltered.
    health = [0, 0]

    if K == 1:
        cp = _mix_tally_fit(
            health,
            _mix_fit_component,
            formula,
            data,
            family,
            weights=np.ones(n),
            sp=control.sp,
            optimizer=(opt if control.sp is None else None),
            **gam_kwargs,
        )
        ll = float(_mix_rowsum(_cmp_logpdf(cp, data)[:, None], grp, n_units).sum())
        return {
            "n_fit": health[0],
            "n_stall": health[1],
            "components": [cp],
            "pi": np.ones(1),
            "gamma": np.ones((nu, 1)),
            "cluster": np.zeros(nu, dtype=int),
            "loglik": ll,
            "ll_path": [ll],
            "iter": 1,
            "converged": True,
            "monotone": True,
            "worst_drop": 0.0,
            "K": 1,
        }

    components: List[Any] = [None] * K
    start_k: List[Any] = [None] * K  # coefficient warm starts
    warm_k: List[Any] = [None] * K  # last selected sp per component: warm-starts
    #                                 the next REML search (hea in_out)
    # per-component smoothing parameters held between re-selections:
    #   "auto"      -> None every iter (REML re-selects each M-step)
    #   "fixed"     -> control.sp (a pooled pilot's sp), never moves -> monotone
    #   "scheduled" -> START at the pooled pilot's sp, then re-select every
    #                  sp_every iters AFTER the first
    sp_k: List[Any] = [
        control.sp if penalty in ("fixed", "scheduled") else None for _ in range(K)
    ]
    pi = g.mean(axis=0)
    prev = -np.inf
    ll_path: List[float] = []
    conv = False
    parametric: Optional[bool] = None
    worst_drop = 0.0
    ll = np.nan
    z = prev_z = None
    best_ll = -np.inf
    stall = 0
    cc = float(control.degen_strength or 0.0)
    has_degen = bool(getattr(family, "degen", ()) or ())
    # the degeneracy guard's strength lambda_k = c / N_k moves with the
    # responsibilities, so while it is live the unpenalised loglik may dip
    degen_moves = cc > 0 and has_degen
    t_start = time.monotonic()
    it = 0

    for it in range(1, control.max_iter + 1):
        # per-restart wall-clock backstop (defence in depth behind the
        # degeneracy penalty): a degenerate near-singular M-step grind is caught
        # and the restart is treated exactly like a failed one.
        if (
            control.time_budget is not None
            and time.monotonic() - t_start > control.time_budget
        ):
            raise RuntimeError(
                f"circ_mix: an EM run exceeded time_budget "
                f"({control.time_budget}s) at iter {it}; restart aborted."
            )
        reselect = {
            "auto": True,
            "fixed": False,
            "scheduled": it > 1 and it % sp_every == 0,
        }[penalty]
        # M-step: weighted component refits + mixing weights, each warm-started
        # from the previous iteration's coefficients (tracks the moving weighted
        # MLE, so the parametric / fixed-sp EM stays monotone and converges in
        # fewer iterations). A unit's responsibility is broadcast to all its rows
        # (g[grp, k]) — the subject-level weight under a group, identity for rows.
        pi = g.mean(axis=0)

        def fit_k(k, wk, fam_k, start):
            cp = _mix_tally_fit(
                health,
                _mix_fit_component,
                formula,
                data,
                fam_k,
                weights=wk,
                start=start,
                sp=None if reselect else sp_k[k],
                optimizer=opt if reselect else None,
                warm=warm_k[k] if reselect else None,
                **gam_kwargs,
            )
            if reselect:  # remember the selected sp: warm_k warm-starts the next
                spk = _cmp_sp(cp)  # search; sp_k holds it between
                warm_k[k] = spk  # scheduled re-selections
                if penalty != "auto":
                    sp_k[k] = spk
            return cp

        pending, stalled = [], []
        for k in range(K):
            wk = np.maximum(g[grp, k], control.wfloor)
            fam_k = family
            # size-aware MAP degeneracy guard: pull this component's
            # concentration / shape toward the diffuse model with strength
            # λ_k = c / N_k, N_k the effective component size. Vanishes for a
            # well-populated component, bites only as one collapses onto a thin
            # subset. Set on a per-component family COPY (the families are module
            # singletons); l0 is unpenalised, so the E-step is unchanged.
            if degen_moves:
                Nk = float(wk.sum())
                fam_k = copy.copy(family)
                fam_k.map_lambda = (cc / Nk) if Nk > 0 else None
            try:
                cp = fit_k(k, wk, fam_k, start_k[k])
            except Exception as exc:  # retried below from a neutral start
                components[k] = None
                pending.append((k, wk, fam_k, exc))
                continue
            components[k] = cp
            # Sanity-check the M-step against the component's own weighted
            # moments -- a closed-form estimate the maximiser must beat. A fit
            # that scores WORSE than that start did not maximise anything, and
            # the engine does not always say so: warm-started below the diffuse
            # concentration it can return the diffuse model exactly, reporting
            # convergence and issuing no warning. Pricing the start costs one
            # ll(deriv=0), no fit; the refit below runs only when it is beaten.
            # Two independent tells that the M-step did not maximise anything,
            # each catching failures the other misses:
            #   * the engine reports it did not converge -- a stalled fit hands
            #     back its own start, and that dead point would become the next
            #     iteration's warm start, pinning the component for the run;
            #   * it scores below the component's own weighted moments, a
            #     closed-form estimate any maximiser must beat. The engine does
            #     NOT always report this one: warm-started below the diffuse
            #     concentration it can return the diffuse model exactly, claim
            #     convergence, and issue no warning.
            # Pricing the moments costs one ll(deriv=0); the refit runs only on
            # a tell, and is kept only if it scores better.
            ms = _mix_moment_start(cp, wk, ctr_arg)
            lam_k = getattr(fam_k, "map_lambda", None)
            at_moments, at_fit = _cmp_obj_at(cp, [ms, _cmp_coef(cp)], wk, lam_k)
            if at_moments > at_fit or not all(
                bool(getattr(f, "converged", True)) for f in cp.fits
            ):
                stalled.append((k, wk, fam_k, ms, lam_k))
        # A component whose weighted M-step will not start is retried from its
        # own weighted moments, laid out by a sibling that did fit. The failing
        # fits are the CONCENTRATED components, so the retry start decides
        # whether they survive: a diffuse one collapses them and EM descends.
        if pending or stalled:
            sib = next((cp for cp in components if cp is not None), None)
            if sib is None:
                raise RuntimeError(
                    f"circ_mix: every component's M-step failed at iter {it} "
                    f"({pending[0][3]})."
                ) from pending[0][3]
            for k, wk, fam_k, _exc in pending:
                components[k] = fit_k(k, wk, fam_k, _mix_moment_start(sib, wk, ctr_arg))
            # A fit the moments beat is refitted FROM those moments. Left alone
            # it would also poison the next iteration, whose warm start is this
            # coefficient vector — that is how one bad step pins a component for
            # a whole run. The refit is kept only if it scores better, so the
            # check can never lose ground.
            for k, wk, fam_k, ms, lam_k in stalled:
                try:
                    alt = fit_k(k, wk, fam_k, ms)
                except Exception:
                    continue
                keep = components[k]
                if (
                    _cmp_obj_at(alt, [_cmp_coef(alt)], wk, lam_k)[0]
                    > _cmp_obj_at(keep, [_cmp_coef(keep)], wk, lam_k)[0]
                ):
                    components[k] = alt
        start_k = [_cmp_coef(cp) for cp in components]
        if parametric is None:  # fixed across iterations
            parametric = not any(_mix_has_smooth(cp) for cp in components)

        # E-step: per-row densities summed within unit -> per-unit log-lik and
        # responsibilities. Soft EM uses the log-sum-exp mixture log-lik and
        # softmax gamma; hard CEM seats each unit (curve) wholly at its argmax
        # and uses the classification log-lik (sum of per-unit maxima).
        Lr = np.column_stack([_cmp_logpdf(cp, data) for cp in components])
        L = _mix_rowsum(Lr, grp, n_units)
        with np.errstate(divide="ignore"):
            logmix = L + np.log(pi)[None, :]
        if hard:
            cl = _mix_classify(logmix)
            ll, g, z = cl["loglik"], cl["gamma"], cl["z"]
        else:
            ll = float(_mix_loglik_rows(logmix).sum())
            g = _mix_responsibilities(logmix)
        ll_path.append(ll)
        if control.verbose:
            pis = ", ".join(f"{p:.3f}" for p in pi)
            print(f"    iter {it:3d}  ll = {ll:.5f}  pi = ({pis})")

        if it > 1:
            worst_drop = max(worst_drop, prev - ll)  # how far ll ever fell
            if abs(ll - prev) < control.tol * (abs(prev) + control.tol):
                conv = True
                break
            if hard and prev_z is not None and np.array_equal(z, prev_z):
                conv = True  # CEM: partition stable
                break
        # stall-abort: a quadrature family whose log-likelihood wobbles at its
        # numerical noise floor can keep changing by just over `tol` forever and
        # never trip the test above, grinding to max_iter. If no NEW BEST appears
        # for 12 consecutive iterations the run has effectively converged (it is
        # oscillating around an optimum, not climbing), so stop and keep it — a
        # data-size-independent guard, unlike a wall-clock cap.
        if not np.isfinite(best_ll) or ll > best_ll + control.tol * (
            abs(best_ll) + control.tol
        ):
            best_ll = ll
            stall = 0
        else:
            stall += 1
            if stall >= 12:
                conv = True
                break
        prev = ll
        if hard:
            prev_z = z

    # monotone is EXPECTED only when NO penalty moves between iterations. Two
    # penalties can move:
    #   * the REML smoothing penalty -- still selected every M-step under
    #     penalty = "auto"/"scheduled", held under "fixed", absent for a
    #     parametric spec;
    #   * the size-aware degeneracy guard, whose strength lambda_k = c / N_k
    #     tracks the effective component size N_k and so moves whenever the
    #     responsibilities do.
    # The M-step maximises the PENALISED objective, so while either penalty is
    # live the reported (unpenalised) mixture log-likelihood may dip; that is
    # expected and is not flagged. Both clauses are required: at any positive
    # degen_strength the guard's penalty moves even for a parametric spec.
    penalty_moves = penalty in ("auto", "scheduled") and not parametric
    monotone_expected = not (penalty_moves or degen_moves)
    monotone = (not monotone_expected) or worst_drop <= 1e-6 * (abs(ll) + 1)
    return {
        "n_fit": health[0],
        "n_stall": health[1],
        "components": components,
        "pi": pi,
        "gamma": g,
        "cluster": np.argmax(g, axis=1),
        "loglik": ll,
        "ll_path": ll_path,
        "iter": it,
        "converged": conv,
        "monotone": monotone,
        "worst_drop": worst_drop,
        "K": K,
    }


def _mix_em_once(
    formula,
    data,
    family,
    K,
    control,
    resp,
    init_method,
    grp,
    n_units,
    rng,
    **gam_kwargs,
) -> Dict[str, Any]:
    """One EM run from a chosen init method."""
    feat = _mix_init_features(formula, data, family, grp, n_units)
    g = _mix_init_gamma(init_method, feat, K, feat["x"].shape[0], rng)
    return _mix_em_core(
        formula, data, family, control, resp, g, grp, n_units, **gam_kwargs
    )


def _mix_restarts(
    formula, data, family, K, control, resp, R, grp, n_units, rng, **gam_kwargs
) -> Dict[str, Any]:
    """Random-restart wrapper: keep the largest-loglik run.

    Restart 1 uses ``control.init`` (kmeans by default — a strong, separated
    start); later restarts are random, for basin diversity. Each restart is
    seeded from a stream drawn ONCE here, so the runs are independent and the
    result does not depend on the order they are executed in.
    """
    seeds = _mix_seeds(rng, R)
    fits: List[Optional[Dict[str, Any]]] = []
    lls = np.full(R, np.nan)
    for r in range(R):
        init_r = control.init if r == 0 else "random"
        try:
            st = _mix_em_once(
                formula,
                data,
                family,
                K,
                control,
                resp,
                init_r,
                grp,
                n_units,
                RMersenneTwister(int(seeds[r])),
                **gam_kwargs,
            )
        except Exception as exc:  # a failed restart is skipped, not fatal
            if control.verbose:
                print(f"   restart {r + 1} FAILED: {exc}")
            st = None
        fits.append(st)
        if st is not None:
            lls[r] = st["loglik"]
    ok = [i for i, f in enumerate(fits) if f is not None]
    if not ok:
        raise RuntimeError(f"all {R} restarts failed for K = {K}.")
    best = fits[max(ok, key=lambda i: lls[i])]
    best["restart_lls"] = lls
    best["basin_hits"] = int(np.sum(np.abs(lls - best["loglik"]) < 1e-3))
    return best


def _mix_search_fixed(
    formula, data, family, K, control, resp, grp, n_units, lam, rng, **gam_kwargs
) -> Dict[str, Any]:
    """``search="fixed"``: K=1 is a single weighted-trivial circ_gam; K>1 is
    best-of-restarts. Returns ``{state, trace=None}``."""
    if K == 1:
        st = _mix_em_once(
            formula,
            data,
            family,
            1,
            control,
            resp,
            "kmeans",
            grp,
            n_units,
            rng,
            **gam_kwargs,
        )
        st["R"] = 1
        st["restart_lls"] = np.array([st["loglik"]])
        st["basin_hits"] = 1
    else:
        st = _mix_restarts(
            formula,
            data,
            family,
            K,
            control,
            resp,
            control.restarts,
            grp,
            n_units,
            rng,
            **gam_kwargs,
        )
        st["R"] = control.restarts
    return {"state": st, "trace": None}


# =========================================================================== #
#  circ_mix — automatic K: the four structure moves
# =========================================================================== #
# A move reshapes the per-unit responsibility matrix, re-fits by a warm local
# EM, and reports its objective. The reshapes are pure (and unit-tested); the
# fitting is the ordinary engine. Each move returns None when it cannot apply.
#
# Sources: split/merge follow Ueda et al.'s SMEM, death is Figueiredo & Jain's
# annihilation, birth is the DP-means / CRP "new table".
def _mix_gamma_split(g: np.ndarray, j: int, label: np.ndarray) -> np.ndarray:
    """K → K+1: split column ``j`` by a per-unit 0/1 ``label``."""
    lab = np.asarray(label)
    a = g[:, j] * (lab == 0)
    b = g[:, j] * (lab == 1)
    left = g[:, :j]
    right = g[:, j + 1 :]
    return np.column_stack([left, a, b, right])


def _mix_gamma_merge(g: np.ndarray, j: int, m: int) -> np.ndarray:
    """K → K−1: merge columns ``j`` and ``m`` (their mass is summed)."""
    keep = [c for c in range(g.shape[1]) if c not in (j, m)]
    return np.column_stack([g[:, keep], g[:, j] + g[:, m]])


def _mix_gamma_death(g: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """K → |keep|: drop the other columns and renormalise."""
    sub = np.array(g[:, list(keep)], dtype=float, copy=True)
    rs = sub.sum(axis=1)
    dead = rs <= 0
    if dead.any():  # a unit with no mass left seats uniformly
        sub[dead, :] = 1.0 / len(keep)
        rs = sub.sum(axis=1)
    return sub / rs[:, None]


def _mix_gamma_birth(g: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """K → K+1: seed a fresh component from the units in ``idx``."""
    out = np.column_stack([g, np.zeros(g.shape[0])])
    out[np.asarray(idx, dtype=int), :] = 0.0
    out[np.asarray(idx, dtype=int), -1] = 1.0
    return out


def _mix_resid_feature(cp: _MixComponent, cx: Dict[str, Any]) -> Dict[str, Any]:
    """The split residual feature for a component: per response, the wrapped
    angular residual θ − μ̂ (circular) or the scaled residual (linear),
    column-bound over all factors. For a product component this is the JOINT
    residual feature, so the circular 2-means splits the worst blob along
    whichever response it is most over-dispersed in. Returns the per-ROW feature
    plus its geometry flag; the caller reduces it to per-unit and clusters it."""
    resps = _mix_responses(cx["formula"])
    preds = _cmp_predict(cp, cx["data"], "response")
    if not isinstance(preds, list):
        preds = [preds]
    cols = []
    for j, r in enumerate(resps):
        mu = np.asarray(preds[j], dtype=float)[:, 0]  # location param of factor j
        v = np.asarray(cx["data"][r].to_numpy(), dtype=float)
        if cx["resp_circ"]:
            cols.append(_wrap(v - mu))  # wrapped angular residual
        else:
            d = v - mu
            sd = float(np.std(d, ddof=1))
            cols.append((d - float(np.mean(d))) / (sd if sd > 0 else 1.0))
    return {"x": np.column_stack(cols), "circular": cx["resp_circ"]}


def _mix_eval(g_new, move, cx, J_cur, **gam_kwargs) -> Optional[Dict[str, Any]]:
    """Fit + score a proposed K, robustly but cheaply.

    The move's WARM responsibilities are tried first: if that single EM already
    lowers J below the current state, it is accepted immediately (the common,
    fast path — an informative split/merge warm-starts almost perfectly). Only
    when the warm fit FAILS to improve does it pay for independent restarts at
    the new K — the robustness net that (a) confirms a non-improving move really
    has no better optimum at that K before greedy gives up, and (b) recovers
    from a poor warm seed or component collapse. So growing is cheap, and the
    decision to stop is still made against grid-quality optima. Returns None
    only if every candidate fit failed.
    """
    try:
        warm = _mix_em_core(
            cx["formula"],
            cx["data"],
            cx["family"],
            cx["control"],
            cx["resp"],
            g_new,
            cx["grp"],
            cx["n_units"],
            **gam_kwargs,
        )
    except Exception:
        warm = None
    tolJ = 1e-6 * (abs(J_cur) + 1)
    if warm is not None:
        Jw = _mix_objective(warm, cx["lam"])
        if Jw < J_cur - tolJ:
            return {"move": move, "state": warm, "J": Jw}
    # warm did not improve (or failed): verify with restarts at the new K
    Kp = g_new.shape[1]
    R = cx["control"].restarts
    seeds = _mix_seeds(cx["rng"], R)
    fits = [] if warm is None else [warm]
    for r in range(R):
        init_r = cx["control"].init if r == 0 else "random"
        try:
            fits.append(
                _mix_em_once(
                    cx["formula"],
                    cx["data"],
                    cx["family"],
                    Kp,
                    cx["control"],
                    cx["resp"],
                    init_r,
                    cx["grp"],
                    cx["n_units"],
                    RMersenneTwister(int(seeds[r])),
                    **gam_kwargs,
                )
            )
        except Exception:
            continue
    if not fits:
        return None
    best = max(fits, key=lambda f: f["loglik"])
    return {"move": move, "state": best, "J": _mix_objective(best, cx["lam"])}


def _mix_move_split(state, cx, J_cur, **gam_kwargs) -> Optional[Dict[str, Any]]:
    """SPLIT (grow): split the worst-fit component (lowest mean per-obs density
    among its MAP members) by circular 2-means on its angular residuals — the
    over-dispersed / bimodal angular-residual signal."""
    K = state["K"]
    if K >= cx["control"].kmax:
        return None
    grp, n_units = cx["grp"], cx["n_units"]
    z = state["cluster"]  # per UNIT
    Lr = np.column_stack([_cmp_logpdf(cp, cx["data"]) for cp in state["components"]])
    L = _mix_rowsum(Lr, grp, n_units)  # n_units x K per-unit log-density
    sz = np.bincount(z, minlength=K)
    md = np.array(
        [float(L[z == k, k].mean()) if sz[k] > 0 else np.inf for k in range(K)]
    )
    md[sz < 2 * cx["control"].min_size] = np.inf  # need room to split
    if not np.isfinite(md).any():
        return None
    j = int(np.argmin(md))
    rf = _mix_resid_feature(state["components"][j], cx)  # per-ROW residuals
    ftx = (
        _mix_unit_angle(rf["x"], grp, n_units)
        if rf["circular"]
        else _mix_unit_feature(rf["x"], grp, n_units)
    )
    mem = np.flatnonzero(z == j)  # the units in component j
    try:
        km = _mix_kmeans(ftx[mem], 2, rf["circular"], cx["rng"])
    except Exception:
        return None
    label = _mix_assign(ftx, km["centers"], rf["circular"])  # per-unit split label
    return _mix_eval(
        _mix_gamma_split(state["gamma"], j, label), "split", cx, J_cur, **gam_kwargs
    )


def _mix_move_merge(state, cx, J_cur, **gam_kwargs) -> Optional[Dict[str, Any]]:
    """MERGE (shrink): merge the two most similar components (smallest wrapped
    distance between their responsibility-weighted mean directions, summed over
    responses — joint on the torus)."""
    K = state["K"]
    if K <= cx["control"].kmin or K < 2:
        return None
    grp, n_units = cx["grp"], cx["n_units"]
    # Responsibilities are per UNIT, so reduce each response to a per-unit value
    # first (its circular/arithmetic mean within the unit) -- identity for rows.
    ctrs = []
    for r in _mix_responses(cx["formula"]):
        vr = np.asarray(cx["data"][r].to_numpy(), dtype=float)
        if cx["resp_circ"]:
            vu = _mix_unit_angle(vr, grp, n_units)
        else:
            vu = _mix_unit_feature(vr[:, None], grp, n_units)[:, 0]
        col = []
        for k in range(K):
            w = state["gamma"][:, k]
            if cx["resp_circ"]:
                col.append(
                    float(np.arctan2(np.sum(w * np.sin(vu)), np.sum(w * np.cos(vu))))
                )
            else:
                sw = float(w.sum())
                col.append(float(np.sum(w * vu) / sw) if sw > 0 else np.nan)
        ctrs.append(np.array(col))
    pair, bd = None, np.inf
    for a in range(K - 1):
        for b in range(a + 1, K):
            d = 0.0
            for c0 in ctrs:
                d += (
                    float(_wrap(c0[a] - c0[b])) ** 2
                    if cx["resp_circ"]
                    else (c0[a] - c0[b]) ** 2
                )
            if d < bd:
                bd, pair = d, (a, b)
    if pair is None:
        return None
    return _mix_eval(
        _mix_gamma_merge(state["gamma"], pair[0], pair[1]),
        "merge",
        cx,
        J_cur,
        **gam_kwargs,
    )


def _mix_move_death(state, cx, J_cur, **gam_kwargs) -> Optional[Dict[str, Any]]:
    """DEATH (shrink): drop components whose soft size n_k = Σ_i γ_ik falls
    below ``min_size`` (this folds in the collapse guards)."""
    K = state["K"]
    nk = state["gamma"].sum(axis=0)
    keep = np.flatnonzero(nk >= cx["control"].min_size)
    if keep.size >= K:
        return None  # nothing undersized
    if keep.size < cx["control"].kmin:  # never fall below kmin
        keep = np.sort(np.argsort(-nk)[: cx["control"].kmin])
    if keep.size >= K:
        return None
    return _mix_eval(
        _mix_gamma_death(state["gamma"], np.sort(keep)),
        "death",
        cx,
        J_cur,
        **gam_kwargs,
    )


def _mix_move_birth(state, cx, J_cur, **gam_kwargs) -> Optional[Dict[str, Any]]:
    """BIRTH (grow): seed a fresh component from the worst-explained units."""
    K = state["K"]
    if K >= cx["control"].kmax:
        return None
    grp, n_units = cx["grp"], cx["n_units"]
    nu = state["gamma"].shape[0]
    Lr = np.column_stack([_cmp_logpdf(cp, cx["data"]) for cp in state["components"]])
    L = _mix_rowsum(Lr, grp, n_units)
    with np.errstate(divide="ignore"):
        ld = _mix_loglik_rows(L + np.log(state["pi"])[None, :])
    m = min(2 * cx["control"].min_size, nu // 2)
    if m < cx["control"].min_size:
        return None
    idx = np.argsort(ld)[:m]  # the worst-explained units
    return _mix_eval(
        _mix_gamma_birth(state["gamma"], idx), "birth", cx, J_cur, **gam_kwargs
    )


_MIX_MOVE_FNS = {
    "split": _mix_move_split,
    "merge": _mix_move_merge,
    "death": _mix_move_death,
    "birth": _mix_move_birth,
}


# =========================================================================== #
#  circ_mix — automatic K: the search strategies
# =========================================================================== #
def _mix_search_grid(
    formula, data, family, K, control, resp, grp, n_units, lam, rng, **gam_kwargs
) -> Dict[str, Any]:
    """``search="grid"``: the brute ``kmin:kmax`` sweep + information criterion
    — a validation cross-check on the greedy moves. Picks the minimum-J K."""
    Ks = [k for k in range(control.kmin, control.kmax + 1) if k <= n_units]
    rows, best, bestJ = [], None, np.inf
    for k in Ks:
        try:
            fk = _mix_search_fixed(
                formula,
                data,
                family,
                k,
                control,
                resp,
                grp,
                n_units,
                lam,
                rng,
                **gam_kwargs,
            )["state"]
        except Exception as exc:
            if control.verbose:
                print(f"  [grid] K={k} FAILED: {exc}")
            continue
        J = _mix_objective(fk, lam)
        df = (fk["K"] - 1) + sum(_cmp_edf(cp) for cp in fk["components"])
        rows.append(
            {
                "K": k,
                "loglik": fk["loglik"],
                "df": df,
                "bic": -2.0 * fk["loglik"] + df * float(np.log(n_units)),
                "J": J,
            }
        )
        if control.verbose:
            print(f"  [grid] K={k}  loglik={fk['loglik']:.2f}  df={df:.1f}  J={J:.2f}")
        if J < bestJ:
            bestJ, best = J, fk
    if best is None:
        raise RuntimeError(
            f"grid search: all K in {control.kmin}:{control.kmax} failed."
        )
    return {"state": best, "trace": pl.DataFrame(rows)}


def _mix_search_greedy(
    formula, data, family, K, control, resp, grp, n_units, lam, rng, **gam_kwargs
) -> Dict[str, Any]:
    """``search="greedy"``: start at the init K, run EM to convergence, then
    attempt structure moves; accept the move that drops J most, repeat until
    none improves J. Greedy + strict decrease ⇒ deterministic, monotone in J,
    and cannot cycle."""
    base = _mix_search_fixed(
        formula, data, family, K, control, resp, grp, n_units, lam, rng, **gam_kwargs
    )
    state = base["state"]
    J = _mix_objective(state, lam)
    cx = {
        "formula": formula,
        "data": data,
        "family": family,
        "control": control,
        "resp": resp,
        "grp": grp,
        "n_units": n_units,
        "lam": lam,
        "rng": rng,
        "resp_circ": _mix_resp_circular(family),
    }
    fns = {m: _MIX_MOVE_FNS[m] for m in control.moves if m in _MIX_MOVE_FNS}
    trace: List[Dict[str, Any]] = []
    maxrounds = 3 * control.kmax + 5
    for step in range(1, maxrounds + 1):
        cands = []
        for name, fn in fns.items():
            try:
                c = fn(state, cx, J, **gam_kwargs)
            except Exception:
                c = None
            if c is not None:
                cands.append(c)
        if not cands:
            break
        cand = min(cands, key=lambda c: c["J"])
        if cand["J"] >= J - 1e-6 * (abs(J) + 1):  # no improving move
            break
        trace.append(
            {
                "step": step,
                "move": cand["move"],
                "K_from": state["K"],
                "K_to": cand["state"]["K"],
                "J_from": J,
                "J_to": cand["J"],
                "dJ": cand["J"] - J,
            }
        )
        if control.verbose:
            print(
                f"  [greedy] step {step}: {cand['move']:<5s}  "
                f"K {state['K']}->{cand['state']['K']}  "
                f"J {J:.2f}->{cand['J']:.2f} (dJ {cand['J'] - J:.2f})"
            )
        state, J = cand["state"], cand["J"]
    # carry the init-K restart health onto the final state (the basin signal)
    for key in ("R", "restart_lls", "basin_hits"):
        state[key] = base["state"][key]
    return {"state": state, "trace": pl.DataFrame(trace) if trace else None}


# =========================================================================== #
#  circ_mix — the fitted object
# =========================================================================== #
def _mix_logmix(obj: "CircMix", newdata) -> np.ndarray:
    """``logmix[i,k] = log π_k + log f_k(y_i | x_i)`` — shared by the E-step and
    :meth:`CircMix.predict`."""
    pmat = _mix_gate(obj.gating, newdata.height, obj.K)
    L = np.column_stack([_cmp_logpdf(cp, newdata) for cp in obj.components])
    with np.errstate(divide="ignore"):
        return L + np.log(pmat)


def _mix_palette(K: int) -> List[str]:
    """A fixed, version-stable categorical palette (recycled past 10)."""
    base = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
    ]
    return [base[k % len(base)] for k in range(K)]


def _mix_band_circular(ax, grid, mid, csd, color, view=(0.0, 2.0 * np.pi)):
    """The ± circular-SD band of a component's location, tiled across the 0/2π
    cut, in a per-component colour. Built on the same unwrap/tile primitives as
    ``regression._band_fill_circular``, which is fixed to one series colour."""
    grid = np.asarray(grid, dtype=float)
    csd = np.broadcast_to(np.asarray(csd, dtype=float), np.asarray(mid).shape)
    for idx, phase in _unwrap_runs(mid):
        g, lo, hi = grid[idx], phase - csd[idx], phase + csd[idx]
        for k in _tile_k(float(lo.min()), float(hi.max()), view):
            ax.fill_between(
                g,
                lo + 2 * np.pi * k,
                hi + 2 * np.pi * k,
                color=color,
                alpha=0.22,
                linewidth=0.0,
            )


def _mix_loc_band(cp: _MixComponent, nd, resp_circular: bool):
    """Per-component band for the LOCATION on the response scale, matching
    ``CircGAM.circ_plot``. For a circular response it is the half-width ``csd``
    of the ± circular-SD *predictive* band — the component law's own spread, by
    quadrature, which covers every family including pnlss's derived direction.
    For a linear response it is a delta-method 2-SE lo–hi on the mean. ``None``
    on a failed predict."""
    fit = cp.fit
    try:
        if resp_circular:
            fm = np.asarray(_cmp_predict(cp, nd, "response"), dtype=float)
            return {"csd": _circ_sd_quad(fit.family, fm)}
        pr = fit.predict(_to_polars(nd), type="response", se_fit=True)
        mid = pr["fit"].to_numpy()
        se = pr["se.fit"].to_numpy()
        return {"lo": mid - 2.0 * se, "hi": mid + 2.0 * se}
    except Exception:
        return None


def _mix_leg(kind) -> str:
    """Human-readable leg label, mirroring ``CircGAM``'s geometry switch."""
    return {
        "cl": "circular-linear (c~l)",
        "cc": "circular-circular (c~c)",
        "lc": "linear-circular (l~c)",
        "ll": "location-scale (l~l)",
        "joint": "joint torus density",
    }.get(kind, "circular")


def _mix_deparse(formula) -> str:
    """Render a spec (a string, or a list of strings / nested lists) as one
    compact line for printing."""
    if isinstance(formula, str):
        return formula

    def one(f):
        return f if isinstance(f, str) else (f[0] if len(f) else "")

    return " | ".join(one(f) for f in formula)


class CircMix:
    """A fitted finite mixture of circular distributional GAMs.

    Returned by :func:`circ_mix`; see that function for the model. Fitted
    per-observation arrays carry a trailing underscore.

    Attributes
    ----------
    K, K_init : int
        The fitted and the starting number of components.
    search : {"fixed", "greedy", "grid"}
        How K was decided.
    components : list
        The K fitted components — each wrapping a weighted :func:`circ_gam`, or,
        for a joint density, a *product* of several (one per chain-rule factor).
    gating : dict
        The mixing-weight object; ``gating["pi"]`` are the component
        proportions.
    gamma_ : ndarray, shape (n_units, K)
        The soft responsibilities.
    labels_ : ndarray, shape (n_units,)
        The per-unit MAP cluster labels (0-based).
    nk : ndarray
        MAP cluster sizes; ``Gtilde`` is the number of non-empty components.
    loglik, df, edf, bic, aic : float / ndarray
        The mixture log-likelihood, degrees of freedom (K−1) + Σ_k edf_k, the
        per-component edf, and the two information criteria.
    objective : dict
        ``J``, ``loglik``, ``df``, ``bic`` and the ``lambda_`` used.
    unit : dict
        The clustering unit — ``kind`` ("row"/"subject"), the per-row ``index``,
        ``n_units`` and the group ``labels``.
    converged, iter, ll_path, monotone : bool / int / list / bool
        Convergence flag, iteration count, the recorded log-likelihood path of
        the selected run, and whether it ascended monotonically.
    restarts : dict
        ``R``, the per-restart log-likelihoods ``lls`` and the ``basin_hits``
        count (how many restarts reached the kept optimum — a health signal).
    fit_health : dict
        ``n_fit`` weighted M-step fits were run in the kept EM run, of which
        ``n_stall`` were reported by the engine as not converged. This is the
        denominator for the ``gam.fit5 step failed`` warnings on stderr, which
        are left unfiltered: a handful out of hundreds of fits is routine — the
        engine's final Newton step often cannot improve on an already-converged
        fit, and each stall is retried from the component's weighted moments
        with the better fit kept — while a count approaching ``n_fit`` means the
        M-step is genuinely struggling.
    move_trace : polars.DataFrame or None
        ``None`` for ``search="fixed"``; the accepted moves for ``"greedy"``;
        the K sweep for ``"grid"``.
    degen : dict or None
        The degeneracy guard at the fitted optimum, or ``None`` when it is off
        (``control.degen_strength = 0``) or the family declares no guarded
        parameter. ``penalty`` is the log-likelihood each component traded away
        to the guard, in nats, and ``by_param`` splits it by guarded parameter;
        ``lambda_`` is the per-component strength λ_k = c/N_k. Because
        ``loglik`` and ``bic`` are reported *without* the penalty, a component
        with a large ``penalty`` is not a failed fit — it is a deliberately
        shrunk one, and lowering ``degen_strength`` will move it.
    geometry : str
        The leg — ``"cl"``, ``"cc"``, ``"lc"``, ``"ll"`` or ``"joint"``.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    # ---- information criteria ------------------------------------------- #
    @property
    def aic(self) -> float:
        """Akaike information criterion, −2 logLik + 2 df."""
        return -2.0 * self.loglik + 2.0 * self.df

    # ---- component parameters -------------------------------------------- #
    def coef(self) -> Dict[str, Any]:
        """One coefficient vector per component (a product component's factors
        are returned as a list, in chain-rule order)."""
        return {
            f"component{k + 1}": _cmp_coef(cp) for k, cp in enumerate(self.components)
        }

    # ---- prediction ------------------------------------------------------- #
    def predict(self, newdata=None, type: str = "cluster", log: bool = False):
        """Predict from the fitted mixture.

        Parameters
        ----------
        newdata : DataFrame, optional
            Omitting it uses the training frame. Angles must be on the same
            (original, un-centred) scale as the frame ``circ_mix`` was fitted
            on; each component's own centring rotation is applied internally.
        type : {"cluster", "density", "response"}
            ``"cluster"`` (default) returns the ``n × K`` responsibility matrix,
            ``"density"`` the mixture density per row, ``"response"`` the
            per-component response-scale fitted values (a list of length K).
        log : bool
            For ``type="density"``, return the log-density.
        """
        if type not in ("cluster", "density", "response"):
            raise ValueError('`type` must be one of "cluster", "density", "response".')
        # the ORIGINAL training frame, not a component's fit frame: components
        # may each carry a different centring rotation, and _circ_logpdf applies
        # each one itself.
        newdata = _to_polars(self.data if newdata is None else newdata)
        if type == "response":
            return [_cmp_predict(cp, newdata, "response") for cp in self.components]
        if self.response not in newdata.columns:
            raise ValueError(
                f'predict(type="{type}") needs the response {self.response!r} '
                "in `newdata`."
            )
        logmix = _mix_logmix(self, newdata)
        if type == "cluster":
            return _mix_responsibilities(logmix)
        ld = _mix_loglik_rows(logmix)
        return ld if log else np.exp(ld)

    def predict_proba(self, newdata=None) -> np.ndarray:
        """Component posterior probabilities — ``predict(type="cluster")``."""
        return self.predict(newdata, type="cluster")

    def predict_density(self, newdata=None, log: bool = False) -> np.ndarray:
        """Mixture density per row — ``predict(type="density")``."""
        return self.predict(newdata, type="density", log=log)

    def score_samples(self, newdata=None) -> np.ndarray:
        """Per-row mixture log-density."""
        return self.predict(newdata, type="density", log=True)

    # ---- display ---------------------------------------------------------- #
    def __repr__(self) -> str:
        hard = self.control.assign == "hard"
        pi = self.gating["pi"]
        units = (
            f"{self.unit['n_units']} subjects ({len(self.unit['index'])} rows)"
            if self.unit["kind"] == "subject"
            else f"{self.unit['n_units']} obs"
        )
        fam_name = getattr(self.family, "name", None) or type(self.family).__name__
        out = [
            f"Finite mixture of circular GAMs (circ_mix) -- {_mix_leg(self.geometry)}",
            f"  family {fam_name} | K = {self.K} component"
            f"{'s' if self.K > 1 else ''} | {units}"
            f"{' | hard (CEM)' if hard else ''}",
            f"  formula: {_mix_deparse(self.formula)}",
            f"  logLik{' (classification)' if hard else ''} = {self.loglik:.2f} | "
            f"df = {self.df:.2f} | BIC = {self.bic:.2f}",
            "  components (MAP):",
        ]
        for k in range(self.K):
            out.append(
                f"    {k + 1:2d}:  pi = {pi[k]:.3f}   n = {self.nk[k]:3d}   "
                f"edf = {self.edf[k]:.2f}"
            )
        if self.Gtilde < self.K:
            out.append(
                f"  {self.K - self.Gtilde} of {self.K} components are empty under MAP."
            )
        # auto-K provenance: how the component count was decided
        if self.search == "greedy":
            nm = 0 if self.move_trace is None else self.move_trace.height
            line = (
                f"  auto-K (greedy): K_init {self.K_init} -> {self.K} via "
                f"{nm} accepted move{'' if nm == 1 else 's'}"
            )
            if nm:
                line += f" [{', '.join(self.move_trace['move'].to_list())}]"
            out.append(line + ".")
        elif self.search == "grid" and self.move_trace is not None:
            ks = self.move_trace["K"].to_list()
            out.append(
                f"  auto-K (grid): swept K = {min(ks)}..{max(ks)}, selected "
                f"K = {self.K} by min "
                f"{'BIC' if self.control.lambda_ is None else 'J'}."
            )
        tail = (
            f"  {'converged' if self.converged else 'STOPPED (max_iter)'} in "
            f"{self.iter} iterations"
        )
        if self.K > 1:
            tail += (
                f"; restart basin hits {self.restarts['basin_hits']}"
                f"/{self.restarts['R']}"
            )
        out.append(tail + ".")
        if self.K > 1 and not self.monotone:
            out.append("  note: a non-monotone EM step was seen -- inspect .ll_path.")
        fh = getattr(self, "fit_health", None)
        if fh and fh["n_stall"]:
            out.append(
                f"  note: {fh['n_stall']} of {fh['n_fit']} weighted M-step fits "
                "did not converge; each was retried from its weighted moments "
                "and the better kept -- see .fit_health."
            )
        # logLik/BIC above are UNPENALISED, so a component the guard shrank looks
        # like a failed fit unless the guard says otherwise.
        dg = getattr(self, "degen", None)
        if dg is not None and np.any(dg["binding"]):
            which = np.flatnonzero(dg["binding"])
            hit = ", ".join(f"{k + 1} ({dg['penalty'][k]:.1f} nats)" for k in which)
            plural = "s" if which.size > 1 else ""
            out.append(
                f"  note: the degeneracy guard (degen_strength = "
                f"{dg['strength']:g}) is shrinking component{plural} {hit}; "
                "logLik/BIC are reported unpenalised -- see .degen."
            )
        return "\n".join(out)

    def params(self, newdata=None) -> Dict[str, Any]:
        """Each component's fitted parameters on the RESPONSE scale.

        The values the family is parameterised by — ``(mu, kappa)`` for
        ``vmlss``, and so on — one row per observation, in the ORIGINAL
        (un-centred) frame: a component that
        :func:`~pycircstat2.regression.circ_gam` rotated off the tan-half wall
        reports its direction back where the data live, not where it was
        fitted. This is what :meth:`coef` cannot give: coefficients are on the
        LINK scale and in the component's own fit frame, so a centred
        component's location intercept reads as 0 whatever its actual mean
        direction.

        Returns a ``{"component1": DataFrame, ...}`` mapping — a list of frames,
        one per chain-rule factor, for a joint (product) component.

        Parameters
        ----------
        newdata : DataFrame, optional
            Where to evaluate. Omitting it uses the training frame. For an
            intercept-only (density-clustering) spec the parameters are constant
            and every row is identical.
        """
        nd = _to_polars(self.data if newdata is None else newdata)

        def frame(cp, arr):
            f0 = cp.fits[0]
            names = getattr(f0.family, "params", None) or [
                f"p{j + 1}" for j in range(np.shape(arr)[1])
            ]
            a = np.asarray(arr, dtype=float)
            return pl.DataFrame(
                {nm: a[:, j] for j, nm in enumerate(names[: a.shape[1]])}
            )

        out: Dict[str, Any] = {}
        for k, cp in enumerate(self.components):
            pr = _cmp_predict(cp, nd, "response")
            out[f"component{k + 1}"] = (
                [frame(_MixComponent(f), a) for f, a in zip(cp.fits, pr)]
                if isinstance(cp, _MixProduct)
                else frame(cp, pr)
            )
        return out

    def summary(self) -> None:
        """Print the model header, the per-component response-scale parameters
        and the link-scale coefficients."""
        print(self)
        print("\n  per-component parameters (response scale, original frame):")
        for k, (_, pf) in enumerate(self.params().items()):
            print(f"  [component {k + 1}]")
            for j, fr in enumerate([pf] if isinstance(pf, pl.DataFrame) else pf):
                pre = (
                    "    " if isinstance(pf, pl.DataFrame) else f"    factor {j + 1}: "
                )
                parts = []
                for nm in fr.columns:
                    v = fr[nm].to_numpy()
                    parts.append(
                        f"{nm} = {v[0]:.4f}"
                        if np.ptp(v) <= 1e-8
                        else f"{nm} in [{v.min():.4f}, {v.max():.4f}]"
                    )
                print(pre + "  ".join(parts))
        print("\n  per-component coefficients (link scale, fit frame):")
        for k, (_, cf) in enumerate(self.coef().items()):
            ctr = [
                float(getattr(f, "circ_center", 0.0) or 0.0)
                for f in self.components[k].fits
            ]
            tag = (
                f"  [centred at {', '.join(f'{c:+.4g}' for c in ctr)} rad]"
                if any(ctr)
                else ""
            )
            print(f"  [component {k + 1}]{tag}")
            if isinstance(cf, list):
                for j, c in enumerate(cf):
                    print(f"    factor {j + 1}: {np.round(c, 4)}")
            else:
                print(f"    {np.round(cf, 4)}")

    # ---- plotting ---------------------------------------------------------- #
    def circ_plot(self, view: str = "flat", n: int = 200, se: bool = True, **kw):
        """Clustered views of the fitted mixture — the ``plot.circ_mix`` twin.

        The flat view colours the observations by their MAP cluster, with — for
        a regression cell — each component's fitted location curve over the
        single covariate, or — for a density cell (``theta ~ 1``) — the
        per-cluster spread of the response with each component's fitted mean
        direction. The geometry view draws those per-component curves on the
        leg's natural 3-D surface (cylinder for c~l, torus for c~c, upright can
        for l~c), sharing ``CircGAM.circ_plot``'s canvas.

        Parameters
        ----------
        view : {"flat", "geometry", "both"}
            Which view to draw. A joint (product) component has only the flat
            torus-square projection, so it draws that whatever the ``view``.
        n : int, default=200
            Grid points for each component's fitted curve.
        se : bool, default=True
            Band each component's curve: ± the component law's circular
            standard deviation (its predictive angular spread) for a circular
            response, a pointwise 2-SE interval for a linear response.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if view not in ("flat", "geometry", "both"):
            raise ValueError('`view` must be one of "flat", "geometry", "both".')
        cols = _mix_palette(self.K)
        z = np.asarray(self.labels_)[self.unit["index"]]  # per ROW

        if self.geometry == "joint":
            if view != "flat":
                warnings.warn(
                    "circ_plot: the joint geometry surface is not drawn yet; "
                    "showing the flat torus-square.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            fig, ax = plt.subplots(figsize=(5.6, 5.2))
            self._plot_joint(ax, cols, z)
            fig.tight_layout()
            return fig

        cov = self.components[0].fit._covariate()
        kind, resp_circ, cov_circ, _ = self.components[0].fit._geometry()
        surface = {"cl": "cylinder", "cc": "torus", "lc": "can"}.get(kind)
        if view != "flat" and (cov is None or surface is None):
            warnings.warn(
                "circ_plot: no surface for this fit (geometry needs exactly one "
                "covariate); drawing the flat view.",
                RuntimeWarning,
                stacklevel=2,
            )
            view = "flat"

        if view == "flat":
            fig, ax = plt.subplots(figsize=(6.4, 4.6))
            self._plot_flat(ax, cols, z, cov, resp_circ, cov_circ, n, se)
        elif view == "geometry":
            fig = plt.figure(figsize=(6.0, 5.4))
            ax = fig.add_subplot(111, projection="3d")
            self._plot_geometry(ax, cols, z, cov, resp_circ, cov_circ, surface, n)
        else:
            fig = plt.figure(figsize=(11.6, 4.8))
            ax3 = fig.add_subplot(1, 2, 1, projection="3d")
            self._plot_geometry(ax3, cols, z, cov, resp_circ, cov_circ, surface, n)
            ax = fig.add_subplot(1, 2, 2)
            self._plot_flat(ax, cols, z, cov, resp_circ, cov_circ, n, se)
        fig.tight_layout()
        return fig

    # -- the per-view drawers ------------------------------------------------ #
    def _plot_grid(self, cov, cov_circ, n):
        """The covariate grid and its one-column newdata frame."""
        xv = np.asarray(self.data[cov].to_numpy(), dtype=float)
        lo, hi = float(np.nanmin(xv)), float(np.nanmax(xv))
        if cov_circ:
            lo, hi = (0.0, 2 * np.pi) if lo >= 0 else (-np.pi, np.pi)
        grid = np.linspace(lo, hi, n)
        return xv, grid, pl.DataFrame({cov: grid})

    def _plot_flat(self, ax, cols, z, cov, resp_circ, cov_circ, n, se):
        yobs = np.asarray(self.data[self.response].to_numpy(), dtype=float)
        yw = _to_02pi(yobs) if resp_circ else yobs

        # density cell (no covariate): the clusters of the response alone
        if cov is None:
            for k in range(self.K):
                m = z == k
                if not m.any():
                    continue
                ax.scatter(
                    yw[m],
                    np.full(int(m.sum()), k)
                    + 0.14 * np.random.default_rng(k).uniform(-1, 1, int(m.sum())),
                    s=12,
                    color=cols[k],
                    alpha=0.65,
                    linewidths=0,
                )
            if resp_circ:
                nd = self.data.head(1)
                for k in range(self.K):
                    mu = float(
                        np.asarray(_cmp_predict(self.components[k], nd, "response"))[
                            0, 0
                        ]
                    )
                    ax.axvline(_to_02pi(mu), color=cols[k], lw=2, ls="--")
                ax.set_xlim(0, 2 * np.pi)
            ax.set_xlabel(self.response)
            ax.set_ylabel("cluster")
            ax.set_yticks(range(self.K))
            ax.set_yticklabels([f"{k + 1}" for k in range(self.K)])
            ax.set_title(f"circ_mix: {self.Gtilde} clusters")
            return

        # regression cell: response vs covariate, per-component location
        xv, grid, nd = self._plot_grid(cov, cov_circ, n)
        for k in range(self.K):
            m = z == k
            if m.any():
                ax.scatter(xv[m], yw[m], s=12, color=cols[k], alpha=0.55, linewidths=0)
        yk = [
            np.asarray(_cmp_predict(cp, nd, "response"), dtype=float)[:, 0]
            for cp in self.components
        ]
        if se:
            for k in range(self.K):
                bd = _mix_loc_band(self.components[k], nd, resp_circ)
                if bd is None:
                    continue
                if resp_circ:
                    _mix_band_circular(ax, grid, yk[k], bd["csd"], cols[k])
                else:
                    ax.fill_between(
                        grid, bd["lo"], bd["hi"], color=cols[k], alpha=0.22, lw=0
                    )
        for k in range(self.K):
            if resp_circ:
                _lines_circular(ax, grid, yk[k], color=cols[k], lw=2.4)
            else:
                ax.plot(grid, yk[k], color=cols[k], lw=2.4)
        ax.set_xlabel(cov)
        ax.set_ylabel(self.response)
        ax.set_ylim(0, 2 * np.pi) if resp_circ else None
        ax.set_title(f"circ_mix: {self.K} components")

    def _plot_geometry(self, ax, cols, z, cov, resp_circ, cov_circ, surface, n):
        xv, grid, nd = self._plot_grid(cov, cov_circ, n)
        yobs = np.asarray(self.data[self.response].to_numpy(), dtype=float)
        vobs = _wrap(yobs) if resp_circ else yobs
        yk = [
            np.asarray(_cmp_predict(cp, nd, "response"), dtype=float)[:, 0]
            for cp in self.components
        ]
        ykw = [_wrap(v) if resp_circ else v for v in yk]
        yspan = (
            np.concatenate([np.concatenate(ykw), vobs]) if surface == "can" else None
        )
        to_u, to_v, xyz, (mx, my, mz) = _surface_maps(surface, grid, xv, yspan)
        ax.plot_wireframe(mx, my, mz, color="0.85", linewidth=0.4)
        ax.set_box_aspect((np.ptp(mx), np.ptp(my), np.ptp(mz)))
        for k in range(self.K):
            m = z == k
            if m.any():
                px, py, pz = xyz(to_u(xv[m]), to_v(vobs[m]))
                ax.scatter(px, py, pz, s=6, color=cols[k], alpha=0.6, linewidths=0)
        for k in range(self.K):
            cx, cy, cz = xyz(to_u(grid), to_v(ykw[k]))
            ax.plot(cx, cy, cz, color=cols[k], lw=3)
        ax.set_axis_off()
        ax.set_title(f"{surface} · circ_mix ({self.K} components)")

    def _plot_joint(self, ax, cols, z):
        """The joint flat view: each observation on the response square,
        coloured by MAP cluster, with each component's weighted (parent,
        conditional) centroid marked. The parent response goes on x, the
        conditional on y."""
        resps = _mix_responses(self.formula)  # (conditional, parent) order
        circ = _mix_resp_circular(self.family)
        vals = [np.asarray(self.data[r].to_numpy(), dtype=float) for r in resps[:2]]

        def w(v):
            return _to_02pi(v) if circ else v

        for k in range(self.K):
            m = z == k
            if m.any():
                ax.scatter(
                    w(vals[1])[m],
                    w(vals[0])[m],
                    s=12,
                    color=cols[k],
                    alpha=0.6,
                    linewidths=0,
                )
        for k in range(self.K):
            wk = self.gamma_[:, k]
            if circ:
                cx = float(
                    np.arctan2(
                        np.sum(wk * np.sin(vals[1])), np.sum(wk * np.cos(vals[1]))
                    )
                )
                cy = float(
                    np.arctan2(
                        np.sum(wk * np.sin(vals[0])), np.sum(wk * np.cos(vals[0]))
                    )
                )
            else:
                sw = float(wk.sum())
                cx, cy = (
                    float(np.sum(wk * vals[1]) / sw),
                    float(np.sum(wk * vals[0]) / sw),
                )
            ax.scatter(
                w(cx),
                w(cy),
                s=140,
                facecolor=cols[k],
                edgecolor="black",
                linewidths=2,
                zorder=5,
            )
        ax.set_xlabel(resps[1])
        ax.set_ylabel(resps[0])
        if circ:
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)
        ax.set_title(f"circ_mix joint: {self.Gtilde} torus clusters")


# =========================================================================== #
#  circ_mix — the front door
# =========================================================================== #
def circ_mix(
    formula,
    data,
    family=vmlss,
    K: int = 2,
    search: str = "fixed",
    assign: str = "soft",
    group: Optional[str] = None,
    control=None,
    **gam_kwargs,
) -> CircMix:
    """Finite mixtures of circular distributional GAMs, by EM.

    Fits a K-component finite mixture of circular distributional GAMs by the EM
    algorithm. It does not touch the families or the GAM internals: each M-step
    is a weighted :func:`~pycircstat2.regression.circ_gam` fit and each E-step
    reads the family's per-observation density. Because a component is reached
    only through that small interface, one engine spans density clustering, the
    circular–linear / circular–circular / linear–circular regression trio, and
    everything in between — the response geometry is set entirely by ``family``.

    Parameters
    ----------
    formula : str or list
        A model spec for ONE component, exactly as
        :func:`~pycircstat2.regression.circ_gam` expects: ``"theta ~ 1"``
        (density clustering), ``"theta ~ x1 + x2"`` (circular–linear),
        ``"theta ~ cos(phi) + sin(phi)"`` or ``"theta ~ s(phi, bs='cc')"``
        (circular–circular), or ``"y ~ s(phi, bs='cc')"`` with a linear-response
        family (linear–circular). A *list* carrying two or more distinct
        responses fits a joint (torus) density by the chain rule — e.g.
        ``["psi ~ cos(phi) + sin(phi)", "phi ~ 1"]`` factorises
        f(ψ, φ) = f(ψ | φ) f(φ) into two factors; a response named in another
        formula's right-hand side is conditioned on it, and the list order is
        the chain-rule order. (A single response with two or more
        *location-scale* predictors is still one component, written
        ``["theta ~ s(x)", "~ s(x)"]``; the joint reading needs two or more
        distinct left-hand sides. Joint densities over more than two responses
        are not yet supported.)
    data : DataFrame
        A polars (or pandas) frame holding the response and covariates. Angles
        are in **radians**.
    family : default=vmlss
        A location-scale family: any circular family (``vmlss``, ``pnlss``, …)
        for a circular response, or ``"gaulss"``/``"gammals"`` for the
        linear–circular leg. The family is the only thing that sets the response
        geometry; the EM machinery is identical across all of them.
    K : int, default=2
        The number of mixture components. Under ``search="fixed"`` it is held;
        under ``search="greedy"`` it is the *starting* count from which the data
        grows or shrinks K (not a ceiling or floor); it is ignored under
        ``search="grid"`` (which sweeps ``control.kmin:control.kmax``).
    search : {"fixed", "greedy", "grid"}
        How the number of components is decided. ``"fixed"`` (default) holds K.
        Automatic-K search is opt-in: ``"greedy"`` runs bidirectional split /
        merge / death moves from the init K, accepting any move that lowers the
        penalised objective J = −2 logLik + λ·df (= BIC when λ = log n) — the
        warm heuristic, which grows reliably from a small init K; ``"grid"``
        fits every K in ``kmin:kmax`` with restarts and picks the minimum-J K —
        the robust selector and the cross-check on the moves.
    assign : {"soft", "hard"}
        The E-step assignment rule. ``"soft"`` (default) is EM with fractional
        responsibilities. ``"hard"`` is classification EM (CEM): each unit seats
        wholly at its argmax component, and the engine maximises the
        classification log-likelihood Σ_u max_k (log π_k + log f_k(y_u)) rather
        than the mixture log-likelihood (its ``loglik``/``bic`` are on that
        classification scale). Combined with ``search="greedy"`` it is the
        circular DP-means / k-means-style hard clustering.
    group : str, optional
        The clustering unit. ``None`` (default) clusters *rows* — one
        responsibility per observation. A column name, ``group="id"``, clusters
        *subjects / curves*: a subject's whole trajectory seats at one component
        (the longitudinal / latent-class-growth case). Under a group the
        responsibilities, MAP labels and the BIC sample size are all per
        subject.
    control : CircMixControl or dict, optional
        Tuning parameters; see :class:`CircMixControl`.
    **gam_kwargs
        Further arguments forwarded to the per-component
        :func:`~pycircstat2.regression.circ_gam` M-step (``knots``, ``method``,
        ``center``, …).

    Returns
    -------
    CircMix

    Notes
    -----
    **EM and restarts.** Each run alternates a weighted M-step (one ``circ_gam``
    per component, weighted by the responsibilities) with an E-step that records
    the observed-data mixture log-likelihood and updates the responsibilities.
    For parametric (penalty-free) components the EM is monotone. The fit is
    repeated from ``control.restarts`` random responsibility seeds and the
    largest-log-likelihood run is kept; ``.restarts["basin_hits"]`` reports how
    many restarts reached it (a health signal). Under ``penalty="auto"`` a
    moving smoothing penalty makes small dips expected, so they are not flagged.

    **Model selection.** ``df = (K−1) + Σ_k edf_k`` and
    ``bic = −2 logLik + df·log(n)``. For a joint component the per-component
    edf is summed over its factors.

    **Joint (torus) density.** A multi-response ``formula`` (two distinct
    left-hand sides) makes each component a *product* of weighted ``circ_gam``
    fits — one per chain-rule factor — whose joint log-density is the sum of the
    factor log-densities. The EM loop, restarts, MAP, J/BIC and the automatic-K
    moves are unchanged: the joint case is a component-implementation swap, not
    a different engine. The circular k-means initialisation
    (:func:`circ_kmeans`) seeds on all angular responses jointly — one torus
    coordinate per response — and a greedy split divides the worst component on
    its joint angular residuals, so it grows along whichever response is the
    more over-dispersed.

    **Longitudinal / curve clustering.** With ``group="id"`` the unit is a
    subject: the E-step sums each component's per-row log-densities within
    subject (so the whole trajectory shares one responsibility), the M-step
    broadcasts that responsibility back to the subject's rows, and the BIC
    counts n = the number of subjects.


    See Also
    --------
    circ_kmeans : the initialiser and split rule.
    pycircstat2.regression.circ_gam : the per-component M-step.

    Examples
    --------
        import numpy as np, polars as pl
        from pycircstat2.clustering import circ_mix
        rng = np.random.default_rng(1)
        n = 400
        z = rng.integers(2, size=n)
        x = rng.uniform(-1, 1, n)
        mu = 2 * np.arctan(np.where(z == 0, 0.9, -0.9) + np.where(z == 0, 2.2, -2.2) * x)
        y = np.mod(rng.vonmises(mu, 6.0), 2 * np.pi)
        df = pl.DataFrame({"y": y, "x": x})
        m = circ_mix("y ~ x", df, K=2)          # a two-component vM regression
        m.labels_                                # MAP cluster per observation
    """
    if search not in ("fixed", "greedy", "grid"):
        raise ValueError('`search` must be one of "fixed", "greedy", "grid".')
    if assign not in ("soft", "hard"):
        raise ValueError('`assign` must be one of "soft", "hard".')

    # ---- scope guards -------------------------------------------------- #
    fam = _resolve_gam_family(family)
    n_lp = getattr(fam, "n_lp", None)
    if not (isinstance(n_lp, int) and n_lp >= 2):
        raise ValueError(
            "`family` must be a location-scale family — one declaring several "
            "linear predictors (a `n_lp` >= 2), e.g. vmlss, pnlss, 'gaulss'."
        )
    K = int(K)
    if K < 1:
        raise ValueError("`K` must be a single integer >= 1.")

    data = _to_polars(data)
    n = data.height
    resps = _mix_responses(formula)  # >= 2 ==> a joint product component
    if not resps:
        raise ValueError(
            'the first formula must name the response, e.g. "theta ~ s(x)".'
        )
    resp = resps[0]  # the primary (conditional) response
    if len(resps) > 2:
        raise ValueError(
            "joint densities over more than two responses (d > 2) are not "
            "supported; use a two-response (torus) spec."
        )
    miss = [r for r in resps if r not in data.columns]
    if miss:
        raise ValueError(f"response(s) not found in `data`: {', '.join(miss)}.")

    ctl = _mix_control(control)
    if ctl.kmin > ctl.kmax:
        raise ValueError(f"control.kmin ({ctl.kmin}) exceeds kmax ({ctl.kmax}).")

    # ---- the clustering unit: rows, or subjects via group="id" ---------- #
    u = _mix_group_index(group, data, n)
    grp, n_units = u["grp"], u["n_units"]
    if K > n_units:
        raise ValueError(f"K = {K} exceeds the number of {u['kind']}s ({n_units}).")

    # ---- carry the E-step rule + penalty handling on the control -------- #
    # `assign` rides the control so the EM core reads it without a new threaded
    # argument. For penalty = "fixed"/"scheduled" the M-step smoothing
    # parameters are seeded ONCE here, from a pooled single-component pilot fit
    # on all the data. It matters only for smooth components: a parametric
    # formula has no penalty, so the three modes coincide and no pilot is
    # needed. A user-supplied control.sp wins (the pilot is skipped).
    ctl = copy.copy(ctl)
    ctl.assign = assign
    if (
        ctl.penalty in ("fixed", "scheduled")
        and ctl.sp is None
        and _mix_formula_smooth(formula)
    ):
        try:
            ctl.sp = _mix_pooled_sp(formula, data, fam, **gam_kwargs)
        except Exception:
            ctl.sp = None
        if ctl.sp is None:
            warnings.warn(
                f'circ_mix: penalty="{ctl.penalty}" pilot fit failed; falling '
                'back to penalty="auto" (per-iteration REML).',
                RuntimeWarning,
                stacklevel=2,
            )
            ctl.penalty = "auto"

    rng = _mix_rng(ctl.seed)
    lam = float(np.log(n_units)) if ctl.lambda_ is None else float(ctl.lambda_)
    args = (formula, data, fam, K, ctl, resp, grp, n_units, lam, rng)
    if search == "fixed":
        res = _mix_search_fixed(*args, **gam_kwargs)
    elif search == "grid":
        res = _mix_search_grid(*args, **gam_kwargs)
    else:
        res = _mix_search_greedy(*args, **gam_kwargs)
    best = res["state"]
    Kf = best["K"]
    if not best["monotone"]:
        warnings.warn(
            "circ_mix: an EM run's log-likelihood was non-monotone (worst drop "
            f"{best['worst_drop']:.3g}); parametric or fixed-penalty EM should "
            'ascend -- inspect .ll_path (use penalty="fixed" for monotone '
            "smooth EM).",
            RuntimeWarning,
            stacklevel=2,
        )

    # ---- assemble the fitted object ------------------------------------- #
    edf_k = np.array([_cmp_edf(cp) for cp in best["components"]], dtype=float)
    df = (Kf - 1) + float(edf_k.sum())
    bic = -2.0 * best["loglik"] + df * float(np.log(n_units))
    J = -2.0 * best["loglik"] + df * lam
    nk = np.bincount(best["cluster"], minlength=Kf)
    cp0 = best["components"][0]
    geometry = "joint" if isinstance(cp0, _MixProduct) else cp0.fit._geometry()[0]

    return CircMix(
        formula=formula,
        data=data,
        family=fam,
        response=resp,
        K=Kf,
        K_init=K,
        search=search,
        components=best["components"],
        gating={"type": "constant", "pi": best["pi"]},
        unit={
            "kind": u["kind"],
            "index": grp,
            "n_units": n_units,
            "labels": u["labels"],
        },
        gamma_=best["gamma"],
        labels_=best["cluster"],
        nk=nk,
        Gtilde=int(np.sum(nk > 0)),
        loglik=best["loglik"],
        df=df,
        edf=edf_k,
        bic=bic,
        objective={
            "J": J,
            "loglik": best["loglik"],
            "df": df,
            "bic": bic,
            "lambda_": lam,
        },
        iter=best["iter"],
        converged=best["converged"],
        ll_path=best["ll_path"],
        monotone=best["monotone"],
        restarts={
            "R": best["R"],
            "lls": best["restart_lls"],
            "basin_hits": best["basin_hits"],
        },
        fit_health={
            "n_fit": int(best.get("n_fit", 0)),
            "n_stall": int(best.get("n_stall", 0)),
        },
        move_trace=res["trace"],
        geometry=geometry,
        degen=_mix_degen_report(best["components"], best["gamma"], grp, ctl),
        control=ctl,
    )
