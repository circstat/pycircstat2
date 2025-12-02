from __future__ import annotations

import inspect
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import logsumexp

from .descriptive import circ_dist, circ_kappa, circ_mean_and_r
from .distributions import CircularContinuous, vonmises, katojones
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
            raise RuntimeError("Failed to initialise clusters without empty components.")

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
        self.n = n = alpha.size

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
                print(
                    f"Reached max iter {self.n_iters}. Final nLL = {nLL:.3f}\n"
                )

        self.nLL = nLL_history[~np.isnan(nLL_history)]

        self.m_ = means
        self.r_ = resultants
        self.p_ = p
        self.kappa_ = kappa
        self.params_ = [
            {"mu": float(self.m_[i]), "kappa": float(self.kappa_[i])} for i in range(self.n_clusters)
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
            raise ValueError("Number of clusters exceeds sample size during initialization.")

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
                raise RuntimeError("Failed to initialize clusters without empty components.")

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

    def _regularise_params(self, params: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
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
            s *= (1.0 - self._s_shrink)
            rho, lam = katojones._rho_lam_from_aux(gamma, s, phi)
            rho = float(np.clip(rho, 0.0, 1.0 - self._rho_margin))
            lam = float(np.mod(lam, 2.0 * np.pi))

        return mu, gamma, rho, lam

    def _violates_or_degenerate(self, params: Tuple[float, float, float, float]) -> bool:
        mu, gamma, rho, lam = params
        if not np.all(np.isfinite([mu, gamma, rho, lam])):
            return True
        if gamma <= self._gamma_floor or rho >= 1.0 - self._rho_margin:
            return True
        limit = (1.0 - gamma) ** 2
        if limit <= 0.0:
            return True
        return self._constraint_value(gamma, rho, lam) >= limit - self._constraint_margin / 2.0

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
                katojones.logpdf(alpha, mu=mu[k], gamma=gamma[k], rho=rho[k], lam=lam[k])
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
                initial_params = self._regularise_params((mu[k], gamma[k], rho[k], lam[k]))
                for start_params in (initial_params, moment_est):
                    try:
                        est, _info = katojones.fit(
                            alpha,
                            method="mle",
                            weights=w,
                            initial=start_params,
                            optimizer="L-BFGS-B",
                            options={"maxiter": self.mle_maxiter, "ftol": self.mle_ftol},
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
        nLL = self._nll(self.alpha, self.p_, self.mu_, self.gamma_, self.rho_, self.lam_)
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
        log_resp = self._log_gamma(alpha, self.p_, self.mu_, self.gamma_, self.rho_, self.lam_)
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
        for pc, muc, gc, rhoc, lamc in zip(self.p_, self.mu_, self.gamma_, self.rho_, self.lam_):
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
            raise TypeError("`distribution` must be an instance of CircularContinuous (e.g. vonmises).")
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
                inferred_names = [name.strip() for name in shapes.split(",") if name.strip()]

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
            (fit_method is None or (isinstance(fit_method, str) and fit_method.lower() == "auto"))
            and distribution_name in {"vonmises_flattopped", "inverse_batschelet"}
        ):
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

    def _array_to_params(self, values: Union[Dict[str, float], Tuple[float, ...], List[float]]) -> Dict[str, float]:
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
                params_est, _info = self.distribution.fit(alpha, return_info=True, **fit_options)
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

    def _initialize(self, alpha: np.ndarray) -> Tuple[List[Dict[str, float]], np.ndarray]:
        n = alpha.size
        if self.n_clusters > n:
            raise ValueError("Number of clusters cannot exceed number of observations during initialisation.")

        for _ in range(128):
            labels = self._rng.integers(self.n_clusters, size=n)
            if all(np.any(labels == c) for c in range(self.n_clusters)):
                break
        else:
            raise RuntimeError("Failed to initialise mixture components without empty clusters.")

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
                params_updated.append(self._fit_component(alpha, weights, current_params=params_list[c]))
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
                print(f"Reached max iter {self.n_iters}. Final nLL = {nLL_history[self.n_iters - 1]:.3f}\n")

        self.nLL = nLL_history[~np.isnan(nLL_history)]
        self.p_ = p
        self.params_ = params_list
        self.param_matrix_ = np.vstack([self._params_to_array(params) for params in params_list])

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
        if (
            self.n_init_clusters is None
            or self.n_init_clusters >= n_samples
        ):
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
            means = {cid: circ_mean_and_r(alpha[indices])[0] for cid, indices in clusters.items()}
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
            merges.append([cid_i, cid_j, float(abs(best_dist)), float(len(merged_indices))])

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
        self.merges_ = np.array(merges, dtype=float) if merges else np.empty((0, 4), dtype=float)
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

        k = self.centers_.size
        labels = np.zeros(alpha.size, dtype=int)
        for i, angle in enumerate(alpha):
            distances = [abs(circ_dist(angle, center, metric=self.metric)) for center in self.centers_]
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
