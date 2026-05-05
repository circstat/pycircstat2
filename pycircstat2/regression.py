import re
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from hea import lm as _hea_lm
from scipy.linalg import lstsq
from scipy.special import i0e
from scipy.stats import chi2, norm, t as student_t

from .utils import A1, A1inv, significance_code

__all__ = ["CLRegression", "CCRegression", "LCRegression"]


def _safe_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


class CLRegression:
    """
    Circular-Linear Regression.

    Fits a circular response to linear predictors using iterative optimization.

    Parameters
    ----------
    formula : str, optional
        A formula string like 'θ ~ x1 + x2 + x3' specifying the model.
    data : pd.DataFrame, optional
        A pandas DataFrame containing the response and predictors.
    theta : np.ndarray, optional
        A numpy array of circular response values in radians.
    X : np.ndarray, optional
        A numpy array of predictor values.
    model_type : str, optional
        Type of model to fit. Must be one of 'mean', 'kappa', or 'mixed'.

        - 'mean': Fit a model for the mean direction.
        - 'kappa': Fit a model for the concentration parameter.
        - 'mixed': Fit a mixed circular-linear model.

    beta0 : np.ndarray, optional
        Initial values for the beta coefficients.
    alpha0 : float, optional
        Initial value for the intercept.
    gamma0 : np.ndarray, optional
        Initial values for the gamma coefficients.
    tol : float, optional
        Convergence tolerance for the optimization.
    max_iter : int, optional
        Maximum number of iterations for the optimization.
    verbose : bool, optional
        Whether to print optimization progress.

    Attributes
    ----------
    result : dict
        A dictionary containing the estimated coefficients and other statistics.

        - beta : np.ndarray
            Estimated beta coefficients for the mean direction. Used by
            'mean' and 'mixed' models; zero for 'kappa'.
        - alpha : float
            Estimated intercept for the concentration parameter.
        - gamma : np.ndarray
            Estimated coefficients for the concentration parameter.
        - mu : float
            Estimated mean direction of the circular response.
        - kappa : float or np.ndarray
            Concentration parameter. Scalar for 'mean'; n-element array
            of per-observation values κ_i = exp(α + X_iᵀγ) for 'kappa'
            and 'mixed'.
        - log_likelihood : float
            Log-likelihood of the model.

    Methods
    -------
    summary()
        Print the coefficient table, mean direction, concentration, and fit
        metrics.
    predict(X_new)
        Predict mean direction at new X (constant μ for ``model_type='kappa'``).
    predict_kappa(X_new)
        Predict per-observation κ̂(X) for ``model_type`` in
        ``{'kappa', 'mixed'}``.
    plot(figsize=None, n_curve=200, axes=None)
        Two-panel diagnostic figure (fit overlay / κ curve / residuals,
        depending on ``model_type`` and dimensionality).
    AIC(), BIC()
        Information criteria for the fitted model.

    Notes
    -----
    The 'mean' branch is ported from ``lm.circular.cl`` in the ``circular``
    R package (Agostinelli & Lund); SE formulas follow Fisher (1993)
    eq. 6.62-6.64. The 'kappa' and 'mixed' branches extend that framework
    to model the concentration as a log-linear function of predictors,
    following Fisher (1993) §6.4.3-§6.4.4 (eq. 6.81, 6.82, 6.86, 6.87).
    Per-observation SE for κ̂_i uses the delta method on (α̂, γ̂).

    References
    ----------
    - Fisher, N. I. (1993). Statistical analysis of circular data. Cambridge University Press.
    - Pewsey, A., Neuhäuser, M., & Ruxton, G. D. (2014) Circular Statistics in R. Oxford University Press.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        theta: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        model_type: str = "mixed",
        beta0: Union[np.ndarray, None] = None,
        alpha0: Union[float, None] = None,
        gamma0: Union[np.ndarray, None] = None,
        tol: float = 1e-8,
        max_iter: int = 100,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.model_type = model_type

        # Parse inputs
        if formula and data is not None:
            theta_arr, X_arr, feature_names = self._parse_formula(formula, data)
        elif theta is not None and X is not None:
            feature_names = None
            theta_arr, X_arr = theta, X
        else:
            raise ValueError("Provide either a formula + data or theta and X.")

        self.theta, self.X = self._prepare_design(theta_arr, X_arr)
        if feature_names is None:
            self.feature_names = [f"x{i}" for i in range(self.X.shape[1])]
        else:
            self.feature_names = feature_names

        # Validate model type
        if model_type not in ["mean", "kappa", "mixed"]:
            raise ValueError("Model type must be 'mean', 'kappa', or 'mixed'.")

        # Initialize parameters
        p = self.X.shape[1]
        self.alpha = float(alpha0) if alpha0 is not None else 0.0
        self.beta = self._coerce_vector(beta0, p, name="beta")
        self.gamma = self._coerce_vector(gamma0, p, name="gamma")

        # Fit the model
        self.result = self._fit()

    @staticmethod
    def _coerce_vector(vec: Optional[np.ndarray], length: int, name: str) -> np.ndarray:
        if vec is None:
            return np.zeros(length, dtype=float)
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size != length:
            raise ValueError(f"Initial {name} must have length {length} (got {arr.size}).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Initial {name} contains non-finite values.")
        return arr

    @staticmethod
    def _prepare_design(theta: Iterable[float], X: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        if theta_arr.size == 0:
            raise ValueError("`theta` must contain at least one observation.")
        if not np.all(np.isfinite(theta_arr)):
            raise ValueError("`theta` contains non-finite values.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.ndim != 2:
            raise ValueError("`X` must be convertible to a 2D numeric array.")
        if X_arr.shape[0] != theta_arr.size:
            raise ValueError("`theta` and `X` must have matching numbers of rows.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("`X` contains non-finite values.")
        return theta_arr, X_arr

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(
                f"Formula must contain exactly one '~'; got: {formula!r}"
            )
        theta_col, x_cols = parts
        theta_series = data[theta_col.strip()]
        if theta_series.isnull().any():
            raise ValueError("Response column contains missing values.")
        theta = theta_series.to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+") if col.strip()]
        if not x_cols:
            raise ValueError(f"No predictors found in formula: {formula!r}")
        X_df = data[x_cols]
        if X_df.isnull().any().any():
            raise ValueError("Predictor columns contain missing values.")
        X = X_df.to_numpy()
        return theta, X, x_cols

    @staticmethod
    def _A1_prime(kappa: np.ndarray) -> np.ndarray:
        a1 = A1(kappa)
        return 1 - a1 / kappa - a1**2

    @staticmethod
    def _safe_exp_kappa(eta: np.ndarray) -> np.ndarray:
        # Bound the log-concentration to avoid exp overflow during iterations.
        # exp(±50) ≈ {5e21, 2e-22}, comfortably finite.
        return np.exp(np.clip(eta, -50.0, 50.0))

    @staticmethod
    def _log_i0(kappa: np.ndarray) -> np.ndarray:
        # log I_0(κ) computed via the exponentially scaled Bessel to stay finite
        # for large κ (raw i0 overflows around κ ≈ 710).
        return np.asarray(kappa) + np.log(i0e(kappa))

    @staticmethod
    def _delta_se_kappa(
        kappa: np.ndarray, X1: np.ndarray, cov_alpha_gamma: np.ndarray
    ) -> np.ndarray:
        # κ_i = exp(α + X_iᵀ γ); ∂κ_i/∂(α,γ) = κ_i · z_i with z_i = [1, X_i].
        # Var(κ_i) ≈ κ_i² · z_iᵀ Σ z_i (delta method).
        z_cov = X1 @ cov_alpha_gamma
        quad = np.einsum("ij,ij->i", z_cov, X1)
        var_kappa = (kappa**2) * np.clip(quad, 0.0, None)
        return np.sqrt(var_kappa)

    def _fit(self):
        theta = self.theta
        n = len(theta)
        X = self.X
        X1 = np.column_stack((np.ones(n), X))  # Add intercept
        beta, alpha, gamma = self.beta, self.alpha, self.gamma
        diff = self.tol + 1
        log_likelihood_old = -np.inf

        # Tiny ridge added to the normal-equation LHS to keep solves finite
        # when XtX is near-singular. Hoisted out of the loop body.
        ridge_X = 1e-8 * np.eye(X.shape[1])
        ridge_X1 = 1e-8 * np.eye(X1.shape[1])

        for iter_count in range(self.max_iter):
            if self.model_type == "mean":
                # Step 1: Compute mu and kappa
                raw_deviation = theta - 2 * np.arctan(X @ beta)
                S = np.mean(np.sin(raw_deviation))
                C = np.mean(np.cos(raw_deviation))
                R = np.hypot(S, C)
                kappa = float(A1inv(R))
                mu = np.arctan2(S, C)

                # Step 2: Update beta
                denom = 1 + (X @ beta) ** 2
                G = 2 * X / denom[:, None]
                weight = float(kappa * A1(kappa))
                u = kappa * np.sin(raw_deviation - mu)
                XtX = G.T @ G
                rhs = G.T @ u + weight * XtX @ beta
                mat = weight * XtX + ridge_X
                beta_new = _safe_solve(mat, rhs)
                alpha_new, gamma_new = alpha, gamma

                # Log-likelihood
                log_likelihood = -n * float(self._log_i0(kappa)) + kappa * np.sum(
                    np.cos(raw_deviation - mu)
                )

            elif self.model_type == "kappa":
                # Step 1: Compute mu and kappa
                kappa = self._safe_exp_kappa(alpha + X @ gamma)
                S = float(np.sum(kappa * np.sin(theta)))
                C = float(np.sum(kappa * np.cos(theta)))
                mu = np.arctan2(S, C)

                # Step 2: Update gamma
                a1_kappa = A1(kappa)
                # Floor A1'(κ) to keep the IRLS step finite when some κ_i are
                # very large (A1'(κ) → 0 as κ → ∞ ⇒ y_gamma blows up).
                a1_prime = np.maximum(self._A1_prime(kappa), 1e-12)
                residuals_gamma = np.cos(theta - mu) - a1_kappa
                y_gamma = residuals_gamma / (a1_prime * kappa)
                weights = (kappa**2) * a1_prime
                XtWX = X1.T @ (weights[:, None] * X1)
                XtWy = X1.T @ (weights * y_gamma)
                update = _safe_solve(XtWX + ridge_X1, XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]
                beta_new = beta
                # Log-likelihood
                log_likelihood = -np.sum(self._log_i0(kappa)) + np.sum(
                    kappa * np.cos(theta - mu)
                )

            elif self.model_type == "mixed":
                # Step 1: Compute mu and kappa
                kappa = self._safe_exp_kappa(alpha + X @ gamma)
                raw_deviation = theta - 2 * np.arctan(X @ beta)
                S = np.sum(kappa * np.sin(raw_deviation))
                C = np.sum(kappa * np.cos(raw_deviation))
                mu = np.arctan2(S, C)

                # Step 2: Update beta — Fisher scoring step from current β.
                # Score s(β) = Gᵀ (κ ⊙ sin(rdev − μ)); info I(β) = Gᵀ diag(κ A1(κ)) G.
                # β_new solves I β_new = I β + s.
                denom = 1 + (X @ beta) ** 2
                G = 2 * X / denom[:, None]
                weights_beta = kappa * A1(kappa)
                XtWX_beta = G.T @ (weights_beta[:, None] * G)
                u_beta = kappa * np.sin(raw_deviation - mu)
                rhs_beta = G.T @ u_beta + XtWX_beta @ beta
                beta_new = _safe_solve(XtWX_beta + ridge_X, rhs_beta)

                # Step 3: Update gamma
                a1_kappa = A1(kappa)
                a1_prime = np.maximum(self._A1_prime(kappa), 1e-12)
                residuals_gamma = np.cos(raw_deviation - mu) - a1_kappa
                y_gamma = residuals_gamma / (a1_prime * kappa)
                weights_gamma = (kappa**2) * a1_prime
                XtWX = X1.T @ (weights_gamma[:, None] * X1)
                XtWy = X1.T @ (weights_gamma * y_gamma)
                update = _safe_solve(XtWX + ridge_X1, XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]

                # Log-likelihood
                log_likelihood = -np.sum(self._log_i0(kappa)) + np.sum(
                    kappa * np.cos(raw_deviation - mu)
                )

            # Convergence check
            diff = np.abs(log_likelihood - log_likelihood_old)
            if self.verbose:
                print(
                    f"Iteration {iter_count + 1}: Log-Likelihood = {log_likelihood:.5f}, diff = {diff:.2e}"
                )
            if diff < self.tol:
                break

            beta, alpha, gamma = beta_new, alpha_new, gamma_new
            log_likelihood_old = log_likelihood
        else:
            warnings.warn(
                f"CLRegression did not converge in {self.max_iter} iterations "
                f"(last diff={diff:.2e}, tol={self.tol:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )

        result = {
            "beta": beta,
            "alpha": alpha,
            "gamma": gamma,
            "mu": mu,
            "kappa": kappa,
            "log_likelihood": log_likelihood,
        }

        se_result = self._compute_standard_errors(result)

        result.update(se_result)

        return result

    def _compute_standard_errors(self, result):
        """
        Compute standard errors for the parameters based on the fitted model.
        """
        theta = self.theta
        X = self.X
        n = len(theta)
        kappa = result["kappa"]
        beta = result["beta"]

        se_results = {}

        if self.model_type == "mean":
            # Mean Direction Model
            denom = 1 + (X @ beta) ** 2
            G = 2 * X / denom[:, None]
            weight = float(kappa * A1(kappa))
            XtAX = weight * (G.T @ G)
            cov_beta = _safe_inverse(XtAX)
            se_beta = np.sqrt(np.diag(cov_beta))

            denom_mu = max((n - X.shape[1]) * kappa * A1(kappa), 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)
            denom_kappa = n * (1 - A1(kappa) ** 2 - A1(kappa) / kappa)
            se_kappa = np.sqrt(1 / max(denom_kappa, 1e-12))

            se_results.update(
                {
                    "se_beta": se_beta,
                    "se_mu": se_mu,
                    "se_kappa": se_kappa,
                }
            )

        elif self.model_type == "kappa":
            # Concentration Parameter Model
            X1 = np.column_stack((np.ones(n), X))  # Add intercept
            weights = (kappa**2) * self._A1_prime(kappa)
            XtWX = X1.T @ (weights[:, None] * X1)

            cov_gamma_alpha = _safe_inverse(XtWX)
            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            # Fisher (1993), eq. 6.82: σ̂_μ = (Σ κ̂_i A1(κ̂_i) − 1/2)^(−1/2).
            denom_mu = max(float(np.sum(kappa * A1(kappa))) - 0.5, 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)

            se_kappa = self._delta_se_kappa(kappa, X1, cov_gamma_alpha)

            se_results.update(
                {
                    "se_alpha": se_alpha,
                    "se_gamma": se_gamma,
                    "se_mu": se_mu,
                    "se_kappa": se_kappa,
                }
            )

        elif self.model_type == "mixed":
            # Mixed Model
            denom = 1 + (X @ beta) ** 2
            G = 2 * X / denom[:, None]
            weights_beta = kappa * A1(kappa)
            XtGKGX = G.T @ (weights_beta[:, None] * G)

            cov_beta = _safe_inverse(XtGKGX)
            se_beta = np.sqrt(np.diag(cov_beta))

            X1 = np.column_stack((np.ones(n), X))  # Add intercept
            weights_gamma = (kappa**2) * self._A1_prime(kappa)
            XtWX_gamma = X1.T @ (weights_gamma[:, None] * X1)

            cov_gamma_alpha = _safe_inverse(XtWX_gamma)
            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            # Fisher (1993), eq. 6.82: σ̂_μ = (Σ κ̂_i A1(κ̂_i) − 1/2)^(−1/2).
            denom_mu = max(float(np.sum(kappa * A1(kappa))) - 0.5, 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)
            se_kappa = self._delta_se_kappa(kappa, X1, cov_gamma_alpha)
            se_results.update(
                {
                    "se_beta": se_beta,
                    "se_alpha": se_alpha,
                    "se_gamma": se_gamma,
                    "se_mu": se_mu,
                    "se_kappa": se_kappa,
                }
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return se_results

    def AIC(self):
        """
        Calculate Akaike Information Criterion (AIC).
        """
        if self.result is None:
            raise ValueError("Model must be fitted before calculating AIC.")

        log_likelihood = self.result["log_likelihood"]
        if self.model_type == "mean":
            n_params = len(self.result["beta"])  # Only beta
        elif self.model_type == "kappa":
            n_params = 1 + len(self.result["gamma"])  # alpha + gamma
        elif self.model_type == "mixed":
            n_params = (
                1 + len(self.result["beta"]) + len(self.result["gamma"])
            )  # alpha + beta + gamma
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return -2 * log_likelihood + 2 * n_params

    def BIC(self):
        """
        Calculate Bayesian Information Criterion (BIC).
        """
        if self.result is None:
            raise ValueError("Model must be fitted before calculating BIC.")

        log_likelihood = self.result["log_likelihood"]
        n = len(self.theta)
        if self.model_type == "mean":
            n_params = len(self.result["beta"])  # Only beta
        elif self.model_type == "kappa":
            n_params = 1 + len(self.result["gamma"])  # alpha + gamma
        elif self.model_type == "mixed":
            n_params = (
                1 + len(self.result["beta"]) + len(self.result["gamma"])
            )  # alpha + beta + gamma
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return -2 * log_likelihood + n_params * np.log(n)

    def predict(self, X_new):
        """
        Predict circular response values for new predictor values.

        Parameters
        ----------
        X_new: array-like, shape (n_samples, n_features)
            New predictor data.

        Returns
        -------
        theta_new: array-like, shape(n_samples, )
            New circular response values.
        """
        if self.result is None:
            raise ValueError("Model must be fitted before making predictions.")

        X_arr = np.asarray(X_new, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("`X_new` contains non-finite values.")
        if X_arr.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Expected {self.X.shape[1]} predictors, received {X_arr.shape[1]}."
            )

        mu = self.result["mu"]
        if self.model_type == "kappa":
            # Conditional mean is constant μ (β is not part of the model).
            return np.full(X_arr.shape[0], np.mod(mu, 2 * np.pi))

        beta = self.result.get("beta")
        if beta is None or np.any(~np.isfinite(beta)):
            raise ValueError("Model does not contain beta coefficients for prediction.")
        return np.mod(mu + 2 * np.arctan(X_arr @ beta), 2 * np.pi)

    def predict_kappa(self, X_new) -> np.ndarray:
        """Predict per-observation concentration κ_i = exp(α + X_iᵀγ).

        Only meaningful for ``model_type`` in ``{"kappa", "mixed"}``; for the
        ``"mean"`` model the concentration is a single scalar already in
        ``self.result["kappa"]``.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features) or (n_features,)
            New predictor data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.model_type == "mean":
            raise ValueError(
                "predict_kappa() is for model_type in {'kappa', 'mixed'}; "
                "the 'mean' model has a scalar κ in result['kappa']."
            )
        X_arr = np.asarray(X_new, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Expected {self.X.shape[1]} predictors, received {X_arr.shape[1]}."
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("`X_new` contains non-finite values.")
        return self._predict_kappa(X_arr)

    def _predict_kappa(self, X_arr: np.ndarray) -> np.ndarray:
        """Internal: numpy-only κ̂(X) without input validation."""
        alpha = self.result["alpha"]
        gamma = self.result["gamma"]
        eta = alpha + X_arr @ gamma
        return np.exp(np.clip(eta, -50.0, 50.0))

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        n_curve: int = 200,
        axes=None,
    ):
        """Two-panel diagnostic figure.

        Layout depends on ``model_type`` and the number of predictors:

        - 1D X, ``model_type`` in ``{"mean", "mixed"}``: fit overlay
          (data and curve replicated at θ and θ+2π) and residuals vs X.
        - 1D X, ``model_type`` == ``"kappa"``: data scatter with the
          constant μ line, plus fitted κ_i = exp(α + X_iᵀγ) on the right.
        - Multi-D X: residuals vs fitted angle, plus residual histogram.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_features = self.X.shape[1]
        is_1d = n_features == 1

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5))
        else:
            axes = list(axes)
            if len(axes) != 2:
                raise ValueError("`axes` must be a sequence of length 2.")
            fig = axes[0].figure

        if not is_1d:
            self._plot_residual_diagnostic(axes)
            fig.tight_layout()
            return fig

        x_data = self.X[:, 0]
        theta_data = np.mod(self.theta, 2 * np.pi)
        feature_label = self.feature_names[0]

        x_grid = np.linspace(x_data.min(), x_data.max(), n_curve)
        x_grid_2d = x_grid[:, None]

        ax = axes[0]
        if self.model_type in ("mean", "mixed"):
            mu = self.result["mu"]
            beta = self.result["beta"]
            curve = np.mod(mu + 2 * np.arctan(x_grid * beta[0]), 2 * np.pi)
            curve_plot = curve.astype(float).copy()
            jumps = np.where(np.abs(np.diff(curve)) > np.pi)[0]
            curve_plot[jumps] = np.nan
            ax.plot(x_grid, curve_plot, color="C1", lw=2, label="fit")
            ax.plot(x_grid, curve_plot + 2 * np.pi, color="C1", lw=2)
        else:  # kappa-only: conditional mean is the constant μ.
            mu = self.result["mu"]
            ax.axhline(mu, color="C1", lw=2, label=f"μ = {mu:.3f}")
            ax.axhline(mu + 2 * np.pi, color="C1", lw=2)

        ax.scatter(x_data, theta_data, color="C0", s=20, alpha=0.6, edgecolors="none", label="data")
        ax.scatter(x_data, theta_data + 2 * np.pi, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax.set_ylim(0, 4 * np.pi)
        ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        ax.set_yticklabels(["0", "π", "2π", "3π", "4π"])
        ax.set_xlabel(feature_label)
        ax.set_ylabel("θ")
        ax.set_title("Fit overlay")
        ax.legend(loc="best", frameon=False)

        ax = axes[1]
        if self.model_type == "kappa":
            kappa_curve = self._predict_kappa(x_grid_2d)
            ax.plot(x_grid, kappa_curve, color="C1", lw=2)
            ax.set_ylabel("κ̂(X) = exp(α + Xγ)")
            ax.set_title("Fitted concentration")
        else:
            residuals = np.angle(np.exp(1j * (self.theta - self._fitted_mean())))
            ax.scatter(x_data, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs X")
        ax.set_xlabel(feature_label)

        fig.tight_layout()
        return fig

    def _fitted_mean(self) -> np.ndarray:
        """Conditional mean angle at the training X (constant μ for kappa-only)."""
        mu = self.result["mu"]
        if self.model_type == "kappa":
            return np.full(self.theta.shape, mu)
        beta = self.result["beta"]
        return mu + 2 * np.arctan(self.X @ beta)

    def _plot_residual_diagnostic(self, axes) -> None:
        residuals = np.angle(np.exp(1j * (self.theta - self._fitted_mean())))
        fitted = np.mod(self._fitted_mean(), 2 * np.pi)
        ax = axes[0]
        ax.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("Fitted θ (rad)")
        ax.set_ylabel("Residual (rad)")
        ax.set_title("Residuals vs fitted")

        ax = axes[1]
        ax.hist(residuals, bins=20, color="C0", alpha=0.7, edgecolor="black")
        ax.axvline(0.0, color="k", lw=0.5)
        ax.set_xlabel("Residual (rad)")
        ax.set_ylabel("Count")
        ax.set_title("Residual histogram")

    @staticmethod
    def _two_sided_p(t_value: float) -> float:
        if np.isnan(t_value):
            return np.nan
        return float(2.0 * norm.sf(np.abs(t_value)))

    def summary(self):
        if self.result is None:
            raise ValueError("Model must be fitted before summarizing.")

        # Title based on model type
        if self.model_type == "mean":
            print("\nCircular Regression for the Mean Direction\n")
        elif self.model_type == "kappa":
            print("\nCircular Regression for the Concentration Parameter\n")
        elif self.model_type == "mixed":
            print("\nMixed Circular-Linear Regression\n")

        # Call
        print("Call:")
        print(f"  CLRegression(model_type='{self.model_type}')\n")

        # Coefficients for mean direction (Beta)
        se_beta = self.result.get("se_beta")
        if (
            self.model_type in ["mean", "mixed"]
            and self.result.get("beta") is not None
            and se_beta is not None
        ):
            print("Coefficients for Mean Direction (Beta):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)'}"
            )
            for i, coef in enumerate(self.result["beta"]):
                se_val = se_beta[i]
                t_value = coef / se_val if se_val else np.nan
                p_value = self._two_sided_p(t_value)
                print(
                    f"β{i:<3} {coef:<12.5f} {se_val:<12.5f} {t_value:<10.2f} {p_value:<12.5f}{significance_code(p_value):<3}"
                )

        # Coefficients for concentration parameter (Gamma)
        se_gamma = self.result.get("se_gamma")
        se_alpha = self.result.get("se_alpha")
        if (
            self.model_type in ["kappa", "mixed"]
            and self.result.get("gamma") is not None
            and se_gamma is not None
            and se_alpha is not None
        ):
            print("\nCoefficients for Concentration (Gamma):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)':<12}"
            )
            # Report alpha as the first coefficient
            alpha = self.result["alpha"]
            t_value_alpha = alpha / se_alpha if se_alpha else np.nan
            p_value_alpha = self._two_sided_p(t_value_alpha)
            print(
                f"α{'':<5} {alpha:<12.5f} {se_alpha:<12.5f} {t_value_alpha:<10.2f} {p_value_alpha:<12.5f}{significance_code(p_value_alpha)}"
            )
            for i, coef in enumerate(self.result["gamma"]):
                se_val = se_gamma[i]
                t_value = coef / se_val if se_val else np.nan
                p_value = self._two_sided_p(t_value)
                print(
                    f"γ{i:<5} {coef:<12.5f} {se_val:<12.5f} {t_value:<10.2f} {p_value:<12.5f}{significance_code(p_value)}"
                )

        # Summary for mu and kappa
        print("\nSummary:")
        print("  Mean Direction (mu) in radians:")
        mu = self.result["mu"]
        se_mu = self.result.get("se_mu")
        if se_mu is not None:
            print(f"    μ: {mu:.5f} (SE: {se_mu:.5f})")
        else:
            print(f"    μ: {mu:.5f}")

        print("\n  Concentration Parameter (kappa):")
        kappa = self.result["kappa"]
        se_kappa = self.result.get("se_kappa")
        if isinstance(kappa, np.ndarray):
            print("    Index    kappa        Std. Error")
            for i, k in enumerate(kappa, start=1):
                se_val = se_kappa[i - 1] if se_kappa is not None else float("nan")
                print(f"    [{i}]    {k:>10.5f}    {se_val:>10.5f}")
            # Per-obs κ_i are correlated (shared α, γ), so averaging individual
            # SEs is not the SE of the mean — report only the point estimate.
            print(f"    Mean:    {np.mean(kappa):.5f}")
        else:
            if se_kappa is not None:
                print(f"    κ: {kappa:.5f} (SE: {se_kappa:.5f})")
            else:
                print(f"    κ: {kappa:.5f}")

        # Summary for model fit metrics
        print("\nModel Fit Metrics:\n")
        print(f"{'Metric':<12} {'Value':<12}")
        log_likelihood = self.result.get("log_likelihood", float("nan"))
        nll = -log_likelihood  # Negative log-likelihood
        print(f"{'nLL':<12} {nll:<12.5f}")
        print(f"{'AIC':<12} {self.AIC():<12.5f}")
        print(f"{'BIC':<12} {self.BIC():<12.5f}")

        # Notes
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("p-values are approximated using the normal distribution.\n")


class CCRegression:
    """
    Circular-Circular Regression.

    Fits a circular response to circular predictors using a specified order of harmonics.

    Parameters
    ----------
    theta : np.ndarray
        A numpy array of circular response values in radians.
    x : np.ndarray
        A numpy array of circular predictor values in radians.
    order : int, optional
        Order of harmonics to include in the model (default is 1).
    level : float, optional
        Significance level for testing higher-order terms (default is 0.05).

    Attributes
    ----------
    rho : float
        Circular correlation coefficient.
    fitted : np.ndarray
        Fitted values of the circular response in radians.
    residuals : np.ndarray
        Residuals of the circular response in radians.
    coefficients : dict
        Coefficients of the cos and sin terms for each harmonic order.
    p_values : np.ndarray
        P-values for higher-order terms.
    kappa : float
        Concentration of the residuals, A1⁻¹(mean cos(residuals)).
    A_k : float
        Mean cosine of the residuals (input to A1⁻¹).
    message : str
        Message indicating the significance of higher-order terms.

    Methods
    -------
    summary()
        Print the harmonic coefficient table, ρ, residual κ, and the test
        of higher-order terms.
    predict(x)
        Predict the circular response at new ``x``.
    plot(figsize=None, n_curve=200, axes=None)
        Two-panel diagnostic figure (fit overlay for 1-D ``x``; residuals
        vs fitted + histogram for multi-D).

    Notes
    -----
    The implementation is ported from the ``lm.circular.cc`` in the
    ``circular`` R package (Agostinelli & Lund).

    References
    ----------
    - Jammalamadaka, S. R., & Sengupta, A. (2001) Topics in Circular Statistics. World Scientific.
    - Pewsey, A., Neuhäuser, M., & Ruxton, G. D. (2014) Circular Statistics in R. Oxford University Press.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        theta: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        order: int = 1,
        level: float = 0.05,
    ):
        if formula and data is not None:
            theta_arr, x_arr, self.feature_names = self._parse_formula(formula, data)
            self.theta = self._validate_input(theta_arr)
            self.x = self._validate_input(x_arr)
            if self.x.ndim == 1:
                self.x = self.x[:, None]
        elif theta is not None and x is not None:
            self.theta = self._validate_input(theta)
            self.x = self._validate_input(x)
            if self.x.ndim == 1:
                self.x = self.x[:, None]
            self.feature_names = [f"x{i}" for i in range(self.x.shape[1])]
        else:
            raise ValueError("Provide either a formula + data or theta and x.")

        if self.theta.ndim != 1:
            raise ValueError(
                f"`theta` must be 1-dimensional (got shape {self.theta.shape})."
            )
        if self.theta.size != self.x.shape[0]:
            raise ValueError("`theta` and `x` must have matching numbers of rows.")

        self.order = order
        self.level = level

        if self.order < 1:
            raise ValueError("`order` must be a positive integer.")
        if not (0 < self.level < 1):
            raise ValueError("`level` must lie between 0 and 1.")

        n_params = 1 + 2 * self.x.shape[1] * self.order
        if self.theta.size <= n_params:
            raise ValueError(
                f"order={self.order} requires more than {n_params} observations "
                f"(got {self.theta.size}); reduce `order` or provide more data."
            )

        # Fit the model
        self.result = self._fit()

    @staticmethod
    def _validate_input(arr: np.ndarray) -> np.ndarray:
        """Validate angular input and wrap to ``[0, 2π)``.

        The model is 2π-periodic, so values are normalised modulo ``2π``.
        Input is expected to be in radians; degrees would silently wrap to
        the wrong range (e.g. 360° → 360 mod 2π ≈ 5.97 rad ≈ 342°).
        """
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.ndim == 0:
            raise ValueError("Input must be at least one-dimensional.")
        if not np.all(np.isfinite(arr_np)):
            raise ValueError("Circular input contains non-finite values.")
        if arr_np.size and float(np.max(np.abs(arr_np))) > 4 * np.pi:
            warnings.warn(
                "Circular input contains values with |x| > 4π; expected "
                "radians. Degree-valued input will be silently wrapped "
                "modulo 2π and produce incorrect results — use np.deg2rad.",
                UserWarning,
                stacklevel=3,
            )
        return np.mod(arr_np, 2 * np.pi)

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(
                f"Formula must contain exactly one '~'; got: {formula!r}"
            )
        theta_col, x_cols = parts
        theta = data[theta_col.strip()].to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+") if col.strip()]
        if not x_cols:
            raise ValueError(f"No predictors found in formula: {formula!r}")
        X = data[x_cols].to_numpy()
        return theta, X, x_cols

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Harmonic design matrix [1 | cos(kx_j) | sin(kx_j)] for given x."""
        if x.ndim == 1:
            x = x[:, None]
        n, n_features = x.shape
        cos_terms, sin_terms = [], []
        for j in range(n_features):
            for k in range(1, self.order + 1):
                cos_terms.append(np.cos(k * x[:, j]))
                sin_terms.append(np.sin(k * x[:, j]))
        return np.column_stack([np.ones(n)] + cos_terms + sin_terms)

    def _fit(self):
        n = self.x.shape[0]
        order = self.order
        n_features = self.x.shape[1]

        # Track which (feature, harmonic) each design column corresponds to.
        cos_labels: List[Tuple[int, int]] = []
        sin_labels: List[Tuple[int, int]] = []
        for j in range(n_features):
            for k in range(1, order + 1):
                cos_labels.append((j, k))
                sin_labels.append((j, k))

        Y_cos = np.cos(self.theta)
        Y_sin = np.sin(self.theta)

        X = self._design_matrix(self.x)
        beta_cos, _, _, _ = lstsq(X, Y_cos)
        beta_sin, _, _, _ = lstsq(X, Y_sin)

        # Fitted values
        cos_fit = X @ beta_cos
        sin_fit = X @ beta_sin
        fitted = np.mod(np.arctan2(sin_fit, cos_fit), 2 * np.pi)

        # Residuals (angular for diagnostics + raw OLS residuals on cos/sin)
        residuals = np.angle(np.exp(1j * (self.theta - fitted)))
        residual_cos = Y_cos - cos_fit
        residual_sin = Y_sin - sin_fit

        # Circular correlation coefficient
        rho = float(np.clip(np.sqrt(np.mean(cos_fit**2 + sin_fit**2)), 0.0, 1.0))

        # Per-coefficient OLS SEs for the cos/sin sub-models. Used by summary()
        # to print a CL/LC-style coefficient table; not part of the R parity
        # surface (lm.circular.cc returns only the coefficient matrix).
        XtX_inv = _safe_inverse(X.T @ X)
        diag_inv = np.maximum(np.diag(XtX_inv), 0.0)
        df_resid = max(n - X.shape[1], 1)
        sigma2_cos = float(residual_cos @ residual_cos) / df_resid
        sigma2_sin = float(residual_sin @ residual_sin) / df_resid
        se_beta_cos = np.sqrt(sigma2_cos * diag_inv)
        se_beta_sin = np.sqrt(sigma2_sin * diag_inv)

        # Test higher-order terms
        higher_order_cos = []
        higher_order_sin = []
        for j in range(n_features):
            x_col = self.x[:, j]
            higher_order_cos.append(np.cos((order + 1) * x_col))
            higher_order_sin.append(np.sin((order + 1) * x_col))
        if higher_order_cos:
            W = np.column_stack(higher_order_cos + higher_order_sin)
        else:
            W = np.empty((n, 0))

        # Projection matrix for the current model
        if W.size:
            M = X @ XtX_inv @ X.T
            H = W.T @ (np.eye(n) - M) @ W
            H_inv = _safe_inverse(H)
            N = W @ H_inv @ W.T

            denom_cos = float(residual_cos @ residual_cos)
            denom_sin = float(residual_sin @ residual_sin)
            adj = max(n - (2 * order + 1), 1)
            T1 = (
                adj
                * float(residual_cos @ N @ residual_cos)
                / max(denom_cos, 1e-12)
            )
            T2 = (
                adj
                * float(residual_sin @ N @ residual_sin)
                / max(denom_sin, 1e-12)
            )

            p1 = 1 - chi2.cdf(T1, W.shape[1])
            p2 = 1 - chi2.cdf(T2, W.shape[1])
            p_values = np.array([p1, p2], dtype=float)
        else:
            p_values = np.array([np.nan, np.nan], dtype=float)

        # Message about higher-order terms
        if np.all(np.isnan(p_values)):
            message = "No additional harmonics available for testing."
        elif np.all(p_values > self.level):
            message = (
                f"Higher-order terms are not significant at the {self.level} level."
            )
        else:
            message = f"Higher-order terms are significant at the {self.level} level."

        # Residual concentration (R parity): A1inv of the mean cosine of residuals.
        A_k = float(np.mean(np.cos(residuals)))
        if A_k < 0:
            warnings.warn(
                f"Mean residual cosine A_k={A_k:.4f} is negative — residuals "
                "are systematically anti-aligned with the fitted direction. "
                "κ has been clamped to 0; check for sign errors or model "
                "misspecification.",
                UserWarning,
                stacklevel=3,
            )
        kappa_residual = float(A1inv(A_k))

        return {
            "rho": rho,
            "fitted": fitted,
            "residuals": residuals,
            "coefficients": {
                "cos": beta_cos,
                "sin": beta_sin,
            },
            "se_coefficients": {
                "cos": se_beta_cos,
                "sin": se_beta_sin,
            },
            "df_resid": df_resid,
            "cos_labels": cos_labels,
            "sin_labels": sin_labels,
            "p_values": p_values,
            "A_k": A_k,
            "kappa": kappa_residual,
            "message": message,
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the circular response at new predictor values.

        Parameters
        ----------
        x : array-like, shape (n,) or (n, n_features)
            New predictor values in radians. For multi-feature models the
            second axis must match ``self.x.shape[1]``.

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted angles wrapped to ``[0, 2π)``.
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[1] != self.x.shape[1]:
            raise ValueError(
                f"Expected {self.x.shape[1]} predictor column(s); received "
                f"{x_arr.shape[1]}."
            )
        x_arr = np.mod(x_arr, 2 * np.pi)
        design = self._design_matrix(x_arr)
        cos_pred = design @ self.result["coefficients"]["cos"]
        sin_pred = design @ self.result["coefficients"]["sin"]
        return np.mod(np.arctan2(sin_pred, cos_pred), 2 * np.pi)

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        n_curve: int = 200,
        axes=None,
    ):
        """Two-panel diagnostic figure.

        For a single circular predictor, the left panel is a fit overlay
        with both data and curve replicated at ``θ`` and ``θ + 2π`` (Pewsey
        Fig 6.10 convention) so the wrap-around does not visually break the
        relationship; the right panel shows the wrapped residuals against
        the predictor.

        For multiple circular predictors, the left panel shows residuals
        vs the fitted angle and the right panel a residual histogram.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_features = self.x.shape[1]

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5))
        else:
            axes = list(axes)
            if len(axes) != 2:
                raise ValueError("`axes` must be a sequence of length 2.")
            fig = axes[0].figure

        if n_features == 1:
            x_data = self.x[:, 0]
            theta_data = self.theta
            residuals = self.result["residuals"]
            x_grid = np.linspace(0.0, 2 * np.pi, n_curve)
            theta_pred = self.predict(x_grid)
            # Break the curve where it wraps so plot() doesn't draw a
            # vertical jump connecting 2π to 0.
            theta_plot = theta_pred.astype(float).copy()
            jumps = np.where(np.abs(np.diff(theta_pred)) > np.pi)[0]
            theta_plot[jumps] = np.nan

            ax = axes[0]
            ax.plot(x_grid, theta_plot, color="C1", lw=2, label="fit")
            ax.plot(x_grid, theta_plot + 2 * np.pi, color="C1", lw=2)
            ax.scatter(x_data, theta_data, color="C0", s=20, alpha=0.6, edgecolors="none", label="data")
            ax.scatter(x_data, theta_data + 2 * np.pi, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 4 * np.pi)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
            ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
            ax.set_yticklabels(["0", "π", "2π", "3π", "4π"])
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel("θ")
            ax.set_title("Fit overlay")
            ax.legend(loc="best", frameon=False)

            ax = axes[1]
            ax.scatter(x_data, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs predictor")
        else:
            residuals = self.result["residuals"]
            fitted = self.result["fitted"]

            ax = axes[0]
            ax.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_xlabel("Fitted θ (rad)")
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs fitted")

            ax = axes[1]
            ax.hist(residuals, bins=20, color="C0", alpha=0.7, edgecolor="black")
            ax.axvline(0.0, color="k", lw=0.5)
            ax.set_xlabel("Residual (rad)")
            ax.set_ylabel("Count")
            ax.set_title("Residual histogram")

        fig.tight_layout()
        return fig

    def summary(self):
        """
        Print a summary of the regression results.
        """
        print("\nCircular-Circular Regression\n")
        print(f"Circular Correlation Coefficient (rho): {self.result['rho']:.5f}")
        print(f"Mean Residual Cosine (A_k):             {self.result['A_k']:.5f}")
        print(f"Residual Concentration (kappa):         {self.result['kappa']:.5f}\n")

        cos_coeffs = self.result["coefficients"]["cos"]
        sin_coeffs = self.result["coefficients"]["sin"]
        se_cos = self.result["se_coefficients"]["cos"]
        se_sin = self.result["se_coefficients"]["sin"]
        df_resid = self.result["df_resid"]
        cos_labels = self.result.get("cos_labels", [])
        sin_labels = self.result.get("sin_labels", [])

        intercept_label = "(Intercept)"
        cos_label_strs = [f"cos(x{f + 1},k={k})" for (f, k) in cos_labels]
        sin_label_strs = [f"sin(x{f + 1},k={k})" for (f, k) in sin_labels]
        row_labels = [intercept_label, *cos_label_strs, *sin_label_strs]
        label_width = max(12, *(len(s) for s in row_labels))

        def _print_block(title: str, coefs: np.ndarray, ses: np.ndarray) -> None:
            print(f"{title}:\n")
            print(
                f"{'':<{label_width}} {'Estimate':<12} {'Std. Error':<12} "
                f"{'t value':<10} {'Pr(>|t|)':<12}"
            )
            for label, coef, se_val in zip(row_labels, coefs, ses):
                t_val = coef / se_val if se_val else np.nan
                if np.isnan(t_val):
                    p_val = np.nan
                else:
                    p_val = float(2.0 * student_t.sf(np.abs(t_val), df=df_resid))
                print(
                    f"{label:<{label_width}} {coef:<12.5f} {se_val:<12.5f} "
                    f"{t_val:<10.2f} {p_val:<12.5f}{significance_code(p_val)}"
                )
            print()

        _print_block("Coefficients (Cosine Model)", cos_coeffs, se_cos)
        _print_block("Coefficients (Sine Model)", sin_coeffs, se_sin)

        # Higher-order test (parity with R's lm.circular.cc): jointly tests
        # whether the order+1 cos/sin pair adds explanatory power, separately
        # for the cosine and sine sub-models.
        p1, p2 = self.result["p_values"]
        print("Higher-Order Terms Test:\n")
        print(f"{'':<{label_width}} {'Pr(>χ²)':<12}")
        print(f"{'cosine model':<{label_width}} {p1:<12.5f}{significance_code(p1)}")
        print(f"{'sine model':<{label_width}} {p2:<12.5f}{significance_code(p2)}")

        print(f"\n{self.result['message']}")
        print(
            "\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )
        print(
            "Per-coefficient p-values use the t distribution; the higher-order "
            "test uses χ² (Jammalamadaka & Sengupta 2001).\n"
        )


# Markers used by LCRegression's formula parser.
# `[^\W\d_]\w*` matches a Python-style identifier including Unicode letters
# (e.g. Greek `θ`), while still forbidding a leading digit.
_LC_IDENT = r"[^\W\d_]\w*"
# Accept both `harmonic(theta, k=K)` and `harmonic(theta, K)`.
_LC_HARMONIC_RE = re.compile(
    rf"harmonic\s*\(\s*({_LC_IDENT})\s*(?:,\s*(?:k\s*=\s*)?(\d+)\s*)?\)"
)
_LC_UNSUPPORTED_RE = re.compile(r"\b(skew|flat)\s*\(")
# Coefficient-name pattern as emitted by hea: e.g. "cos(theta)",
# "sin(2 * theta)", or "cos(theta * 2)" — multiplier may appear on either side.
_LC_TRIG_RE = re.compile(
    rf"^(cos|sin)\(\s*"
    rf"(?:(?P<lmult>\d+)\s*\*\s*(?P<lvar>{_LC_IDENT})"
    rf"|(?P<rvar>{_LC_IDENT})(?:\s*\*\s*(?P<rmult>\d+))?)"
    rf"\s*\)$"
)


class LCRegression:
    """
    Linear–Circular Regression.

    Models a linear response Y as a function of a circular regressor θ
    (in radians). Backed by ``hea.lm``.

    Formula syntax
    --------------
    The right-hand side accepts either a marker that expands to a Fourier
    basis, or fully explicit ``cos(...) / sin(...)`` terms (or both).

    - ``"y ~ harmonic(theta)"`` — basic cosine model (Pewsey et al. 2014, §8.4.1)
    - ``"y ~ harmonic(theta, k=K)"`` — extended model with K harmonics (§8.4.2)
    - ``"y ~ cos(theta) + sin(theta) + cos(3*theta) + sin(3*theta)"`` —
      fully explicit; useful for non-contiguous harmonic orders
    - ``"y ~ harmonic(theta, k=2) + temperature"`` — mix marker with extra
      linear covariates

    Markers ``skew(theta)`` and ``flat(theta)`` are reserved for the
    nonlinear models in §8.4.3 / §8.4.4 and currently raise
    ``NotImplementedError`` (they need a nonlinear least-squares backend
    that ``hea`` does not yet provide).

    Parameters
    ----------
    formula : str
        R-style formula. See above.
    data : pandas.DataFrame or polars.DataFrame
        Input data. Pandas inputs are converted to polars internally.

    Attributes
    ----------
    formula : str
        The original formula passed in.
    expanded_formula : str
        Formula after marker expansion, as actually fit by ``hea.lm``.
    lm_fit : hea.lm
        The underlying linear-model fit. Use it for diagnostics
        (``.plot()``, ``.summary()``, ``.r_squared``, etc.).
    result : dict
        - coefficients : dict of {name: value} from the linear fit
        - harmonics : list of dicts, one per matched ``cos(k·θ)/sin(k·θ)``
          pair, each with ``variable``, ``k``, ``cos_coef``, ``sin_coef``,
          ``amplitude``, ``phase``, ``se_amplitude``, ``se_phase`` (the
          last two via the delta method on the (cos, sin) covariance).
        - sigma, r_squared, aic, bic : scalars
        - fitted, residuals : np.ndarray

    Notes
    -----
    The harmonic-pair detector recognises only **integer** multipliers,
    written on either side of ``*`` (e.g. ``cos(theta)``, ``cos(2*theta)``,
    ``cos(theta*2)``). A term like ``cos(0.5*theta)`` is treated as a regular
    linear predictor and won't appear in ``result['harmonics']``.

    References
    ----------
    Pewsey, A., Neuhäuser, M., Ruxton, G. D. (2014). *Circular Statistics
    in R*. Oxford University Press, §8.4.
    """

    def __init__(
        self,
        formula: str,
        data: Union[pd.DataFrame, "pl.DataFrame"],
    ):
        if not isinstance(formula, str) or "~" not in formula:
            raise ValueError(
                f"Formula must be a string containing '~'; got {formula!r}"
            )

        self.formula = formula
        self.response = formula.split("~", 1)[0].strip()
        self.data = self._to_polars(data)
        self.expanded_formula = self._expand_formula(formula)
        self.lm_fit = _hea_lm(self.expanded_formula, self.data)
        self.result = self._build_result()

    @staticmethod
    def _to_polars(data: Union[pd.DataFrame, "pl.DataFrame"]) -> "pl.DataFrame":
        if isinstance(data, pl.DataFrame):
            return data
        if isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        raise TypeError(
            f"`data` must be a pandas or polars DataFrame; got {type(data).__name__}"
        )

    @staticmethod
    def _expand_formula(formula: str) -> str:
        lhs, _, rhs = formula.partition("~")
        if _LC_UNSUPPORTED_RE.search(rhs):
            raise NotImplementedError(
                "skew() and flat() markers require a nonlinear least-squares "
                "backend (hea.nls), which is not yet available."
            )

        def _expand(match: "re.Match[str]") -> str:
            col = match.group(1)
            k = int(match.group(2)) if match.group(2) else 1
            if k < 1:
                raise ValueError(f"harmonic(..., k={k}): k must be a positive integer.")
            terms = []
            for j in range(1, k + 1):
                if j == 1:
                    terms.append(f"cos({col}) + sin({col})")
                else:
                    terms.append(f"cos({j}*{col}) + sin({j}*{col})")
            return " + ".join(terms)

        expanded_rhs = _LC_HARMONIC_RE.sub(_expand, rhs)
        return f"{lhs.strip()} ~ {expanded_rhs.strip()}"

    def _build_result(self) -> dict:
        bhat_df = self.lm_fit.bhat
        coef_names = list(bhat_df.columns)
        coef_values = list(bhat_df.row(0))
        coefficients = dict(zip(coef_names, coef_values))

        cov = np.asarray(self.lm_fit.V_bhat, dtype=float)
        column_names = list(self.lm_fit.column_names)
        name_to_idx = {n: i for i, n in enumerate(column_names)}

        # Group cos/sin terms by (variable, multiplier).
        groups: dict = {}
        for name, value in coefficients.items():
            m = _LC_TRIG_RE.match(name)
            if not m:
                continue
            func = m.group(1)
            if m.group("lvar") is not None:
                var = m.group("lvar")
                k = int(m.group("lmult"))
            else:
                var = m.group("rvar")
                k = int(m.group("rmult")) if m.group("rmult") else 1
            slot = groups.setdefault((var, k), {})
            slot[func] = (name, value)

        harmonics = []
        for (var, k), pair in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            cos_entry = pair.get("cos")
            sin_entry = pair.get("sin")
            cos_val = cos_entry[1] if cos_entry else None
            sin_val = sin_entry[1] if sin_entry else None

            amplitude = phase = se_amp = se_phase = None
            if cos_val is not None and sin_val is not None:
                amplitude = float(np.hypot(cos_val, sin_val))
                phase = float(np.arctan2(sin_val, cos_val))
                # Delta-method SEs from the (c, s) covariance block.
                ic = name_to_idx.get(cos_entry[0])
                isn = name_to_idx.get(sin_entry[0])
                if ic is not None and isn is not None and amplitude > 0:
                    var_c = cov[ic, ic]
                    var_s = cov[isn, isn]
                    cov_cs = cov[ic, isn]
                    r2 = amplitude ** 2
                    var_amp = (
                        cos_val ** 2 * var_c
                        + 2 * cos_val * sin_val * cov_cs
                        + sin_val ** 2 * var_s
                    ) / r2
                    var_phase = (
                        sin_val ** 2 * var_c
                        - 2 * cos_val * sin_val * cov_cs
                        + cos_val ** 2 * var_s
                    ) / (r2 ** 2)
                    se_amp = float(np.sqrt(max(var_amp, 0.0)))
                    se_phase = float(np.sqrt(max(var_phase, 0.0)))

            harmonics.append(
                {
                    "variable": var,
                    "k": k,
                    "cos_coef": cos_val,
                    "sin_coef": sin_val,
                    "amplitude": amplitude,
                    "phase": phase,
                    "se_amplitude": se_amp,
                    "se_phase": se_phase,
                }
            )

        residuals = self.lm_fit.residuals
        if isinstance(residuals, pl.DataFrame):
            residuals = residuals.to_numpy().ravel()
        fitted = self.lm_fit.yhat
        if isinstance(fitted, pl.DataFrame):
            fitted = fitted.to_numpy().ravel()

        return {
            "coefficients": coefficients,
            "harmonics": harmonics,
            "sigma": float(self.lm_fit.sigma),
            "r_squared": float(self.lm_fit.r_squared),
            "aic": float(self.lm_fit.AIC),
            "bic": float(self.lm_fit.BIC),
            "fitted": np.asarray(fitted, dtype=float),
            "residuals": np.asarray(residuals, dtype=float),
        }

    def predict(
        self, data: Union[pd.DataFrame, "pl.DataFrame"]
    ) -> np.ndarray:
        """Predict the linear response for new values of the regressors."""
        new = self._to_polars(data)
        out = self.lm_fit.predict(new=new)
        if isinstance(out, pl.DataFrame):
            out = out.to_numpy().ravel()
        return np.asarray(out, dtype=float)

    def summary(self) -> None:
        """Print a full diagnostic summary.

        Reuses ``hea.lm.summary()`` for the standard regression block
        (residual quantiles, coefficient table with SEs/CIs/t/p, fit metrics)
        and appends a harmonic-decomposition table with delta-method SEs and
        95% CIs for each cos/sin amplitude and phase.
        """
        print("\nLinear-Circular Regression")
        if self.expanded_formula != self.formula:
            print(f"User formula:     {self.formula}")
            print(f"Expanded formula: {self.expanded_formula}")
        print()

        # hea.lm.summary() prints to stdout and returns None.
        self.lm_fit.summary()

        if self.result["harmonics"]:
            self._print_harmonic_table()

    def _print_harmonic_table(self) -> None:
        z = float(norm.ppf(0.975))

        def _fmt(value):
            return "n/a" if value is None else f"{value:.4f}"

        def _ci(value, se):
            if value is None or se is None:
                return "n/a"
            return f"[{value - z * se:.4f}, {value + z * se:.4f}]"

        rows = []
        for h in self.result["harmonics"]:
            amp, ph = h["amplitude"], h["phase"]
            se_a, se_p = h["se_amplitude"], h["se_phase"]
            label = f"{h['variable']}, k={h['k']}"
            rows.append(
                (label, _fmt(amp), _fmt(se_a), _ci(amp, se_a), _fmt(ph), _fmt(se_p), _ci(ph, se_p))
            )

        headers = (
            "term",
            "amplitude",
            "SE",
            "CI[2.5%, 97.5%]",
            "phase",
            "SE",
            "CI[2.5%, 97.5%]",
        )
        widths = [
            max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]

        print("\nHarmonic decomposition:")
        line = "  ".join(f"{h:<{w}s}" for h, w in zip(headers, widths))
        print(line)
        print("-" * len(line))
        for r in rows:
            print("  ".join(f"{c:<{w}s}" for c, w in zip(r, widths)))
        print(
            "Phase in radians; SEs and CIs from the delta method on (cos, sin) "
            "coefficients.\n"
        )

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        n_curve: int = 200,
        ci: bool = True,
        pi: bool = False,
        level: float = 0.95,
        axes=None,
    ):
        """Two-panel diagnostic figure.

        Left:  scatter (θ, y) with the fitted curve over the data's θ range,
               optionally with confidence and/or prediction bands.
        Right: residuals vs fitted values.

        For models with extra non-circular covariates (e.g. ``y ~
        harmonic(θ) + temperature``), the curve is drawn fixing those
        covariates at their column means.

        Parameters
        ----------
        figsize : tuple, optional
            Matplotlib figure size; defaults to ``(11, 4.5)``.
        n_curve : int
            Number of θ points used to draw the fitted curve.
        ci, pi : bool
            Whether to shade a confidence band (``ci``) and/or prediction
            band (``pi``) at the requested ``level``.
        level : float
            Coverage probability for the bands (default 0.95).
        axes : sequence of matplotlib Axes, optional
            Two pre-existing axes to draw into. If omitted, a fresh figure
            is created.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if not self.result["harmonics"]:
            raise ValueError(
                "plot() requires at least one matched cos/sin pair "
                "(harmonic decomposition)."
            )

        theta_var = self.result["harmonics"][0]["variable"]
        theta_data = self.data[theta_var].to_numpy()
        y_data = self.data[self.response].to_numpy()
        fitted = self.result["fitted"]
        residuals = self.result["residuals"]

        # Build a θ grid spanning the data; hold any other covariates at their mean.
        t_lo = float(min(theta_data.min(), 0.0))
        t_hi = float(max(theta_data.max(), 2 * np.pi))
        theta_grid = np.linspace(t_lo, t_hi, n_curve)
        grid: dict = {theta_var: theta_grid}
        for col in self.data.columns:
            if col in (theta_var, self.response):
                continue
            series = self.data[col]
            if series.dtype.is_numeric():
                grid[col] = np.full(n_curve, float(series.mean()))
            else:
                # Hold non-numeric columns at their mode.
                grid[col] = [series.mode()[0]] * n_curve
        grid_df = pl.DataFrame(grid)

        yhat_df = self.lm_fit.predict(grid_df)
        yhat = np.asarray(yhat_df.to_numpy()).ravel()
        alpha = 1.0 - level
        ci_lo = ci_hi = pi_lo = pi_hi = None
        if ci:
            arr = self.lm_fit.compute_ci_yhat(yhat=yhat_df, Xnew=grid_df, alpha=alpha).to_numpy()
            ci_lo, ci_hi = arr[:, 0], arr[:, 1]
        if pi:
            arr = self.lm_fit.compute_pi_yhat(yhat=yhat_df, Xnew=grid_df, alpha=alpha).to_numpy()
            pi_lo, pi_hi = arr[:, 0], arr[:, 1]

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 4.5))
        else:
            axes = list(axes)
            if len(axes) != 2:
                raise ValueError("`axes` must be a sequence of length 2.")
            fig = axes[0].figure

        ax = axes[0]
        if pi_lo is not None:
            ax.fill_between(
                theta_grid, pi_lo, pi_hi,
                color="C1", alpha=0.15, label=f"{int(level*100)}% PI",
            )
        if ci_lo is not None:
            ax.fill_between(
                theta_grid, ci_lo, ci_hi,
                color="C1", alpha=0.30, label=f"{int(level*100)}% CI",
            )
        ax.plot(theta_grid, yhat, color="C1", lw=2, label="fit")
        ax.scatter(theta_data, y_data, color="C0", s=20, alpha=0.6, edgecolors="none", label="data")
        ax.set_xlabel(theta_var)
        ax.set_ylabel(self.response)
        ax.set_title("Fit overlay")
        ax.legend(loc="best", frameon=False)

        # If the grid covers a full 2π span, mark the canonical ticks.
        if t_lo <= 0 and t_hi >= 2 * np.pi:
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

        ax = axes[1]
        ax.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs fitted")

        fig.tight_layout()
        return fig
