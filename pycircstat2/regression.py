from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.special import i0, i1
from scipy.stats import chi2, norm

from .utils import A1, A1inv, significance_code

__all__ = ["CLRegression", "CCRegression"]


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
            Estimated beta coefficients for the mean direction.
        - alpha : float
            Estimated intercept for the concentration parameter..
        - gamma : np.ndarray
            Estimated coefficients for the concentration parameter.
        - mu : float
            Estimated mean direction of the circular response.
        - kappa : float
            Estimated concentration parameter of the circular response.
        - log_likelihood : float
            Log-likelihood of the model.

    Methods
    -------
    summary()
        Print a summary of the regression results.


    Notes
    -----
    The implementation is ported from the `lm.circular.cl` in the `circular` R package.

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
        theta_col, x_cols = formula.split("~")
        theta_series = data[theta_col.strip()]
        if theta_series.isnull().any():
            raise ValueError("Response column contains missing values.")
        theta = theta_series.to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+")]
        X_df = data[x_cols]
        if X_df.isnull().any().any():
            raise ValueError("Predictor columns contain missing values.")
        X = X_df.to_numpy()
        return theta, X, x_cols

    def _A1(self, kappa: np.ndarray) -> np.ndarray:
        return i1(kappa) / i0(kappa)

    def _A1inv(self, R: float) -> float:
        if 0 <= R < 0.53:
            return 2 * R + R**3 + (5 * R**5) / 6
        elif R < 0.85:
            return -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            return 1 / (R**3 - 4 * R**2 + 3 * R)

    def _A1_prime(self, kappa: np.ndarray) -> np.ndarray:
        a1 = A1(kappa)
        return 1 - a1 / kappa - a1**2

    def _fit(self):
        theta = self.theta
        n = len(theta)
        X = self.X
        X1 = np.column_stack((np.ones(n), X))  # Add intercept
        beta, alpha, gamma = self.beta, self.alpha, self.gamma
        diff = self.tol + 1
        log_likelihood_old = -np.inf

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
                weight = float(kappa * A1(np.array([kappa]))[0])
                u = kappa * np.sin(raw_deviation - mu)
                XtX = G.T @ G
                rhs = G.T @ u + weight * XtX @ beta
                mat = weight * XtX + 1e-8 * np.eye(X.shape[1])
                beta_new = _safe_solve(mat, rhs)
                alpha_new, gamma_new = alpha, gamma

                # Log-likelihood
                log_likelihood = -n * np.log(i0(kappa)) + kappa * np.sum(
                    np.cos(raw_deviation - mu)
                )

            elif self.model_type == "kappa":
                # Step 1: Compute mu and kappa
                kappa = np.exp(alpha + X @ gamma)
                S = float(np.sum(kappa * np.sin(theta)))
                C = float(np.sum(kappa * np.cos(theta)))
                mu = np.arctan2(S, C)

                # Step 2: Update gamma
                a1_kappa = self._A1(kappa)
                a1_prime = self._A1_prime(kappa)
                if np.any(np.isclose(a1_prime, 0.0)):
                    raise ValueError("Encountered zero derivative in concentration update.")
                residuals_gamma = np.cos(theta - mu) - a1_kappa
                y_gamma = residuals_gamma / (a1_prime * kappa)
                weights = (kappa**2) * a1_prime
                XtWX = X1.T @ (weights[:, None] * X1)
                XtWy = X1.T @ (weights * y_gamma)
                update = _safe_solve(XtWX + 1e-8 * np.eye(X1.shape[1]), XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]
                beta_new = beta
                # Log-likelihood
                log_likelihood = -np.sum(np.log(i0(kappa))) + np.sum(
                    kappa * np.cos(theta - mu)
                )

            elif self.model_type == "mixed":
                # Step 1: Compute mu and kappa
                kappa = np.exp(alpha + X @ gamma)
                raw_deviation = theta - 2 * np.arctan(X @ beta)
                S = np.sum(kappa * np.sin(raw_deviation))
                C = np.sum(kappa * np.cos(raw_deviation))
                mu = np.arctan2(S, C)
                residuals = theta - mu

                # Step 2: Update beta
                denom = 1 + (X @ beta) ** 2
                G = 2 * X / denom[:, None]
                weights_beta = kappa * self._A1(kappa)
                XtWX_beta = G.T @ (weights_beta[:, None] * G)
                rhs_beta = G.T @ (weights_beta * np.sin(residuals))
                beta_new = _safe_solve(
                    XtWX_beta + 1e-8 * np.eye(X.shape[1]), rhs_beta
                )

                # Step 3: Update gamma
                a1_kappa = self._A1(kappa)
                a1_prime = self._A1_prime(kappa)
                if np.any(np.isclose(a1_prime, 0.0)):
                    raise ValueError("Encountered zero derivative in concentration update.")
                residuals_gamma = np.cos(raw_deviation - mu) - a1_kappa
                y_gamma = residuals_gamma / (a1_prime * kappa)
                weights_gamma = (kappa**2) * a1_prime
                XtWX = X1.T @ (weights_gamma[:, None] * X1)
                XtWy = X1.T @ (weights_gamma * y_gamma)
                update = _safe_solve(XtWX + 1e-8 * np.eye(X1.shape[1]), XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]

                # Log-likelihood
                log_likelihood = -np.sum(np.log(i0(kappa))) + np.sum(
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
        gamma = result["gamma"]
        alpha = result["alpha"]

        se_results = {}

        if self.model_type == "mean":
            # Mean Direction Model
            denom = 1 + (X @ beta) ** 2
            G = 2 * X / denom[:, None]
            weight = float(kappa * self._A1(np.array([kappa]))[0])
            XtAX = weight * (G.T @ G) + 1e-8 * np.eye(X.shape[1])
            cov_beta = _safe_inverse(XtAX)
            se_beta = np.sqrt(np.diag(cov_beta))

            denom_mu = max((n - X.shape[1]) * kappa * self._A1(np.array([kappa]))[0], 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)
            denom_kappa = n * (1 - self._A1(np.array([kappa]))[0] ** 2 - self._A1(np.array([kappa]))[0] / kappa)
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
            weights = (np.exp(X1 @ np.hstack([alpha, gamma])) ** 2) * self._A1_prime(kappa)
            XtWX = X1.T @ (weights[:, None] * X1) + 1e-8 * np.eye(X1.shape[1])

            cov_gamma_alpha = _safe_inverse(XtWX)

            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            denom_mu = max(np.sum(kappa * self._A1(kappa)) - 0.5, 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)

            se_kappa = np.sqrt(1 / np.clip(n * self._A1_prime(kappa), 1e-12, None))

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
            weights_beta = kappa * self._A1(kappa)
            XtGKGX = G.T @ (weights_beta[:, None] * G) + 1e-8 * np.eye(X.shape[1])

            cov_beta = _safe_inverse(XtGKGX)
            se_beta = np.sqrt(np.diag(cov_beta))

            X1 = np.column_stack((np.ones(n), X))  # Add intercept
            weights_gamma = (np.exp(X1 @ np.hstack([alpha, gamma])) ** 2) * self._A1_prime(kappa)
            XtWX_gamma = X1.T @ (weights_gamma[:, None] * X1) + 1e-8 * np.eye(X1.shape[1])

            cov_gamma_alpha = _safe_inverse(XtWX_gamma)
            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            denom_mu = max(np.sum(kappa * self._A1(kappa)) - 0.5, 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)
            a1_vals = self._A1(kappa)
            denom_kappa = n * (1 - a1_vals**2 - a1_vals / kappa)
            se_kappa = np.sqrt(1 / np.clip(denom_kappa, 1e-12, None))
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

        beta = self.result.get("beta")
        if beta is None or np.any(~np.isfinite(beta)):
            raise ValueError("Model does not contain beta coefficients for prediction.")

        X_arr = np.asarray(X_new, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.shape[1] != beta.size:
            raise ValueError(
                f"Expected {beta.size} predictors, received {X_arr.shape[1]}."
            )
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("`X_new` contains non-finite values.")

        mu = self.result["mu"]
        return mu + 2 * np.arctan(X_arr @ beta)

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
                t_value = abs(coef / se_val) if se_val else np.nan
                p_value = (
                    2 * (1 - norm.cdf(np.abs(t_value)))
                    if not np.isnan(t_value)
                    else np.nan
                )
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
            p_value_alpha = (
                2 * (1 - norm.cdf(np.abs(t_value_alpha)))
                if not np.isnan(t_value_alpha)
                else np.nan
            )
            print(
                f"α{'':<5} {alpha:<12.5f} {se_alpha:<12.5f} {t_value_alpha:<10.2f} {p_value_alpha:<12.5f}{significance_code(p_value_alpha)}"
            )
            for i, coef in enumerate(self.result["gamma"]):
                se_val = se_gamma[i]
                t_value = coef / se_val if se_val else np.nan
                p_value = (
                    2 * (1 - norm.cdf(np.abs(t_value)))
                    if not np.isnan(t_value)
                    else np.nan
                )
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
            if se_kappa is not None:
                print(f"    Mean:    {np.mean(kappa):.5f} (SE: {np.mean(se_kappa):.5f})")
            else:
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
    message : str
        Message indicating the significance of higher-order terms.

    Methods
    -------
    summary()
        Print a summary of the regression results.


    Notes
    -----
    The implementation is ported from the `lm.circular.cc` in the `circular` R package.

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

        self.order = order
        self.level = level

        if self.order < 1:
            raise ValueError("`order` must be a positive integer.")
        if not (0 < self.level < 1):
            raise ValueError("`level` must lie between 0 and 1.")

        # Fit the model
        self.result = self._fit()

    @staticmethod
    def _validate_input(arr: np.ndarray) -> np.ndarray:
        """
        Validate input array and ensure it is in radians.
        """
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.ndim == 0:
            raise ValueError("Input must be at least one-dimensional.")
        if not np.all(np.isfinite(arr_np)):
            raise ValueError("Circular input contains non-finite values.")
        return np.mod(arr_np, 2 * np.pi)

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        theta_col, x_cols = formula.split("~")
        theta = data[theta_col.strip()].to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+")]
        X = data[x_cols].to_numpy()
        return theta, X, x_cols

    def _fit(self):
        n = self.x.shape[0]
        order = self.order
        n_features = self.x.shape[1]

        # Create harmonic terms
        cos_terms = []
        sin_terms = []
        cos_labels: List[Tuple[int, int]] = []
        sin_labels: List[Tuple[int, int]] = []
        for j in range(n_features):
            x_col = self.x[:, j]
            for k in range(1, order + 1):
                cos_terms.append(np.cos(k * x_col))
                sin_terms.append(np.sin(k * x_col))
                cos_labels.append((j, k))
                sin_labels.append((j, k))

        # Linear models for cos(theta) and sin(theta)
        Y_cos = np.cos(self.theta)
        Y_sin = np.sin(self.theta)

        design_matrix = [np.ones(n)] + cos_terms + sin_terms
        X = np.column_stack(design_matrix)
        beta_cos, _, _, _ = lstsq(X, Y_cos)
        beta_sin, _, _, _ = lstsq(X, Y_sin)

        # Fitted values
        cos_fit = X @ beta_cos
        sin_fit = X @ beta_sin
        fitted = np.mod(np.arctan2(sin_fit, cos_fit), 2 * np.pi)

        # Residuals
        residuals = np.angle(np.exp(1j * (self.theta - fitted)))

        # Circular correlation coefficient
        rho = float(np.clip(np.sqrt(np.mean(cos_fit**2 + sin_fit**2)), 0.0, 1.0))

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
            XtX = X.T @ X
            M = X @ _safe_inverse(XtX) @ X.T
            H = W.T @ (np.eye(n) - M) @ W
            H_inv = _safe_inverse(H)
            N = W @ H_inv @ W.T

            residual_cos = Y_cos - X @ beta_cos
            residual_sin = Y_sin - X @ beta_sin

            denom_cos = float(residual_cos.T @ residual_cos)
            denom_sin = float(residual_sin.T @ residual_sin)
            adj = max(n - (2 * order + 1), 1)
            T1 = (
                adj
                * float(residual_cos.T @ N @ residual_cos)
                / max(denom_cos, 1e-12)
            )
            T2 = (
                adj
                * float(residual_sin.T @ N @ residual_sin)
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

        return {
            "rho": rho,
            "fitted": fitted,
            "residuals": residuals,
            "coefficients": {
                "cos": beta_cos,
                "sin": beta_sin,
            },
            "cos_labels": cos_labels,
            "sin_labels": sin_labels,
            "p_values": p_values,
            "message": message,
        }

    def summary(self):
        """
        Print a summary of the regression results.
        """
        print("\nCircular-Circular Regression\n")
        print(f"Circular Correlation Coefficient (rho): {self.result['rho']:.5f}\n")

        print("Coefficients:")
        cos_coeffs = self.result["coefficients"]["cos"]
        sin_coeffs = self.result["coefficients"]["sin"]
        cos_labels = self.result.get("cos_labels", [])
        sin_labels = self.result.get("sin_labels", [])

        # Headers
        print(f"{'Harmonic':<12} {'Cosine Coeff':<14} {'Sine Coeff':<14}")

        # Intercept
        print(f"{'(Intercept)':<12} {cos_coeffs[0]:<14.5f} {sin_coeffs[0]:<14.5f}")

        # Cosine harmonics
        offset = 1
        for idx, (feature_idx, harmonic) in enumerate(cos_labels):
            label = f"cos(x{feature_idx + 1},k={harmonic})"
            print(
                f"{label:<12} {cos_coeffs[offset + idx]:<14.5f} {sin_coeffs[offset + idx]:<14.5f}"
            )

        # Sine harmonics
        sine_offset = offset + len(cos_labels)
        for idx, (feature_idx, harmonic) in enumerate(sin_labels):
            label = f"sin(x{feature_idx + 1},k={harmonic})"
            print(
                f"{label:<12} {cos_coeffs[sine_offset + idx]:<14.5f} {sin_coeffs[sine_offset + idx]:<14.5f}"
            )

        print("\nP-values for Higher-Order Terms:")
        print(
            f"p1: {self.result['p_values'][0]:.5f}, p2: {self.result['p_values'][1]:.5f}"
        )

        print(f"\n{self.result['message']}\n")
