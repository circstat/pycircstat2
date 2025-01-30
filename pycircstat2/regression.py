from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.special import i0, i1
from scipy.stats import chi2, norm

from .utils import A1, A1inv, significance_code

__all__ = ["CLRegression", "CCRegression"]


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
            self.theta, self.X, self.feature_names = self._parse_formula(formula, data)
        elif theta is not None and X is not None:
            self.theta = theta
            self.X = X
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]
        else:
            raise ValueError("Provide either a formula + data or theta and X.")

        # Validate model type
        if model_type not in ["mean", "kappa", "mixed"]:
            raise ValueError("Model type must be 'mean', 'kappa', or 'mixed'.")

        # Initialize parameters
        self.alpha = alpha0 if alpha0 is not None else 0.0
        self.beta = beta0 if beta0 is not None else np.zeros(self.X.shape[1])
        self.gamma = gamma0 if gamma0 is not None else np.zeros(self.X.shape[1])

        # Fit the model
        self.result = self._fit()

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        theta_col, x_cols = formula.split("~")
        theta = data[theta_col.strip()].to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+")]
        X = data[x_cols].to_numpy()
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
                R = np.sqrt(S**2 + C**2)
                kappa = A1inv(R)
                mu = np.arctan2(S, C)

                # Step 2: Update beta
                G = 2 * X / (1 + (X @ beta) ** 2)[:, None]
                A = np.eye(n) * (kappa * A1(kappa))
                u = kappa * np.sin(raw_deviation - mu)
                beta_new = np.linalg.solve(G.T @ A @ G, G.T @ (u + A @ G @ beta))
                alpha_new, gamma_new = np.nan, np.nan

                # Log-likelihood
                log_likelihood = -n * np.log(i0(kappa)) + kappa * np.sum(
                    np.cos(raw_deviation - mu)
                )

            elif self.model_type == "kappa":
                # Step 1: Compute mu and kappa
                kappa = np.exp(alpha + X @ gamma)
                S = np.sum(kappa * np.sin(theta))
                C = np.sum(kappa * np.cos(theta))
                mu = np.arctan2(S, C)

                # Step 2: Update gamma
                residuals_gamma = np.cos(theta - mu) - self._A1(kappa)
                y_gamma = residuals_gamma / (self._A1_prime(kappa) * kappa)
                W_gamma = np.diag((kappa**2) * self._A1_prime(kappa))
                XtWX = X1.T @ W_gamma @ X1
                XtWy = X1.T @ W_gamma @ y_gamma
                update = np.linalg.solve(XtWX, XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]
                beta_new = np.nan
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
                G = 2 * X / (1 + (X @ beta) ** 2)[:, None]
                W_kappa = np.diag(kappa * self._A1(kappa))
                beta_new = np.linalg.solve(
                    G.T @ W_kappa @ G, G.T @ W_kappa @ np.sin(residuals)
                )

                # Step 3: Update gamma
                residuals_gamma = np.cos(raw_deviation - mu) - self._A1(kappa)
                y_gamma = residuals_gamma / (self._A1_prime(kappa) * kappa)
                W_gamma = np.diag((kappa**2) * self._A1_prime(kappa))
                XtWX = X1.T @ W_gamma @ X1
                XtWy = X1.T @ W_gamma @ y_gamma
                update = np.linalg.solve(XtWX, XtWy)
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
            G = 2 * X / (1 + (X @ beta) ** 2)[:, None]
            A = np.eye(n) * (kappa * self._A1(kappa))
            XtAX = G.T @ A @ G
            cov_beta = np.linalg.solve(XtAX, np.eye(XtAX.shape[0]))
            se_beta = np.sqrt(np.diag(cov_beta))

            se_mu = 1 / np.sqrt((n - X.shape[1]) * kappa * self._A1(kappa))
            se_kappa = np.sqrt(
                1 / (n * (1 - self._A1(kappa) ** 2 - self._A1(kappa) / kappa))
            )

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
            W = np.diag(
                (np.exp(X1 @ np.hstack([alpha, gamma])) ** 2) * self._A1_prime(kappa)
            )
            XtWX = X1.T @ W @ X1

            cov_gamma_alpha = np.linalg.solve(XtWX, np.eye(XtWX.shape[0]))

            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            se_mu = 1 / np.sqrt(np.sum(kappa * self._A1(kappa)) - 0.5)

            se_kappa = np.sqrt(1 / (n * self._A1_prime(kappa)))

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
            G = 2 * X / (1 + (X @ beta) ** 2)[:, None]
            K = np.diag(kappa * self._A1(kappa))
            XtGKGX = G.T @ K @ G

            cov_beta = np.linalg.solve(XtGKGX, np.eye(XtGKGX.shape[0]))
            se_beta = np.sqrt(np.diag(cov_beta))

            X1 = np.column_stack((np.ones(n), X))  # Add intercept
            W_gamma = np.diag(
                (np.exp(X1 @ np.hstack([alpha, gamma])) ** 2) * self._A1_prime(kappa)
            )
            XtWX_gamma = X1.T @ W_gamma @ X1

            # Check positive definiteness and regularize if needed
            eigenvalues_gamma = np.linalg.eigvals(XtWX_gamma)
            if np.any(eigenvalues_gamma <= 0):
                XtWX_gamma += np.eye(XtWX_gamma.shape[0]) * 1e-8

            cov_gamma_alpha = np.linalg.solve(XtWX_gamma, np.eye(XtWX_gamma.shape[0]))
            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            se_mu = 1 / np.sqrt(np.sum(kappa * self._A1(kappa)) - 0.5)
            se_kappa = np.sqrt(
                1 / (n * (1 - self._A1(kappa) ** 2 - self._A1(kappa) / kappa))
            )
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

        mu = self.result["mu"]
        beta = self.result["beta"]
        return mu + 2 * np.arctan(X_new @ beta)

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
        if self.model_type in ["mean", "mixed"] and self.result["beta"] is not None:
            print("Coefficients for Mean Direction (Beta):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)'}"
            )
            for i, coef in enumerate(self.result["beta"]):
                # Placeholder for standard error and p-values
                se_beta = self.result["se_beta"][i]
                t_value = np.abs(coef / se_beta) if se_beta else np.nan
                p_value = (
                    2 * (1 - norm.cdf(np.abs(t_value)))
                    if not np.isnan(t_value)
                    else np.nan
                )
                print(
                    f"β{i:<3} {coef:<12.5f} {se_beta:<12.5f} {t_value:<10.2f} {p_value:<12.5f}{significance_code(p_value):<3}"
                )

        # Coefficients for concentration parameter (Gamma)
        if self.model_type in ["kappa", "mixed"] and self.result["gamma"] is not None:
            print("\nCoefficients for Concentration (Gamma):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)':<12}"
            )
            # Report alpha as the first coefficient
            alpha = self.result["alpha"]
            se_alpha = self.result["se_alpha"]
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
                # Placeholder for standard error and p-values
                se_gamma = self.result["se_gamma"][i]
                t_value = coef / se_gamma if se_gamma else np.nan
                p_value = (
                    2 * (1 - norm.cdf(np.abs(t_value)))
                    if not np.isnan(t_value)
                    else np.nan
                )
                print(
                    f"γ{i:<5} {coef:<12.5f} {se_gamma:<12.5f} {t_value:<10.2f} {p_value:<12.5f}{significance_code(p_value)}"
                )

        # Summary for mu and kappa
        print("\nSummary:")
        print("  Mean Direction (mu) in radians:")
        mu = self.result["mu"]
        se_mu = self.result["se_mu"]
        print(f"    μ: {mu:.5f} (SE: {se_mu:.5f})")

        print("\n  Concentration Parameter (kappa):")
        kappa = self.result["kappa"]
        se_kappa = self.result["se_kappa"]
        if isinstance(kappa, np.ndarray):
            print("    Index    kappa        Std. Error")
            for i, (k, se) in enumerate(zip(kappa, se_kappa), start=1):
                print(f"    [{i}]    {k:>10.5f}    {se:>10.5f}")
            print(f"    Mean:    {np.mean(kappa):.5f} (SE: {np.mean(se_kappa):.5f})")
        else:
            print(f"    κ: {kappa:.5f} (SE: {se_kappa:.5f})")

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
            self.theta, self.x, self.feature_names = self._parse_formula(formula, data)
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

        # Fit the model
        self.result = self._fit()

    @staticmethod
    def _validate_input(arr: np.ndarray) -> np.ndarray:
        """
        Validate input array and ensure it is in radians.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        return arr % (2 * np.pi)

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

        # Create harmonic terms
        order_matrix = np.arange(1, order + 1)
        cos_x = np.cos(self.x * order_matrix)
        sin_x = np.sin(self.x * order_matrix)

        # Linear models for cos(theta) and sin(theta)
        Y_cos = np.cos(self.theta)
        Y_sin = np.sin(self.theta)

        X = np.column_stack([np.ones(n), cos_x, sin_x])
        beta_cos, _, _, _ = lstsq(X, Y_cos)
        beta_sin, _, _, _ = lstsq(X, Y_sin)

        # Fitted values
        cos_fit = X @ beta_cos
        sin_fit = X @ beta_sin
        fitted = np.arctan2(sin_fit, cos_fit) % (2 * np.pi)

        # Residuals
        residuals = (self.theta - fitted) % (2 * np.pi)

        # Circular correlation coefficient
        rho = np.sqrt(np.mean(cos_fit**2) + np.mean(sin_fit**2))

        # Test higher-order terms
        higher_order_cos = np.cos((order + 1) * self.x)
        higher_order_sin = np.sin((order + 1) * self.x)

        # Projection matrix for the current model
        M = X @ np.linalg.inv(X.T @ X) @ X.T
        W = np.column_stack([higher_order_cos, higher_order_sin])
        H = W.T @ (np.eye(n) - M) @ W
        H_inv = np.linalg.inv(H)
        N = W @ H_inv @ W.T

        residual_cos = (np.eye(n) - M) @ Y_cos
        residual_sin = (np.eye(n) - M) @ Y_sin

        T1 = (
            (n - (2 * order + 1))
            * (residual_cos.T @ N @ residual_cos)
            / (residual_cos.T @ residual_cos)
        )
        T2 = (
            (n - (2 * order + 1))
            * (residual_sin.T @ N @ residual_sin)
            / (residual_sin.T @ residual_sin)
        )

        p1 = 1 - chi2.cdf(T1, 2)
        p2 = 1 - chi2.cdf(T2, 2)

        p_values = np.array([p1, p2])

        # Message about higher-order terms
        if np.all(p_values > self.level):
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

        # Headers
        print(f"{'Harmonic':<12} {'Cosine Coeff':<14} {'Sine Coeff':<14}")

        # Intercept
        print(f"{'(Intercept)':<12} {cos_coeffs[0]:<14.5f} {sin_coeffs[0]:<14.5f}")

        # Group harmonics: Cosine and Sine
        for i in range(1, len(cos_coeffs)):
            if i <= self.order:
                print(
                    f"{'cos.x' + str(i):<12} {cos_coeffs[i]:<14.5f} {sin_coeffs[i]:<14.5f}"
                )
            else:
                print(
                    f"{'sin.x' + str(i - self.order):<12} {cos_coeffs[i]:<14.5f} {sin_coeffs[i]:<14.5f}"
                )

        print("\nP-values for Higher-Order Terms:")
        print(
            f"p1: {self.result['p_values'][0]:.5f}, p2: {self.result['p_values'][1]:.5f}"
        )

        print(f"\n{self.result['message']}\n")
