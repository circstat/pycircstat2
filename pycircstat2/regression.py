from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import i0, i1, ive
from scipy.stats import norm

__all__ = ["CLRegression", "LCRegression", "CCRegression"]


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
    summary(digits=3)
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
        A1 = self._A1(kappa)
        return 1 - A1 / kappa - A1**2

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
                kappa = self._A1inv(R)
                mu = np.arctan2(S, C)
                residuals = theta - mu

                # Step 2: Update beta
                G = 2 * X / (1 + (X @ beta) ** 2)[:, None]
                A = np.eye(n) * (kappa * self._A1(kappa))
                u = kappa * np.sin(raw_deviation - mu)
                beta_new = np.linalg.solve(G.T @ A @ G, G.T @ (u + A @ G @ beta))
                alpha_new, gamma_new = np.nan, np.nan

                # Log-likelihood
                log_likelihood = -n * np.log(i0(kappa)) + kappa * np.sum(
                    np.cos(residuals)
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

        return {
            "beta": beta,
            "alpha": alpha,
            "gamma": gamma,
            "mu": mu,
            "kappa": kappa,
            "log_likelihood": log_likelihood,
        }

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
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)':<12}"
            )
            for i, coef in enumerate(self.result["beta"]):
                # Placeholder for standard error and p-values
                se_beta = np.nan  # Replace with actual computation later
                t_value = coef / se_beta if se_beta else np.nan
                p_value = (
                    2 * (1 - norm.cdf(np.abs(t_value)))
                    if not np.isnan(t_value)
                    else np.nan
                )
                print(
                    f"β_{i:<3} {coef:<12.5f} {se_beta:<12.5f} {t_value:<10.2f} {p_value:<12.5f}"
                )

        # Coefficients for concentration parameter (Gamma)
        if self.model_type in ["kappa", "mixed"] and self.result["gamma"] is not None:
            print("\nCoefficients for Concentration (Gamma):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)':<12}"
            )
            for i, coef in enumerate(self.result["gamma"]):
                # Placeholder for standard error and p-values
                se_gamma = np.nan  # Replace with actual computation later
                t_value = coef / se_gamma if se_gamma else np.nan
                p_value = (
                    2 * (1 - norm.cdf(np.abs(t_value)))
                    if not np.isnan(t_value)
                    else np.nan
                )
                print(
                    f"γ_{i:<3} {coef:<12.5f} {se_gamma:<12.5f} {t_value:<10.2f} {p_value:<12.5f}"
                )

        # Intercept (Alpha) for concentration
        if self.model_type in ["kappa", "mixed"] and not np.isnan(self.result["alpha"]):
            print("\nIntercept for Concentration (Alpha):")
            print(f"  Estimate: {self.result['alpha']:.5f}")

        # Summary for mu and kappa
        print("\nSummary:")
        print("  Mean Direction (mu) in radians:")
        mu = self.result["mu"]
        se_mu = np.nan  # Placeholder for standard error
        print(f"    mu: {mu:.5f} (SE: {se_mu:.5f})")

        print("\n  Concentration Parameter (kappa):")
        kappa = self.result["kappa"]
        se_kappa = np.nan  # Placeholder for standard error
        if isinstance(kappa, np.ndarray):
            print(f"    kappa: {np.mean(kappa):.5f} (SE: {se_kappa:.5f})")
        else:
            print(f"    kappa: {kappa:.5f} (SE: {se_kappa:.5f})")

        # Log-likelihood
        print(f"\nLog-Likelihood: {self.result.get('log_likelihood', 'NA'):.5f}")

        # Notes
        print("\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1")
        print("p-values are approximated using the normal distribution.\n")


class LCRegression:
    pass


class CCRegression:
    pass
