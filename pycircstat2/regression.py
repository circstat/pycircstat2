from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import i0, ive
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
    beta0 : np.ndarray, optional
        Initial values for the beta coefficients.
    tol : float, optional
        Convergence tolerance for the optimization.
    verbose : bool, optional
        Whether to print optimization progress.

    Attributes
    ----------
    result : dict
        A dictionary containing the estimated coefficients and other statistics.

        - beta : np.ndarray
            Estimated beta coefficients.
        - se_beta : np.ndarray
            Standard errors of the beta coefficients.
        - t_values : np.ndarray
            t-values of the beta coefficients.
        - p_values : np.ndarray
            p-values of the beta coefficients.
        - mu : float
            Estimated mean direction of the circular response.
        - se_mu : float
            Standard error of the mean direction.
        - kappa : float
            Estimated concentration parameter of the circular response.
        - se_kappa : float
            Standard error of the concentration parameter.
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
        beta0: Optional[np.ndarray] = None,
        tol: float = 1e-10,
        verbose: bool = False,
    ):
        self.tol = tol
        self.verbose = verbose
        self.result = None

        # Initialize via formula or directly via θ and X
        if formula and data is not None:
            self.formula = formula
            self.data = data
            self.y, self.X, self.feature_names = self._parse_formula(formula, data)
        elif theta is not None and X is not None:

            if theta is not None:
                if not np.all((theta >= 0) & (theta <= 2 * np.pi)):
                    raise ValueError(
                        "Response `theta` must be in radians within [0, 2π]."
                    )

            if np.linalg.cond(X.T @ X) > 1e10:
                raise Warning(
                    "Predictors may be collinear, leading to unstable results."
                )

            self.y = theta
            self.X = X
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]
        else:
            raise ValueError(
                "Either provide formula and data, or theta and X directly."
            )

        # Initialize beta coefficients
        if beta0 is None:
            beta0 = np.zeros(self.X.shape[1])
        self.beta0 = beta0

        # Fit the model
        self.result = self._fit_lc_model()

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Parse a formula like 'θ ~ X' into response θ and predictors X.
        """
        y_col, x_cols = formula.split("~")
        y_col = y_col.strip()
        x_cols = [col.strip() for col in x_cols.split("+")]

        y = data[y_col].to_numpy()
        X = data[x_cols].to_numpy()

        return y, X, x_cols

    def _fit_lc_model(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Perform iterative optimization to fit the circular-linear regression model.
        """
        y = self.y % (2 * np.pi)  # Wrap response to [0, 2pi)
        X = self.X
        beta = self.beta0.copy()

        n = len(y)
        diff = self.tol + 1
        iter_count = 0

        while diff > self.tol:
            iter_count += 1
            residual = y - 2 * np.arctan(X @ beta)
            S = np.sum(np.sin(residual)) / n
            C = np.sum(np.cos(residual)) / n
            R = np.sqrt(S**2 + C**2)
            mu = np.arctan2(S, C)
            kappa = self._A1inv(R)

            u = kappa * np.sin(residual - mu)
            D = 2 * X / (1 + (X @ beta) ** 2)[:, None]

            A = np.eye(n) * (kappa * self._A1(kappa))
            beta_new = np.linalg.solve(D.T @ A @ D, D.T @ (u + A @ D @ beta))

            diff = np.max(np.abs(beta_new - beta))
            beta = beta_new

            if self.verbose:
                log_likelihood = -n * np.log(i0(kappa)) + kappa * np.sum(
                    np.cos(residual - mu)
                )
                print(f"Iteration {iter_count}: Log-Likelihood = {log_likelihood:.5f}")

        residual = y - 2 * np.arctan(X @ beta)
        log_likelihood = -n * np.log(i0(kappa)) + kappa * np.sum(np.cos(residual - mu))

        # Standard errors
        D = 2 * X / (1 + (X @ beta) ** 2)[:, None]
        A = np.eye(n) * (kappa * self._A1(kappa))
        XtAX = D.T @ A @ D
        cov_beta = np.linalg.solve(XtAX, np.eye(XtAX.shape[0]))
        se_beta = np.sqrt(np.diag(cov_beta))

        # Other estimates
        se_mu = 1 / np.sqrt((n - X.shape[1]) * kappa * self._A1(kappa))
        se_kappa = np.sqrt(
            1 / (n * (1 - self._A1(kappa) ** 2 - self._A1(kappa) / kappa))
        )

        t_values = beta / se_beta
        p_values = 2 * (1 - norm.cdf(np.abs(t_values)))

        return {
            "beta": beta,
            "se_beta": se_beta,
            "t_values": t_values,
            "p_values": p_values,
            "mu": mu,
            "se_mu": se_mu,
            "kappa": kappa,
            "se_kappa": se_kappa,
            "log_likelihood": log_likelihood,
        }

    def _A1(self, kappa: float) -> float:
        """
        Exact calculation of the first moment of the von Mises distribution.
        """
        if kappa == 0:
            return 0  # A1(0) is 0 by definition
        return ive(1, kappa) / ive(0, kappa)

    def _A1inv(self, R: float) -> float:
        """
        Approximation of the inverse of the first moment of the von Mises distribution.
        """
        if 0 <= R < 0.53:
            return 2 * R + R**3 + (5 * R**5) / 6
        elif R < 0.85:
            return -0.4 + 1.39 * R + 0.43 / (1 - R)
        else:
            return 1 / (R**3 - 4 * R**2 + 3 * R)

    def summary(self, digits: int = 3):
        """
        Print a summary of the regression results.
        """
        print("\nCircular-Linear Regression Summary")
        print("\nCall:")
        if hasattr(self, "formula"):
            print(f"  {self.formula}\n")
        else:
            print(f"  θ and X provided directly\n")

        print("Circular-Linear Regression\n")
        print("Coefficients:\n")
        print(
            f"{'':<12} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)':<12}"
        )
        for name, beta, se, t, p in zip(
            self.feature_names,
            self.result["beta"],
            self.result["se_beta"],
            self.result["t_values"],
            self.result["p_values"],
        ):
            print(
                f"{name:<12} {beta:<12.{digits}f} {se:<12.{digits}f} {t:<10.{digits}f} {p:<12.{digits}g}"
            )

        print("\nLog-Likelihood:", round(self.result["log_likelihood"], digits))
        print("\nSummary: (mu in radians)")
        print(
            f"  mu: {self.result['mu']:.{digits}f} ({self.result['se_mu']:.{digits}f})"
        )
        print(
            f"  kappa: {self.result['kappa']:.{digits}f} ({self.result['se_kappa']:.{digits}f})\n"
        )


class LCRegression:
    pass


class CCRegression:
    pass
