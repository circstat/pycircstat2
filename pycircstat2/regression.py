from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from formulaic import model_matrix
from scipy.linalg import cholesky, lu, qr, solve_triangular
from scipy.optimize import minimize
from scipy.stats import f, norm, t

__all__ = ["LCRegression"]


class lm:
    """Linear Models in R-Style."""

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        weights: Union[None, np.array] = None,
        method: str = "qr",
    ):

        # meta
        self.formula = formula
        self.data = data
        self.weights = weights
        self.method = method

        # design matrix
        self.y, self.X = model_matrix(formula, data)

        self.column_names = self.X.columns
        self.feature_names = (
            self.column_names[1:]
            if "Intercept" in self.column_names
            else self.column_names
        )

        X = self.X.values
        y = self.y.values.flatten()

        self.n, self.p = (
            n,
            p,
        ) = X.shape  # n_samples x n_features (intercept included if available)
        if weights is not None and len(weights) != n:
            raise ValueError(
                "Length of weights should be the same as the number of rows in the dataframe"
            )
        self.W = W = np.eye(n) if weights is None else np.diag(weights)

        # model degree of freedom
        self.df_model = self.p - 1 if "Intercept" in self.column_names else self.p

        # residual degrees of freedom (n - p)
        self.df_residuals = (
            self.n - self.df_model - 1
            if "Intercept" in self.column_names
            else self.n - self.df_model
        )

        ##############
        # Estimation #
        ##############

        if method == "nll":
            bhat, self.sigma, self.XtX, self.Xty, self.loss = self.compute_bhat(
                X, y, W, "nll"
            )
        elif method == "sse":
            bhat, self.XtX, self.Xty, self.loss = self.compute_bhat(X, y, W, "sse")
        else:
            bhat, self.XtX, self.Xty = self.compute_bhat(X, y, W, method)

        self.bhat = pd.DataFrame(
            bhat.reshape(1, -1), columns=self.column_names, index=["Estimate"]
        )

        # compute predicted (fitted values ŷ = Xβ̂)
        self.yhat = self.compute_yhat()
        yhat = self.yhat.values.flatten()

        # compute residuals ϵ̂
        residuals = y - yhat
        self.residuals = pd.DataFrame(residuals[:, None], columns=["residuals"])

        # compute residual sum of squares (RSS)
        self.rss = np.squeeze(residuals.T @ residuals)

        # compute standard deviation of model coefficients
        # aka Residual SE: σ^2 = RSS / df_residuals
        if method == "nll":
            self.sigma_squared = self.sigma**2
        else:
            self.sigma_squared = self.rss / self.df_residuals
            self.sigma = np.sqrt(self.sigma_squared)

        # compute standard error for β̂
        self.XtXinv = self.compute_XtXinv()

        se_bhat, V_bhat = self.compute_se_bhat()
        self.V_bhat = V_bhat
        self.se_bhat = pd.DataFrame(
            se_bhat.reshape(1, -1), columns=self.column_names, index=["Std. Error"]
        )

        # compute confidence interval for β̂
        self.ci_bhat = self.compute_ci_bhat()

        # compute t values of model coefficients
        self.t_values = self.compute_t_values()

        # p values
        self.p_values = self.compute_p_values()

        # compute r2 and r2adjusted, aka coefficient of determination
        # aka percentage of variance explained. Noted that the formulae
        # are different for cases with and without intercept
        (
            self.tss,
            self.r_squared,
            self.r_squared_adjusted,
        ) = self.compute_goodness_of_fit()

        # compute F-statistics with scipy.stats.f.sf
        # H0: all coefficients == 0
        # H1: at least one coefficient != 0
        self.fstats, self.f_p_value = self.compute_fstats()

        # compute log-likelihood
        self.loglike = self.compute_loglikelihood()

        # compute AIC (Akaike Information criterion): -2logL + 2p, p is the total number of parameters
        self.AIC = self.compute_AIC()

        # compute BIC (Bayes Information criterian): -2logL + p * log(n)
        self.BIC = self.compute_BIC()

    def __repr__(self):

        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += "Coefficients:\n"
        docstring += self.bhat.to_string()

        return docstring

    def __str__(self):

        return self.__repr__()

    def compute_XtXinv(self):
        U, S, Vt = np.linalg.svd(self.XtX, full_matrices=False)
        XtXinv = Vt.T @ np.diag(1 / S) @ U.T
        return XtXinv

    def compute_bhat(self, X, y, W, return_ss=True):

        bhat, XtX, Xty = _qr(X, y, W)

        if return_ss:
            return bhat, XtX, Xty
        else:
            return bhat

    def compute_se_bhat(self):
        V_bhat = self.sigma_squared * self.XtXinv
        se_bhat = np.sqrt(np.diag(V_bhat))[:, None]
        return se_bhat, V_bhat

    def compute_ci_bhat(self, alpha=0.05):

        se_bhat = self.se_bhat.values.T
        bhat = self.bhat.values.T
        ci_bhat = (
            t.ppf(1 - alpha / 2, self.df_residuals) * se_bhat * np.array([-1, 1]) + bhat
        )
        ci_bhat = pd.DataFrame(
            ci_bhat,
            index=self.column_names,
            columns=[f"CI[{alpha/2*100}%]", f"CI[{100-alpha/2*100}%]"],
        )
        return ci_bhat

    def compute_ci_bhat_bootstrap(self, num_bootstrap=4000, alpha=0.05):

        X = self.X.values
        W = self.W
        bhat = self.bhat.values.T
        residuals = self.residuals.values.flatten()
        bhat_stars = np.zeros([num_bootstrap, self.p])
        for i in range(num_bootstrap):
            residuals_star = np.random.choice(
                residuals, size=len(residuals), replace=True
            )
            y_star = X @ bhat + residuals_star[:, None]
            bhat_star = self.compute_bhat(X, y_star.flatten(), W, return_ss=False)
            bhat_stars[i] = bhat_star

        ci_bhat_bootstrap = np.quantile(
            bhat_stars, q=[alpha / 2, 1 - alpha / 2], axis=0
        ).T
        self.ci_bhat_bootstrap = ci_bhat_bootstrap = pd.DataFrame(
            ci_bhat_bootstrap,
            index=self.column_names,
            columns=[f"CI[{alpha/2*100}%]", f"CI[{100-alpha/2*100}%]"],
        )
        self.bhat_bootstrap = pd.DataFrame(bhat_stars, columns=self.column_names)

        return ci_bhat_bootstrap

    def compute_yhat(self, Xnew=None, interval=None, alpha=0.05):
        if Xnew is None:
            X = self.X.values
        else:
            X = self.X.model_spec.get_model_matrix(Xnew).values
        # compute predicted or fitted values ŷ = Xβ̂
        bhat = self.bhat.values.T
        yhat = X @ bhat
        yhat = pd.DataFrame(yhat, columns=["Fitted"])

        if interval is None:
            return yhat
        elif interval is True:
            ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha)
            pi_yhat = self.compute_pi_yhat(yhat, Xnew, alpha)
            return pd.concat([yhat, ci_yhat, pi_yhat], axis=1)
        elif interval == "prediction":
            pi_yhat = self.compute_pi_yhat(yhat, Xnew, alpha)
            return pd.concat([yhat, pi_yhat], axis=1)
        elif interval == "confidence":
            ci_yhat = self.compute_ci_yhat(yhat, Xnew, alpha)
            return pd.concat([yhat, ci_yhat], axis=1)
        else:
            raise ValueError(
                "Please enter a valid value: [None, True, 'prediction', 'confidence']"
            )

    def compute_ci_yhat(self, yhat, Xnew=None, alpha=0.05):

        if Xnew is None:
            X = self.X.values
        else:
            X = self.X.model_spec.get_model_matrix(Xnew).values

        sigma = self.sigma
        sigma_squared = self.sigma_squared

        V_yhat = X @ self.XtXinv @ X.T * sigma_squared

        se_yhat_mean = np.sqrt(np.diag(V_yhat)) * sigma
        ci_yhat = (
            t.ppf(1 - alpha / 2, self.df_residuals)
            * se_yhat_mean[:, None]
            * np.array([-1, 1])
            + yhat.values
        )
        ci_yhat = pd.DataFrame(
            ci_yhat, columns=[f"CI[{alpha/2*100}%]", f"CI[{100-alpha/2*100}%]"]
        )

        return ci_yhat

    def compute_pi_yhat(self, yhat, Xnew=None, alpha=0.05):

        if Xnew is None:
            X = self.X.values
        else:
            X = self.X.model_spec.get_model_matrix(Xnew).values

        sigma = self.sigma
        sigma_squared = self.sigma_squared

        V_yhat = X @ self.XtXinv @ X.T * sigma_squared

        se_yhat = np.sqrt(1 + np.diag(V_yhat)) * sigma
        pi_yhat = (
            t.ppf(1 - alpha / 2, self.df_residuals)
            * se_yhat[:, None]
            * np.array([-1, 1])
            + yhat.values
        )
        pi_yhat = pd.DataFrame(
            pi_yhat, columns=[f"PI[{alpha/2*100}%]", f"PI[{100-alpha/2*100}%]"]
        )

        return pi_yhat

    def compute_t_values(self):

        t_values = self.bhat.values / self.se_bhat.values

        return pd.DataFrame(t_values, columns=self.column_names, index=["t values"])

    def compute_p_values(self):
        # compute p values of model coefficients with scipy.stats.t.sf
        # H0: βi==0
        # H1: βi!=0
        p_values = 2 * t.sf(np.abs(self.t_values.values), self.df_residuals)
        return pd.DataFrame(p_values, columns=self.column_names, index=["Pr(>|t|)"])

    def compute_goodness_of_fit(self):

        y = self.y.values

        if "Intercept" in self.column_names:
            tss = np.sum((y - y.mean()) ** 2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y - ȳ)**2)
            r_squared = (1 - self.rss / tss).squeeze()
            # Eq: r2adj = 1 - (1 - r2) * (n - 1) / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * (self.n - 1) / (
                self.df_residuals
            )
        else:
            tss = np.sum(y**2)
            # Eq: r2 = 1 - RSS / TSS = 1 -  sum((ŷ - yi)**2) / sum((y)**2)
            r_squared = (1 - self.rss / tss).squeeze()
            # Eq: r2adj = 1 - (1 - r2) * n / df_residuals
            r_squared_adjusted = 1 - (1 - r_squared) * self.n / (self.df_residuals)

        return tss, r_squared, r_squared_adjusted

    def compute_fstats(self):
        if self.df_model != 0:
            fstats = np.squeeze(
                ((self.tss - self.rss) / self.df_model) / (self.rss / self.df_residuals)
            )
            f_p_value = f.sf(fstats, self.df_model, self.df_residuals).squeeze()
        else:
            fstats, f_p_value = None, None
        return fstats, f_p_value

    def compute_loglikelihood(self):
        return np.squeeze(
            -0.5 * self.n * (np.log(self.rss / self.n) + np.log(2 * np.pi) + 1)
        )

    def compute_AIC(self):
        # add 1 to p to keep consistent with R
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + 2 * (self.p + 1)

    def compute_BIC(self):
        # add 1 to p to keep consistent with R
        # https://stackoverflow.com/q/37917437
        return -2 * self.loglike + np.log(self.n) * (self.p + 1)

    def predict(self, Xnew=None, interval=None, alpha=0.05):
        self.Xnew = Xnew  # save it for later use.
        return self.compute_yhat(Xnew=Xnew, interval=interval, alpha=alpha)

    def summary(self, digits=3, cor=False):

        docstring = f"""Formula: {self.formula}\n\n"""
        docstring += "Coefficients:\n"

        sig = significance_code(self.p_values.values.T)
        res = pd.concat(
            [self.bhat, self.se_bhat, self.ci_bhat.T, self.t_values, self.p_values],
            axis=0,
        ).T.round(digits)

        res[""] = sig
        self.results = res

        docstring += res.to_string()
        docstring += "\n---"
        docstring += "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"

        docstring += f"\n\nn = {self.n}, p = {self.p}, Residual SE = {np.sqrt(self.sigma_squared):.3f} on {self.df_residuals} DF\n"
        docstring += f"R-Squared = {self.r_squared:.4f}, adjusted R-Squared = {self.r_squared_adjusted:.4f}\n"

        if self.fstats is not None:
            docstring += f'F-statistics = {self.fstats:.4f} on {self.df_model} and {self.df_residuals} DF, p-value: {self.f_p_value:{".2f" if self.f_p_value > 1e-5 else "e"}}\n\n'

        docstring += f"Log Likelihood = {self.loglike:.4f}, AIC = {self.AIC:.4f}, BIC = {self.BIC:.4f}"

        if cor is True:

            docstring += f"\n\nCorrelation of Coefficients:\n"
            if "Intercept" in self.column_names:
                docstring += (
                    self.X.drop("Intercept", axis=1)
                    .corr()
                    .to_string(
                        formatters={col: "{:.2f}".format for col in self.X.columns}
                    )
                )
            else:
                docstring += self.X.corr().to_string(
                    formatters={col: "{:.2f}".format for col in self.X.columns}
                )

        print(docstring)


class LCRegression(lm):

    """Linear~Circular Regression.

    Basically, it's fitting a circle on the surface of a cylinder.

    y: Linear variable
    θ: Circular variable
    β: coefficients

    y ~ β_0 + β_1 * cos(θ) + β_2 * sin(θ)
    """

    def __init__(
        self,
        formula: str,  # Wilkinson formulas
        data: pd.DataFrame,
        weights: Union[None, np.array] = None,
        method: str = "qr",
    ):

        super().__init__(formula=formula, data=data, weights=weights, method=method)

    def plot(
        self, Xnew=None, ax=None, interval=False, cylinder=False, feature_name="θ"
    ):

        from mpl_toolkits.mplot3d import Axes3D

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # plot raw scatter
        if "Intercept" in self.X.columns:
            design_matrix = self.X.drop("Intercept", axis=1)
        else:
            design_matrix = self.X

        X = design_matrix[design_matrix.columns[0]].values
        ax.set_xlabel(f"{design_matrix.columns[0]}")
        Y = design_matrix[design_matrix.columns[1]].values
        ax.set_ylabel(f"{design_matrix.columns[1]}")
        Z = self.y.values

        ax.scatter(X, Y, Z, color="black")

        # plot fitted curve
        if Xnew is None:
            Xnew = pd.DataFrame(
                data=np.linspace(-np.pi, np.pi, 50), columns=[feature_name]
            )

        design_matrix_new = self.X.model_spec.get_model_matrix(Xnew)
        Xn = design_matrix_new[design_matrix_new.columns[1]].values
        Yn = design_matrix_new[design_matrix_new.columns[2]].values
        Zn = self.compute_yhat(Xnew, interval=True)
        ax.plot(Xn, Yn, Zn["Fitted"].values, color="black")

        if interval:
            ax.plot(Xn, Yn, Zn["CI[97.5%]"], linestyle="--", color="gray")
            ax.plot(Xn, Yn, Zn["CI[2.5%]"], linestyle="--", color="gray")

            z = np.linspace(Zn["CI[2.5%]"].min(), Zn["CI[97.5%]"].max(), 50)

        else:
            z = np.linspace(Z.min(), Z.max(), 50)

        # plot cylindar
        if cylinder:
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, Zc = np.meshgrid(theta, z)
            Xc = np.cos(theta_grid)
            Yc = np.sin(theta_grid)

            ax.plot_surface(Xc, Yc, Zc, alpha=0.2, color="gray")

        ax.set_zlabel(f"{self.y.columns[0]}")


#################
# Estimate bhat #
#################


def _nq(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    Solving the normal equations by directly inverting
    the gram matrix.

    return_ss: return sufficient statistics

    """
    XtX = X.T @ W @ X
    Xty = X.T @ W @ y
    b = np.linalg.inv(XtX) @ (Xty)
    if return_ss:
        return b, XtX, Xty
    else:
        return b


def _lu(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    LU decomposition.
    The same as using numpy.linalg.solve()
    """
    XtX = X.T @ W @ X
    Xty = X.T @ W @ y
    P, L, U = lu(XtX, permute_l=False)
    z = solve_triangular(L, P @ Xty, lower=True)
    b = solve_triangular(U, z)

    if return_ss:
        return b, XtX, Xty
    else:
        return b


def _chol(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    Cholesky decomposition.
    """
    XtX = X.T @ W @ X
    Xty = X.T @ W @ y
    L = cholesky(XtX, lower=True)
    b = solve_triangular(L.T, solve_triangular(L, Xty, lower=True))
    if return_ss:
        return b, XtX, Xty
    else:
        return b


def _svd(X: np.array, y: np.array, return_ss: bool = True):
    """
    Single value decomposition.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Sinv = np.diag(1 / S)
    b = Vt.T @ Sinv @ U.T @ y
    if return_ss:
        XtX = X.T @ X
        Xty = X.T @ y
        return b, XtX, Xty
    else:
        return b


def _qr(X: np.array, y: np.array, W: np.array, return_ss: bool = True):
    """
    QR decomposition.
    """

    L = cholesky(W, lower=True)
    Xhat = L.T @ X
    yhat = L.T @ y

    Q, R = qr(Xhat, mode="economic")
    f = Q.T @ yhat
    b = solve_triangular(R, f)

    if return_ss:
        XtX = X.T @ W @ X
        Xty = X.T @ W @ y
        return b, XtX, Xty
    else:
        return b


def _nll(X, y):

    """
    Negative log-likelihood.
    """

    y = y.flatten()
    n, m = X.shape
    b = np.zeros(m)
    sigma = 1e-5
    p = np.hstack([sigma, b])

    def cost(p, X, y):
        mu = X @ p[1:]
        L = -np.sum(norm.logpdf(y, loc=mu, scale=p[0]))
        return L

    res = minimize(cost, p, args=(X, y), method="L-BFGS-B")
    popt = res.x

    return popt[1:], popt[0], res.fun


def _sse(X, y):

    """
    Sum squared error.
    """

    y = y.flatten()
    n, m = X.shape
    p = np.zeros(m)

    def cost(p, X, y):
        mu = X @ p
        L = np.sum((y - mu) ** 2)
        return L

    res = minimize(cost, p, args=(X, y), method="L-BFGS-B")
    return res.x, res.fun


#########
# Utils #
#########


def AIC(*ms):

    aic = [m.AIC for m in ms]
    df = [m.p + 1 for m in ms]
    formuli = [m.formula for m in ms]

    df = pd.DataFrame.from_dict({"formula": formuli, "df": df, "AIC": aic}).set_index(
        "formula"
    )

    print(df.to_string(formatters={"AIC": "{:.2f}".format}))


def anova(m0, m1):

    models = [m0, m1]

    docstring = "Analysis of Variance Table\n\n"
    for i, model in enumerate(models):
        docstring += f"model {i}: {model.formula}\n"

    df0 = m0.df_residuals
    df1 = m1.df_residuals

    rss0 = m0.rss
    rss1 = m1.rss

    fstat = ((rss0 - rss1) / (df0 - df1)) / (rss1 / df1)
    f_p_value = f.sf(fstat, df0 - df1, df1)
    res = significance_code([f_p_value])[0]

    df_model = ["", f"{df0-df1:.0f}"]
    SoS = ["", f"{np.sum((m1.yhat.values - m1.y.values.mean())**2):.3f}"]

    df = pd.DataFrame.from_dict(
        {
            "Res.Df": [df0, df1],
            "RSS": [rss0, rss1],
            "Df": df_model,
            "Sum of Sq": SoS,
            "F": ["", f"{fstat:.3f}"],
            "Pr(>F)": ["", f"{f_p_value:.3f}"],
            " ": ["", res],
        }
    )

    print(docstring)
    print(df.to_string(formatters={"RSS": "{:.3f}".format}))
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")


def significance_code(p_values):

    sig = []
    for p in p_values:

        if p < 0.001:
            sig.append("***")
        elif p < 0.01:
            sig.append("**")
        elif p < 0.05:
            sig.append("*")
        elif p < 0.1:
            sig.append(".")
        else:
            sig.append(" ")

    return sig
