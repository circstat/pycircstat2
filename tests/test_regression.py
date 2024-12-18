import numpy as np
import pandas as pd

from pycircstat2 import load_data
from pycircstat2.regression import CCRegression, CLRegression


def test_cc_regression_against_r():
    df = load_data(
        "milwaukee",
        source="jammalamadaka",
    )
    ctheta = np.deg2rad(df["theta"].values)
    cpsi = np.deg2rad(df["psi"].values)

    # Expected results from R for order=2
    expected_order2 = {
        "rho": 0.635871,
        "coefficients": {
            "cos": [0.1441268, 0.6414811, 0.1165915, 0.2171076, -0.4374547],
            "sin": [-0.2191974, -0.4509745, 0.1831359, 0.2225796, 0.2924121],
        },
        "p_values": [0.8645504, 0.2263628],
    }

    # Expected results from R for order=4
    expected_order4 = {
        "rho": 0.7164767,
        "coefficients": {
            "cos": [
                0.041599429,
                0.457900406,
                0.088764564,
                -0.028686850,
                0.008402543,
                0.208183091,
                -0.376345616,
                0.157329438,
                0.253633917,
            ],
            "sin": [
                -0.13009026,
                -0.48505260,
                0.06348012,
                -0.25924308,
                -0.06191083,
                0.36991639,
                0.20301592,
                -0.01059473,
                -0.46305354,
            ],
        },
        "p_values": [0.991511, 0.7997684],
    }
    # Test order=2
    model_order2 = CCRegression(theta=ctheta, x=cpsi, order=2)
    np.testing.assert_allclose(
        model_order2.result["rho"], expected_order2["rho"], atol=1e-4
    )
    np.testing.assert_allclose(
        model_order2.result["coefficients"]["cos"],
        expected_order2["coefficients"]["cos"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        model_order2.result["coefficients"]["sin"],
        expected_order2["coefficients"]["sin"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        model_order2.result["p_values"], expected_order2["p_values"], atol=1e-4
    )

    # Test order=4
    model_order4 = CCRegression(theta=ctheta, x=cpsi, order=4)
    np.testing.assert_allclose(
        model_order4.result["rho"], expected_order4["rho"], atol=1e-4
    )
    np.testing.assert_allclose(
        model_order4.result["coefficients"]["cos"],
        expected_order4["coefficients"]["cos"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        model_order4.result["coefficients"]["sin"],
        expected_order4["coefficients"]["sin"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        model_order4.result["p_values"], expected_order4["p_values"], atol=1e-4
    )

    df_rad = df.copy()
    df_rad["theta"] = ctheta
    df_rad["psi"] = cpsi

    # Test formula parsing for order=2
    formula_model = CCRegression(formula="theta ~ psi", data=df_rad, order=2)
    np.testing.assert_allclose(
        formula_model.result["rho"], expected_order2["rho"], atol=1e-4
    )
    np.testing.assert_allclose(
        formula_model.result["coefficients"]["cos"],
        expected_order2["coefficients"]["cos"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        formula_model.result["coefficients"]["sin"],
        expected_order2["coefficients"]["sin"],
        atol=1e-4,
    )
    np.testing.assert_allclose(
        formula_model.result["p_values"], expected_order2["p_values"], atol=1e-4
    )


def test_cl_regression_against_r():
    # Load dataset
    df = load_data("B20", source="fisher")

    X = df["x"].values
    θ = np.deg2rad(df["θ"].values)

    data_cl = pd.DataFrame({"X": X, "θ": θ})

    # Fit the model
    cl_model = CLRegression(
        formula="θ ~ X", data=data_cl, model_type="mean", tol=1e-10, verbose=False
    )

    # Extract results
    result = cl_model.result

    # Expected values from R
    expected_beta = -0.008317
    expected_se_beta = 0.001359
    expected_mu = 2.426
    expected_se_mu = 0.1119
    expected_kappa = 3.224
    expected_se_kappa = 0.7159
    expected_log_likelihood = 27.76

    # Assert coefficients
    assert np.isclose(
        result["beta"][0], expected_beta, atol=1e-3
    ), f"Expected beta: {expected_beta}, got: {result['beta'][0]}"
    assert np.isclose(
        result["se_beta"][0], expected_se_beta, atol=1e-3
    ), f"Expected SE(beta): {expected_se_beta}, got: {result['se_beta'][0]}"

    # Assert mean direction (mu)
    assert np.isclose(
        result["mu"], expected_mu, atol=1e-2
    ), f"Expected mu: {expected_mu}, got: {result['mu']}"
    assert np.isclose(
        result["se_mu"], expected_se_mu, atol=1e-2
    ), f"Expected SE(mu): {expected_se_mu}, got: {result['se_mu']}"

    # Assert concentration parameter (kappa)
    assert np.isclose(
        result["kappa"], expected_kappa, atol=1e-2
    ), f"Expected kappa: {expected_kappa}, got: {result['kappa']}"
    assert np.isclose(
        result["se_kappa"], expected_se_kappa, atol=1e-2
    ), f"Expected SE(kappa): {expected_se_kappa}, got: {result['se_kappa']}"

    # Assert log-likelihood
    assert np.isclose(
        result["log_likelihood"], expected_log_likelihood, atol=1e-2
    ), f"Expected log-likelihood: {expected_log_likelihood}, got: {result['log_likelihood']}"
