import numpy as np
import pandas as pd
import pytest

from pycircstat2 import load_data
from pycircstat2.regression import CCRegression, CLRegression, LCRegression
from pycircstat2.utils import A1inv


_PEWSEY_LUNG_DEATHS = [
    [3035, 2552, 2704, 2554, 2014, 1655, 1721, 1524, 1596, 2074, 2199, 2512],
    [2933, 2889, 2938, 2497, 1870, 1726, 1607, 1545, 1396, 1787, 2076, 2837],
    [2787, 3891, 3179, 2011, 1636, 1580, 1489, 1300, 1356, 1653, 2013, 2823],
    [2996, 2523, 2540, 2520, 1994, 1641, 1691, 1479, 1696, 1877, 2032, 2484],
    [2899, 2990, 2890, 2379, 1933, 1734, 1617, 1495, 1440, 1777, 1970, 2745],
    [2841, 3535, 3010, 2091, 1667, 1589, 1518, 1349, 1392, 1619, 1954, 2633],
]


def _lung_dataframe(drop_feb_outliers: bool = True) -> pd.DataFrame:
    """Pewsey, Neuhäuser & Ruxton (2014) §8.4.1 lung-disease deaths."""
    y = np.array(_PEWSEY_LUNG_DEATHS, dtype=float).ravel()
    month = np.tile(np.arange(1, 13), 6)
    theta = (np.pi / 6) * month
    df = pd.DataFrame({"y": y, "theta": theta, "month": month})
    if drop_feb_outliers:
        # Book sets Feb 1976 (index 25) and Feb 1979 (index 61) to NA.
        df = df.drop(index=[25, 61]).reset_index(drop=True)
    return df


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


def _simulate_cl(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1))
    mu_true, beta_true, kappa_true = 0.7, np.array([0.9]), 5.0
    eps = rng.vonmises(0, kappa_true, size=n)
    theta = mu_true + 2 * np.arctan(X @ beta_true) + eps
    return X, theta, mu_true, beta_true, kappa_true


def test_mixed_matches_mean_when_kappa_is_constant():
    """With constant true κ, mixed and mean should agree on β/μ to high precision."""
    X, theta, _, _, _ = _simulate_cl()

    m_mean = CLRegression(theta=theta, X=X, model_type="mean", tol=1e-10, max_iter=500)
    m_mixed = CLRegression(theta=theta, X=X, model_type="mixed", tol=1e-10, max_iter=500)

    np.testing.assert_allclose(
        m_mixed.result["beta"], m_mean.result["beta"], atol=5e-3
    )
    np.testing.assert_allclose(m_mixed.result["mu"], m_mean.result["mu"], atol=5e-3)
    # exp(α) should be near the mean model's scalar κ since γ should be ≈ 0
    np.testing.assert_allclose(
        np.exp(m_mixed.result["alpha"]), m_mean.result["kappa"], rtol=0.1
    )
    assert abs(m_mixed.result["gamma"][0]) < 0.1


def test_mixed_recovers_true_parameters():
    """Mixed model should recover the simulation parameters within sampling error."""
    X, theta, mu_true, beta_true, kappa_true = _simulate_cl(seed=1, n=800)
    m = CLRegression(theta=theta, X=X, model_type="mixed", tol=1e-10, max_iter=500)
    np.testing.assert_allclose(m.result["beta"], beta_true, atol=0.15)
    np.testing.assert_allclose(m.result["mu"], mu_true, atol=0.1)
    np.testing.assert_allclose(np.exp(m.result["alpha"]), kappa_true, rtol=0.25)


def test_kappa_model_fits_and_predicts_constant_mean():
    """Kappa-only model: κ depends on X but conditional mean is constant μ."""
    X, theta, _, _, _ = _simulate_cl(seed=2, n=300)
    m = CLRegression(theta=theta, X=X, model_type="kappa", tol=1e-8, max_iter=200)

    assert np.all(np.isfinite(m.result["kappa"]))
    assert np.all(m.result["kappa"] > 0)

    pred = m.predict(X)
    assert pred.shape == (X.shape[0],)
    np.testing.assert_allclose(pred, m.result["mu"])


def test_predict_mean_model_round_trip():
    X, theta, _, _, _ = _simulate_cl(seed=3, n=200)
    m = CLRegression(theta=theta, X=X, model_type="mean", tol=1e-10)
    pred = m.predict(X)
    assert pred.shape == theta.shape
    assert np.all(np.isfinite(pred))


def test_se_kappa_delta_method_shape_and_finiteness():
    X, theta, _, _, _ = _simulate_cl(seed=4, n=300)
    for model_type in ("kappa", "mixed"):
        m = CLRegression(theta=theta, X=X, model_type=model_type, tol=1e-8, max_iter=300)
        se = m.result["se_kappa"]
        assert se.shape == (X.shape[0],)
        assert np.all(np.isfinite(se))
        assert np.all(se > 0)


def test_a1inv_clamps_at_unit_radius():
    # A1 maps κ≥0 to [0,1); A1inv at R≥1 must not explode.
    assert np.isfinite(A1inv(1.0))
    assert np.isfinite(A1inv(1.5))
    assert A1inv(0.0) == 0.0


def test_cc_regression_rejects_oversize_order():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 5)
    x = rng.uniform(0, 2 * np.pi, 5)
    with pytest.raises(ValueError, match="more than"):
        CCRegression(theta=theta, x=x, order=5)


def test_cc_regression_exposes_residual_kappa():
    df = load_data("milwaukee", source="jammalamadaka")
    ctheta = np.deg2rad(df["theta"].values)
    cpsi = np.deg2rad(df["psi"].values)
    m = CCRegression(theta=ctheta, x=cpsi, order=2)
    assert "kappa" in m.result and "A_k" in m.result
    assert np.isfinite(m.result["kappa"]) and m.result["kappa"] >= 0
    assert -1 <= m.result["A_k"] <= 1


def test_a1_stable_at_extreme_kappa():
    from pycircstat2.utils import A1

    # i0/i1 overflow around κ ≈ 710; A1 must remain finite via i0e/i1e.
    for k in (700.0, 5_000.0, 1e6):
        val = float(A1(k))
        assert np.isfinite(val)
        assert 0.0 < val < 1.0


def test_log_likelihood_stable_at_high_concentration():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    theta = 0.5 + rng.vonmises(0, 50.0, n)
    m = CLRegression(theta=theta, X=X, model_type="kappa", tol=1e-8, max_iter=300)
    assert np.isfinite(m.result["log_likelihood"])
    assert np.all(np.isfinite(m.result["kappa"]))


def test_formula_parser_rejects_malformed_formulas():
    df = pd.DataFrame({"y": [0.1, 0.2], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="exactly one '~'"):
        CLRegression(formula="y ~ x ~ z", data=df)
    with pytest.raises(ValueError, match="No predictors"):
        CLRegression(formula="y ~ ", data=df)


def test_cc_summary_label_widths(capsys):
    rng = np.random.default_rng(0)
    n = 60
    theta = rng.uniform(0, 2 * np.pi, n)
    x = rng.uniform(0, 2 * np.pi, n)
    m = CCRegression(theta=theta, x=x, order=4)
    m.summary()
    captured = capsys.readouterr().out
    # Long labels like "cos(x1,k=4)" must appear intact (not truncated).
    assert "cos(x1,k=4)" in captured
    assert "sin(x1,k=4)" in captured


# ----------------------------- LCRegression -------------------------------


def test_lc_regression_against_pewsey_lung_disease():
    """§8.4.1 reduced extended model: y ~ cos(θ) + sin(θ) + sin(2θ)."""
    df = _lung_dataframe(drop_feb_outliers=True)
    m = LCRegression("y ~ cos(theta) + sin(theta) + sin(2*theta)", df)
    coefs = m.result["coefficients"]
    assert np.isclose(coefs["(Intercept)"], 2125.12, atol=1e-1)
    assert np.isclose(coefs["cos(theta)"], 454.18, atol=1e-1)
    assert np.isclose(coefs["sin(theta)"], 601.96, atol=1e-1)
    assert np.isclose(coefs["sin(2 * theta)"], 108.69, atol=1e-1)
    assert np.isclose(m.result["sigma"], 171.3, atol=1e-1)
    assert np.isclose(m.result["r_squared"], 0.9093, atol=1e-3)


def test_lc_marker_matches_explicit():
    """harmonic(theta, k=K) must produce identical fit to the explicit form."""
    rng = np.random.default_rng(0)
    n = 240
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = (
        2.0
        + 1.5 * np.cos(theta - 0.7)
        + 0.6 * np.cos(2 * theta - 1.2)
        + rng.normal(0, 0.2, n)
    )
    df = pd.DataFrame({"y": y, "theta": theta})

    marker = LCRegression("y ~ harmonic(theta, k=2)", df)
    explicit = LCRegression(
        "y ~ cos(theta) + sin(theta) + cos(2*theta) + sin(2*theta)", df
    )
    np.testing.assert_allclose(
        list(marker.result["coefficients"].values()),
        list(explicit.result["coefficients"].values()),
        atol=1e-10,
    )
    assert marker.expanded_formula == explicit.expanded_formula
    np.testing.assert_allclose(
        [h["amplitude"] for h in marker.result["harmonics"]],
        [h["amplitude"] for h in explicit.result["harmonics"]],
        atol=1e-10,
    )


def test_lc_amplitude_phase_recovery():
    """Generated data with known γ₁ and φ₁ — recovered to good precision."""
    rng = np.random.default_rng(1)
    n = 1000
    theta = rng.uniform(0, 2 * np.pi, n)
    true_amp, true_phase = 2.5, 0.9
    y = 5.0 + true_amp * np.cos(theta - true_phase) + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"y": y, "theta": theta})

    m = LCRegression("y ~ harmonic(theta)", df)
    h = m.result["harmonics"]
    assert len(h) == 1
    assert h[0]["k"] == 1
    assert np.isclose(h[0]["amplitude"], true_amp, atol=0.05)
    assert np.isclose(h[0]["phase"], true_phase, atol=0.05)


def test_lc_marker_with_extra_covariate():
    """harmonic(theta) + temperature: marker expands, covariate passes through."""
    rng = np.random.default_rng(2)
    n = 300
    theta = rng.uniform(0, 2 * np.pi, n)
    temperature = rng.normal(20, 5, n)
    y = 1.0 + 2.0 * np.cos(theta - 0.4) + 0.3 * temperature + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"y": y, "theta": theta, "temperature": temperature})

    m = LCRegression("y ~ harmonic(theta) + temperature", df)
    coefs = m.result["coefficients"]
    assert "temperature" in coefs
    assert np.isclose(coefs["temperature"], 0.3, atol=0.05)
    h = m.result["harmonics"][0]
    assert np.isclose(h["amplitude"], 2.0, atol=0.05)


def test_lc_predict_round_trip():
    df = _lung_dataframe(drop_feb_outliers=True)
    m = LCRegression("y ~ harmonic(theta, k=2)", df)
    pred = m.predict(df)
    assert pred.shape == (len(df),)
    np.testing.assert_allclose(pred, m.result["fitted"], atol=1e-10)


def test_lc_skew_and_flat_not_implemented():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "theta": [0.0, 1.0, 2.0]})
    with pytest.raises(NotImplementedError, match="hea.nls"):
        LCRegression("y ~ skew(theta)", df)
    with pytest.raises(NotImplementedError, match="hea.nls"):
        LCRegression("y ~ flat(theta)", df)


def test_lc_formula_validation():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "theta": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="'~'"):
        LCRegression("not a formula", df)
    with pytest.raises(ValueError, match="positive integer"):
        LCRegression("y ~ harmonic(theta, k=0)", df)


def test_lc_accepts_polars_data():
    import polars as pl

    df = _lung_dataframe(drop_feb_outliers=True)
    m_pd = LCRegression("y ~ harmonic(theta)", df)
    m_pl = LCRegression("y ~ harmonic(theta)", pl.from_pandas(df))
    np.testing.assert_allclose(
        list(m_pd.result["coefficients"].values()),
        list(m_pl.result["coefficients"].values()),
        atol=1e-12,
    )
