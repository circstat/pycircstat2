import numpy as np
import pandas as pd
import polars as pl
import pytest

from pycircstat2 import load_data
from pycircstat2.regression import CCRegression, CLRegression, LCRegression
from pycircstat2.utils import A1inv


def _lung_dataframe(drop_feb_outliers: bool = True) -> "pl.DataFrame":
    """Pewsey, Neuhäuser & Ruxton (2013) §8.4.1 lung-disease deaths."""
    df = load_data("lung_deaths", source="pewsey")
    df = df.with_columns(((np.pi / 6) * pl.col("month")).alias("theta"))
    df = df.rename({"deaths": "y"})
    if drop_feb_outliers:
        df = df.filter(~((pl.col("month") == 2) & pl.col("year").is_in([1976, 1979])))
    return df


def test_cc_regression_against_r():
    df = load_data(
        "milwaukee",
        source="jammalamadaka",
    )
    ctheta = np.deg2rad(df["theta"].to_numpy())
    cpsi = np.deg2rad(df["psi"].to_numpy())

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

    df_rad = df.with_columns(
        pl.Series("theta", ctheta),
        pl.Series("psi", cpsi),
    )

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

    X = df["x"].to_numpy()
    θ = np.deg2rad(df["θ"].to_numpy())

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
    np.testing.assert_allclose(pred, np.mod(m.result["mu"], 2 * np.pi))


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
    ctheta = np.deg2rad(df["theta"].to_numpy())
    cpsi = np.deg2rad(df["psi"].to_numpy())
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
    # The single-string formula parser is the fisher-lee path (the default
    # gam backend routes single strings through hea's own parser).
    df = pd.DataFrame({"y": [0.1, 0.2], "x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="exactly one '~'"):
        CLRegression(formula="y ~ x ~ z", data=df, backend="fisher-lee")
    with pytest.raises(ValueError, match="No predictors"):
        CLRegression(formula="y ~ ", data=df, backend="fisher-lee")


def test_cl_plot_mean_model_1d():
    import matplotlib

    matplotlib.use("Agg")
    df = load_data("B20", source="fisher")
    data = pd.DataFrame({"X": df["x"].to_numpy(), "θ": np.deg2rad(df["θ"].to_numpy())})
    m = CLRegression(formula="θ ~ X", data=data, model_type="mean")
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Fit overlay" in titles
    assert "Residuals vs X" in titles
    overlay = next(ax for ax in fig.axes if ax.get_title() == "Fit overlay")
    ylo, yhi = overlay.get_ylim()
    assert ylo == 0.0 and np.isclose(yhi, 4 * np.pi)


def test_cl_plot_kappa_only_shows_kappa_curve():
    import matplotlib

    matplotlib.use("Agg")
    df = load_data("B20", source="fisher")
    data = pd.DataFrame({"X": df["x"].to_numpy(), "θ": np.deg2rad(df["θ"].to_numpy())})
    m = CLRegression(formula="θ ~ X", data=data, model_type="kappa")
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Fitted concentration" in titles
    kappa_ax = next(ax for ax in fig.axes if ax.get_title() == "Fitted concentration")
    # κ curve must be strictly positive.
    line = kappa_ax.get_lines()[0]
    ys = line.get_ydata()
    assert np.all(ys > 0)


def test_cl_predict_kappa_kappa_and_mixed():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    theta = 0.5 + rng.vonmises(0, 50.0, n)
    for model_type in ("kappa", "mixed"):
        m = CLRegression(theta=theta, X=X, model_type=model_type, tol=1e-8, max_iter=200)
        kappa_pred = m.predict_kappa(np.array([0.0, 1.0, -1.0]))
        assert kappa_pred.shape == (3,)
        assert np.all(kappa_pred > 0) and np.all(np.isfinite(kappa_pred))
        # κ at X=0 must equal exp(α̂).
        np.testing.assert_allclose(
            kappa_pred[0], np.exp(m.result["alpha"]), atol=1e-10
        )


def test_cl_predict_kappa_rejects_mean_model():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 1))
    theta = 0.5 + 2 * np.arctan(X[:, 0] * 0.3) + rng.vonmises(0, 5.0, n)
    m = CLRegression(theta=theta, X=X, model_type="mean")
    with pytest.raises(ValueError, match="model_type in"):
        m.predict_kappa(np.array([0.0]))


def test_lc_harmonic_positional_k():
    """Both `harmonic(theta, k=2)` and `harmonic(theta, 2)` must work."""
    df = _lung_dataframe(drop_feb_outliers=True)
    m_kw = LCRegression("y ~ harmonic(theta, k=2)", df)
    m_pos = LCRegression("y ~ harmonic(theta, 2)", df)
    assert m_kw.expanded_formula == m_pos.expanded_formula
    np.testing.assert_allclose(
        list(m_kw.result["coefficients"].values()),
        list(m_pos.result["coefficients"].values()),
        atol=1e-12,
    )


def test_cl_summary_does_not_print_mean_se(capsys):
    """Per-obs SEs are correlated; averaging them is meaningless. Drop it."""
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 1))
    theta = 0.5 + rng.vonmises(0, 5.0, n)
    m = CLRegression(theta=theta, X=X, model_type="kappa")
    m.summary()
    out = capsys.readouterr().out
    # Old format included "Mean: ... (SE: ...)"; new format drops the SE.
    mean_lines = [ln for ln in out.splitlines() if ln.strip().startswith("Mean:")]
    assert mean_lines, "summary should print a Mean kappa line"
    for ln in mean_lines:
        assert "SE" not in ln


def test_cl_plot_multi_feature_fallback():
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(size=(n, 2))
    theta = 0.7 + 2 * np.arctan(X @ np.array([0.5, -0.3])) + rng.vonmises(0, 5.0, n)
    m = CLRegression(theta=theta, X=X, model_type="mean")
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Residuals vs fitted" in titles
    assert "Residual histogram" in titles


def test_cc_predict_round_trip_on_training_data():
    df = load_data("milwaukee", source="jammalamadaka")
    ctheta = np.deg2rad(df["theta"].to_numpy())
    cpsi = np.deg2rad(df["psi"].to_numpy())
    m = CCRegression(theta=ctheta, x=cpsi, order=2)
    pred = m.predict(cpsi)
    diff = np.angle(np.exp(1j * (pred - m.result["fitted"])))
    assert np.max(np.abs(diff)) < 1e-10


def test_cc_predict_rejects_wrong_feature_count():
    rng = np.random.default_rng(0)
    n = 30
    theta = rng.uniform(0, 2 * np.pi, n)
    x = rng.uniform(0, 2 * np.pi, (n, 2))
    m = CCRegression(theta=theta, x=x, order=1)
    with pytest.raises(ValueError, match="Expected 2"):
        m.predict(np.ones(5))


def test_cc_plot_single_feature_two_panels():
    import matplotlib

    matplotlib.use("Agg")
    df = load_data("milwaukee", source="jammalamadaka")
    ctheta = np.deg2rad(df["theta"].to_numpy())
    cpsi = np.deg2rad(df["psi"].to_numpy())
    m = CCRegression(theta=ctheta, x=cpsi, order=2)
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Fit overlay" in titles
    assert "Residuals vs predictor" in titles
    overlay_ax = next(ax for ax in fig.axes if ax.get_title() == "Fit overlay")
    # y-axis should span [0, 4π] for the stacked-copy display.
    ylo, yhi = overlay_ax.get_ylim()
    assert ylo == 0.0 and np.isclose(yhi, 4 * np.pi)


def test_cc_plot_multi_feature_fallback():
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    n = 80
    x = rng.uniform(0, 2 * np.pi, (n, 2))
    theta = np.mod(0.5 + 0.3 * np.sin(x[:, 0]) + 0.2 * np.cos(x[:, 1]), 2 * np.pi)
    m = CCRegression(theta=theta, x=x, order=1)
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Residuals vs fitted" in titles
    assert "Residual histogram" in titles
    with pytest.raises(ValueError, match="single circular predictor"):
        m.plot(polar=True)


def test_cc_plot_band_toggle():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection

    df = load_data("milwaukee", source="jammalamadaka")
    ctheta = np.deg2rad(df["theta"].to_numpy())
    cpsi = np.deg2rad(df["psi"].to_numpy())
    m = CCRegression(theta=ctheta, x=cpsi, order=1)

    def overlay_ax(fig):
        return next(ax for ax in fig.axes if ax.get_title() == "Fit overlay")

    def n_bands(fig):
        return sum(
            isinstance(c, PolyCollection) for c in overlay_ax(fig).collections
        )

    # Band on by default (one polygon per visible 2π replica), removable.
    assert n_bands(m.plot()) >= 2
    assert n_bands(m.plot(band=False)) == 0

    # The unwrapped curve is continuous: no NaN seam-breaks in any replica.
    for line in overlay_ax(m.plot()).get_lines():
        assert np.all(np.isfinite(line.get_ydata()))

    # Band half-width is the circular SD implied by ρ̂: positive, ≤ π.
    grid = np.linspace(0.0, 2 * np.pi, 50)
    cos_fit, sin_fit = m._predict_embedding(grid[:, None])
    rho = np.clip(np.hypot(cos_fit, sin_fit), 1e-9, 1.0)
    sd = np.minimum(np.sqrt(-2.0 * np.log(rho)), np.pi)
    assert np.all((sd > 0) & (sd <= np.pi))


def test_cc_plot_polar_clock():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.quiver import Quiver

    df = load_data("milwaukee", source="jammalamadaka")
    ctheta = np.deg2rad(df["theta"].to_numpy())
    cpsi = np.deg2rad(df["psi"].to_numpy())
    for m in (
        CCRegression(theta=ctheta, x=cpsi, order=1),
        CCRegression(
            "theta ~ s(psi, bs='cc')",
            pl.DataFrame({"theta": ctheta, "psi": cpsi}),
        ),
    ):
        fig = m.plot(polar=True)
        clock = fig.axes[0]
        assert "Fit clock" in clock.get_title()
        assert not clock.axison  # compass frame, no cartesian axes
        # Two quivers: fitted-direction arrows and data arrows.
        assert sum(isinstance(c, Quiver) for c in clock.collections) == 2
        # The residual panel is untouched by the polar switch.
        assert fig.axes[1].get_title() == "Residuals vs predictor"


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
    # marker expands to the same explicit cos/sin terms (identical fit and
    # harmonic decomposition); the marker is shorthand for the explicit form.
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


def test_lc_accepts_pandas_data():
    """Polars is the native input; pandas is still accepted via the soft path."""
    pytest.importorskip("pandas")

    df_pl = _lung_dataframe(drop_feb_outliers=True)
    m_pl = LCRegression("y ~ harmonic(theta)", df_pl)
    m_pd = LCRegression("y ~ harmonic(theta)", df_pl.to_pandas())
    np.testing.assert_allclose(
        list(m_pl.result["coefficients"].values()),
        list(m_pd.result["coefficients"].values()),
        atol=1e-12,
    )


def test_lc_harmonic_se_and_ci_present():
    """Delta-method SEs for amplitude/phase should be finite and positive."""
    df = _lung_dataframe(drop_feb_outliers=True)
    m = LCRegression("y ~ harmonic(theta, k=2)", df)
    for h in m.result["harmonics"]:
        assert h["se_amplitude"] is not None and h["se_amplitude"] > 0
        assert h["se_phase"] is not None and h["se_phase"] > 0
        # Sanity: SE_amp should not exceed the amplitude itself by orders of magnitude.
        assert h["se_amplitude"] < 10 * h["amplitude"]


def test_lc_accepts_unicode_identifiers():
    """Greek/Unicode column names should work in formulas (e.g. `θ`)."""
    df = _lung_dataframe(drop_feb_outliers=True)
    df_unicode = df.rename({"theta": "θ"})
    m = LCRegression("y ~ harmonic(θ, k=2)", df_unicode)
    assert "cos(θ)" in m.expanded_formula  # expands to explicit cos/sin terms
    assert np.isclose(m.result["r_squared"], 0.9094, atol=1e-3)
    assert all(h["variable"] == "θ" for h in m.result["harmonics"])


def test_lc_plot_returns_figure_with_two_panels():
    import matplotlib

    matplotlib.use("Agg")
    df = _lung_dataframe(drop_feb_outliers=True)
    m = LCRegression("y ~ harmonic(theta, k=2)", df)
    fig = m.plot(ci=True, pi=True)
    titles = [ax.get_title() for ax in fig.axes]
    assert "Fit overlay" in titles
    assert "Residuals vs fitted" in titles
    # Fit overlay axes should at least contain fit + data + CI + PI artists.
    overlay_ax = next(ax for ax in fig.axes if ax.get_title() == "Fit overlay")
    labels = [ln.get_label() for ln in overlay_ax.get_lines()]
    assert "fit" in labels


def test_lc_plot_with_extra_covariate_holds_at_mean():
    """When extra covariates exist, the fit curve fixes them at the column mean."""
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.default_rng(0)
    n = 200
    theta = rng.uniform(0, 2 * np.pi, n)
    temp = rng.normal(20, 5, n)
    y = 2 + 1.5 * np.cos(theta - 0.5) + 0.1 * temp + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"y": y, "theta": theta, "temp": temp})
    m = LCRegression("y ~ harmonic(theta) + temp", df)
    fig = m.plot(ci=False, pi=False)
    overlay_ax = next(ax for ax in fig.axes if ax.get_title() == "Fit overlay")
    fit_line = next(ln for ln in overlay_ax.get_lines() if ln.get_label() == "fit")
    xs, ys = fit_line.get_xdata(), fit_line.get_ydata()
    # At the curve midpoint of θ, the value should match the analytical
    # "fit at theta=π, temp=mean" prediction within numerical tolerance.
    coefs = m.result["coefficients"]
    h = m.result["harmonics"][0]
    expected_at_pi = (
        coefs["(Intercept)"]
        + h["cos_coef"] * np.cos(np.pi)
        + h["sin_coef"] * np.sin(np.pi)
        + coefs["temp"] * float(temp.mean())
    )
    idx = np.argmin(np.abs(xs - np.pi))
    # Tolerance reflects the grid spacing (200 points across [0, 2π]).
    assert np.isclose(ys[idx], expected_at_pi, atol=0.05)


def test_lc_summary_includes_lm_block_and_harmonic_table(capsys):
    df = _lung_dataframe(drop_feb_outliers=True)
    m = LCRegression("y ~ harmonic(theta, k=2)", df)
    m.summary()
    out = capsys.readouterr().out
    # hea.lm summary content
    assert "Coefficients:" in out
    assert "Pr(>|t|)" in out
    assert "R-Squared" in out
    # Our additions
    assert "Harmonic decomposition" in out
    assert "amplitude" in out
    assert "phase" in out


# --- A.1: GAM backend (smooth-term dispatch) ---------------------------------


def _smooth_lc_data(n: int = 160, seed: int = 2) -> "pl.DataFrame":
    """Linear response over a circular predictor (LC), smooth in θ."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = 2 * np.sin(x) + 0.5 * np.cos(2 * x) + rng.normal(0, 0.3, n)
    return pl.DataFrame({"x": x, "y": y})


def _smooth_cc_data(n: int = 160, seed: int = 4) -> "pl.DataFrame":
    """Circular response with a cyclic-in-x mean direction (CC)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2 * np.pi, n)
    theta = np.mod(x + 0.6 * np.sin(x) + 0.25 * rng.standard_normal(n), 2 * np.pi)
    return pl.DataFrame({"x": x, "theta": theta})


def test_lc_gam_dispatch_on_smooth_term():
    df = _smooth_lc_data()
    m = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    assert m.backend == "gam"
    assert m.lm_fit is None and m.gam_fit is not None
    # smooth backend → no harmonic decomposition, but edf + fit metrics present
    assert m.result["harmonics"] == []
    assert m.result["edf_total"] > 1.0
    assert "s(x)" in m.result["edf_by_smooth"]
    assert 0.0 < m.result["deviance_explained"] <= 1.0
    n = df.height
    assert m.result["fitted"].shape == (n,)
    assert m.result["residuals"].shape == (n,)


def test_lc_parametric_still_uses_lm():
    df = _smooth_lc_data()
    m = LCRegression("y ~ harmonic(x, k=2)", df)
    assert m.backend == "lm"
    assert m.gam_fit is None and m.lm_fit is not None
    assert len(m.result["harmonics"]) == 2


def test_lc_gam_cyclic_period_continuity():
    """A cc smooth wraps at the period: f(0) == f(2π)."""
    df = _smooth_lc_data()
    m = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    ends = m.predict(pl.DataFrame({"x": [0.0, 2 * np.pi]}))
    assert np.isclose(ends[0], ends[1], atol=1e-6)


def test_lc_gam_predict_matches_in_sample_fit():
    df = _smooth_lc_data()
    m = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    pred = m.predict(df.select("x"))
    assert np.allclose(pred, m.result["fitted"], atol=1e-6)


def test_lc_gam_knots_rejected_on_parametric_formula():
    df = _smooth_lc_data()
    with pytest.raises(ValueError, match="smooth"):
        LCRegression("y ~ harmonic(x)", df, knots={"x": [0.0, 2 * np.pi]})


def test_lc_gam_plot_two_panels_and_polar():
    import matplotlib

    matplotlib.use("Agg")
    df = _smooth_lc_data()
    m = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Fit overlay" in titles
    assert "Residuals vs fitted" in titles
    figp = m.plot(polar=True)
    overlay = next(ax for ax in figp.axes if ax.get_title() == "Fit overlay")
    assert overlay.name == "polar"


def test_lc_gam_summary_shows_smooth_block(capsys):
    df = _smooth_lc_data()
    m = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    m.summary()
    out = capsys.readouterr().out
    assert "Linear-Circular Regression" in out
    # mgcv-style gam block printed by hea.gam.summary()
    assert "smooth terms" in out


def test_cc_gam_dispatch_and_result():
    df = _smooth_cc_data()
    m = CCRegression("theta ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    assert m.backend == "gam"
    # no harmonic coefficients / no higher-order χ² test on the smooth backend
    assert m.result["coefficients"] is None
    assert np.all(np.isnan(m.result["p_values"]))
    assert set(m.result["edf_total"]) == {"cos", "sin"}
    assert 0.0 <= m.result["rho"] <= 1.0
    assert np.isfinite(m.result["kappa"])
    assert m.result["fitted"].shape == (df.height,)


def test_cc_gam_predict_is_circular_and_cyclic():
    df = _smooth_cc_data()
    m = CCRegression("theta ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    pred = m.predict(np.array([0.0, 2 * np.pi]))
    assert np.all((pred >= 0) & (pred < 2 * np.pi + 1e-9))
    # cyclic smooths → μ̂(0) == μ̂(2π) (allowing a wrap of exactly 2π)
    gap = abs(pred[0] - pred[1])
    assert min(gap, 2 * np.pi - gap) < 1e-3


def test_cc_gam_plot_and_summary(capsys):
    import matplotlib

    matplotlib.use("Agg")
    df = _smooth_cc_data()
    m = CCRegression("theta ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    fig = m.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert "Fit overlay" in titles
    m.summary()
    out = capsys.readouterr().out
    assert "gam backend" in out
    assert "rho" in out.lower()


def test_cc_gam_knots_rejected_on_parametric():
    df = _smooth_cc_data()
    with pytest.raises(ValueError, match="smooth"):
        CCRegression(
            theta=df["theta"].to_numpy(),
            x=df["x"].to_numpy(),
            knots={"x": [0.0, 2 * np.pi]},
        )


def test_lc_gam_cyclic_knots_default_to_period():
    """A cyclic smooth wraps at [0, 2π] without the caller passing knots —
    identical to the explicit knots, and f(0) == f(2π)."""
    df = _smooth_lc_data()
    auto = LCRegression("y ~ s(x, bs='cc')", df)
    explicit = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    assert np.allclose(auto.result["fitted"], explicit.result["fitted"], atol=1e-9)
    ends = auto.predict(pl.DataFrame({"x": [0.0, 2 * np.pi]}))
    assert np.isclose(ends[0], ends[1], atol=1e-6)


def test_lc_gam_knots_override_period():
    """An explicit knots= overrides the default circular period."""
    df = _smooth_lc_data()
    default = LCRegression("y ~ s(x, bs='cc')", df)
    other = LCRegression("y ~ s(x, bs='cc')", df, knots={"x": [0.0, 12.0]})
    assert not np.allclose(default.result["fitted"], other.result["fitted"])


def test_lc_noncyclic_smooth_gets_no_default_knots():
    """Only cyclic bases get the [0, 2π] default; a plain smooth fits fine."""
    df = _smooth_lc_data()
    m = LCRegression("y ~ s(x)", df)
    assert m.backend == "gam"
    assert m.result["edf_total"] > 1.0


def test_cc_gam_cyclic_knots_default_to_period():
    df = _smooth_cc_data()
    auto = CCRegression("theta ~ s(x, bs='cc')", df)
    explicit = CCRegression("theta ~ s(x, bs='cc')", df, knots={"x": [0.0, 2 * np.pi]})
    assert np.allclose(auto.result["fitted"], explicit.result["fitted"], atol=1e-9)


# ---------------------------------------------------------------------------
# Phase 2: CircularLL — circular distributions as hea general families
# ---------------------------------------------------------------------------

from hea.models import gam as hea_gam  # noqa: E402

from pycircstat2.distributions import (  # noqa: E402
    projectednormal,
    vonmises,
    wrapcauchy,
)
from pycircstat2.regression import CircularLL  # noqa: E402


def test_circularll_requires_regression_ready():
    # triangular stays off the regression contract (no location parameter,
    # non-smooth density — see the regression plan's "not worth promoting")
    from pycircstat2.distributions import triangular

    with pytest.raises(TypeError, match="regression-ready"):
        CircularLL(triangular)


def test_katojones_family_dispatch():
    """katojones regresses in disc-chart coordinates (u1/u2 are not logpdf
    parameters): the bare distribution must auto-route to KatoJonesLL in
    CLRegression, and base CircularLL must refuse it with guidance."""
    from pycircstat2.distributions import katojones
    from pycircstat2.regression import KatoJonesLL

    with pytest.raises(TypeError, match="KatoJonesLL"):
        CircularLL(katojones)

    rng = np.random.default_rng(3)
    theta = np.array([
        float(katojones.rvs(mu=2.0, gamma=0.4, rho=0.3, lam=0.5,
                            size=1, random_state=rng)[0])
        for _ in range(80)
    ])
    df = pl.DataFrame({"theta": theta})
    m = CLRegression(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family=katojones)
    assert isinstance(m.family, KatoJonesLL)
    assert np.isfinite(m.result["log_likelihood"])
    # the auto-route fits the same model as an explicit KatoJonesLL()
    m2 = CLRegression(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df,
                      family=KatoJonesLL())
    assert np.isclose(m.result["log_likelihood"],
                      m2.result["log_likelihood"], rtol=1e-8)


def test_lss_alias_surface():
    """The *lss instance aliases (validation plan §3.5): module-level
    pre-built families sharing the circlss names — ``family=vmlss`` is the
    primary explicit-gam spelling. Checks identity wiring, name parity,
    the ``__call__`` re-configurator (fresh instance, alias untouched),
    chart-coordinate subclass routing, and the regression.py re-export."""
    from pycircstat2 import distributions as D
    from pycircstat2.regression import KatoJonesLL

    aliases = {
        "cardlss": D.cardioid, "cartlss": D.cartwright, "wnlss": D.wrapnorm,
        "wclss": D.wrapcauchy, "vmlss": D.vonmises,
        "pnlss": D.projectednormal, "jplss": D.jonespewsey,
        "ssjplss": D.jonespewsey_sineskewed, "kjlss": D.katojones,
        "ibslss": D.inverse_batschelet,
    }
    for name, dist in aliases.items():
        fam = getattr(D, name)
        assert isinstance(fam, D.CircularLL)
        assert fam.dist is dist
        assert fam.name == name           # circlss family$family parity
        assert fam.n_theta == 0           # the shared-instance safety condition
        assert fam.n_lp == len(dist.param_roles)
    assert isinstance(D.kjlss, D.KatoJonesLL)  # disc-chart routing baked in

    # __call__ is the family-side freeze idiom: a fresh configured family,
    # the module-level alias stays pristine, the alias name propagates.
    clone = D.vmlss(links=["identity", "log"])
    assert clone is not D.vmlss and type(clone) is D.CircularLL
    assert [lnk.name for lnk in clone.links] == ["identity", "log"]
    assert [lnk.name for lnk in D.vmlss.links] == ["tanhalf", "log"]
    assert clone.name == "vmlss"
    fresh = D.vmlss()                     # the R parens spelling, verbatim
    assert fresh is not D.vmlss
    assert [lnk.name for lnk in fresh.links] == ["tanhalf", "log"]
    kj = D.kjlss()
    assert type(kj) is D.KatoJonesLL and kj.name == "kjlss"

    # the moved classes re-export unchanged from regression.py
    assert CircularLL is D.CircularLL and KatoJonesLL is D.KatoJonesLL


def test_ibslss_intercept_only_matches_marginal_mle():
    """The M3 end-to-end gate for the inverse-Batschelet *lss family: an
    intercept-only ``ibslss`` fit must reproduce ``inverse_batschelet.fit``
    (the trusted marginal MLE) — both maximize the same likelihood, and for
    intercept-only the mgcv-inside-link and the marginal parameterizations
    coincide. This pins the whole bridge at once: ll, dlogpdf, the FD-grade
    d2logpdf Hessian, the closed-form null start, the EFS optimizer, and the
    per-observation normalizer (dev/plans/vectorize-distributions-and-ibslss.md
    §7)."""
    from pycircstat2.distributions import ibslss, inverse_batschelet
    from pycircstat2.regression import circ_gam

    data = inverse_batschelet.rvs(
        xi=2.3, kappa=2.5, nu=0.35, lmbd=-0.3, size=4000, random_state=7
    )
    df = pl.DataFrame({"theta": data})
    g = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family=ibslss,
                 method="REML")
    pred = np.asarray(
        g.predict(pl.DataFrame({"_dummy": [0.0]}), type="response")
    )[0]
    xi_g, kappa_g, nu_g, lmbd_g = pred
    xi_m, kappa_m, nu_m, lmbd_m = inverse_batschelet.fit(data)

    ang = (xi_g - xi_m + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(ang) < 1e-3
    np.testing.assert_allclose([kappa_g, nu_g, lmbd_g],
                               [kappa_m, nu_m, lmbd_m], atol=1e-3, rtol=0.0)


def test_ibslss_recovers_covariate_location():
    """M4: an ``ibslss`` GAM with a covariate-driven location exercises the
    per-observation (array) parameter path — the one intercept-only parity
    cannot reach (constant params collapse to the scalar path). Simulate
    ξ(x) = 1.5·sin x with κ,ν,λ fixed (rotate a base sample), fit a location
    smooth, and confirm the recovered direction tracks the truth."""
    from pycircstat2.distributions import ibslss, inverse_batschelet
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(11)
    n = 1500
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    xi_true = 1.5 * np.sin(x)
    base = inverse_batschelet.rvs(xi=0.0, kappa=4.0, nu=0.3, lmbd=-0.2,
                                  size=n, random_state=rng)
    theta = np.mod(base + xi_true, 2.0 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x})

    g = circ_gam(["theta ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family=ibslss,
                 method="REML")
    assert g.converged
    grid = np.linspace(x.min(), x.max(), 200)
    mu = np.angle(np.exp(1j * np.asarray(
        g.predict(pl.DataFrame({"x": grid}), type="response"))[:, 0]))
    truth = 1.5 * np.sin(grid)
    a = mu - np.angle(np.mean(np.exp(1j * mu)))
    b = truth - np.angle(np.mean(np.exp(1j * truth)))
    corr = (np.sum(np.sin(a) * np.sin(b))
            / np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)))
    assert corr > 0.95


def test_ibslss_recovers_covariate_concentration():
    """A *distributional* `ibslss` smooth — κ(x) on the log link — is the only
    fit that drives the per-observation normalizer derivative (the location
    smooth leaves κ,λ constant, so its normalizer block stays on the cheap
    scalar path). Simulate κ(x)=exp(0.7+0.9 sin x) (binned scalar draws), fit a
    concentration smooth, and confirm the recovered log-κ tracks the truth.
    This is the permanent guard on the vectorized normalizer that the κ(x)/λ(x)
    perf rewrite enabled (dev/plans/vectorize-distributions-and-ibslss.md)."""
    from pycircstat2.distributions import ibslss, inverse_batschelet
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(7)
    n = 700
    x = np.sort(rng.uniform(-np.pi, np.pi, n))
    theta = np.empty(n)
    edges = np.linspace(x.min(), x.max(), 21)
    idx = np.clip(np.digitize(x, edges) - 1, 0, 19)
    for b in range(20):
        m = idx == b
        if not m.any():
            continue
        kb = float(np.exp(0.7 + 0.9 * np.sin(x[m].mean())))
        theta[m] = inverse_batschelet.rvs(xi=0.4, kappa=kb, nu=0.2, lmbd=-0.2,
                                          size=int(m.sum()), random_state=rng)
    df = pl.DataFrame({"theta": np.mod(theta, 2.0 * np.pi), "x": x})

    g = circ_gam(["theta ~ 1", "~ s(x)", "~ 1", "~ 1"], df, family=ibslss,
                 method="REML")
    assert g.converged
    grid = np.linspace(x.min(), x.max(), 150)
    eta_k = np.asarray(g.predict(pl.DataFrame({"x": grid}), type="link"))[:, 1]
    log_k_true = 0.7 + 0.9 * np.sin(grid)
    corr = np.corrcoef(eta_k, log_k_true)[0, 1]
    assert corr > 0.9


def test_lss_alias_is_clregression_default():
    """``family=None``, ``family=vmlss`` and ``family=CircularLL(vonmises)``
    fit the same model; the default *is* the shared vmlss alias, so fitted
    summaries print the cross-language family name."""
    from pycircstat2.distributions import vmlss

    rng = np.random.default_rng(7)
    theta = np.mod(rng.vonmises(1.0, 2.0, size=60), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    m0 = CLRegression("theta ~ 1", df)
    assert m0.family is vmlss
    m1 = CLRegression("theta ~ 1", df, family=vmlss)
    m2 = CLRegression("theta ~ 1", df, family=CircularLL(vonmises))
    assert np.isclose(m0.result["log_likelihood"],
                      m1.result["log_likelihood"], rtol=1e-10)
    assert np.isclose(m1.result["log_likelihood"],
                      m2.result["log_likelihood"], rtol=1e-10)


def test_katojones_shape_inference():
    """Delta-method SEs/CIs for the KJ shape parameters through the disc
    chart (validation plan §3.4's deferred item): finite, ordered, inside
    the natural ranges, link-scale γ interval inside (0, 1), and the
    delta SEs consistent with Monte-Carlo coefficient-propagation."""
    from pycircstat2.distributions import katojones
    from pycircstat2.regression import KatoJonesLL
    from scipy.special import expit

    rng = np.random.default_rng(9)
    n = 150
    x = rng.uniform(-1.0, 1.0, n)
    mu_t = np.mod(1.0 + 0.6 * x, 2 * np.pi)
    a_t, b_t = katojones.disc_chart(0.45, 0.6, -0.4)
    rho_t = float(np.hypot(a_t, b_t))
    lam_t = float(np.mod(np.arctan2(b_t, a_t), 2 * np.pi))
    theta = np.array([
        float(katojones.rvs(mu_t[i], 0.45, rho_t, lam_t, random_state=rng))
        for i in range(n)
    ])
    df = pl.DataFrame({"theta": theta, "x": x})
    m = CLRegression(["theta ~ x", "~ 1", "~ 1", "~ 1"], df,
                     family=KatoJonesLL())

    si = m.shape_inference()  # all shape LPs intercept-only: data=None OK
    for name in ("gamma", "a", "b", "rho", "lam", "u1", "u2"):
        e = si[name]
        assert np.all(np.isfinite(e["estimate"]))
        assert np.all(e["se"] > 0)
        assert np.all(e["lo"] <= e["estimate"]) and np.all(
            e["estimate"] <= e["hi"])
    assert 0.0 < si["gamma"]["lo"][0] < si["gamma"]["hi"][0] < 1.0
    assert 0.0 <= si["rho"]["lo"][0] and si["rho"]["hi"][0] <= 1.0

    # delta SEs vs Monte-Carlo propagation of N(beta_hat, Vp)
    g = m.gam_fit
    beta = np.asarray(g.bhat.row(0), dtype=float)
    V = np.asarray(g.Vp, dtype=float)
    L = np.linalg.cholesky(V + 1e-12 * np.eye(V.shape[0]))
    X = np.asarray(
        g.predict(newdata=df[:1], type="lpmatrix"), dtype=float)
    lpi = [np.asarray(ix, dtype=int) for ix in g.lpi]
    draws = beta[None, :] + rng.standard_normal((8000, beta.size)) @ L.T
    g_d = expit(X[:, lpi[1]] @ draws[:, lpi[1]].T)
    a_d, b_d = katojones.disc_chart(
        g_d, X[:, lpi[2]] @ draws[:, lpi[2]].T,
        X[:, lpi[3]] @ draws[:, lpi[3]].T)
    for name, mc in (("gamma", g_d), ("a", a_d), ("b", b_d),
                     ("rho", np.hypot(a_d, b_d))):
        assert np.isclose(si[name]["se"][0], mc.std(), rtol=0.15)

    # non-KJ family refuses with guidance
    d2 = pl.DataFrame({
        "theta": np.mod(rng.vonmises(1.0, 3.0, 80), 2 * np.pi),
        "x": rng.uniform(-1, 1, 80),
    })
    m_vm = CLRegression(["theta ~ x", "~ 1"], d2, backend="gam")
    with pytest.raises(ValueError, match="KatoJones"):
        m_vm.shape_inference()


def test_circularll_vonmises_intercept_only_matches_mle():
    """gam(["theta ~ 1", "~ 1"], family=CircularLL(vonmises)) is the
    unpenalized von Mises MLE — pins the whole ll() derivative stack
    (Newton lands on the score root) and initialize_coef."""
    rng = np.random.default_rng(1)
    theta = np.mod(rng.vonmises(1.0, 3.0, 500), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    m = hea_gam(
        ["theta ~ 1", "~ 1"], data=df, family=CircularLL(vonmises), method="REML"
    )
    assert m.converged
    coef = np.asarray(m.coefficients, dtype=float)
    mu_mle, kappa_mle = vonmises.fit(theta)
    assert np.mod(2 * np.arctan(coef[0]), 2 * np.pi) == pytest.approx(mu_mle, abs=1e-4)
    assert np.exp(coef[1]) == pytest.approx(kappa_mle, rel=1e-3)


def test_circularll_wrapcauchy_intercept_only_matches_mle():
    """Same MLE-equivalence through the logit-linked wrapped Cauchy."""
    theta = np.asarray(wrapcauchy.rvs(2.2, 0.55, size=800, random_state=11))
    df = pl.DataFrame({"theta": theta})
    m = hea_gam(
        ["theta ~ 1", "~ 1"], data=df, family=CircularLL(wrapcauchy), method="REML"
    )
    assert m.converged
    coef = np.asarray(m.coefficients, dtype=float)
    mu_mle, rho_mle = wrapcauchy.fit(theta)
    assert np.mod(2 * np.arctan(coef[0]), 2 * np.pi) == pytest.approx(mu_mle, abs=1e-3)
    assert 1.0 / (1.0 + np.exp(-coef[1])) == pytest.approx(rho_mle, abs=1e-3)


def test_circularll_vonmises_recovers_smooth_mu_and_kappa():
    """The §5 target: smooth μ(x) AND log κ(z) by REML, jointly. True curves
    stay inside the tanhalf principal branch (the link cannot cross ±π)."""
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mu_true = np.pi / 2 + 1.2 * np.sin(2 * np.pi * x)
    kap_true = np.exp(0.8 + 1.2 * z)
    theta = np.mod(mu_true + rng.vonmises(0.0, kap_true, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x, "z": z})

    m = hea_gam(
        ["theta ~ s(x)", "~ s(z)"], data=df, family=CircularLL(vonmises),
        method="REML",
    )
    assert m.converged
    fv = np.asarray(m.fitted_values)
    circ_err = np.abs(np.angle(np.exp(1j * (fv[:, 0] - mu_true))))
    logk_err = np.abs(np.log(fv[:, 1]) - np.log(kap_true))
    assert circ_err.mean() < 0.08
    assert logk_err.mean() < 0.10


def test_circularll_projectednormal_fits_full_circle_sweep():
    """A full-circle μ(x) sweep — unrepresentable through tanhalf (pole at
    ±π) — fits cleanly through the projected normal's two identity LPs."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.uniform(0, 1, n)
    gamma = 2.0
    mu1, mu2 = gamma * np.cos(2 * np.pi * x), gamma * np.sin(2 * np.pi * x)
    theta = np.mod(
        np.arctan2(mu2 + rng.standard_normal(n), mu1 + rng.standard_normal(n)),
        2 * np.pi,
    )
    df = pl.DataFrame({"theta": theta, "x": x})
    m = hea_gam(
        ["theta ~ s(x)", "~ s(x)"], data=df,
        family=CircularLL(projectednormal), method="REML",
    )
    assert m.converged
    fv = np.asarray(m.fitted_values)
    dir_hat = np.arctan2(fv[:, 1], fv[:, 0])
    dir_true = np.arctan2(mu2, mu1)
    err = np.abs(np.angle(np.exp(1j * (dir_hat - dir_true))))
    assert err.mean() < 0.05


def test_circularll_postproc_binds_both_hea_conventions():
    """The seam moved under us once (hea unified postproc on mgcv's 6-arg
    hook) — pin that our signature binds BOTH calling conventions: hea
    0.1.4's positional ``(y, fitted)`` and the keyword form of the mgcv
    hook later hea uses. Only ``y`` enters the null refit, so the two
    must agree exactly."""
    rng = np.random.default_rng(5)
    n = 200
    theta = np.mod(rng.vonmises(1.0, 2.0, n), 2 * np.pi)
    fam = CircularLL(vonmises)
    fitted = np.column_stack([np.full(n, 1.0), np.full(n, 2.0)])
    old = fam.postproc(theta, fitted)  # hea 0.1.4 call shape
    new = fam.postproc(                # hea > 0.1.4 (mgcv 6-arg hook)
        theta, prior_weights=np.ones(n), fitted=fitted,
        linear_predictors=fitted, offset=None, intercept=True,
    )
    assert np.isfinite(old["null_deviance"])
    assert old["null_deviance"] == pytest.approx(new["null_deviance"])


def test_circularll_rejects_prior_weights():
    """gam(weights=) must fail loudly: no mgcv gamlss family uses prior
    weights in its ll, so a weighted circular fit would have no R
    reference to pin against — refusing beats silently fitting
    unweighted. Unit weights (hea's default when the caller passes
    nothing) must keep fitting."""
    rng = np.random.default_rng(9)
    n = 120
    theta = np.mod(rng.vonmises(1.0, 3.0, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    with pytest.raises(NotImplementedError, match="prior weights"):
        hea_gam(["theta ~ 1", "~ 1"], data=df, family=CircularLL(vonmises),
                method="REML", weights=np.full(n, 2.0))
    m = hea_gam(["theta ~ 1", "~ 1"], data=df, family=CircularLL(vonmises),
                method="REML")
    assert m.converged


def _cl_gam_sim(n=900, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mu = np.pi / 2 + 1.2 * np.sin(2 * np.pi * x)
    kap = np.exp(0.8 + 1.2 * z)
    theta = np.mod(mu + rng.vonmises(0.0, kap, n), 2 * np.pi)
    return pl.DataFrame({"theta": theta, "x": x, "z": z}), mu


def test_cl_gam_dispatch_and_surfaces():
    """A formula list routes CLRegression onto the CircularLL gam backend:
    smooth μ(x) recovered, result keys populated, predict/AIC/BIC work."""
    df, mu_true = _cl_gam_sim()
    m = CLRegression(["theta ~ s(x)", "~ s(z)"], df)
    assert m.backend == "gam"
    err = np.abs(np.angle(np.exp(1j * (m.result["mu"] - mu_true))))
    assert err.mean() < 0.1
    assert m.result["edf_total"] > 2
    assert np.isfinite(m.AIC()) and np.isfinite(m.BIC())
    new = pl.DataFrame({"x": [0.25], "z": [0.5]})
    assert 0.0 <= float(m.predict(new)[0]) < 2 * np.pi
    assert float(m.predict_kappa(new)[0]) > 0


def test_cl_gam_single_formula_implies_constant_kappa():
    df, _ = _cl_gam_sim()
    m = CLRegression("theta ~ s(x)", df)
    assert m.backend == "gam"
    assert m.formula == ["theta ~ s(x)", "~ 1"]
    k = m.result["kappa"]
    assert np.allclose(k, k[0])


def test_cl_gam_family_kwarg_and_fisher_lee_guard():
    """family= accepts a regression-ready distribution (auto-wrapped); gam
    options on the fisher-lee (array) path raise."""
    from pycircstat2.distributions import projectednormal

    df, _ = _cl_gam_sim()
    m = CLRegression("theta ~ s(x)", df, family=projectednormal)
    assert m.result["param_names"] == ["mu1", "mu2"]
    assert m.result["kappa"] is None
    with pytest.raises(ValueError, match="gam"):
        CLRegression(
            theta=np.array([0.1, 0.5, 1.0]),
            X=np.array([[0.0], [0.5], [1.0]]),
            family=projectednormal,
        )


def test_cl_backend_dispatch_unified_grammar():
    """One formula grammar, backend= selects the engine: default gam; smooth
    forces gam; fisher-lee expresses mean/kappa/mixed via the list; arrays and
    model_type= imply fisher-lee; the conflicting/invalid combinations raise."""
    df = load_data("B20", source="fisher")
    d = pl.DataFrame(
        {"X": df["x"].to_numpy(), "Z": df["x"].to_numpy() * 2.0,
         "θ": np.deg2rad(df["θ"].to_numpy())}
    )

    # default backend is gam
    assert CLRegression("θ ~ X", d).backend == "gam"
    assert CLRegression(["θ ~ X", "~ X"], d).backend == "gam"

    # fisher-lee: list configuration -> model_type
    assert CLRegression(["θ ~ X", "~ 1"], d, backend="fisher-lee").model_type == "mean"
    assert CLRegression(["θ ~ 1", "~ X"], d, backend="fisher-lee").model_type == "kappa"
    assert CLRegression(["θ ~ X", "~ X"], d, backend="fisher-lee").model_type == "mixed"
    # single string sugar -> mean model
    assert CLRegression("θ ~ X", d, backend="fisher-lee").model_type == "mean"

    # raises
    with pytest.raises(ValueError, match="shared design"):
        CLRegression(["θ ~ X", "~ Z"], d, backend="fisher-lee")
    with pytest.raises(ValueError, match="require backend='gam'"):
        CLRegression("θ ~ s(X)", d, backend="fisher-lee")
    with pytest.raises(ValueError, match="fisher-lee alias"):
        CLRegression("θ ~ X", d, model_type="mean", backend="gam")
    with pytest.raises(ValueError, match="arrays use backend"):
        CLRegression(theta=d["θ"].to_numpy(), X=d["X"].to_numpy()[:, None], backend="gam")


def test_cl_fisher_lee_list_matches_legacy_model_type():
    """The fisher-lee formula list and the legacy model_type= alias are the
    same fit (the list is just the new spelling)."""
    df = load_data("B20", source="fisher")
    d = pl.DataFrame({"X": df["x"].to_numpy(), "θ": np.deg2rad(df["θ"].to_numpy())})

    for mt, formulas in [
        ("mean", ["θ ~ X", "~ 1"]),
        ("kappa", ["θ ~ 1", "~ X"]),
        ("mixed", ["θ ~ X", "~ X"]),
    ]:
        legacy = CLRegression("θ ~ X", d, model_type=mt, tol=1e-10)
        unified = CLRegression(formulas, d, backend="fisher-lee", tol=1e-10)
        assert unified.model_type == mt
        np.testing.assert_allclose(unified.result["mu"], legacy.result["mu"], atol=1e-8)
        np.testing.assert_allclose(
            np.atleast_1d(unified.result["kappa"]),
            np.atleast_1d(legacy.result["kappa"]), rtol=1e-6,
        )


def test_cl_gam_plot_draws_both_bands_with_toggles():
    """The gam overlay draws a CI band (μ̂ ± z·se, like LC) and a κ-implied
    ±1 circ-SD dispersion band (like CC), each independently toggleable;
    projected normal (derived direction, no κ role) draws neither, no crash."""
    import matplotlib
    matplotlib.use("Agg")
    from pycircstat2.distributions import projectednormal

    df, _ = _cl_gam_sim(n=400)
    m = CLRegression("theta ~ s(x)", df)

    def _labels(fig):
        return {t.get_text() for t in fig.axes[0].get_legend().get_texts()}

    assert {"95% CI", "±1 circ-SD"} <= _labels(m.plot())
    assert "95% CI" not in _labels(m.plot(ci=False))
    assert "±1 circ-SD" not in _labels(m.plot(pi=False))
    assert _labels(m.plot(level=0.80)) >= {"80% CI"}

    mp = CLRegression("theta ~ s(x)", df, family=projectednormal)
    assert _labels(mp.plot()) == {"fit", "data"}  # no bands, but renders


@pytest.mark.parametrize("model_type", ["mean", "kappa", "mixed"])
def test_cl_parametric_plot_draws_both_bands(model_type):
    """The parametric overlays (mean/kappa/mixed) carry the same CI +
    dispersion bands as the gam backend, each toggleable."""
    import matplotlib
    matplotlib.use("Agg")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 1))
    theta = np.mod(0.7 + 2 * np.arctan(0.9 * X[:, 0]) + rng.vonmises(0, 5.0, 120), 2 * np.pi)
    m = CLRegression(theta=theta, X=X, model_type=model_type, tol=1e-8, max_iter=300)

    def _labels(fig):
        return {t.get_text() for t in fig.axes[0].get_legend().get_texts()}

    assert {"95% CI", "±1 circ-SD"} <= _labels(m.plot())
    assert "95% CI" not in _labels(m.plot(ci=False))
    assert "±1 circ-SD" not in _labels(m.plot(pi=False))


def test_cl_predict_accepts_array_and_dataframe_both_backends():
    """predict()/predict_kappa() take either a design array or a DataFrame on
    both backends, with matching results (the cross-backend comparison surface)."""
    df = load_data("B20", source="fisher")
    d = pl.DataFrame({"X": df["x"].to_numpy(), "θ": np.deg2rad(df["θ"].to_numpy())})
    grid = np.linspace(0.0, d["X"].max(), 7)
    g_arr, g_df = grid[:, None], pl.DataFrame({"X": grid})

    for backend in ("fisher-lee", "gam"):
        m = CLRegression(["θ ~ X", "~ X"], d, backend=backend)
        np.testing.assert_allclose(m.predict(g_arr), m.predict(g_df), atol=1e-9)
        np.testing.assert_allclose(m.predict_kappa(g_arr), m.predict_kappa(g_df), atol=1e-9)

    # gam: wrong column count is a clear error
    m = CLRegression(["θ ~ X", "~ X"], d, backend="gam")
    with pytest.raises(ValueError, match="predictor column"):
        m.predict(np.zeros((5, 2)))


# --- circ_gam / circ_lm front doors (dev/plans/circ_gam_unified_api.md) ----

from pycircstat2.distributions import vmlss  # noqa: E402
from pycircstat2.regression import circ_gam, circ_lm  # noqa: E402


def test_circ_gam_b2_twin_and_knot_defaults():
    """The circlss §4 twin call fits through circ_gam with family/method
    defaulted; with knots omitted, cyclic boundaries default to the full
    period; a single formula auto-expands to a constant second LP."""
    rng = np.random.default_rng(8)
    n = 400
    phi = rng.uniform(-np.pi, np.pi, n)
    mu_true = np.pi / 2 + 1.2 * np.sin(phi)
    theta = np.mod(mu_true + rng.vonmises(0.0, 4.0, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "phi": phi})

    b2 = circ_gam(["theta ~ s(phi, bs='cc')", "~ s(phi, bs='cc')"], df,
                  knots={"phi": [-np.pi, np.pi]})
    assert b2.converged
    fv = np.asarray(b2.fitted_values)
    err = np.abs(np.angle(np.exp(1j * (np.mod(fv[:, 0], 2 * np.pi) - mu_true))))
    assert err.mean() < 0.15

    df2 = pl.DataFrame({"theta": theta, "phi": np.mod(phi, 2 * np.pi)})
    m = circ_gam("theta ~ s(phi, bs='cc')", df2)
    assert m.converged


def test_circ_gam_matches_clregression_gam_backend():
    """Phase-1 DoD pin: circ_gam reproduces CLRegression's gam backend
    exactly while both entries exist (same engine call underneath)."""
    df, _ = _cl_gam_sim(n=500, seed=3)
    g = circ_gam(["theta ~ s(x)", "~ s(z)"], df)
    m = CLRegression(["theta ~ s(x)", "~ s(z)"], df)
    assert float(g.logLik) == pytest.approx(
        m.result["log_likelihood"], rel=1e-10
    )
    got = dict(zip(g.bhat.columns, g.bhat.row(0)))
    assert got == pytest.approx(m.result["coefficients"], rel=1e-10)


def test_circ_gam_family_resolution():
    """Strings resolve through the circular catalog (alias or distribution
    name); unknown names raise with guidance."""
    df, _ = _cl_gam_sim(n=300, seed=11)
    a = circ_gam(["theta ~ s(x)", "~ 1"], df, family="vonmises")
    b = circ_gam(["theta ~ s(x)", "~ 1"], df, family=vmlss)
    assert float(a.logLik) == pytest.approx(float(b.logLik), rel=1e-10)
    with pytest.raises(ValueError, match="unknown family"):
        circ_gam("theta ~ s(x)", df, family="nope")


def test_circ_gam_gaussian_passthrough():
    """No gatekeeping: a linear response rides through to hea untouched
    (the old LC-smooth case), with the period-knot default still applied."""
    import hea.family as hea_family

    rng = np.random.default_rng(21)
    n = 300
    phi = rng.uniform(0, 2 * np.pi, n)
    y = 2.0 + np.sin(phi) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"y": y, "phi": phi})
    g = circ_gam("y ~ s(phi, bs='cc')", df, family="gaussian")
    direct = hea_gam("y ~ s(phi, bs='cc')", df, family=hea_family.gaussian,
                     knots={"phi": [0.0, 2 * np.pi]}, method="REML")
    assert float(g.AIC) == pytest.approx(float(direct.AIC), rel=1e-12)


def test_circ_gam_k3_k4_families_post_gate():
    """jplss (3-LP) and kjlss (4-LP) ride hea's efsud K=3/4 paths — now
    R-pinned hea-side (twlss/shash landed) — through circ_gam by name."""
    rng = np.random.default_rng(5)
    theta = np.mod(rng.vonmises(1.0, 3.0, 80), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    j = circ_gam(["theta ~ 1", "~ 1", "~ 1"], df, family="jplss")
    assert np.isfinite(float(j.logLik))
    k = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family="kjlss")
    assert np.isfinite(float(k.logLik))


def test_circ_gam_closed_form_null_start_avoids_indefinite_hessian():
    """Closed-form null start in ``CircularLL._null_params`` (the circlss
    ``initialize`` convention): location = mean direction, concentration =
    the Rbar-based estimator, every shape/skewness parameter = 0 (the
    von-Mises/symmetric reduction member).

    The marginal joint MLE start it replaces (``dist.fit``) pushes a shape
    parameter to an extreme on covariate-driven-location data: the pooled 2nd
    moment of angles with a swinging mean inflates toward the boundary, so
    ssjplss gets psi-hat ~ -4 (vs the data-generating 0.5). That start makes
    ``gam.fit5``'s penalized Hessian indefinite ("indefinite penalized
    likelihood"); the neutral shape=0 start stays in the well-conditioned
    basin. kjlss is the same failure mode via the disc-chart blow-up
    |u| -> ~1e4 (now bounded at 8). See dev/plans/pycircstat2-divergences.md.
    """
    from pycircstat2 import distributions as D

    # (a) the start *is* the closed form: shape/skew exactly 0, concentration
    #     from Rbar via the per-distribution hook, location = mean direction.
    rng = np.random.default_rng(3)
    y = np.mod(rng.vonmises(1.0, 2.5, 300), 2 * np.pi)
    sy, cy = float(np.mean(np.sin(y))), float(np.mean(np.cos(y)))
    Rbar = float(np.hypot(sy, cy))
    np.testing.assert_allclose(
        D.ssjplss._null_params(y),
        [np.arctan2(sy, cy), D.ssjplss.dist._concentration_start(Rbar), 0.0, 0.0],
        atol=1e-12,
    )
    # von Mises concentration start is exactly Fisher's A1-inverse, clamped
    assert D.vmlss._null_params(y)[1] == pytest.approx(
        float(np.clip(A1inv(Rbar), 0.01, 500.0)), abs=1e-12)
    # the 2-component projected normal has no concentration hook -> MLE fallback
    assert len(D.pnlss._null_params(y)) == 2

    # (b) ssjplss with a covariate-driven location: the marginal-MLE start
    #     raised FloatingPointError("indefinite penalized likelihood") here;
    #     the closed-form start converges to a finite fit.
    rng = np.random.default_rng(11)
    n = 150
    x = np.sort(rng.uniform(0.0, 1.0, n))
    mu = np.mod(2.0 * np.arctan(2.0 * np.sin(2 * np.pi * x)), 2 * np.pi)
    th = np.mod(np.array([
        float(D.jonespewsey_sineskewed.rvs(
            xi=float(m), kappa=2.0, psi=0.5, lmbd=0.6, size=1, random_state=rng)[0])
        for m in mu]), 2 * np.pi)
    df = pl.DataFrame({"theta": th, "x": x})
    g = circ_gam(["theta ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family="ssjplss")
    assert np.isfinite(float(g.logLik)) and g.converged

    # (c) kjlss with a covariate-driven location: the uncapped disc-chart
    #     inverse handed gam.fit5 a |u| ~ 1e4 start and crashed; the |u| <= 8
    #     cap keeps it finite.
    rng = np.random.default_rng(7)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    mu = np.mod(2 * np.pi * x, 2 * np.pi)
    th = np.mod(np.array([
        float(D.katojones.rvs(mu=float(m), gamma=0.4, rho=0.3, lam=0.5,
                              size=1, random_state=rng)[0])
        for m in mu]), 2 * np.pi)
    df = pl.DataFrame({"theta": th, "x": x})
    k = circ_gam(["theta ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family="kjlss")
    assert np.isfinite(float(k.logLik))


def test_circ_gam_cartlss_location_warm_start_recovers_wiggly_mu():
    """Role-aware location warm start in ``CircularLL.initialize_coef``.

    Cartwright's density is exactly 0 at the antipode for every ζ, so a
    flat-μ start strands antipodal observations on the log-likelihood cliffs
    and EFS collapses μ to a constant (a coupled bad basin: μ̂ flat, ρ̂ → 0).
    The projected pilot starts the location LP near the data and escapes it.
    On wiggly-μ, moderate-ζ data (ρ ≈ 0.67) the flat start gave mean angular
    error ≈ 1.0 rad — no recovery; the pilot recovers μ to < 0.1 rad with the
    fitted location's circular spread matching the truth's (not collapsed).
    """
    from pycircstat2.distributions import cartwright

    rng = np.random.default_rng(4)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.mod(2.0 * np.arctan(np.sin(2.0 * np.pi * x)), 2 * np.pi)
    theta = np.mod(np.array([
        float(cartwright.rvs(mu=float(m), zeta=0.5, size=1, random_state=rng)[0])
        for m in mu_true
    ]), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x})

    g = circ_gam(["theta ~ s(x)", "~ 1"], df, family="cartlss")
    assert g.converged
    mu_fit = np.mod(np.asarray(g.fitted_values)[:, 0], 2 * np.pi)
    err = np.abs(np.angle(np.exp(1j * (mu_fit - mu_true))))
    assert err.mean() < 0.3                      # ≈ 1.0 without the pilot
    # fitted μ is not collapsed to a constant: its circular dispersion
    # (1 − R̄) tracks the truth's (≈ 0.58) instead of falling toward 0
    fit_disp = 1.0 - np.abs(np.mean(np.exp(1j * mu_fit)))
    assert fit_disp > 0.4


def test_circ_lm_modes_and_equivalence():
    """circ_lm is a pure dispatcher: every spelling (plain and R-style
    hyphenated) reaches the right class and reproduces its fit exactly."""
    X, theta, _, _, _ = _simulate_cl()
    lung = _lung_dataframe(drop_feb_outliers=True)
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 2 * np.pi, 60)
    th = np.mod(x + 0.4 * np.sin(x) + rng.vonmises(0.0, 5.0, 60), 2 * np.pi)

    ref_cl = CLRegression(theta=theta, X=X, model_type="mean")
    for mode in ("cl", "c-l"):
        m = circ_lm(mode, theta=theta, X=X, model_type="mean")
        assert isinstance(m, CLRegression)
        np.testing.assert_allclose(m.result["beta"], ref_cl.result["beta"],
                                   atol=1e-12)
    ref_cc = CCRegression(theta=th, x=x, order=2)
    for mode in ("cc", "c-c"):
        m = circ_lm(mode, theta=th, x=x, order=2)
        assert isinstance(m, CCRegression)
        np.testing.assert_allclose(m.result["fitted"], ref_cc.result["fitted"],
                                   atol=1e-12)
    ref_lc = LCRegression("y ~ harmonic(theta, k=2)", lung)
    for mode in ("lc", "l-c"):
        m = circ_lm(mode, "y ~ harmonic(theta, k=2)", lung)
        assert isinstance(m, LCRegression)
        np.testing.assert_allclose(
            list(m.result["coefficients"].values()),
            list(ref_lc.result["coefficients"].values()), atol=1e-12)
    with pytest.raises(ValueError, match="mode must be"):
        circ_lm("xy")


def test_circ_gam_cyclic_summary_general_family(capsys):
    """summary() on a fully-penalized smooth (bs='cc' → penalty null space
    0 → hea's reTest/_recov path) under a general family. Crashed with
    AttributeError('_fisher_w') before hea@97a244b; fixed by consuming the
    stored gam.fit5.post.proc R factor (R'R = −lbb, mgcv's object$R) in
    _recov — see hea/.claude/plans/fit5-recov-summary-fix.md."""
    rng = np.random.default_rng(8)
    n = 150
    phi = rng.uniform(0, 2 * np.pi, n)
    theta = np.mod(np.pi / 2 + np.sin(phi) + rng.vonmises(0.0, 4.0, n),
                   2 * np.pi)
    df = pl.DataFrame({"theta": theta, "phi": phi})
    m = circ_gam("theta ~ s(phi, bs='cc')", df)
    assert m.converged
    m.summary()
    out = capsys.readouterr().out
    assert "Approximate significance of smooth terms:" in out
