import numpy as np
import pytest

from pycircstat2 import Circular, load_data
from pycircstat2.distributions import vonmises
from pycircstat2.hypothesis import (
    V_test,
    batschelet_test,
    binomial_test,
    change_point_test,
    chisquare_test,
    circ_anova_test,
    circ_range_test,
    concentration_test,
    harrison_kanji_test,
    kuiper_test,
    omnibus_test,
    one_sample_test,
    rao_homogeneity_test,
    rao_spacing_test,
    rayleigh_test,
    symmetry_test,
    wallraff_test,
    watson_test,
    watson_u2_test,
    watson_williams_test,
    wheeler_watson_test,
)


def test_rayleigh_test():

    # Ch27 Example 1 (Zar, 2010, P667)
    # Using data from Ch26 Example 2.
    data_zar_ex2_ch26 = load_data("D1", source="zar")
    circ_zar_ex1_ch27 = Circular(data=data_zar_ex2_ch26["θ"].values)

    # computed directly from r and n
    result = rayleigh_test(n=circ_zar_ex1_ch27.n, r=circ_zar_ex1_ch27.r)
    np.testing.assert_approx_equal(result.z, 5.448, significant=3)
    assert 0.001 < result.pval < 0.002

    # computed directly from alpha
    result = rayleigh_test(alpha=circ_zar_ex1_ch27.alpha)
    np.testing.assert_approx_equal(result.z, 5.448, significant=3)
    assert 0.001 < result.pval < 0.002


def test_V_test():

    # Ch27 Example 2 (Zar, 2010, P669)
    data_zar_ex2_ch27 = load_data("D7", source="zar")
    circ_zar_ex2_ch27 = Circular(data=data_zar_ex2_ch27["θ"].values)

    # computed directly from r and n
    V, u, p = V_test(
        angle=np.deg2rad(90),
        mean=circ_zar_ex2_ch27.mean,
        n=circ_zar_ex2_ch27.n,
        r=circ_zar_ex2_ch27.r,
    )

    np.testing.assert_approx_equal(V, 9.498, significant=3)
    np.testing.assert_approx_equal(u, 4.248, significant=3)
    assert p < 0.0005

    # computed directly from alpha
    V, u, p = V_test(
        alpha=circ_zar_ex2_ch27.alpha,
        angle=np.deg2rad(90),
    )

    np.testing.assert_approx_equal(V, 9.498, significant=3)
    np.testing.assert_approx_equal(u, 4.248, significant=3)
    assert p < 0.0005


def test_one_sample_test():

    # Ch27 Example 3 (Zar, 2010, P669)
    # Using data from Ch27 Example 2
    data_zar_ex2_ch27 = load_data("D7", source="zar")
    circ_zar_ex3_ch27 = Circular(data=data_zar_ex2_ch27["θ"].values, unit="degree")

    # computed directly from lb and ub
    reject_null = one_sample_test(
        lb=circ_zar_ex3_ch27.mean_lb,
        ub=circ_zar_ex3_ch27.mean_ub,
        angle=np.deg2rad(90),
    )

    assert reject_null is False

    # computed directly from alpha
    reject_null = one_sample_test(alpha=circ_zar_ex3_ch27.alpha, angle=np.deg2rad(90))

    assert reject_null is False


def test_omnibus_test():

    data_zar_ex4_ch27 = load_data("D8", source="zar")
    circ_zar_ex4_ch27 = Circular(data=data_zar_ex4_ch27["θ"].values, unit="degree")

    A, pval = omnibus_test(alpha=circ_zar_ex4_ch27.alpha, scale=1)

    np.testing.assert_approx_equal(pval, 0.0043, significant=2)


def test_batschelet_test():

    data_zar_ex5_ch27 = load_data("D8", source="zar")
    circ_zar_ex5_ch27 = Circular(data=data_zar_ex5_ch27["θ"].values, unit="degree")

    C, pval = batschelet_test(
        angle=np.deg2rad(45),
        alpha=circ_zar_ex5_ch27.alpha,
    )
    np.testing.assert_equal(C, 5)
    np.testing.assert_approx_equal(pval, 0.00661, significant=3)


def test_chisquare_test():

    d2 = load_data("D2", source="zar")
    c2 = Circular(data=d2["θ"].values, w=d2["w"].values)

    result = chisquare_test(c2.w)
    np.testing.assert_approx_equal(result.chi2, 66.543, significant=3)
    assert result.pval < 0.001


def test_symmetry_test():

    data_zar_ex6_ch27 = load_data("D9", source="zar")
    circ_zar_ex6_ch27 = Circular(data=data_zar_ex6_ch27["θ"].values, unit="degree")

    d, p = symmetry_test(median=circ_zar_ex6_ch27.median, alpha=circ_zar_ex6_ch27.alpha)
    assert p > 0.5


def test_watson_williams_test():

    data = load_data("D10", source="zar")
    s1 = Circular(data=data[data["sample"] == 1]["θ"].values)
    s2 = Circular(data=data[data["sample"] == 2]["θ"].values)
    F, pval = watson_williams_test(circs=[s1, s2])

    np.testing.assert_approx_equal(F, 1.61, significant=3)
    np.testing.assert_approx_equal(pval, 0.22, significant=2)

    data = load_data("D11", source="zar")
    s1 = Circular(data=data[data["sample"] == 1]["θ"].values)
    s2 = Circular(data=data[data["sample"] == 2]["θ"].values)
    s3 = Circular(data=data[data["sample"] == 3]["θ"].values)

    F, pval = watson_williams_test(circs=[s1, s2, s3])

    np.testing.assert_approx_equal(F, 1.86, significant=3)
    np.testing.assert_approx_equal(pval, 0.19, significant=2)


def test_watson_u2_test():

    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values)
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values)
    U2, pval = watson_u2_test(circs=[c0, c1])

    np.testing.assert_approx_equal(U2, 0.1458, significant=3)
    assert 0.1 < pval < 0.2

    d = load_data("D13", source="zar")
    c0 = Circular(
        data=d[d["sample"] == 1]["θ"].values, w=d[d["sample"] == 1]["w"].values
    )
    c1 = Circular(
        data=d[d["sample"] == 2]["θ"].values, w=d[d["sample"] == 2]["w"].values
    )
    U2, pval = watson_u2_test(circs=[c0, c1])

    np.testing.assert_approx_equal(U2, 0.0612, significant=3)
    assert pval > 0.5


def test_wheeler_watson_test():
    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values)
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values)

    W, pval = wheeler_watson_test(circs=[c0, c1])
    np.testing.assert_approx_equal(W, 3.678, significant=3)
    assert 0.1 < pval < 0.25


def test_wallraff_test():

    d = load_data("D14", source="zar")
    c0 = Circular(data=d[d["sex"] == "male"]["θ"].values)
    c1 = Circular(data=d[d["sex"] == "female"]["θ"].values)
    U, pval = wallraff_test(angle=np.deg2rad(135), circs=[c0, c1])
    np.testing.assert_approx_equal(U, 18.5, significant=3)
    assert pval > 0.20

    from pycircstat2.utils import time2float

    d = load_data("D15", source="zar")
    c0 = Circular(data=time2float(d[d["sex"] == "male"]["time"].values))
    c1 = Circular(data=time2float(d[d["sex"] == "female"]["time"].values))
    U, pval = wallraff_test(
        angle=np.deg2rad(time2float(["7:55", "8:15"])),
        circs=[c0, c1],
        verbose=True,
    )
    np.testing.assert_equal(U, 13)
    assert pval > 0.05


def test_kuiper_test():

    d = load_data("B5", source="fisher")["θ"].values
    c = Circular(data=d, unit="degree", n_intervals=180)
    V, pval = kuiper_test(alpha=c.alpha)
    np.testing.assert_approx_equal(V, 1.5864, significant=3)
    assert pval > 0.05


def test_watson_test():

    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c_pigeon = Circular(data=pigeon)
    U2, pval = watson_test(alpha=c_pigeon.alpha, n_simulation=9999)
    np.testing.assert_approx_equal(U2, 0.137, significant=3)
    assert pval > 0.10


def test_rao_spacing_test():
    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c_pigeon = Circular(data=pigeon)
    U, pval = rao_spacing_test(alpha=c_pigeon.alpha, n_simulation=9999)
    np.testing.assert_approx_equal(U, 161.92308, significant=3)
    assert 0.05 < pval < 0.10

def test_circ_range_test():


    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 72.0, 108.0, 108.0, 169.2, 324.0])
    range_stat, pval = circ_range_test(x)
    np.testing.assert_approx_equal(range_stat, 4.584073, significant=5)
    np.testing.assert_approx_equal(pval, 0.01701148, significant=5)


def test_binomial_test_uniform():
    """Test binomial_test with uniform circular data (should not reject H0)."""
    np.random.seed(42)
    alpha = np.random.uniform(0, 2 * np.pi, 100)  # Uniformly distributed angles
    md = np.pi  # Test median at π (should be non-significant)
    
    pval = binomial_test(alpha, md)
    
    assert 0.05 < pval < 1.0, f"Unexpected p-value for uniform data: {pval}"

def test_binomial_test_skewed():
    """Test binomial_test with a skewed circular distribution (should reject H0)."""
    np.random.seed(42)
    alpha = np.random.vonmises(mu=np.pi/4, kappa=3, size=100)  # Clustered around π/4
    md = np.pi  # Incorrect median hypothesis
    
    pval = binomial_test(alpha, md)
    
    assert pval < 0.05, f"Expected significant p-value but got {pval}"

def test_binomial_test_symmetric():
    """Test binomial_test with symmetric distribution around π (should fail to reject H0)."""
    alpha = np.array([-np.pi/4, np.pi/4, np.pi/2, -np.pi/2, np.pi])
    md = np.pi  # This should be a valid median
    
    pval = binomial_test(alpha, md)
    
    assert pval > 0.05, f"Unexpected p-value for symmetric data: {pval}"

def test_binomial_test_extreme_case():
    """Test binomial_test with all points clustered at π (extreme case)."""
    alpha = np.full(20, np.pi)  # All angles at π
    md = np.pi
    
    pval = binomial_test(alpha, md)
    
    assert np.isclose(pval, 1.0), f"Expected p-value of 1 for identical data but got {pval}"



def test_concentration_identical():
    """Test concentration_test with identical von Mises distributions (should fail to reject H0)."""
    np.random.seed(42)
    alpha1 = vonmises.rvs(mu=0, kappa=3, size=50)
    alpha2 = vonmises.rvs(mu=0, kappa=3, size=50)

    f_stat, pval = concentration_test(alpha1, alpha2)

    assert pval > 0.05, f"Unexpectedly small p-value: {pval}, should not reject H0."

def test_concentration_different():
    """Test concentration_test with different kappa values (should reject H0)."""
    np.random.seed(42)
    alpha1 = vonmises.rvs(mu=0, kappa=3, size=50)  # Higher concentration
    alpha2 = vonmises.rvs(mu=0, kappa=1, size=50)  # Lower concentration

    f_stat, pval = concentration_test(alpha1, alpha2)

    assert pval < 0.05, f"Expected small p-value, but got {pval}"

def test_concentration_high_dispersion():
    """Test concentration_test with very dispersed data (should fail to reject H0)."""
    np.random.seed(42)
    alpha1 = np.random.uniform(0, 2*np.pi, 50)  # Uniformly spread
    alpha2 = np.random.uniform(0, 2*np.pi, 50)

    f_stat, pval = concentration_test(alpha1, alpha2)

    assert pval > 0.05, f"Unexpectedly small p-value: {pval}, should not reject H0."


def test_concentration_extreme_case():
    """Test concentration_test when both samples have extremely high concentration (should fail to reject H0)."""
    np.random.seed(42)
    alpha1 = vonmises.rvs(mu=0, kappa=100, size=50)
    alpha2 = vonmises.rvs(mu=0, kappa=100, size=50)

    f_stat, pval = concentration_test(alpha1, alpha2)

    assert pval > 0.05, f"Unexpectedly small p-value: {pval}, should not reject H0."



def test_rao_homogeneity_identical():
    """Test with identical von Mises distributions (should fail to reject H0)."""
    np.random.seed(42)
    samples = [vonmises.rvs(mu=0, kappa=2, size=50) for _ in range(3)]

    results = rao_homogeneity_test(samples)

    assert results["pval_polar"] > 0.05, f"Unexpectedly small p-value: {results['pval_polar']}"
    assert results["pval_disp"] > 0.05, f"Unexpectedly small p-value: {results['pval_disp']}"

def test_rao_homogeneity_different_means():
    """Test with different mean directions (should reject H0 for mean equality)."""
    np.random.seed(42)
    samples = [
        vonmises.rvs(kappa=2, mu=0, size=50),
        vonmises.rvs(kappa=2, mu=np.pi/4, size=50),
        vonmises.rvs(kappa=2, mu=np.pi/2, size=50)
    ]
    results = rao_homogeneity_test(samples)

    assert results["pval_polar"] < 0.05, f"Expected rejection but got p={results['pval_polar']}"

def test_rao_homogeneity_different_dispersion():
    """Test with different kappa values (should reject H0 for dispersion equality)."""
    np.random.seed(42)
    samples = [
        vonmises.rvs(mu=0, kappa=5, size=50),
        vonmises.rvs(mu=0, kappa=2, size=50),
        vonmises.rvs(mu=0, kappa=1, size=50)
    ]

    results = rao_homogeneity_test(samples)

    assert results["pval_disp"] < 0.05, f"Expected rejection but got p={results['pval_disp']}"

def test_rao_homogeneity_small_samples():
    """Test with very small sample sizes (should handle without error)."""
    np.random.seed(42)
    samples = [vonmises.rvs(mu=0, kappa=3, size=5) for _ in range(3)]

    results = rao_homogeneity_test(samples)

    assert "pval_polar" in results and "pval_disp" in results

def test_rao_homogeneity_invalid_input():
    """Test invalid input (should raise ValueError)."""
    with pytest.raises(ValueError):
        rao_homogeneity_test("invalid_input")

    with pytest.raises(ValueError):
        rao_homogeneity_test([np.array([0, np.pi/2]), "invalid_array"])

def test_change_point_basic():
    """Test change_point_test() on a simple dataset matching R."""
    alpha = np.array([3.03, 0.28, 3.90, 5.56, 5.77, 5.06, 5.96, 
                      0.16, 0.51, 1.21, 6.03, 1.05, 0.45, 1.47, 6.09])
    
    result = change_point_test(alpha)

    # Expected values based on R output
    expected_rho = 0.52307
    expected_rmax = 2.237654
    expected_k_r = 6
    expected_rave = 1.066862
    expected_tmax = 0.602549
    expected_k_t = 6
    expected_tave = 0.460675

    assert np.isclose(result["rho"].iloc[0], expected_rho, atol=1e-5)
    assert np.isclose(result["rmax"].iloc[0], expected_rmax, atol=1e-5)
    assert result["k.r"].iloc[0] == expected_k_r
    assert np.isclose(result["rave"].iloc[0], expected_rave, atol=1e-5)
    assert np.isclose(result["tmax"].iloc[0], expected_tmax, atol=1e-5)
    assert result["k.t"].iloc[0] == expected_k_t
    assert np.isclose(result["tave"].iloc[0], expected_tave, atol=1e-5)

def test_harrison_kanji_test():
    """Test Harrison-Kanji two-way ANOVA for circular data."""
    np.random.seed(42)
    alpha = np.random.vonmises(0, 2, 50)
    idp = np.random.choice([1, 2, 3], 50)
    idq = np.random.choice([1, 2], 50)

    pval, anova_table = harrison_kanji_test(alpha, idp, idq)

    assert len(pval) == 3  # Should return three p-values
    assert anova_table.shape[0] >= 3  # At least 3 sources in ANOVA table
    assert all(0 <= p <= 1 for p in pval if p is not None)  # Valid p-values

@pytest.mark.skip(reason="Skipped unless explicitly called with `-k test_harrison_kanji_vs_pycircstat`")
def test_harrison_kanji_vs_pycircstat():
    """Compare PyCircStat2 `harrison_kanji_test` with original PyCircStat `hktest`."""
    def hktest(alpha, idp, idq, inter=True, fn=None):
        """copied and fixed from pycircstat.hktest"""
        import pandas as pd
        from scipy import special, stats

        from pycircstat2.descriptive import circ_kappa, circ_mean, circ_r

        if fn is None:
            fn = ['A', 'B']
        p = len(np.unique(idp))
        q = len(np.unique(idq))
        df = pd.DataFrame({fn[0]: idp, fn[1]: idq, 'dependent': alpha})
        n = len(df)
        tr = n * circ_r(df['dependent'])
        kk = circ_kappa(tr / n)

        # both factors
        gr = df.groupby(fn)
        cn = gr.count()
        cr = gr.agg(circ_r) * cn
        cn = cn.unstack(fn[1])
        cr = cr.unstack(fn[1])

        # factor A
        gr = df.groupby(fn[0])
        pn = gr.count()['dependent']
        pr = gr.agg(circ_r)['dependent'] * pn
        pm = gr.agg(circ_mean)['dependent']
        # factor B
        gr = df.groupby(fn[1])
        qn = gr.count()['dependent']
        qr = gr.agg(circ_r)['dependent'] * qn
        qm = gr.agg(circ_mean)['dependent']

        if kk > 2:  # large kappa
            # effect of factor 1
            eff_1 = sum(pr ** 2 / cn.sum(axis=1)) - tr ** 2 / n
            df_1 = p - 1
            ms_1 = eff_1 / df_1

            # effect of factor 2
            eff_2 = sum(qr ** 2. / cn.sum(axis=0)) - tr ** 2 / n
            df_2 = q - 1
            ms_2 = eff_2 / df_2

            # total effect
            eff_t = n - tr ** 2 / n
            df_t = n - 1
            m = cn.values.mean()

            if inter:
                # correction factor for improved F statistic
                beta = 1 / (1 - 1 / (5 * kk) - 1 / (10 * (kk ** 2)))
                # residual effects
                eff_r = n - (cr**2./cn).values.sum()
                df_r = p*q*(m-1)
                ms_r = eff_r / df_r

                # interaction effects
                eff_i = (cr**2./cn).values.sum() - sum(qr**2./qn) - sum(pr**2./pn) + tr**2/n
                df_i = (p-1)*(q-1)
                ms_i = eff_i/df_i;

                # interaction test statistic
                FI = ms_i / ms_r
                pI = 1 - stats.f.cdf(FI,df_i,df_r)
            else:
                # residual effect
                eff_r = n - sum(qr**2./qn)- sum(pr**2./pn) + tr**2/n
                df_r = (p-1)*(q-1)
                ms_r = eff_r / df_r

                # interaction effects
                eff_i = None
                df_i = None
                ms_i = None

                # interaction test statistic
                FI = None
                pI = np.nan
                beta = 1


            F1 = beta * ms_1 / ms_r
            p1 = 1 - stats.f.cdf(F1,df_1,df_r)

            F2 = beta * ms_2 / ms_r
            p2 = 1 - stats.f.cdf(F2,df_2,df_r)

        else: #small kappa
            # correction factor
            # special.iv is Modified Bessel function of the first kind of real order
            rr = special.iv(1,kk) / special.iv(0,kk)
            f = 2/(1-rr**2)

            chi1 = f * (sum(pr**2./pn)- tr**2/n)
            df_1 = 2*(p-1)
            p1 = 1 - stats.chi2.cdf(chi1, df=df_1)

            chi2 = f * (sum(qr**2./qn)- tr**2/n)
            df_2 = 2*(q-1)
            p2 = 1 - stats.chi2.cdf(chi2, df=df_2)

            chiI = f * ( (cr**2./cn).values.sum() - sum(pr**2./pn) - sum(qr**2./qn) + tr**2/n)
            df_i = (p-1) * (q-1)
            pI = stats.chi2.sf(chiI, df=df_i)



        pval = (p1.squeeze(), p2.squeeze(), pI.squeeze())

        if kk>2:
            table = pd.DataFrame({
                'Source': fn + ['Interaction', 'Residual', 'Total'],
                'DoF': [df_1, df_2, df_i, df_r, df_t],
                'SS': [eff_1, eff_2, eff_i, eff_r, eff_t],
                'MS': [ms_1, ms_2, ms_i, ms_r, np.nan],
                'F': [F1.squeeze(), F2.squeeze(), FI, np.nan, np.nan],
                'p': list(pval) + [np.nan, np.nan]
            })
            table = table.set_index('Source')
        else:
            table = pd.DataFrame({
                'Source': fn + ['Interaction'],
                'DoF': [df_1, df_2, df_i],
                'chi2': [chi1.squeeze(), chi2.squeeze(), chiI.squeeze()],
                'p': pval
            })
            table = table.set_index('Source')

        return pval, table

    alpha = np.random.vonmises(0, 2, 50)
    idp = np.random.choice([1, 2, 3], 50)
    idq = np.random.choice([1, 2], 50)

    # Run original PyCircStat test
    pval_orig, table_orig = hktest(alpha, idp, idq)

    # Run PyCircStat2 version
    pval_new, table_new = harrison_kanji_test(alpha, idp, idq)

    # Compare p-values
    assert np.allclose(pval_orig, pval_new, atol=1e-6), f"P-values mismatch:\nOriginal: {pval_orig}\nNew: {pval_new}"

    # Compare ANOVA table values (ignoring index differences)
    table_orig_values = table_orig.to_numpy()
    table_new_values = table_new.to_numpy()

    assert np.allclose(table_orig_values, table_new_values, atol=1e-6, equal_nan=True), f"ANOVA tables differ:\nOriginal:\n{table_orig}\nNew:\n{table_new}"

def test_circ_anova_test():
    """Test the Circular ANOVA (F-test & LRT) for multiple samples."""

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate von Mises distributed samples with different mean directions
    group1 = np.random.vonmises(mu=0, kappa=5, size=50)
    group2 = np.random.vonmises(mu=np.pi / 4, kappa=5, size=50)
    group3 = np.random.vonmises(mu=np.pi / 2, kappa=5, size=50)

    samples = [group1, group2, group3]

    # Run F-test
    result_f = circ_anova_test(samples, method="F-test")
    assert "statistic" in result_f, "F-test did not return a statistic"
    assert "p_value" in result_f, "F-test did not return a p-value"
    assert result_f["p_value"] >= 0 and result_f["p_value"] <= 1, "F-test p-value out of range"
    assert result_f["df"] == (2, 147, 149), f"F-test degrees of freedom mismatch: {result_f['df']}"
    
    # Run Likelihood Ratio Test (LRT)
    result_lrt = circ_anova_test(samples, method="LRT")
    assert "statistic" in result_lrt, "LRT did not return a statistic"
    assert "p_value" in result_lrt, "LRT did not return a p-value"
    assert result_lrt["p_value"] >= 0 and result_lrt["p_value"] <= 1, "LRT p-value out of range"
    assert result_lrt["df"] == 2, f"LRT degrees of freedom mismatch: {result_lrt['df']}"

    # Edge case: All groups have the same mean direction
    identical_group = np.random.vonmises(mu=0, kappa=5, size=50)
    result_identical = circ_anova_test([identical_group] * 3, method="F-test")
    assert result_identical["p_value"] > 0.05, "F-test should not reject H0 for identical groups"

    # Edge case: Small sample sizes
    small_group1 = np.random.vonmises(mu=0, kappa=5, size=5)
    small_group2 = np.random.vonmises(mu=np.pi / 4, kappa=5, size=5)
    small_group3 = np.random.vonmises(mu=np.pi / 2, kappa=5, size=5)

    result_small = circ_anova_test([small_group1, small_group2, small_group3], method="F-test")
    assert result_small["p_value"] >= 0 and result_small["p_value"] <= 1, "Small-sample p-value out of range"

    # Invalid method check
    with pytest.raises(ValueError, match="Invalid method. Choose 'F-test' or 'LRT'."):
        circ_anova_test(samples, method="INVALID")

    # Single group should raise error
    with pytest.raises(ValueError, match="At least two groups are required for ANOVA."):
        circ_anova_test([group1])
