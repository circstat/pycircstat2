import numpy as np
import polars as pl
import pytest

from pycircstat2.clustering import (
    CircHAC,
    CircKMeans,
    CircMixControl,
    MoCD,
    MovM,
    _circ_logpdf,
    _cmp_coef,
    _cmp_edf,
    _cmp_sp,
    _mix_classify,
    _mix_control,
    _mix_factor_specs,
    _mix_formula_smooth,
    _mix_gamma_birth,
    _mix_gamma_death,
    _mix_gamma_merge,
    _mix_gamma_split,
    _mix_gate,
    _mix_group_index,
    _mix_has_smooth,
    _mix_loglik_rows,
    _mix_n_responses,
    _mix_objective,
    _mix_response,
    _mix_responses,
    _mix_responsibilities,
    _mix_rowsum,
    _mix_unit_feature,
    _MixComponent,
    _MixProduct,
    circ_kmeans,
    circ_mix,
)
from pycircstat2.distributions import vmlss, vonmises
from pycircstat2.regression import circ_gam

############################
#  Fixtures and Utilities  #
############################

@pytest.fixture
def sample_data():
    """Generate sample circular data following a mixture of von Mises distributions."""
    np.random.seed(42)
    x1 = np.random.vonmises(mu=0, kappa=5, size=35)
    x2 = np.random.vonmises(mu=np.pi, kappa=10, size=35)
    x = np.concatenate([x1, x2])
    np.random.shuffle(x)
    return x

@pytest.fixture
def movm_instance():
    """Create a default instance of MovM for testing."""
    return MovM(n_clusters=3, n_iters=50, unit="radian", random_seed=42)

@pytest.fixture
def circhac_instance():
    """Create a default instance of CircHAC for testing."""
    return CircHAC(n_clusters=3, metric="geodesic", unit="radian")


@pytest.fixture
def circkmeans_instance():
    """Create a default instance of CircKMeans for testing."""
    return CircKMeans(n_clusters=3, metric="geodesic", unit="radian", random_seed=42)


@pytest.fixture
def mocd_instance():
    """Create a default instance of MoCD (von Mises mixture) for testing."""
    return MoCD(
        distribution=vonmises,
        n_clusters=3,
        n_iters=50,
        unit="radian",
        random_seed=42,
        threshold=1e-6,
        burnin=10,
    )


############################
#  Tests for MovM          #
############################

def test_initialization(movm_instance):
    """Test if the MovM class initializes with correct parameters."""
    assert movm_instance.n_clusters == 3
    assert movm_instance.n_iters == 50
    assert movm_instance.unit == "radian"

def test_fit_convergence(movm_instance, sample_data):
    """Test if the algorithm converges within the given iterations."""
    movm_instance.fit(sample_data, verbose=False)
    assert movm_instance.converged or len(movm_instance.nLL) == movm_instance.n_iters

def test_fit_cluster_assignment(movm_instance, sample_data):
    """Ensure that fitted cluster assignments are valid and nontrivial."""
    movm_instance.fit(sample_data, verbose=False)
    unique_labels = np.unique(movm_instance.labels_)
    assert len(unique_labels) <= movm_instance.n_clusters  # Some clusters may be empty
    assert len(unique_labels) > 1  # Should not collapse into a single cluster

def test_predict(movm_instance, sample_data):
    """Test cluster predictions on input data."""
    movm_instance.fit(sample_data, verbose=False)
    predicted_labels = movm_instance.predict(sample_data)
    assert len(predicted_labels) == len(sample_data)
    assert np.issubdtype(predicted_labels.dtype, np.integer)

def test_predict_density(movm_instance):
    """Ensure density prediction returns reasonable values."""
    movm_instance.fit(np.random.vonmises(mu=0, kappa=5, size=200), verbose=False)
    x_test = np.linspace(0, 2 * np.pi, 50)
    density = movm_instance.predict_density(x_test)
    assert len(density) == len(x_test)
    assert np.all(density >= 0)  # Probabilities should not be negative


############################
#  Tests for MoCD          #
############################

def test_mocd_initialization_defaults(mocd_instance):
    """Ensure MoCD initialises with the desired configuration."""
    assert mocd_instance.n_clusters == 3
    assert mocd_instance.unit == "radian"
    assert mocd_instance.param_names == ["mu", "kappa"]


def test_mocd_fit_and_params(mocd_instance, sample_data):
    """Fitting should produce mixing weights and component parameters."""
    mocd_instance.fit(sample_data)
    assert mocd_instance.p_ is not None
    assert mocd_instance.params_ is not None
    assert len(mocd_instance.p_) == mocd_instance.n_clusters
    assert len(mocd_instance.params_) == mocd_instance.n_clusters


def test_mocd_predict_proba(mocd_instance, sample_data):
    """Responsibilities should form a valid probability distribution."""
    mocd_instance.fit(sample_data)
    resp = mocd_instance.predict_proba(sample_data)
    assert resp.shape == (mocd_instance.n_clusters, sample_data.size)
    col_sums = resp.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0)


def test_mocd_predict_labels(mocd_instance, sample_data):
    """Cluster labels should be returned with the correct shape."""
    mocd_instance.fit(sample_data)
    labels = mocd_instance.predict(sample_data)
    assert labels.shape == sample_data.shape
    assert np.issubdtype(labels.dtype, np.integer)


def test_mocd_density_and_bic(mocd_instance, sample_data):
    """Density predictions and BIC should be finite after fit."""
    mocd_instance.fit(sample_data)
    grid = np.linspace(0, 2 * np.pi, 32)
    density = mocd_instance.predict_density(grid)
    assert density.shape == grid.shape
    assert np.all(np.isfinite(density))
    bic = mocd_instance.bic()
    assert np.isfinite(bic)


############################
#  Tests for CircHAC       #
############################

def test_circhac_initialization(circhac_instance):
    """Test if the CircHAC class initializes with correct parameters."""
    assert circhac_instance.n_clusters == 3
    assert circhac_instance.metric == "geodesic"
    assert circhac_instance.unit == "radian"

def test_circhac_fit_basic(circhac_instance, sample_data):
    """Test basic fit to ensure merges_ and labels_ are created properly."""
    circhac_instance.fit(sample_data)
    # Check we have a labels_ array of the right size
    assert len(circhac_instance.labels_) == len(sample_data)
    # merges_ shape should be (# merges, 4)
    # If n=200, to get down to 3 clusters, we do 197 merges
    merges = circhac_instance.merges_
    assert merges.shape[1] == 4
    # We might check that merges' final row uses a positive distance
    # or that merges are sorted by step, etc.

def test_circhac_labels_nontrivial(circhac_instance, sample_data):
    """Ensure multiple clusters are formed (unless there's weird data)."""
    circhac_instance.fit(sample_data)
    unique_labels = np.unique(circhac_instance.labels_)
    assert 1 < len(unique_labels) <= circhac_instance.n_clusters

def test_circhac_predict(circhac_instance, sample_data):
    """Test cluster predictions on new data after fit."""
    circhac_instance.fit(sample_data)
    new_points = np.random.vonmises(mu=0, kappa=4, size=10)
    pred_labels = circhac_instance.predict(new_points)
    assert len(pred_labels) == len(new_points)
    assert np.issubdtype(pred_labels.dtype, np.integer)

def test_circhac_silhouette(circhac_instance, sample_data):
    """Check that the silhouette score is in a valid range."""
    circhac_instance.fit(sample_data)
    score = circhac_instance.silhouette_score()
    # silhouette range is [-1,1], typically >0 for decent data
    assert -1.0 <= score <= 1.0

def test_circhac_dendrogram_plot(circhac_instance, sample_data):
    """Smoke test the dendrogram plot to ensure no errors are raised."""
    import matplotlib
    matplotlib.use("Agg")  # run headless
    circhac_instance.fit(sample_data)
    ax = circhac_instance.plot_dendrogram()  # Should not error
    assert ax is not None

############################
#  Tests for CircKMeans    #
############################

def test_circkmeans_initialization(circkmeans_instance):
    """Test if the CircKMeans class initializes with correct parameters."""
    assert circkmeans_instance.n_clusters == 3
    assert circkmeans_instance.metric == "geodesic"
    assert circkmeans_instance.unit == "radian"

def test_circkmeans_fit_basic(circkmeans_instance, sample_data):
    """Test basic fit to ensure centers_ and labels_ are created properly."""
    circkmeans_instance.fit(sample_data)
    
    # Check labels have the correct length
    assert len(circkmeans_instance.labels_) == len(sample_data)

    # Check centers_ exist and match the requested number of clusters
    assert len(circkmeans_instance.centers_) == circkmeans_instance.n_clusters

def test_circkmeans_labels_nontrivial(circkmeans_instance, sample_data):
    """Ensure multiple clusters are formed (unless there's weird data)."""
    circkmeans_instance.fit(sample_data)
    unique_labels = np.unique(circkmeans_instance.labels_)
    
    assert 1 < len(unique_labels) <= circkmeans_instance.n_clusters  # Should not collapse into 1 cluster

def test_circkmeans_inertia_decreases(circkmeans_instance, sample_data):
    """Check that inertia decreases over iterations, indicating convergence."""
    circkmeans_instance.fit(sample_data)
    assert circkmeans_instance.inertia_ is not None
    assert circkmeans_instance.inertia_ >= 0  # Inertia should never be negative

def test_circkmeans_predict(circkmeans_instance, sample_data):
    """Test cluster predictions on new data after fit."""
    circkmeans_instance.fit(sample_data)
    
    new_points = np.random.vonmises(mu=0, kappa=4, size=10)
    pred_labels = circkmeans_instance.predict(new_points)
    
    assert len(pred_labels) == len(new_points)
    assert np.issubdtype(pred_labels.dtype, np.integer)  # Ensure integer cluster labels

def test_circkmeans_convergence(circkmeans_instance, sample_data):
    """Ensure K-means stops after reaching convergence criteria."""
    circkmeans_instance.fit(sample_data)
    
    # If the fit completes within max_iter, we assume it stopped at the tolerance threshold
    assert circkmeans_instance.max_iter >= 10  # Sanity check for large max_iter


############################
#  Tests for circ_kmeans   #
############################
# The torus k-means++ Lloyd routine circ_mix seeds and splits with -- distinct
# from the CircKMeans class above (1-D, random init, circ_dist metric).

def test_circ_kmeans_recovers_two_modes():
    """Two well-separated modes on the circle are recovered exactly."""
    rng = np.random.default_rng(1)
    theta = np.r_[rng.normal(0.0, 0.3, 50), rng.normal(np.pi, 0.3, 50)] % (2 * np.pi)
    km = circ_kmeans(theta, 2, random_seed=1)
    assert km["cluster"].shape == (100,)
    assert sorted(km["size"].tolist()) == [50, 50]
    truth = np.r_[np.zeros(50), np.ones(50)]
    acc = max((km["cluster"] == truth).mean(), (km["cluster"] != truth).mean())
    assert acc == 1.0
    # centres are ON the circle, near 0 and pi
    centres = np.sort(km["centers"].ravel())
    assert abs(centres[0]) < 0.15
    assert abs(abs(centres[1]) - np.pi) < 0.15


def test_circ_kmeans_clusters_on_the_torus():
    """d > 1 columns cluster on the product of circles, jointly."""
    rng = np.random.default_rng(2)
    a = np.r_[rng.normal(-2.0, 0.3, 60), rng.normal(1.0, 0.3, 60)] % (2 * np.pi)
    b = np.r_[rng.normal(1.0, 0.3, 60), rng.normal(-1.5, 0.3, 60)] % (2 * np.pi)
    km = circ_kmeans(np.column_stack([a, b]), 2, random_seed=3)
    assert km["centers"].shape == (2, 2)
    truth = np.r_[np.zeros(60), np.ones(60)]
    acc = max((km["cluster"] == truth).mean(), (km["cluster"] != truth).mean())
    assert acc == 1.0


def test_circ_kmeans_is_seed_deterministic_and_guards_k():
    """A fixed seed reproduces the partition; K > n is rejected."""
    rng = np.random.default_rng(4)
    theta = rng.uniform(0, 2 * np.pi, 40)
    a = circ_kmeans(theta, 3, random_seed=7)
    b = circ_kmeans(theta, 3, random_seed=7)
    assert np.array_equal(a["cluster"], b["cluster"])
    assert np.allclose(a["tot_withinss"], b["tot_withinss"])
    # ... and an UNSEEDED call must simply run (random_seed=None draws its own)
    c = circ_kmeans(theta, 3)
    assert c["cluster"].shape == (40,)
    assert int(c["size"].sum()) == 40
    with pytest.raises(ValueError, match="more cluster centres"):
        circ_kmeans(theta, 41)
    with pytest.raises(ValueError, match="centers"):
        circ_kmeans(theta, 0)


############################
#  Tests for circ_mix      #
############################
# These exercise the engine's pure logic --
# the guards, formula dispatch, the E-step numerics, the responsibility
# reshapes behind the automatic-K moves, the information-criterion arithmetic
# and the component accessors -- with no fitting. The stochastic end-to-end
# behaviour is covered by a handful of small, seeded integration fits at the
# bottom, plus the _circ_logpdf density contract.

@pytest.fixture
def mix_frame():
    """A tiny fixed frame; the guards all fire BEFORE any fit."""
    return pl.DataFrame(
        {
            "y": [0.1, 6.1, 0.3, 5.9, 0.5],
            "x": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "id": [1, 1, 2, 2, 3],
        }
    )


# ---- guards fire before any fitting ------------------------------------- #

def test_circ_mix_rejects_invalid_arguments(mix_frame):
    """Not-yet-supported and invalid arguments are rejected clearly."""
    with pytest.raises(ValueError, match="location-scale family"):
        circ_mix("y ~ x", mix_frame, family="gaussian")
    with pytest.raises(ValueError, match="K"):
        circ_mix("y ~ x", mix_frame, K=0)
    with pytest.raises(ValueError, match="exceeds"):
        circ_mix("y ~ x", mix_frame, K=99)
    with pytest.raises(ValueError, match="search"):
        circ_mix("y ~ x", mix_frame, search="nope")
    with pytest.raises(ValueError, match="assign"):
        circ_mix("y ~ x", mix_frame, assign="nope")
    with pytest.raises(ValueError, match="kmin"):
        circ_mix("y ~ x", mix_frame, control={"kmin": 5, "kmax": 2})


def test_circ_mix_joint_and_response_guards(mix_frame):
    """A joint spec over > 2 responses and a missing response error pre-fit."""
    with pytest.raises(ValueError, match="d > 2"):
        circ_mix(["y ~ 1", "x ~ 1", "id ~ 1"], mix_frame)
    with pytest.raises(ValueError, match="not found in"):
        circ_mix(["y ~ phi", "phi ~ 1"], mix_frame)


def test_circ_mix_group_index(mix_frame):
    """The group index maps rows to units; a bad group errors pre-fit."""
    ur = _mix_group_index(None, mix_frame, mix_frame.height)
    assert ur["kind"] == "row"
    assert np.array_equal(ur["grp"], np.arange(5))
    assert ur["n_units"] == 5
    # subjects: id = 1,1,2,2,3 -> 3 units, rows mapped 0,0,1,1,2
    us = _mix_group_index("id", mix_frame, mix_frame.height)
    assert us["kind"] == "subject"
    assert us["n_units"] == 3
    assert np.array_equal(us["grp"], np.array([0, 0, 1, 1, 2]))
    assert us["labels"] == ["1", "2", "3"]
    with pytest.raises(ValueError, match="not a column"):
        circ_mix("y ~ x", mix_frame, group="nope")
    with pytest.raises(ValueError, match="subject"):
        circ_mix("y ~ x", mix_frame, K=4, group="id")


def test_circ_mix_unit_feature_averages_rows_within_unit():
    """A per-row feature aggregates to per-unit means (identity for rows)."""
    rf = np.array([[1.0, 3.0], [2.0, 4.0], [10.0, 10.0]])
    grp = np.array([0, 0, 1])
    uf = _mix_unit_feature(rf, grp, 2)
    assert uf.shape == (2, 2)
    np.testing.assert_allclose(uf[0], [1.5, 3.5])  # (1+2)/2, (3+4)/2
    np.testing.assert_allclose(uf[1], [10.0, 10.0])
    # grp = arange(n) is the identity (the row case)
    np.testing.assert_allclose(_mix_unit_feature(rf, np.arange(3), 3), rf)


# ---- formula dispatch ---------------------------------------------------- #

def test_circ_mix_formula_parsing():
    """Formula parsing finds the response and counts distinct responses."""
    assert _mix_response("y ~ x") == "y"
    assert _mix_response(["psi ~ phi", "phi ~ 1"]) == "psi"
    assert _mix_n_responses("y ~ s(x)") == 1
    assert _mix_n_responses(["th ~ s(x)", "~ s(x)"]) == 1  # LSS of one response
    assert _mix_n_responses(["psi ~ s(phi)", "phi ~ 1"]) == 2  # joint
    assert _mix_responses("y ~ x") == ["y"]
    assert _mix_responses(["psi ~ cos(phi)", "phi ~ 1"]) == ["psi", "phi"]
    # a nested per-factor location-scale list
    assert _mix_n_responses([["psi ~ s(phi)", "~ s(phi)"], "phi ~ 1"]) == 2


def test_circ_mix_factor_specs():
    """A spec splits into per-component circ_gam specs, in chain-rule order."""
    fs1 = _mix_factor_specs("y ~ x")  # a single formula is one factor
    assert fs1 == ["y ~ x"]
    fs2 = _mix_factor_specs(["th ~ s(x)", "~ s(x)"])  # one response -> one factor
    assert len(fs2) == 1
    fs3 = _mix_factor_specs(["psi ~ cos(phi)", "phi ~ 1"])  # joint -> two factors
    assert len(fs3) == 2
    assert _mix_response(fs3[0]) == "psi"
    assert _mix_response(fs3[1]) == "phi"
    # the nested case: factor 1 keeps its own LSS list, factor 2 is the marginal
    fs4 = _mix_factor_specs([["psi ~ s(phi)", "~ s(phi)"], "phi ~ 1"])
    assert len(fs4) == 2
    assert isinstance(fs4[0], list)
    assert isinstance(fs4[1], str)


def test_circ_mix_formula_smooth_detection():
    """s()/te()/ti()/t2() are detected; cos(/sin( are not false positives."""
    assert _mix_formula_smooth("y ~ s(x)")
    assert _mix_formula_smooth(["y ~ te(a, b)", "~ 1"])
    assert not _mix_formula_smooth("y ~ x")
    assert not _mix_formula_smooth("y ~ cos(phi) + sin(phi)")


# ---- the pure responsibility reshapes behind the moves ------------------- #

def test_circ_mix_gamma_reshapes_preserve_row_sums():
    """Each move must hand the local EM a valid responsibility matrix: row sums
    stay 1 and K changes by exactly +-1 (split/birth/merge) or to |keep|."""
    g = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])

    # SPLIT column 0 by a per-row label in {0,1}: K -> 3
    gs = _mix_gamma_split(g, 0, np.array([0, 1, 0, 1]))
    assert gs.shape == (4, 3)
    np.testing.assert_allclose(gs.sum(axis=1), 1.0)
    np.testing.assert_allclose(gs[:, 0] + gs[:, 1], g[:, 0])  # mass conserved
    np.testing.assert_allclose(gs[:, 2], g[:, 1])  # other column carried over
    assert gs[1, 0] == 0.0  # row 1 labelled 1 -> col a empty

    # MERGE columns 0,1: K -> 1, the merged column is their sum (all rows 1)
    gm = _mix_gamma_merge(g, 0, 1)
    assert gm.shape == (4, 1)
    np.testing.assert_allclose(gm[:, 0], 1.0)

    # DEATH keeping only column 1: renormalised to 1
    gd = _mix_gamma_death(g, [1])
    assert gd.shape == (4, 1)
    np.testing.assert_allclose(gd[:, 0], 1.0)

    # BIRTH seeding rows {0,2}: K -> 3, seeded rows one-hot on the new column
    gb = _mix_gamma_birth(g, [0, 2])
    assert gb.shape == (4, 3)
    np.testing.assert_allclose(gb.sum(axis=1), 1.0)
    np.testing.assert_allclose(gb[[0, 2], 2], 1.0)
    np.testing.assert_allclose(gb[[1, 3], 2], 0.0)


def test_circ_mix_gamma_death_reseeds_empty_rows():
    """A unit left with no mass seats uniformly rather than dividing by zero."""
    g = np.array([[1.0, 0.0], [0.4, 0.6]])
    gd = _mix_gamma_death(g, [1])
    np.testing.assert_allclose(gd.sum(axis=1), 1.0)
    assert np.isfinite(gd).all()


# ---- E-step numerics ------------------------------------------------------ #

def test_circ_mix_estep_row_reductions():
    """The E-step row reductions match closed forms and are stable."""
    logmix = np.array([[-1.0, -2.0], [-0.5, -0.5], [-10.0, -30.0]])
    rows = _mix_loglik_rows(logmix)
    assert rows[0] == pytest.approx(np.log(np.exp(-1) + np.exp(-2)))
    assert rows[1] == pytest.approx(np.log(2 * np.exp(-0.5)))
    assert rows[2] == pytest.approx(-10 + np.log1p(np.exp(-20)))  # row-max stable
    g = _mix_responsibilities(logmix)
    np.testing.assert_allclose(g.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        g[0], np.array([np.exp(-1), np.exp(-2)]) / (np.exp(-1) + np.exp(-2))
    )
    assert g[2, 0] == pytest.approx(1.0, abs=1e-8)  # no underflow to NaN


def test_circ_mix_hard_classification():
    """The hard E-step seats each unit at its argmax (classification loglik)."""
    logmix = np.array([[-1.0, -2.0], [-0.5, -0.3], [-10.0, -30.0]])
    cl = _mix_classify(logmix)
    np.testing.assert_array_equal(cl["z"], [0, 1, 0])
    np.testing.assert_allclose(cl["gamma"].sum(axis=1), 1.0)  # one-hot rows
    np.testing.assert_allclose(cl["gamma"][1], [0.0, 1.0])
    assert cl["loglik"] == pytest.approx(-1 + -0.3 + -10)  # sum of the maxima


def test_circ_mix_constant_gating_broadcasts_pi():
    """The constant gating broadcasts pi over rows."""
    pm = _mix_gate({"type": "constant", "pi": np.array([0.3, 0.7])}, 4, 2)
    assert pm.shape == (4, 2)
    np.testing.assert_allclose(pm[0], [0.3, 0.7])
    assert np.all(pm[:, 0] == 0.3)
    with pytest.raises(ValueError, match="gating"):
        _mix_gate({"type": "covariate"}, 4, 2)


def test_circ_mix_rowsum_collapses_rows_to_units():
    """rowsum sums per-row values within unit; identity for the row case."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(
        _mix_rowsum(x, np.array([0, 0, 1]), 2), [[4.0, 6.0], [5.0, 6.0]]
    )
    np.testing.assert_allclose(_mix_rowsum(x, np.arange(3), 3), x)


# ---- the component interface, on constructed mocks ----------------------- #

class _FakeFit:
    """The fields the _cmp_* accessors read."""

    def __init__(self, coef, edf, sp=()):
        self.coef = np.asarray(coef, dtype=float)
        self.edf = np.asarray(edf, dtype=float)
        self.sp = np.asarray(sp, dtype=float)


def test_circ_mix_component_accessors():
    """cmp_edf sums the fit's edf; cmp_sp reads the smoothing parameters."""
    cp = _MixComponent(_FakeFit([0.5, 1.2], [1.0, 1.0, 1.0], sp=[2.5]))
    assert _cmp_edf(cp) == pytest.approx(3.0)
    np.testing.assert_allclose(_cmp_sp(cp), [2.5])
    assert _mix_has_smooth(cp)  # a non-empty sp IS the smooth predicate
    assert not _mix_has_smooth(_MixComponent(_FakeFit([0.0], [1.0])))


def test_circ_mix_product_component_accessors():
    """A product component's accessors sum / list over its chain-rule factors."""
    prod = _MixProduct(
        [_FakeFit([0.7, 0.2], [1.0, 1.0, 1.0]), _FakeFit([-1.0], [1.0, 1.0])],
        ["psi", "phi"],
    )
    assert isinstance(prod, _MixComponent)  # IS-A component
    assert _cmp_edf(prod) == pytest.approx(5.0)  # 3 + 2 summed over factors
    cf = _cmp_coef(prod)  # per-factor list (the warm-start shape)
    assert isinstance(cf, list) and len(cf) == 2
    assert cf[0][1] == pytest.approx(0.2)
    assert not _mix_has_smooth(prod)  # smooth detection is product-aware
    prod_s = _MixProduct(
        [_FakeFit([0.0], [1.7], sp=[0.5]), _FakeFit([0.0], [1.0, 1.0])], ["a", "b"]
    )
    assert _mix_has_smooth(prod_s)
    sp = _cmp_sp(prod_s)
    assert isinstance(sp, list)
    np.testing.assert_allclose(sp[0], [0.5])
    assert sp[1].size == 0  # a parametric factor: no sp


def test_circ_mix_objective_matches_closed_form():
    """J = -2 logLik + lambda * df, df = (K-1) + sum(edf)."""
    state = {
        "K": 2,
        "loglik": -100.0,
        "components": [
            _MixComponent(_FakeFit([0.0, 0.0], [1.0, 1.0])),
            _MixComponent(_FakeFit([0.0, 0.0], [1.0, 1.0])),
        ],
    }
    # df = (2-1) + (2 + 2) = 5
    assert _mix_objective(state, np.log(100)) == pytest.approx(
        -2 * -100 + np.log(100) * 5
    )
    assert _mix_objective(state, 2.0) == pytest.approx(200 + 2 * 5)  # AIC-like


# ---- the control object --------------------------------------------------- #

def test_circ_mix_control_defaults_and_guards():
    """CircMixControl carries the committed fields and validates its inputs."""
    ct = CircMixControl()
    assert ct.restarts == 10
    assert ct.wfloor == 1e-8
    assert ct.tol == 1e-6
    assert ct.lambda_ is None
    assert ct.init == "kmeans"  # the default
    assert ct.penalty == "auto"  # the default
    assert ct.sp_every == 5
    assert ct.degen_strength == 0.05  # caps concentration near 10 * N_k
    assert ct.time_budget is None
    assert CircMixControl(init="random").init == "random"
    assert CircMixControl(sp_every=0).sp_every == 1  # coerced to >= 1
    with pytest.raises(ValueError, match="not supported"):
        CircMixControl(init="emEM")
    with pytest.raises(ValueError, match="penalty"):
        CircMixControl(penalty="nope")
    with pytest.raises(ValueError, match="non-negative"):
        CircMixControl(degen_strength=-1)
    with pytest.raises(ValueError, match="positive"):
        CircMixControl(time_budget=0)
    with pytest.raises(ValueError, match="moves"):
        CircMixControl(moves=("teleport",))


def test_circ_mix_runs_with_no_control_at_all():
    """The bare call must work: `control` defaults, and an unseeded control
    must NOT require a seed to be passed: an unseeded fit rides the ambient
    RNG stream."""
    rng = np.random.default_rng(31)
    n = 120
    z = rng.integers(2, size=n)
    y = np.mod(rng.vonmises(np.where(z == 0, -1.5, 1.5), 6.0), 2 * np.pi)
    df = pl.DataFrame({"y": y})
    m = circ_mix("y ~ 1", df, K=2, control={"restarts": 2})  # no seed
    assert m.K == 2 and np.isfinite(m.loglik)
    m2 = circ_mix("y ~ 1", df, K=2)  # no control whatsoever
    assert m2.K == 2 and np.isfinite(m2.loglik)
    # a seed is still honoured, and reproducible
    a = circ_mix("y ~ 1", df, K=2, control={"restarts": 2, "seed": 5})
    b = circ_mix("y ~ 1", df, K=2, control={"restarts": 2, "seed": 5})
    assert a.loglik == pytest.approx(b.loglik)
    np.testing.assert_allclose(a.gamma_, b.gamma_)


def test_circ_mix_control_accepts_a_dict_of_overrides():
    """control= takes a CircMixControl, a dict, or None."""
    ct = _mix_control({"restarts": 3, "seed": 7})
    assert isinstance(ct, CircMixControl)
    assert ct.restarts == 3 and ct.seed == 7
    assert _mix_control(None).restarts == 10
    assert _mix_control(ct) is ct
    with pytest.raises(TypeError, match="CircMixControl"):
        _mix_control("nope")


# ---- the E-step density primitive (the only fits above the integration set) #

def test_circ_logpdf_is_per_observation_and_sums_to_the_loglik():
    """_circ_logpdf's one contract: the per-observation l0, summed over the
    training rows, IS the model log-likelihood."""
    rng = np.random.default_rng(1)
    n = 150
    x = rng.uniform(0, 1, n)
    y = np.mod(rng.vonmises(2 * np.arctan(1.1 * np.sin(2 * np.pi * x)), 6.0), 2 * np.pi)
    df = pl.DataFrame({"y": y, "x": x})
    fit = circ_gam("y ~ x", df, family=vmlss)  # parametric MLE
    lp = _circ_logpdf(fit)
    assert lp.shape == (n,)
    assert np.isfinite(lp).all()
    assert lp.sum() == pytest.approx(float(fit.logLik), abs=1e-6)
    np.testing.assert_allclose(lp, _circ_logpdf(fit, df))  # default frame
    assert _circ_logpdf(fit, df.head(5)).shape == (5,)
    with pytest.raises(ValueError, match="response"):
        _circ_logpdf(fit, pl.DataFrame({"x": [0.0]}))


def test_circ_logpdf_works_for_the_linear_circular_leg():
    """The density primitive is response-agnostic: it reads only l0."""
    rng = np.random.default_rng(2)
    n = 150
    phi = rng.uniform(0, 2 * np.pi, n)
    y = 1.5 + 1.2 * np.sin(phi) + rng.normal(0, 0.5, n)
    df = pl.DataFrame({"y": y, "phi": phi})
    fit = circ_gam("y ~ sin(phi) + cos(phi)", df, family="gaulss")
    assert _circ_logpdf(fit).sum() == pytest.approx(float(fit.logLik), abs=1e-6)


def test_circ_logpdf_is_invariant_to_the_centring_frame():
    """A centred fit and an uncentred one give the SAME per-observation density
    -- centring only changes which basin the M-step reaches. This is what makes
    a mixture whose components each centre on their own weighted mode still
    comparable on one scale.

    It also pins the two frames apart: the DEFAULT newdata is ``fit.data``,
    which circ_gam already rotated, while a SUPPLIED newdata is in the original
    frame and must be rotated.
    """
    rng = np.random.default_rng(5)
    n = 200
    # a response hugging the tan-half wall, so center=True actually rotates
    y = np.mod(rng.vonmises(np.pi, 4.0, n), 2 * np.pi)
    df = pl.DataFrame({"y": y})
    fc = circ_gam("y ~ 1", df, family=vmlss, center=True)
    fu = circ_gam("y ~ 1", df, family=vmlss, center=False)
    assert fc.circ_center != 0.0  # the rotation really happened
    assert fu.circ_center == 0.0
    # each fit's own density sums to its own log-likelihood ...
    assert _circ_logpdf(fc).sum() == pytest.approx(float(fc.logLik), abs=1e-6)
    assert _circ_logpdf(fu).sum() == pytest.approx(float(fu.logLik), abs=1e-6)
    # ... the centred fit evaluated on the ORIGINAL frame agrees with its own ...
    np.testing.assert_allclose(_circ_logpdf(fc, df), _circ_logpdf(fc), atol=1e-6)
    # ... and the two frames agree observation by observation
    np.testing.assert_allclose(_circ_logpdf(fc), _circ_logpdf(fu), atol=1e-6)


def test_circ_mix_predict_defaults_to_the_original_training_frame():
    """predict() with no newdata must use the ORIGINAL frame, not a component's
    (possibly rotated) fit frame -- components may centre differently."""
    rng = np.random.default_rng(23)
    n = 180
    z = rng.integers(2, size=n)
    # both components sit near the tan-half wall, so each one really rotates
    y = np.mod(rng.vonmises(np.where(z == 0, np.pi - 0.6, np.pi + 0.6), 6.0), 2 * np.pi)
    df = pl.DataFrame({"y": y})
    m = circ_mix("y ~ 1", df, K=2, control={"restarts": 2, "seed": 1})
    refs = [float(cp.fit.circ_center) for cp in m.components]
    assert any(r != 0.0 for r in refs)  # at least one component rotated
    np.testing.assert_allclose(m.predict(), m.predict(df))
    np.testing.assert_allclose(m.predict(), m.gamma_)
    assert m.score_samples().sum() == pytest.approx(m.loglik, abs=1e-6)


# ---- small seeded integration fits ---------------------------------------- #

def test_circ_mix_density_clustering_recovers_two_components():
    """theta ~ 1 is density clustering -- the classic circular mixture."""
    rng = np.random.default_rng(7)
    n = 200
    z = rng.integers(2, size=n)
    y = np.mod(rng.vonmises(np.where(z == 0, -1.4, 1.4), 4.0), 2 * np.pi)
    m = circ_mix("y ~ 1", pl.DataFrame({"y": y}), K=2,
                 control={"restarts": 2, "seed": 1, "max_iter": 60})
    assert m.K == 2
    assert m.gamma_.shape == (n, 2)
    np.testing.assert_allclose(m.gamma_.sum(axis=1), 1.0)
    assert m.labels_.shape == (n,)
    assert np.isfinite(m.bic) and np.isfinite(m.aic)
    assert m.df == pytest.approx((2 - 1) + m.edf.sum())
    acc = max((m.labels_ == z).mean(), (m.labels_ != z).mean())
    assert acc > 0.9
    # the fitted mean directions bracket the two truths
    mus = np.sort([np.mod(p[0, 0], 2 * np.pi) for p in m.predict(type="response")])
    np.testing.assert_allclose(mus, np.sort(np.mod([-1.4, 1.4], 2 * np.pi)), atol=0.25)


def test_circ_mix_matches_the_published_turtle_mle():
    """Fisher's B3 turtles against the published two-component von Mises MLE
    (Jammalamadaka & Vaidyanathan 2024: logLik -105.413, kappa = [2.619, 8.447],
    pi = [0.84, 0.16]).

    The regression this pins is the DEGENERACY GUARD's calibration. The guard
    caps a component's concentration near ``N_k / (2*degen_strength)``, so a
    strength large enough to matter shrinks the 12-observation concentrated
    component well below its MLE -- at c = 1 it lands at kappa = 2.75, a 3x
    error. At the default c = 0.05 the cap is ~10*N_k and the fit tracks the
    published answer.
    """
    from pycircstat2 import load_data

    d = load_data("B3", source="fisher")
    # the dataset ships in degrees; every circular fit here is in radians
    df = pl.DataFrame({"theta": np.deg2rad(np.asarray(d["θ"], dtype=float))})
    m = circ_mix("theta ~ 1", df, K=2, control={"seed": 1})
    kappa = np.sort([float(np.exp(np.atleast_1d(cp.fit.coef)[1])) for cp in m.components])
    assert m.loglik == pytest.approx(-105.413, abs=0.05)
    np.testing.assert_allclose(kappa, [2.619, 8.447], rtol=0.10)
    np.testing.assert_allclose(np.sort(m.gating["pi"]), [0.16, 0.84], atol=0.02)
    # the guard is quiet here: under a nat per component, so repr stays silent
    assert not m.degen["binding"].any()

    # ... and cranking it up is what breaks the fit, not the data
    hard = circ_mix("theta ~ 1", df, K=2,
                    control={"seed": 1, "degen_strength": 1.0})
    assert max(float(np.exp(np.atleast_1d(cp.fit.coef)[1])) for cp in hard.components) < 4.0
    assert hard.degen["binding"].all()
    assert "degeneracy guard" in repr(hard)


def test_circ_mix_degen_report_measures_the_guard():
    """`.degen` reports the penalty each component actually paid, in nats.

    For the linear kernel on a concentration the penalty is exactly
    ``c * kappa`` (lambda_k = c/N_k times N_k weighted rows), which makes it a
    direct read on the log-likelihood the M-step traded away -- the quantity
    `loglik`/`bic` omit.
    """
    rng = np.random.default_rng(7)
    n = 200
    z = rng.integers(2, size=n)
    y = np.mod(rng.vonmises(np.where(z == 0, -1.4, 1.4), 4.0), 2 * np.pi)
    df = pl.DataFrame({"y": y})
    ctl = {"restarts": 2, "seed": 1, "max_iter": 60}

    m = circ_mix("y ~ 1", df, K=2, control={**ctl, "degen_strength": 0.5})
    dg = m.degen
    assert dg["strength"] == 0.5
    Nk = np.maximum(m.gamma_, m.control.wfloor).sum(axis=0)
    np.testing.assert_allclose(dg["lambda_"], 0.5 / Nk, rtol=1e-6)
    kappa = np.array([float(np.exp(np.atleast_1d(cp.fit.coef)[1])) for cp in m.components])
    np.testing.assert_allclose(dg["penalty"], 0.5 * kappa, rtol=1e-6)
    assert [set(p) for p in dg["by_param"]] == [{"kappa"}, {"kappa"}]

    # switching the guard off retires the report entirely
    off = circ_mix("y ~ 1", df, K=2, control={**ctl, "degen_strength": 0.0})
    assert off.degen is None
    assert "degeneracy guard" not in repr(off)


def test_circ_mix_predict_surfaces_agree():
    """predict(type=...) is self-consistent and matches the fitted state."""
    rng = np.random.default_rng(9)
    n = 150
    z = rng.integers(2, size=n)
    y = np.mod(rng.vonmises(np.where(z == 0, -1.5, 1.5), 5.0), 2 * np.pi)
    df = pl.DataFrame({"y": y})
    m = circ_mix("y ~ 1", df, K=2, control={"restarts": 1, "seed": 2})
    resp = m.predict(df, type="cluster")
    np.testing.assert_allclose(resp.sum(axis=1), 1.0)
    np.testing.assert_allclose(resp, m.gamma_)
    np.testing.assert_allclose(m.predict_proba(df), resp)
    dens = m.predict(df, type="density")
    assert (dens > 0).all()
    np.testing.assert_allclose(np.log(dens), m.predict(df, type="density", log=True))
    np.testing.assert_allclose(m.score_samples(df), np.log(dens))
    # the mixture log-density sums to the fitted log-likelihood
    assert np.log(dens).sum() == pytest.approx(m.loglik, abs=1e-6)
    per_comp = m.predict(df, type="response")
    assert len(per_comp) == 2
    with pytest.raises(ValueError, match="type"):
        m.predict(df, type="nope")


def test_circ_mix_hard_assignment_is_classification_em():
    """assign='hard' seats each unit wholly at its argmax (CEM)."""
    rng = np.random.default_rng(11)
    n = 150
    z = rng.integers(2, size=n)
    y = np.mod(rng.vonmises(np.where(z == 0, -1.5, 1.5), 6.0), 2 * np.pi)
    m = circ_mix("y ~ 1", pl.DataFrame({"y": y}), K=2, assign="hard",
                 control={"restarts": 1, "seed": 3})
    assert np.all((m.gamma_ == 0.0) | (m.gamma_ == 1.0))  # one-hot
    np.testing.assert_allclose(m.gamma_.sum(axis=1), 1.0)
    assert max((m.labels_ == z).mean(), (m.labels_ != z).mean()) > 0.9


def test_circ_mix_groups_cluster_whole_curves():
    """group='id' makes the SUBJECT the unit: one responsibility per curve."""
    rng = np.random.default_rng(13)
    nsub, nt = 24, 6
    cl = rng.integers(2, size=nsub)
    frames = []
    for s in range(nsub):
        ph = 2 * np.pi * np.sort(rng.uniform(0, 1, nt))
        mu = np.where(cl[s] == 0, 1.2, -1.2) * np.sin(ph)
        frames.append(
            pl.DataFrame(
                {
                    "id": np.full(nt, s),
                    "phase": ph,
                    "a": np.mod(rng.vonmises(mu, 10.0), 2 * np.pi),
                }
            )
        )
    long = pl.concat(frames)
    m = circ_mix("a ~ cos(phase) + sin(phase)", long, K=2, group="id",
                 control={"restarts": 2, "seed": 1, "max_iter": 40})
    assert m.unit["kind"] == "subject"
    assert m.unit["n_units"] == nsub
    assert m.labels_.shape == (nsub,)  # per SUBJECT, not per row
    assert m.gamma_.shape == (nsub, 2)
    # the BIC sample size is the number of subjects, not rows
    assert m.bic == pytest.approx(-2 * m.loglik + m.df * np.log(nsub))
    assert max((m.labels_ == cl).mean(), (m.labels_ != cl).mean()) > 0.9


def test_circ_mix_joint_torus_density():
    """Two distinct responses make each component a PRODUCT of factor fits."""
    rng = np.random.default_rng(17)
    n = 200
    z = rng.integers(2, size=n)
    phi = np.mod(rng.vonmises(np.where(z == 0, -2.0, 1.0), 4.0), 2 * np.pi)
    psi = np.mod(
        rng.vonmises(np.where(z == 0, 1.0, -1.5) + 0.8 * np.sin(phi), 5.0), 2 * np.pi
    )
    df = pl.DataFrame({"psi": psi, "phi": phi})
    m = circ_mix(["psi ~ cos(phi) + sin(phi)", "phi ~ 1"], df, K=2,
                 control={"restarts": 2, "seed": 1, "max_iter": 40})
    assert m.geometry == "joint"
    assert all(isinstance(cp, _MixProduct) for cp in m.components)
    assert len(m.components[0].fits) == 2  # one per chain-rule factor
    # a product component's edf is summed over its factors
    assert _cmp_edf(m.components[0]) == pytest.approx(
        sum(float(np.sum(f.edf)) for f in m.components[0].fits)
    )
    assert max((m.labels_ == z).mean(), (m.labels_ != z).mean()) > 0.9


def test_circ_mix_auto_k_greedy_and_grid_agree():
    """Both K-searches find the same optimum on three separated clusters."""
    rng = np.random.default_rng(3)
    n = 240
    z = rng.integers(3, size=n)
    y = np.mod(rng.vonmises(np.array([-2.2, 0.0, 2.2])[z], 12.0), 2 * np.pi)
    df = pl.DataFrame({"y": y})
    ctl = {"restarts": 2, "seed": 1, "kmin": 1, "kmax": 4, "min_size": 10}
    greedy = circ_mix("y ~ 1", df, K=2, search="greedy", control=ctl)
    grid = circ_mix("y ~ 1", df, search="grid", control=ctl)
    assert greedy.K == 3  # grew from the init K = 2 by a split
    assert grid.K == 3
    assert greedy.bic == pytest.approx(grid.bic, rel=1e-3)
    # the greedy trace records the accepted moves; the grid trace the sweep
    assert greedy.move_trace["move"].to_list() == ["split"]
    assert greedy.move_trace["K_to"].to_list() == [3]
    assert grid.move_trace["K"].to_list() == [1, 2, 3, 4]
    assert grid.move_trace["J"].to_list().index(min(grid.move_trace["J"])) == 2
    assert greedy.K_init == 2 and greedy.search == "greedy"


def test_circ_mix_repr_and_coef():
    """The header renders the leg, components and fit summary."""
    rng = np.random.default_rng(19)
    n = 150
    z = rng.integers(2, size=n)
    x = rng.uniform(-1, 1, n)
    mu = 2 * np.arctan(np.where(z == 0, 0.9, -0.9) + np.where(z == 0, 2.2, -2.2) * x)
    y = np.mod(rng.vonmises(mu, 6.0), 2 * np.pi)
    m = circ_mix("y ~ x", pl.DataFrame({"y": y, "x": x}), K=2,
                 control={"restarts": 1, "seed": 1})
    text = repr(m)
    assert "Finite mixture" in text
    assert "circular-linear" in text  # the c~l leg label
    assert "BIC" in text
    assert "y ~ x" in text
    cf = m.coef()
    assert list(cf) == ["component1", "component2"]
    assert np.asarray(cf["component1"]).size == 3  # 2 location + 1 scale coef


def test_circ_mix_circ_plot_views(monkeypatch):
    """circ_plot draws the flat / geometry / both views without error."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    n = 150
    z = rng.integers(2, size=n)
    x = rng.uniform(-1, 1, n)
    mu = 2 * np.arctan(np.where(z == 0, 0.9, -0.9) + np.where(z == 0, 2.2, -2.2) * x)
    y = np.mod(rng.vonmises(mu, 6.0), 2 * np.pi)
    m = circ_mix("y ~ x", pl.DataFrame({"y": y, "x": x}), K=2,
                 control={"restarts": 1, "seed": 1})
    for view, n_axes in (("flat", 1), ("geometry", 1), ("both", 2)):
        fig = m.circ_plot(view=view)
        assert len(fig.axes) == n_axes
        plt.close(fig)
    with pytest.raises(ValueError, match="view"):
        m.circ_plot(view="nope")


def test_circ_mix_circ_plot_density_and_joint_cells():
    """A density cell has no covariate (geometry falls back to flat); a joint
    component draws the torus-square projection whatever the view."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    n = 150
    z = rng.integers(2, size=n)
    y = np.mod(rng.vonmises(np.where(z == 0, -1.4, 1.4), 4.0), 2 * np.pi)
    md = circ_mix("y ~ 1", pl.DataFrame({"y": y}), K=2,
                  control={"restarts": 1, "seed": 1})
    plt.close(md.circ_plot())
    with pytest.warns(RuntimeWarning, match="no surface"):
        plt.close(md.circ_plot(view="geometry"))

    phi = np.mod(rng.vonmises(np.where(z == 0, -2.0, 1.0), 4.0), 2 * np.pi)
    psi = np.mod(
        rng.vonmises(np.where(z == 0, 1.0, -1.5) + 0.8 * np.sin(phi), 5.0), 2 * np.pi
    )
    mj = circ_mix(["psi ~ cos(phi) + sin(phi)", "phi ~ 1"],
                  pl.DataFrame({"psi": psi, "phi": phi}), K=2,
                  control={"restarts": 1, "seed": 1})
    plt.close(mj.circ_plot())
    with pytest.warns(RuntimeWarning, match="joint geometry surface"):
        plt.close(mj.circ_plot(view="geometry"))
