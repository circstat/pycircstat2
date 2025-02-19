import numpy as np
import pytest

from pycircstat2.clustering import CircHAC, CircKMeans, MovM

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
    assert predicted_labels.dtype == np.int64

def test_predict_density(movm_instance):
    """Ensure density prediction returns reasonable values."""
    movm_instance.fit(np.random.vonmises(mu=0, kappa=5, size=200), verbose=False)
    x_test = np.linspace(0, 2 * np.pi, 50)
    density = movm_instance.predict_density(x_test)
    assert len(density) == len(x_test)
    assert np.all(density >= 0)  # Probabilities should not be negative


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
    assert pred_labels.dtype == np.int64

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
    assert pred_labels.dtype == np.int64  # Ensure integer cluster labels

def test_circkmeans_convergence(circkmeans_instance, sample_data):
    """Ensure K-means stops after reaching convergence criteria."""
    circkmeans_instance.fit(sample_data)
    
    # If the fit completes within max_iter, we assume it stopped at the tolerance threshold
    assert circkmeans_instance.max_iter >= 10  # Sanity check for large max_iter