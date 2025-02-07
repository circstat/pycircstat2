import numpy as np
import pytest

from pycircstat2.clustering import MoVM


@pytest.fixture
def sample_data():
    """Generate sample circular data following a mixture of von Mises distributions."""
    np.random.seed(42)
    x1 = np.random.vonmises(mu=0, kappa=5, size=100)
    x2 = np.random.vonmises(mu=np.pi, kappa=10, size=100)
    x = np.concatenate([x1, x2])
    np.random.shuffle(x)
    return x

@pytest.fixture
def movm_instance():
    """Create a default instance of MoVM for testing."""
    return MoVM(n_clusters=3, n_iters=50, unit="radian", random_seed=42)

def test_initialization(movm_instance):
    """Test if the MoVM class initializes with correct parameters."""
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
    unique_labels = np.unique(movm_instance.labels)
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