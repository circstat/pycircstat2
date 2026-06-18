import matplotlib

matplotlib.use("Agg")  # headless: tests never open a GUI window

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures a test opened so they don't accumulate across the
    suite — pyplot retains figures until closed, which triggers matplotlib's
    'More than 20 figures have been opened' memory warning."""
    yield
    plt.close("all")
