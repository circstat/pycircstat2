import numpy as np
import pytest

from pycircstat2.correlation import circ_corrcc


@pytest.mark.parametrize("method", ["fl", "js"])
@pytest.mark.parametrize("test", [False, True])
@pytest.mark.parametrize("scale", [1e-3, 1e-5, 1e-20])
def test_nonconstant_small_angles_have_unit_self_correlation(method, test, scale):
    angles = scale * np.array([0.0, 1.0, 4.0, 2.0, 3.0])
    result = circ_corrcc(angles, angles, method=method, test=test)
    r = result.r if test else result
    assert r == pytest.approx(1.0)
    if method == "js" and test:
        assert np.isfinite(result.test_stat)
        assert 0.0 < result.p_value < 1.0


@pytest.mark.parametrize("method", ["fl", "js"])
@pytest.mark.parametrize("test", [False, True])
@pytest.mark.parametrize("angles", [
    np.zeros(5),
    np.full(5, 0.7),
    np.array([0.0, np.pi, 0.0, np.pi, 0.0]),
    np.array([0.7, 0.7 + 2 * np.pi, 0.7, 0.7 + 2 * np.pi, 0.7]),
])
def test_constant_or_antipodal_angles_remain_degenerate(method, test, angles):
    with pytest.raises(ValueError, match="zero variance"):
        circ_corrcc(angles, angles, method=method, test=test)
