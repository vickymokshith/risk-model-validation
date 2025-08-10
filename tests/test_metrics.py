import numpy as np
from src.utils import psi, ks_stat


def test_psi_zero_on_identical() -> None:
    """PSI should be near zero when comparing identical distributions."""
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = psi(x, x)
    assert result < 1e-6


def test_ks_between_zero_and_one() -> None:
    """KS statistic should fall within [0, 1]."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    labels = np.array([0, 1, 0, 1, 1])
    ks = ks_stat(scores, labels)
    assert 0 <= ks <= 1
