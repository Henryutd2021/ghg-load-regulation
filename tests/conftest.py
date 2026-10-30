import pytest

from ghg_framework import synthetic_region


@pytest.fixture(scope="session")
def spring():
    """Synthetic 'Sunny' region, 1 March to 31 May (three months; quarters Q1 = March, Q2 = April-May)."""
    return synthetic_region("Sunny", hours=92 * 24, start="2025-03-01", seed=3)


@pytest.fixture(scope="session")
def week():
    """Synthetic 'Mixed' region, one week."""
    return synthetic_region("Mixed", hours=7 * 24, start="2025-06-02", seed=5)
