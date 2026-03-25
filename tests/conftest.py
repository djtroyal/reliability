"""Shared fixtures for reliability test suite."""

import numpy as np
import pytest


@pytest.fixture
def weibull_data():
    """Small Weibull-distributed failure dataset."""
    rng = np.random.default_rng(42)
    return rng.weibull(2.0, size=30) * 100.0


@pytest.fixture
def weibull_data_with_censored():
    """Weibull failures + right-censored observations."""
    rng = np.random.default_rng(42)
    failures = rng.weibull(2.0, size=20) * 100.0
    censored = rng.weibull(2.0, size=10) * 120.0
    return failures, censored


@pytest.fixture
def normal_data():
    rng = np.random.default_rng(7)
    return rng.normal(loc=100.0, scale=15.0, size=30)


@pytest.fixture
def lognormal_data():
    rng = np.random.default_rng(13)
    return np.exp(rng.normal(loc=4.0, scale=0.5, size=30))


@pytest.fixture
def exponential_data():
    rng = np.random.default_rng(99)
    return rng.exponential(scale=200.0, size=30)
