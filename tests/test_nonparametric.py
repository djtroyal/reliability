"""Tests for reliability.Nonparametric."""

import numpy as np
import pytest
from reliability.Nonparametric import KaplanMeier, NelsonAalen


@pytest.fixture
def simple_data():
    return np.array([5.0, 10.0, 15.0, 20.0, 25.0])


@pytest.fixture
def censored_data():
    return np.array([5.0, 20.0, 25.0]), np.array([8.0, 12.0])


# --- KaplanMeier ---

def test_km_starts_at_one(simple_data):
    km = KaplanMeier(simple_data)
    assert km.results['SF'].iloc[0] == pytest.approx(1.0)


def test_km_decreasing(simple_data):
    km = KaplanMeier(simple_data)
    sf = km.results['SF'].values
    assert np.all(np.diff(sf) <= 0)


def test_km_results_columns(simple_data):
    km = KaplanMeier(simple_data)
    assert set(km.results.columns) >= {'time', 'SF', 'CI_lower', 'CI_upper'}


def test_km_ci_bounds(simple_data):
    km = KaplanMeier(simple_data)
    assert np.all(km.results['CI_lower'] >= 0)
    assert np.all(km.results['CI_upper'] <= 1)
    assert np.all(km.results['CI_lower'] <= km.results['SF'])
    assert np.all(km.results['CI_upper'] >= km.results['SF'])


def test_km_with_censored(censored_data):
    failures, censored = censored_data
    km = KaplanMeier(failures, right_censored=censored)
    sf = km.results['SF'].values
    assert sf[0] == pytest.approx(1.0)
    assert np.all(np.diff(sf) <= 0)


def test_km_single_failure():
    km = KaplanMeier([10.0])
    assert len(km.results) == 2  # t=0 and t=10


def test_km_ci_width_increases_with_time(simple_data):
    km = KaplanMeier(simple_data)
    widths = km.results['CI_upper'] - km.results['CI_lower']
    # CI width should not decrease (generally increases or stays same)
    assert widths.iloc[-1] >= widths.iloc[0]


# --- NelsonAalen ---

def test_na_starts_at_zero(simple_data):
    na = NelsonAalen(simple_data)
    assert na.results['CHF'].iloc[0] == pytest.approx(0.0)


def test_na_increasing(simple_data):
    na = NelsonAalen(simple_data)
    chf = na.results['CHF'].values
    assert np.all(np.diff(chf) >= 0)


def test_na_has_sf(simple_data):
    na = NelsonAalen(simple_data)
    assert 'SF' in na.results.columns
    sf = na.results['SF'].values
    # SF = exp(-CHF), should be in [0,1]
    assert np.all(sf >= 0) and np.all(sf <= 1)


def test_na_sf_decreasing(simple_data):
    na = NelsonAalen(simple_data)
    sf = na.results['SF'].values
    assert np.all(np.diff(sf) <= 0)


def test_na_with_censored(censored_data):
    failures, censored = censored_data
    na = NelsonAalen(failures, right_censored=censored)
    assert na.results['CHF'].iloc[0] == pytest.approx(0.0)


def test_na_results_columns(simple_data):
    na = NelsonAalen(simple_data)
    assert set(na.results.columns) >= {'time', 'CHF', 'CI_lower', 'CI_upper', 'SF'}
