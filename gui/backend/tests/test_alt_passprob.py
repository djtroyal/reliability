"""Tests for the Poisson pass-probability helper and endpoint."""

import math
import sys
from pathlib import Path

import pytest

# Make the backend package importable when running from the tests/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routers.alt import _poisson_pass_prob


# ─── Unit tests for the helper ────────────────────────────────────────────────

def test_zero_failures_lam_one():
    """c=0, lam=1 → P(pass) = e^{-1} ≈ 0.3679."""
    result = _poisson_pass_prob(lam=1.0, c=0)
    assert abs(result - math.exp(-1)) < 1e-9


def test_one_failure_lam_one():
    """c=1, lam=1 → P(pass) = e^{-1}*(1+1) ≈ 0.7358."""
    expected = math.exp(-1) * (1 + 1)
    result = _poisson_pass_prob(lam=1.0, c=1)
    assert abs(result - expected) < 1e-9


def test_very_high_mtbf_passes():
    """true_mtbf >> T ⟹ lam → 0 ⟹ P(pass) → 1."""
    # T=1000, true_mtbf=1e9 → lam=1e-6
    lam = 1000.0 / 1e9
    result = _poisson_pass_prob(lam=lam, c=0)
    assert result > 0.9999


def test_very_low_mtbf_fails():
    """true_mtbf << T ⟹ lam → ∞ ⟹ P(pass) → 0."""
    # T=1000, true_mtbf=0.001 → lam=1e6
    lam = 1000.0 / 0.001
    result = _poisson_pass_prob(lam=lam, c=0)
    assert result < 1e-6


def test_result_is_clamped_to_unit_interval():
    """Helper must always return a value in [0, 1]."""
    for lam, c in [(0.0, 0), (0.0, 5), (1e9, 0), (1.0, 100)]:
        r = _poisson_pass_prob(lam=lam, c=c)
        assert 0.0 <= r <= 1.0, f"lam={lam}, c={c} → {r}"


def test_lam_zero_returns_one():
    """lam=0 (effectively infinite MTBF) → always pass."""
    assert _poisson_pass_prob(lam=0.0, c=0) == 1.0


def test_negative_c_returns_zero():
    """c<0 is impossible; helper returns 0."""
    assert _poisson_pass_prob(lam=1.0, c=-1) == 0.0


def test_multiple_failures_allowed():
    """c=5, lam=1: manual Poisson CDF check."""
    expected = sum(math.exp(-1) * (1.0 ** k) / math.factorial(k) for k in range(6))
    result = _poisson_pass_prob(lam=1.0, c=5)
    assert abs(result - expected) < 1e-9


def test_t_equals_mtbf_c0():
    """When T == true_mtbf, lam=1, c=0 → P(pass) = e^{-1}."""
    # Consistent with test_zero_failures_lam_one but expressed in MTBF terms.
    T, M = 1000.0, 1000.0
    lam = T / M
    result = _poisson_pass_prob(lam=lam, c=0)
    assert abs(result - math.exp(-1)) < 1e-9
