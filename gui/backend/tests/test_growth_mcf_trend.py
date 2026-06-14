"""Tests for the MCF trend classification helper in growth.py."""

import sys
from pathlib import Path

# Ensure the routers package is importable without a full FastAPI app context.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from routers.growth import _mcf_trend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_mcf(t_start, t_end, slope, n=20):
    """Return (times, mcf_vals) for a perfectly linear MCF segment."""
    times = np.linspace(t_start, t_end, n)
    mcf = slope * (times - t_start)
    return times.tolist(), mcf.tolist()


def _two_half_mcf(slope1, slope2, n_each=15):
    """Stitch two linear segments together to form a complete MCF series."""
    t1 = np.linspace(0, 50, n_each)
    m1 = slope1 * t1

    t2 = np.linspace(50, 100, n_each + 1)[1:]   # avoid duplicate at 50
    # Second half continues from the last MCF value of the first half.
    m2 = m1[-1] + slope2 * (t2 - t2[0])

    times = np.concatenate([t1, t2]).tolist()
    mcf = np.concatenate([m1, m2]).tolist()
    return times, mcf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_improving():
    """A clearly decreasing recurrence rate should be classified as improving."""
    # First-half slope >> second-half slope → ratio well below 0.85
    times, mcf = _two_half_mcf(slope1=0.10, slope2=0.04)
    result = _mcf_trend(times, mcf)
    assert result["trend"] == "improving"
    assert "improving" in result["detail"]
    assert "decreased" in result["detail"]


def test_worsening():
    """A clearly increasing recurrence rate should be classified as worsening."""
    # Second-half slope >> first-half slope → ratio well above 1.15
    times, mcf = _two_half_mcf(slope1=0.04, slope2=0.10)
    result = _mcf_trend(times, mcf)
    assert result["trend"] == "worsening"
    assert "worsening" in result["detail"]
    assert "increased" in result["detail"]


def test_constant():
    """Identical slopes in both halves should be classified as constant."""
    times, mcf = _two_half_mcf(slope1=0.06, slope2=0.06)
    result = _mcf_trend(times, mcf)
    assert result["trend"] == "constant"
    assert "constant" in result["detail"]


def test_short():
    """Fewer than 4 points should return constant with an 'Insufficient' detail."""
    result = _mcf_trend([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert result["trend"] == "constant"
    assert "Insufficient" in result["detail"]

    # Edge: exactly 0 points
    result_empty = _mcf_trend([], [])
    assert result_empty["trend"] == "constant"
    assert "Insufficient" in result_empty["detail"]
