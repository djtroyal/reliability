"""Known-answer tests for SPC control charts."""

import math

import numpy as np
import pytest

from reliability.SPC import (
    control_chart, i_mr_chart, xbar_r_chart, xbar_s_chart,
    p_chart, np_chart, c_chart, u_chart,
)


def test_i_mr_center_lines():
    x = [10, 12, 11, 13, 12, 11, 10, 14, 12, 11]
    res = i_mr_chart(x)
    ind = res["subcharts"][0]
    mr = res["subcharts"][1]
    assert abs(ind["cl"] - np.mean(x)) < 1e-9
    expected_mrbar = np.mean(np.abs(np.diff(x)))
    assert abs(mr["cl"] - expected_mrbar) < 1e-9
    # I-chart limits = xbar +/- 3*(MRbar/1.128)
    sigma = expected_mrbar / 1.128
    assert abs(ind["ucl"] - (np.mean(x) + 3 * sigma)) < 1e-6
    # MR LCL is 0 for n=2 (D3=0).
    assert mr["lcl"] == 0.0


def test_xbar_r_limits():
    data = [[2, 3, 4], [3, 4, 5], [2, 2, 3], [4, 5, 6], [3, 3, 4]]
    res = xbar_r_chart(data)
    xbar = res["subcharts"][0]
    rng = res["subcharts"][1]
    means = [np.mean(g) for g in data]
    ranges = [max(g) - min(g) for g in data]
    assert abs(xbar["cl"] - np.mean(means)) < 1e-9
    rbar = np.mean(ranges)
    assert abs(rng["cl"] - rbar) < 1e-9
    A2 = 1.023  # n=3
    assert abs(xbar["ucl"] - (np.mean(means) + A2 * rbar)) < 1e-6
    D4 = 2.574
    assert abs(rng["ucl"] - D4 * rbar) < 1e-6


def test_xbar_s_limits():
    data = [[2, 3, 4, 5], [3, 4, 5, 6], [2, 2, 3, 4], [4, 5, 6, 7]]
    res = xbar_s_chart(data)
    s = res["subcharts"][1]
    stds = [np.std(g, ddof=1) for g in data]
    assert abs(s["cl"] - np.mean(stds)) < 1e-9


def test_p_chart_center():
    counts = [3, 5, 2, 4, 6]
    sizes = [100, 100, 100, 100, 100]
    res = p_chart(counts, sizes)
    assert abs(res["center"] - sum(counts) / sum(sizes)) < 1e-12
    sub = res["subcharts"][0]
    assert len(sub["points"]) == 5
    assert all(l >= 0 for l in sub["lcl"])


def test_np_chart_center():
    counts = [3, 5, 2, 4, 6]
    res = np_chart(counts, 100)
    pbar = sum(counts) / (100 * len(counts))
    assert abs(res["center"] - 100 * pbar) < 1e-9


def test_c_chart_center():
    counts = [4, 5, 3, 6, 2, 5]
    res = c_chart(counts)
    cbar = np.mean(counts)
    assert abs(res["center"] - cbar) < 1e-9
    sub = res["subcharts"][0]
    assert abs(sub["ucl"] - (cbar + 3 * math.sqrt(cbar))) < 1e-6


def test_u_chart_variable_sizes():
    counts = [5, 8, 3, 10]
    sizes = [10, 12, 8, 15]
    res = u_chart(counts, sizes)
    ubar = sum(counts) / sum(sizes)
    assert abs(res["center"] - ubar) < 1e-12
    sub = res["subcharts"][0]
    # Variable sizes => variable (per-point) limits.
    assert isinstance(sub["ucl"], list)


def test_out_of_control_detected():
    x = [10, 10, 10, 10, 10, 10, 50]  # clear spike
    res = i_mr_chart(x)
    viols = res["subcharts"][0]["violations"]
    assert any(v["rule"] == 1 for v in viols)


def test_dispatch():
    assert control_chart("c", [4, 5, 3, 6])["chart"] == "c"
    with pytest.raises(ValueError):
        control_chart("nonexistent", [1, 2, 3])
