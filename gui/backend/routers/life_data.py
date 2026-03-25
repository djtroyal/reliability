"""Life Data Analysis router."""

import sys
import warnings
import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from reliability.Fitters import (
    Fit_Everything, _FITTER_MAP, ALL_FITTER_NAMES,
    Fit_Weibull_2P, Fit_Weibull_3P,
    Fit_Exponential_1P, Fit_Exponential_2P,
    Fit_Normal_2P, Fit_Lognormal_2P, Fit_Lognormal_3P,
    Fit_Gamma_2P, Fit_Loglogistic_2P, Fit_Beta_2P, Fit_Gumbel_2P,
)
from reliability.Nonparametric import KaplanMeier, NelsonAalen
from reliability.Utils import xy_transform
from schemas import LifeDataFitRequest, NonparametricRequest

router = APIRouter()


def _dist_params(fit, name: str) -> dict:
    """Extract fitted parameters as a flat dict."""
    params = {}
    for attr in ('alpha', 'beta', 'gamma', 'mu', 'sigma', 'Lambda', 'alpha_1', 'alpha_2'):
        if hasattr(fit, attr):
            val = getattr(fit, attr)
            if val is not None:
                params[attr] = round(float(val), 6)
    return params


def _distribution_curves(dist, failures: np.ndarray) -> dict:
    """Generate PDF, CDF, SF, HF curve data for a fitted distribution."""
    lo = failures.min() * 0.5
    hi = failures.max() * 1.5
    x = np.linspace(max(lo, 1e-6), hi, 300)

    # Some distributions have support constraints
    if hasattr(dist, 'gamma') and dist.gamma is not None:
        x = x[x > dist.gamma]
    if len(x) == 0:
        return {}

    return {
        "x": x.tolist(),
        "pdf": dist._pdf(x).tolist(),
        "cdf": dist._cdf(x).tolist(),
        "sf": dist._sf(x).tolist(),
        "hf": dist._hf(x).tolist(),
    }


def _probability_plot_data(fit, name: str, failures: np.ndarray,
                           right_censored) -> dict:
    """Return probability plot scatter + fitted line data."""
    from reliability.Utils import rank_adjustment, median_rank_approximation

    rc = np.asarray(right_censored, dtype=float) if right_censored else None
    adj_ranks, n = rank_adjustment(failures, rc)
    median_ranks = median_rank_approximation(adj_ranks, n)
    sorted_f = np.sort(failures)
    median_ranks = np.clip(median_ranks, 1e-10, 1 - 1e-10)

    x_transform, y_transform, x_label, y_label = xy_transform(name)
    x_pts = x_transform(sorted_f).tolist()
    y_pts = y_transform(median_ranks).tolist()

    # Fitted line
    x_line_raw = np.linspace(sorted_f.min() * 0.8, sorted_f.max() * 1.2, 200)
    x_line_raw = x_line_raw[x_line_raw > 0]
    cdf_vals = np.clip(fit.distribution._cdf(x_line_raw), 1e-10, 1 - 1e-10)
    x_line = x_transform(x_line_raw).tolist()
    y_line = y_transform(cdf_vals).tolist()

    return {
        "scatter_x": x_pts,
        "scatter_y": y_pts,
        "line_x": x_line,
        "line_y": y_line,
        "x_label": x_label,
        "y_label": y_label,
    }


@router.post("/fit")
def fit_distributions(req: LifeDataFitRequest):
    failures = np.asarray(req.failures, dtype=float)
    rc = np.asarray(req.right_censored, dtype=float) if req.right_censored else None

    if len(failures) < 2:
        raise HTTPException(status_code=400, detail="At least 2 failure times required.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fe = Fit_Everything(
                failures=failures,
                right_censored=rc,
                distributions_to_fit=req.distributions_to_fit,
                method=req.method,
                show_probability_plot=False,
                show_histogram_plot=False,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build results rows
    results = []
    for _, row in fe.results.iterrows():
        dist_name = row["Distribution"]
        entry = {
            "Distribution": dist_name,
            "AICc": None if np.isinf(row["AICc"]) else round(float(row["AICc"]), 4),
            "BIC": None if np.isinf(row["BIC"]) else round(float(row["BIC"]), 4),
            "AD": None if np.isinf(row["AD"]) else round(float(row["AD"]), 4),
            "LogLik": round(float(row["Log-Likelihood"]), 4),
        }
        if dist_name in fe.fitted:
            entry["params"] = _dist_params(fe.fitted[dist_name], dist_name)
        results.append(entry)

    # Plot data for best distribution
    best_name = fe.best_distribution_name
    best_fit = fe.fitted.get(best_name)
    plots = {}
    if best_fit and best_fit.distribution:
        try:
            plots["probability"] = _probability_plot_data(
                best_fit, best_name, failures, req.right_censored)
            plots["curves"] = _distribution_curves(best_fit.distribution, failures)
        except Exception:
            pass

    return {
        "results": results,
        "best_distribution": best_name,
        "plots": plots,
        "available_distributions": list(ALL_FITTER_NAMES),
    }


@router.post("/nonparametric")
def nonparametric(req: NonparametricRequest):
    failures = np.asarray(req.failures, dtype=float)
    rc = np.asarray(req.right_censored, dtype=float) if req.right_censored else None

    if len(failures) < 1:
        raise HTTPException(status_code=400, detail="At least 1 failure time required.")

    try:
        if req.method == "KM":
            est = KaplanMeier(failures, right_censored=rc, CI=req.CI)
            df = est.results
            return {
                "method": "Kaplan-Meier",
                "time": df["time"].tolist(),
                "SF": df["SF"].tolist(),
                "CI_lower": df["CI_lower"].tolist(),
                "CI_upper": df["CI_upper"].tolist(),
            }
        else:
            est = NelsonAalen(failures, right_censored=rc, CI=req.CI)
            df = est.results
            return {
                "method": "Nelson-Aalen",
                "time": df["time"].tolist(),
                "CHF": df["CHF"].tolist(),
                "SF": df["SF"].tolist(),
                "CI_lower": df["CI_lower"].tolist(),
                "CI_upper": df["CI_upper"].tolist(),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distributions")
def list_distributions():
    return {"distributions": list(ALL_FITTER_NAMES)}
