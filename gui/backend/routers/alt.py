"""Accelerated Life Testing router."""

import math
import sys
import warnings
import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from reliability.ALT_fitters import Fit_Everything_ALT, ALL_SINGLE_STRESS_NAMES
from reliability.Reliability_testing import (
    sample_size_binomial, parametric_binomial_sample_size,
    parametric_binomial_test_time, binomial_oc_curve,
    one_sample_proportion, two_proportion_test, sample_size_no_failures,
    sequential_sampling_chart, reliability_test_planner,
    reliability_test_duration, chi_squared_test, KS_test,
)
from reliability.Fitters import (
    _FITTER_MAP, Fit_Weibull_2P, Fit_Normal_2P, Fit_Lognormal_2P,
)
from schemas import (
    ALTFitRequest, SampleSizeRequest, AccelerationFactorRequest,
    OneSampleProportionRequest, TwoProportionRequest, NoFailuresRequest,
    SequentialSamplingRequest, TestPlannerRequest, TestDurationRequest,
    GoodnessOfFitRequest, PassProbRequest,
    StepStressRequest, HALTRequest, MarginTestRequest, MultiStressRequest,
    DegradationRequest, ESSRequest, HASSRequest, BurnInRequest,
)

router = APIRouter()


def _poisson_pass_prob(lam: float, c: int) -> float:
    """P(X <= c) where X ~ Poisson(lam). Returns a float in [0, 1]."""
    if lam <= 0.0:
        return 1.0
    if c < 0:
        return 0.0
    try:
        from scipy.stats import poisson as _poisson
        result = float(_poisson.cdf(c, lam))
    except ImportError:
        # Manual Poisson CDF sum
        exp_neg_lam = math.exp(-lam)
        total = 0.0
        power = 1.0
        for k in range(c + 1):
            if k > 0:
                power *= lam / k
            total += exp_neg_lam * power
        result = total
    # Guard against floating-point drift outside [0, 1]
    return max(0.0, min(1.0, result))


@router.post("/pass-probability")
def pass_probability(req: PassProbRequest):
    """Probability of passing a Poisson-model reliability demonstration test."""
    if req.test_duration <= 0:
        raise HTTPException(status_code=400, detail="test_duration must be > 0.")
    if req.true_mtbf <= 0:
        raise HTTPException(status_code=400, detail="true_mtbf must be > 0.")
    if req.allowable_failures < 0:
        raise HTTPException(status_code=400, detail="allowable_failures must be >= 0.")

    lam = req.test_duration / req.true_mtbf
    p_pass = _poisson_pass_prob(lam, req.allowable_failures)

    oc_curve = None
    if req.oc_mtbf_min is not None and req.oc_mtbf_max is not None:
        if req.oc_mtbf_min <= 0 or req.oc_mtbf_max <= 0:
            raise HTTPException(status_code=400, detail="OC curve MTBF bounds must be > 0.")
        mtbf_vals = np.linspace(req.oc_mtbf_min, req.oc_mtbf_max, max(2, req.oc_points))
        p_pass_vals = []
        for m in mtbf_vals:
            lam_i = req.test_duration / float(m)
            p_i = _poisson_pass_prob(lam_i, req.allowable_failures)
            p_pass_vals.append(p_i if math.isfinite(p_i) else None)
        oc_curve = {
            "mtbf": mtbf_vals.tolist(),
            "p_pass": p_pass_vals,
        }

    return {
        "test_duration": req.test_duration,
        "allowable_failures": req.allowable_failures,
        "true_mtbf": req.true_mtbf,
        "lambda": lam,
        "p_pass": p_pass,
        "oc_curve": oc_curve,
    }


@router.post("/one-sample-proportion")
def one_sample_proportion_ep(req: OneSampleProportionRequest):
    try:
        return one_sample_proportion(req.trials, req.successes, CI=req.CI)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/two-proportion-test")
def two_proportion_ep(req: TwoProportionRequest):
    try:
        return two_proportion_test(req.trials_1, req.successes_1,
                                   req.trials_2, req.successes_2, CI=req.CI)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sample-size-no-failures")
def no_failures_ep(req: NoFailuresRequest):
    try:
        return sample_size_no_failures(req.reliability, CI=req.CI,
                                       lifetimes=req.lifetimes,
                                       weibull_shape=req.weibull_shape)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sequential-sampling")
def sequential_sampling_ep(req: SequentialSamplingRequest):
    try:
        return sequential_sampling_chart(req.p1, req.p2, req.alpha, req.beta,
                                         max_samples=req.max_samples)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/test-planner")
def test_planner_ep(req: TestPlannerRequest):
    try:
        return reliability_test_planner(
            MTBF=req.MTBF, test_duration=req.test_duration,
            number_of_failures=req.number_of_failures,
            CI=req.CI, two_sided=req.two_sided)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/test-duration")
def test_duration_ep(req: TestDurationRequest):
    try:
        return reliability_test_duration(
            req.MTBF_required, req.MTBF_design,
            req.consumer_risk, req.producer_risk)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/goodness-of-fit")
def goodness_of_fit_ep(req: GoodnessOfFitRequest):
    """Fit the chosen distribution, then run a chi-squared or KS GoF test."""
    if req.distribution not in _FITTER_MAP:
        raise HTTPException(status_code=400,
                            detail=f"Unknown distribution '{req.distribution}'.")
    failures = np.asarray(req.failures, dtype=float)
    if len(failures) < 5:
        raise HTTPException(status_code=400,
                            detail="At least 5 failures are required.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = _FITTER_MAP[req.distribution](failures=failures)
        dist = fit.distribution
        if req.test == "ks":
            res = KS_test(dist, failures, CI=req.CI)
            res["test"] = "Kolmogorov-Smirnov"
        else:
            res = chi_squared_test(dist, failures, CI=req.CI)
            res["test"] = "Chi-squared"
        res["distribution"] = req.distribution
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fit")
def fit_alt(req: ALTFitRequest):
    failures = np.asarray(req.failures, dtype=float)
    stresses = np.asarray(req.failure_stress, dtype=float)
    rc = np.asarray(req.right_censored, dtype=float) if req.right_censored else None
    rc_stress = np.asarray(req.right_censored_stress, dtype=float) if req.right_censored_stress else None

    if len(failures) < 4:
        raise HTTPException(status_code=400, detail="At least 4 failure times required for ALT.")

    if len(failures) != len(stresses):
        raise HTTPException(status_code=400,
                            detail="failures and failure_stress must have the same length.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fe = Fit_Everything_ALT(
                failures=failures,
                failure_stress=stresses,
                right_censored=rc,
                right_censored_stress=rc_stress,
                use_level_stress=req.use_level_stress,
                models_to_fit=req.models_to_fit,
                sort_by=req.sort_by,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for _, row in fe.results.iterrows():
        entry = {k: (None if (isinstance(v, float) and np.isinf(v)) else
                     (round(v, 4) if isinstance(v, float) else v))
                 for k, v in row.items()}
        results.append(entry)

    # Life-stress plot data for best model
    life_stress_plot = None
    if fe.best_model is not None:
        try:
            model = fe.best_model
            unique_stresses = np.unique(stresses)
            s_range = np.linspace(unique_stresses.min() * 0.8,
                                  unique_stresses.max() * 1.2, 100)
            life_at_stress = []
            for s in s_range:
                try:
                    # life = characteristic life at stress s
                    params = model.life_stress_params
                    dist_info_life = model._life_at_stress(s)
                    life_at_stress.append(float(dist_info_life))
                except Exception:
                    life_at_stress.append(None)

            # Observed median lives per stress level
            obs_stress = []
            obs_life = []
            for s in unique_stresses:
                mask = stresses == s
                median_life = float(np.median(failures[mask]))
                obs_stress.append(float(s))
                obs_life.append(median_life)

            life_stress_plot = {
                "line_stress": s_range.tolist(),
                "line_life": life_at_stress,
                "scatter_stress": obs_stress,
                "scatter_life": obs_life,
                "use_level_stress": req.use_level_stress,
                "use_level_life": (float(model._life_at_stress(req.use_level_stress))
                                   if req.use_level_stress else None),
            }
        except Exception:
            pass

    return {
        "results": results,
        "best_model": fe.best_model_name,
        "life_stress_plot": life_stress_plot,
        "available_models": list(ALL_SINGLE_STRESS_NAMES),
    }


@router.get("/models")
def list_models():
    return {"models": list(ALL_SINGLE_STRESS_NAMES)}


@router.post("/sample-size")
def sample_size(req: SampleSizeRequest):
    """Binomial reliability demonstration test planner.

    Method 1 (nonparametric) solves the binomial equation for sample size.
    Method 2A/2B use a Weibull shape parameter to trade test time against
    samples; both report eta and the reliability demonstrated at test time.
    """
    parametric = req.method in ("parametric_samples", "parametric_time")
    if parametric and (req.mission_time is None or req.beta is None):
        raise HTTPException(status_code=400,
                            detail="mission_time and beta are required for parametric methods.")

    try:
        out = {"method": req.method, "failures": req.failures,
               "R": req.R, "CI": req.CI,
               "n": None, "test_time": None, "eta": None, "R_test": None}

        if req.method == "nonparametric":
            out["n"] = sample_size_binomial(req.R, CI=req.CI, failures=req.failures)
            oc_n, demonstrated_R = out["n"], req.R
        elif req.method == "parametric_samples":
            if req.test_time is None:
                raise ValueError("test_time is required for Method 2A.")
            res = parametric_binomial_sample_size(
                req.R, req.mission_time, req.beta, req.test_time,
                CI=req.CI, failures=req.failures)
            out.update(n=res["n"], eta=round(res["eta"], 4),
                       R_test=round(res["R_test"], 6))
            oc_n, demonstrated_R = res["n"], res["R_test"]
        elif req.method == "parametric_time":
            if req.n is None:
                raise ValueError("n is required for Method 2B.")
            res = parametric_binomial_test_time(
                req.R, req.mission_time, req.beta, req.n,
                CI=req.CI, failures=req.failures)
            out.update(n=req.n, test_time=round(res["T_test"], 4),
                       eta=round(res["eta"], 4), R_test=round(res["R_test"], 6))
            oc_n, demonstrated_R = req.n, res["R_test"]
        else:
            raise ValueError(f"Unknown method: '{req.method}'")

        if req.options_table:
            rows = []
            for f in range(16):
                row = {"f": f}
                try:
                    if req.method == "nonparametric":
                        row["n"] = sample_size_binomial(req.R, CI=req.CI, failures=f)
                    elif req.method == "parametric_samples":
                        row["n"] = parametric_binomial_sample_size(
                            req.R, req.mission_time, req.beta, req.test_time,
                            CI=req.CI, failures=f)["n"]
                    else:  # parametric_time: fixed n, vary f (needs n > f)
                        if req.n >= f + 1:
                            row["test_time"] = round(parametric_binomial_test_time(
                                req.R, req.mission_time, req.beta, req.n,
                                CI=req.CI, failures=f)["T_test"], 2)
                        else:
                            row["test_time"] = None
                except ValueError:
                    row["n" if req.method != "parametric_time" else "test_time"] = None
                rows.append(row)
            out["options_table"] = rows

        if req.oc_curve:
            R_vals, P_acc = binomial_oc_curve(oc_n, failures=req.failures)
            out["oc_curve"] = {
                "R": np.round(R_vals, 6).tolist(),
                "P_accept": np.round(P_acc, 6).tolist(),
                "R_demonstrated": round(float(demonstrated_R), 6),
                "alpha": round(1 - req.CI, 6),
            }

        return out
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acceleration-factor")
def acceleration_factor(req: AccelerationFactorRequest):
    """Compute acceleration factor between test and use stress levels."""
    model = req.model.lower()
    s_test = req.stress_test
    s_use = req.stress_use

    if model == "arrhenius":
        Ea = float(req.params.get("Ea", 0.7))
        k = 8.617e-5
        T_test = s_test + 273.15
        T_use = s_use + 273.15
        if T_test <= 0 or T_use <= 0:
            raise HTTPException(status_code=400, detail="Temperatures must be > -273.15°C.")
        AF = float(np.exp((Ea / k) * (1.0 / T_use - 1.0 / T_test)))
    elif model == "inverse_power":
        n = float(req.params.get("n", 2))
        if s_use <= 0:
            raise HTTPException(status_code=400, detail="Use stress must be > 0.")
        AF = float((s_test / s_use) ** n)
    elif model == "eyring":
        A = float(req.params.get("A", 1))
        T_test = s_test + 273.15
        T_use = s_use + 273.15
        if T_test <= 0 or T_use <= 0:
            raise HTTPException(status_code=400, detail="Temperatures must be > -273.15°C.")
        AF = float(np.exp(A * (1.0 / T_use - 1.0 / T_test)))
    elif model == "coffin_manson":
        # Thermal-cycling fatigue: stress = thermal cycle range ΔT.
        n = float(req.params.get("n", 2.0))
        if s_use <= 0:
            raise HTTPException(status_code=400, detail="Use ΔT must be > 0.")
        AF = float((s_test / s_use) ** n)
    elif model == "peck":
        # Temperature-Humidity (Peck): stress_test/use are temperatures (°C).
        k = 8.617e-5
        n = float(req.params.get("n", 2.7))
        Ea = float(req.params.get("Ea", 0.79))
        RH_test = float(req.params.get("RH_test", 85.0))
        RH_use = float(req.params.get("RH_use", 40.0))
        T_test = s_test + 273.15
        T_use = s_use + 273.15
        if T_test <= 0 or T_use <= 0 or RH_use <= 0:
            raise HTTPException(status_code=400, detail="Invalid temperature/humidity inputs.")
        AF = float((RH_test / RH_use) ** n * np.exp((Ea / k) * (1.0 / T_use - 1.0 / T_test)))
    elif model == "norris_landzberg":
        # Solder-joint thermal cycling: stress_test/use are cycle ranges ΔT.
        k = 8.617e-5
        n = float(req.params.get("n", 1.9))
        m = float(req.params.get("m", 1.0 / 3.0))
        Ea = float(req.params.get("Ea", 0.122))
        f_test = float(req.params.get("f_test", 48.0))
        f_use = float(req.params.get("f_use", 2.0))
        Tmax_test = float(req.params.get("Tmax_test", 100.0)) + 273.15
        Tmax_use = float(req.params.get("Tmax_use", 60.0)) + 273.15
        if s_use <= 0 or f_test <= 0 or Tmax_use <= 0 or Tmax_test <= 0:
            raise HTTPException(status_code=400, detail="Invalid Norris-Landzberg inputs.")
        AF = float((s_test / s_use) ** n * (f_use / f_test) ** m
                   * np.exp((Ea / k) * (1.0 / Tmax_use - 1.0 / Tmax_test)))
    elif model == "black":
        # Electromigration (Black's equation): stress_test/use are temperatures (°C).
        k = 8.617e-5
        n = float(req.params.get("n", 2.0))
        Ea = float(req.params.get("Ea", 0.7))
        J_test = float(req.params.get("J_test", 2.0))
        J_use = float(req.params.get("J_use", 1.0))
        T_test = s_test + 273.15
        T_use = s_use + 273.15
        if T_test <= 0 or T_use <= 0 or J_use <= 0:
            raise HTTPException(status_code=400, detail="Invalid temperature/current-density inputs.")
        AF = float((J_test / J_use) ** n * np.exp((Ea / k) * (1.0 / T_use - 1.0 / T_test)))
    else:
        raise HTTPException(status_code=400,
                            detail=f"Unknown model '{model}'. Use: arrhenius, inverse_power, "
                                   "eyring, coffin_manson, peck, norris_landzberg, black.")

    return {
        "model": model,
        "stress_test": s_test,
        "stress_use": s_use,
        "acceleration_factor": round(AF, 4),
    }


# ---------------------------------------------------------------------------
# Life-distribution fitting helper (shared by degradation analysis)
# ---------------------------------------------------------------------------

def _fit_life_distribution(times, dist_name):
    """Fit a 2-parameter life distribution to projected failure times.

    Returns a dict with the fitted parameters, curve data and summary
    percentiles (mean, median, B10, B50). Raises ValueError on bad input.
    """
    arr = np.asarray([t for t in times if t is not None and np.isfinite(t) and t > 0],
                     dtype=float)
    if len(arr) < 2:
        raise ValueError("Need at least 2 valid projected failure times to fit a distribution.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dist_name == "Normal_2P":
            fit = Fit_Normal_2P(failures=arr, show_probability_plot=False)
            dist = fit.distribution
            params = {"mu": float(fit.mu), "sigma": float(fit.sigma)}
        elif dist_name == "Lognormal_2P":
            fit = Fit_Lognormal_2P(failures=arr, show_probability_plot=False)
            dist = fit.distribution
            params = {"mu": float(fit.mu), "sigma": float(fit.sigma)}
        else:  # Weibull_2P default
            fit = Fit_Weibull_2P(failures=arr, show_probability_plot=False)
            dist = fit.distribution
            params = {"alpha": float(fit.eta), "beta": float(fit.beta)}

    xmax = float(arr.max()) * 1.5
    xs = np.linspace(max(1e-6, float(arr.min()) * 0.3), xmax, 200)
    pdf = dist.PDF(xvals=xs, show_plot=False)
    cdf = dist.CDF(xvals=xs, show_plot=False)

    def _pct(p):
        """Time by which fraction p of the population has failed (inverse CDF)."""
        try:
            return float(dist.quantile(p))
        except Exception:
            return None

    summary = {
        "mean": float(dist.mean),
        "median": _pct(0.5),
        "B10": _pct(0.10),
        "B50": _pct(0.50),
    }
    return {
        "distribution": dist_name,
        "params": params,
        "curve_x": xs.tolist(),
        "pdf": np.asarray(pdf, dtype=float).tolist(),
        "cdf": np.asarray(cdf, dtype=float).tolist(),
        "summary": summary,
    }


@router.post("/degradation")
def degradation(req: DegradationRequest):
    """Wear-to-failure analysis: fit degradation paths, project failure times."""
    from scipy.optimize import curve_fit

    n = len(req.times)
    if not (len(req.unit_ids) == len(req.measurements) == n) or n < 2:
        raise HTTPException(status_code=400,
                            detail="unit_ids, times and measurements must be equal-length (>=2).")

    # Group measurements by unit, preserving order.
    groups: dict = {}
    for uid, t, m in zip(req.unit_ids, req.times, req.measurements):
        groups.setdefault(str(uid), {"t": [], "m": []})
        groups[str(uid)]["t"].append(float(t))
        groups[str(uid)]["m"].append(float(m))

    model = req.degradation_model
    thr = req.threshold

    def _fit_unit(t, y):
        """Fit the chosen degradation model; return (params, predict_fn, t_fail, r2)."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(t) < 2:
            return None
        try:
            if model == "exponential":
                f = lambda x, a, b: a * np.exp(b * x)
                p0 = [max(1e-6, y[0]), 0.001]
                popt, _ = curve_fit(f, t, y, p0=p0, maxfev=10000)
                a, b = popt
                pred = lambda x: a * np.exp(b * x)
                t_fail = math.log(thr / a) / b if (a > 0 and thr / a > 0 and b != 0) else None
            elif model == "power":
                f = lambda x, a, b: a * np.power(np.maximum(x, 1e-9), b)
                popt, _ = curve_fit(f, t, y, p0=[max(1e-6, y[0]), 1.0], maxfev=10000)
                a, b = popt
                pred = lambda x: a * np.power(np.maximum(x, 1e-9), b)
                t_fail = (thr / a) ** (1.0 / b) if (a != 0 and thr / a > 0 and b != 0) else None
            elif model == "logarithmic":
                f = lambda x, a, b: a + b * np.log(np.maximum(x, 1e-9))
                popt, _ = curve_fit(f, t, y, p0=[y[0], 1.0], maxfev=10000)
                a, b = popt
                pred = lambda x: a + b * np.log(np.maximum(x, 1e-9))
                t_fail = math.exp((thr - a) / b) if b != 0 else None
            else:  # linear
                b, a = np.polyfit(t, y, 1)  # slope, intercept
                pred = lambda x: a + b * x
                t_fail = (thr - a) / b if b != 0 else None
            yhat = pred(t)
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            if t_fail is not None and (not np.isfinite(t_fail) or t_fail <= 0):
                t_fail = None
            return pred, t_fail, r2
        except Exception:
            return None

    paths = []
    projected = []
    unit_table = []
    for uid, g in groups.items():
        res = _fit_unit(g["t"], g["m"])
        path = {"unit_id": uid, "t": g["t"], "m": g["m"], "fit_t": None, "fit_m": None}
        if res is not None:
            pred, t_fail, r2 = res
            tmax = max(g["t"]) if g["t"] else 1.0
            extend_to = max(tmax, t_fail if t_fail else tmax) * 1.05
            xs = np.linspace(min(g["t"]), extend_to, 50)
            path["fit_t"] = xs.tolist()
            path["fit_m"] = np.asarray(pred(xs), dtype=float).tolist()
            if t_fail is not None:
                projected.append(t_fail)
            unit_table.append({"unit_id": uid,
                               "projected_failure": (round(t_fail, 4) if t_fail else None),
                               "r2": round(r2, 4)})
        else:
            unit_table.append({"unit_id": uid, "projected_failure": None, "r2": None})
        paths.append(path)

    dist_fit = None
    if len(projected) >= 2:
        try:
            dist_fit = _fit_life_distribution(projected, req.life_distribution)
        except ValueError:
            dist_fit = None

    return {
        "paths": paths,
        "threshold": thr,
        "threshold_direction": req.threshold_direction,
        "projected_failure_times": [round(p, 4) for p in projected],
        "distribution_fit": dist_fit,
        "unit_table": unit_table,
    }


# ---------------------------------------------------------------------------
# Step-Stress ALT — cumulative exposure model
# ---------------------------------------------------------------------------

@router.post("/step-stress")
def step_stress(req: StepStressRequest):
    """Step-stress ALT via the cumulative-exposure (Nelson) model.

    Each step has a stress and duration. Failures observed at their step's
    stress are converted to an equivalent time at a reference (lowest) stress
    using a log-linear life-stress relationship, then a life distribution is
    fitted to those equivalent times.
    """
    steps = req.steps
    if not steps or len(steps) < 2:
        raise HTTPException(status_code=400, detail="At least 2 steps are required.")
    ft = np.asarray(req.failure_times, dtype=float)
    sf = np.asarray(req.stress_at_failure, dtype=float)
    if len(ft) != len(sf) or len(ft) < 2:
        raise HTTPException(status_code=400,
                            detail="failure_times and stress_at_failure must be equal-length (>=2).")

    stresses = [float(s["stress"]) for s in steps]
    durations = [float(s["duration"]) for s in steps]
    ref_stress = min(stresses)

    # Acceleration factor between a stress and the reference, using an
    # inverse-power style relationship AF = (S/S_ref)^p with p estimated from
    # the spread of observed lives across stress levels (fallback p=2).
    uniq = np.unique(sf)
    p = 2.0
    if len(uniq) >= 2:
        med = np.array([np.median(ft[sf == s]) for s in uniq])
        with np.errstate(all="ignore"):
            ls = np.log(uniq / ref_stress)
            lm = np.log(med / med.max())
            denom = float(np.sum(ls * ls))
            if denom > 1e-9:
                p = float(abs(np.sum(ls * lm) / denom))
    if not np.isfinite(p) or p <= 0:
        p = 2.0

    def af(s):
        return (s / ref_stress) ** p if s > 0 else 1.0

    # Cumulative-exposure: equivalent time at reference stress.
    # Time accumulated in prior steps plus time-in-current-step, each scaled
    # by the step's acceleration factor.
    cum_start = {}
    acc = 0.0
    for s, d in zip(stresses, durations):
        cum_start[s] = acc
        acc += d

    equiv = []
    for t, s in zip(ft, sf):
        # equivalent time = AF(s) * (failure time measured within its step window)
        # Approximate the within-step time as t minus the start of that step.
        step_start = cum_start.get(float(s), 0.0)
        in_step = max(0.0, t - step_start)
        prior = step_start  # already-elapsed real time at lower steps
        # Convert prior real-time at the reference baseline (already ref) + accelerated in-step
        equiv.append(prior + af(float(s)) * in_step)
    equiv = np.asarray(equiv, dtype=float)

    dist_map = {"Weibull": "Weibull_2P", "Normal": "Normal_2P", "Lognormal": "Lognormal_2P"}
    dist_name = dist_map.get(req.distribution, "Weibull_2P")
    try:
        fit = _fit_life_distribution(equiv.tolist(), dist_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Cumulative-failure step plot.
    order = np.argsort(ft)
    sorted_t = ft[order]
    cum = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    step_boundaries = list(np.cumsum(durations)[:-1])

    return {
        "exponent_p": round(p, 4),
        "ref_stress": ref_stress,
        "equivalent_times": [round(float(x), 4) for x in equiv],
        "distribution_fit": fit,
        "cumulative_plot": {
            "time": sorted_t.tolist(),
            "cum_fraction": cum.tolist(),
            "step_boundaries": step_boundaries,
        },
        "use_level_stress": req.use_level_stress,
    }


# ---------------------------------------------------------------------------
# HALT — operating / destruct margin determination
# ---------------------------------------------------------------------------

@router.post("/halt")
def halt(req: HALTRequest):
    """Find operating and destruct limits / margins from a HALT step search."""
    levels = np.asarray(req.stress_levels, dtype=float)
    outcomes = [o.lower() for o in req.outcomes]
    if len(levels) != len(outcomes) or len(levels) == 0:
        raise HTTPException(status_code=400,
                            detail="stress_levels and outcomes must be equal-length and non-empty.")

    order = np.argsort(levels)
    levels = levels[order]
    outcomes = [outcomes[i] for i in order]

    operating_limit = None   # first anomaly
    destruct_limit = None     # first hard fail
    for lvl, out in zip(levels, outcomes):
        if out in ("anomaly", "degraded") and operating_limit is None:
            operating_limit = float(lvl)
        if out in ("fail", "destruct") and destruct_limit is None:
            destruct_limit = float(lvl)
    # If a hard failure occurred without a prior anomaly, treat it as the operating limit too.
    if operating_limit is None and destruct_limit is not None:
        operating_limit = destruct_limit

    op_margin = None
    destruct_margin = None
    if req.spec_max is not None:
        if operating_limit is not None:
            op_margin = round(operating_limit - req.spec_max, 4)
        if destruct_limit is not None:
            destruct_margin = round(destruct_limit - req.spec_max, 4)

    return {
        "stress_type": req.stress_type,
        "operating_limit": operating_limit,
        "destruct_limit": destruct_limit,
        "spec_min": req.spec_min,
        "spec_max": req.spec_max,
        "operating_margin": op_margin,
        "destruct_margin": destruct_margin,
        "capability_plot": {
            "levels": levels.tolist(),
            "outcomes": outcomes,
        },
    }


# ---------------------------------------------------------------------------
# Margin Test — reliability demonstration beyond spec
# ---------------------------------------------------------------------------

@router.post("/margin-test")
def margin_test(req: MarginTestRequest):
    """Demonstrate reliability at spec conditions from an over-stress test."""
    from scipy.stats import beta as _beta

    if req.n_units <= 0 or req.n_failures < 0 or req.n_failures > req.n_units:
        raise HTTPException(status_code=400, detail="Invalid unit/failure counts.")
    if req.test_duration <= 0:
        raise HTTPException(status_code=400, detail="test_duration must be > 0.")

    af = req.acceleration_factor
    if af is None:
        # Fall back to a simple inverse-power ratio with exponent 1.
        af = (req.test_stress / req.spec_stress) if req.spec_stress > 0 else 1.0
    if af <= 0:
        raise HTTPException(status_code=400, detail="acceleration_factor must be > 0.")

    # Equivalent demonstrated time at spec conditions.
    equiv_time = req.test_duration * af

    # Lower confidence bound on reliability via Clopper-Pearson (beta) for the
    # demonstrated mission = equivalent test exposure.
    f = req.n_failures
    n = req.n_units
    alpha = 1.0 - req.confidence
    if f == 0:
        r_lower = alpha ** (1.0 / n)
    else:
        r_lower = float(_beta.ppf(alpha, n - f, f + 1))
    r_point = (n - f) / n

    # MTBF estimate at spec (exponential approximation).
    total_equiv = n * equiv_time
    mtbf_spec = total_equiv / f if f > 0 else None

    return {
        "acceleration_factor": round(float(af), 4),
        "equivalent_time_at_spec": round(float(equiv_time), 4),
        "demonstrated_reliability": round(r_point, 5),
        "reliability_lower_bound": round(r_lower, 5),
        "confidence": req.confidence,
        "mtbf_at_spec": (round(mtbf_spec, 2) if mtbf_spec else None),
        "margin_ratio": round(req.test_stress / req.spec_stress, 4) if req.spec_stress > 0 else None,
    }


# ---------------------------------------------------------------------------
# Multi-Stress ALT — two simultaneous stress variables
# ---------------------------------------------------------------------------

@router.post("/multi-stress")
def multi_stress(req: MultiStressRequest):
    """Per-combination statistics and use-condition interpolation for 2 stresses."""
    ft = np.asarray(req.failure_times, dtype=float)
    s1 = np.asarray(req.stress1, dtype=float)
    s2 = np.asarray(req.stress2, dtype=float)
    if not (len(ft) == len(s1) == len(s2)) or len(ft) < 3:
        raise HTTPException(status_code=400,
                            detail="failure_times, stress1, stress2 must be equal-length (>=3).")

    # Per stress-combination summary.
    combos = {}
    for t, a, b in zip(ft, s1, s2):
        key = (float(a), float(b))
        combos.setdefault(key, []).append(float(t))
    combo_table = []
    for (a, b), times in sorted(combos.items()):
        combo_table.append({
            "stress1": a, "stress2": b, "n": len(times),
            "median_life": round(float(np.median(times)), 4),
            "mean_life": round(float(np.mean(times)), 4),
        })

    # Fit log(life) = c0 + c1*s1 + c2*s2 (multiple linear regression) and
    # extrapolate to use conditions.
    use_life = None
    coeffs = None
    with np.errstate(all="ignore"):
        try:
            X = np.column_stack([np.ones_like(s1), s1, s2])
            y = np.log(ft)
            beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
            coeffs = beta_hat.tolist()
            if req.stress1_use is not None and req.stress2_use is not None:
                pred = beta_hat[0] + beta_hat[1] * req.stress1_use + beta_hat[2] * req.stress2_use
                use_life = float(np.exp(pred))
        except Exception:
            pass

    return {
        "stress1_label": req.stress1_label,
        "stress2_label": req.stress2_label,
        "combo_table": combo_table,
        "scatter": {
            "stress1": s1.tolist(), "stress2": s2.tolist(), "life": ft.tolist(),
        },
        "regression_coeffs": coeffs,
        "use_level_life": (round(use_life, 4) if use_life else None),
        "stress1_use": req.stress1_use,
        "stress2_use": req.stress2_use,
    }


# ---------------------------------------------------------------------------
# Stress screening — ESS / HASS / Burn-in
# ---------------------------------------------------------------------------

def _thermal_ss(delta_t, cycles):
    """Thermal-cycling screening strength (fraction of latent defects precipitated)."""
    if delta_t is None or cycles is None or delta_t <= 0 or cycles <= 0:
        return 0.0
    return float(1.0 - math.exp(-0.0017 * (delta_t ** 1.9) * cycles))


def _vibration_ss(grms, minutes):
    """Random-vibration screening strength."""
    if grms is None or minutes is None or grms <= 0 or minutes <= 0:
        return 0.0
    hours = minutes / 60.0
    return float(1.0 - math.exp(-0.0046 * (grms ** 1.71) * hours))


@router.post("/ess")
def ess(req: ESSRequest):
    """Environmental Stress Screening profile development."""
    if not (0.0 <= req.defect_rate <= 1.0):
        raise HTTPException(status_code=400, detail="defect_rate must be in [0, 1].")
    if not (0.0 < req.target_screening_strength < 1.0):
        raise HTTPException(status_code=400, detail="target_screening_strength must be in (0, 1).")

    st = req.screening_type
    curve_x, curve_y, x_label = [], [], ""
    achieved = 0.0
    required = None

    if st == "thermal":
        x_label = "Number of cycles"
        cycles = req.num_cycles or 10
        achieved = _thermal_ss(req.temp_range, cycles)
        # Solve required cycles for the target SS.
        if req.temp_range and req.temp_range > 0:
            required = math.log(1.0 - req.target_screening_strength) / (-0.0017 * (req.temp_range ** 1.9))
            required = max(1.0, required)
        xs = np.linspace(1, max(cycles, (required or cycles)) * 1.5, 60)
        curve_x = xs.tolist()
        curve_y = [_thermal_ss(req.temp_range, c) for c in xs]
    elif st == "vibration":
        x_label = "Duration (minutes)"
        dur = req.vib_duration or 10.0
        achieved = _vibration_ss(req.grms, dur)
        if req.grms and req.grms > 0:
            hours = math.log(1.0 - req.target_screening_strength) / (-0.0046 * (req.grms ** 1.71))
            required = max(1.0, hours * 60.0)
        xs = np.linspace(1, max(dur, (required or dur)) * 1.5, 60)
        curve_x = xs.tolist()
        curve_y = [_vibration_ss(req.grms, m) for m in xs]
    else:  # combined
        x_label = "Number of thermal cycles"
        cycles = req.num_cycles or 10
        ss_t = _thermal_ss(req.temp_range, cycles)
        ss_v = _vibration_ss(req.grms, req.vib_duration)
        achieved = 1.0 - (1.0 - ss_t) * (1.0 - ss_v)
        xs = np.linspace(1, cycles * 2 + 1, 60)
        curve_x = xs.tolist()
        curve_y = [1.0 - (1.0 - _thermal_ss(req.temp_range, c)) * (1.0 - ss_v) for c in xs]

    residual_defect = req.defect_rate * (1.0 - achieved)
    detected = req.defect_rate * achieved

    return {
        "screening_type": st,
        "screening_strength": round(achieved, 5),
        "required": (round(required, 2) if required is not None else None),
        "required_label": x_label,
        "detected_defect_fraction": round(detected, 6),
        "residual_defect_fraction": round(residual_defect, 6),
        "curve": {"x": curve_x, "y": curve_y, "x_label": x_label,
                  "target": req.target_screening_strength},
    }


@router.post("/hass")
def hass(req: HASSRequest):
    """Highly Accelerated Stress Screening: precipitation + detection screens."""
    # Precipitation screen: stress midway between operating and destruct limits.
    precip_temp_low = (req.op_temp_low + req.destruct_temp_low) / 2.0
    precip_temp_high = (req.op_temp_high + req.destruct_temp_high) / 2.0
    precip_dt = precip_temp_high - precip_temp_low
    precip_vib = (req.op_vib + req.destruct_vib) / 2.0

    # Required thermal cycles to hit the target precipitation strength.
    if precip_dt > 0 and 0.0 < req.target_precip_ss < 1.0:
        req_cycles = math.log(1.0 - req.target_precip_ss) / (-0.0017 * (precip_dt ** 1.9))
        req_cycles = max(1.0, req_cycles)
    else:
        req_cycles = None
    precip_ss = _thermal_ss(precip_dt, req_cycles) if req_cycles else 0.0

    # Detection screen at operating limits.
    op_dt = req.op_temp_high - req.op_temp_low
    prob_detect = 1.0 - math.exp(-req.detection_duration / req.use_mtbf) if req.use_mtbf > 0 else 0.0

    return {
        "precipitation_screen": {
            "temp_low": round(precip_temp_low, 2),
            "temp_high": round(precip_temp_high, 2),
            "delta_t": round(precip_dt, 2),
            "vibration": round(precip_vib, 2),
            "required_cycles": (round(req_cycles, 1) if req_cycles else None),
            "screening_strength": round(precip_ss, 5),
        },
        "detection_screen": {
            "temp_low": req.op_temp_low,
            "temp_high": req.op_temp_high,
            "delta_t": round(op_dt, 2),
            "vibration": req.op_vib,
            "duration": req.detection_duration,
            "probability_of_detection": round(prob_detect, 5),
        },
        "stress_levels": {
            "operating": [req.op_temp_low, req.op_temp_high, req.op_vib],
            "precipitation": [precip_temp_low, precip_temp_high, precip_vib],
            "destruct": [req.destruct_temp_low, req.destruct_temp_high, req.destruct_vib],
        },
    }


@router.post("/burn-in")
def burn_in(req: BurnInRequest):
    """Burn-in test design to screen infant-mortality (Weibull, beta < 1)."""
    if req.eta <= 0 or req.beta <= 0 or req.n_units <= 0 or req.duration <= 0:
        raise HTTPException(status_code=400, detail="eta, beta, n_units, duration must be > 0.")

    af = req.acceleration_factor if req.acceleration_factor and req.acceleration_factor > 0 else 1.0
    t_eff = req.duration * af

    def sf(t):
        return math.exp(-((t / req.eta) ** req.beta))

    p_survive = sf(t_eff)
    expected_failures = req.n_units * (1.0 - p_survive)

    # Reliability curves before vs after burn-in (conditional on survival).
    tmax = req.eta * 2.0
    xs = np.linspace(0, tmax, 200)
    r_before = np.array([sf(t) for t in xs])
    r_after = np.array([sf(t + t_eff) / p_survive if p_survive > 0 else 0.0 for t in xs])

    def haz(t):
        if t <= 0:
            return 0.0
        return (req.beta / req.eta) * ((t / req.eta) ** (req.beta - 1.0))

    h_before = np.array([haz(t) for t in xs])
    h_after = np.array([haz(t + t_eff) for t in xs])

    # Post burn-in MTBF (numeric integral of conditional reliability).
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    mtbf_after = float(_trapz(r_after, xs))

    return {
        "effective_burn_in_time": round(t_eff, 3),
        "survival_probability": round(p_survive, 5),
        "expected_failures": round(expected_failures, 3),
        "post_burn_in_mtbf": round(mtbf_after, 3),
        "reliability_plot": {
            "time": xs.tolist(),
            "before": r_before.tolist(),
            "after": r_after.tolist(),
        },
        "hazard_plot": {
            "time": xs.tolist(),
            "before": h_before.tolist(),
            "after": h_after.tolist(),
        },
    }
