"""Accelerated Life Testing router."""

import sys
import warnings
import numpy as np
from fastapi import APIRouter, HTTPException
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from reliability.ALT_fitters import Fit_Everything_ALT, ALL_SINGLE_STRESS_NAMES
from schemas import ALTFitRequest

router = APIRouter()


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
