"""
Utility functions for the reliability engineering suite.

Provides statistical helpers, goodness-of-fit metrics, rank estimation,
and plotting utilities used across all modules.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings


def median_rank_approximation(j, n):
    """Bernard's approximation for median ranks.

    Parameters
    ----------
    j : int or array-like
        Failure order number(s).
    n : int
        Total sample size.

    Returns
    -------
    float or ndarray
        Median rank estimate(s).
    """
    j = np.asarray(j, dtype=float)
    return (j - 0.3) / (n + 0.4)


def rank_adjustment(failures, right_censored=None):
    """Compute adjusted ranks for data with suspensions (right-censored observations).

    Uses the mean order number method to adjust failure order numbers
    when suspensions are present.

    Parameters
    ----------
    failures : array-like
        Failure times.
    right_censored : array-like, optional
        Suspension (right-censored) times.

    Returns
    -------
    adjusted_ranks : ndarray
        Adjusted rank values for each failure (in sorted order of failures).
    n : int
        Total sample size (failures + suspensions).
    """
    failures = np.sort(np.asarray(failures, dtype=float))
    n = len(failures)
    if right_censored is not None:
        right_censored = np.asarray(right_censored, dtype=float)
        n += len(right_censored)

    if right_censored is None or len(right_censored) == 0:
        ranks = np.arange(1, len(failures) + 1, dtype=float)
        return ranks, n

    # Combine and sort all data, tracking type
    all_times = np.concatenate([failures, right_censored])
    all_types = np.concatenate([
        np.ones(len(failures)),       # 1 = failure
        np.zeros(len(right_censored)) # 0 = suspension
    ])
    sort_idx = np.argsort(all_times, kind='stable')
    sorted_types = all_types[sort_idx]

    # Mean order number method
    adjusted_ranks = []
    prev_order = 0
    reverse_rank = n + 1
    for i in range(len(sort_idx)):
        reverse_rank -= 1
        if sorted_types[i] == 1:  # failure
            increment = (n + 1 - prev_order) / (1 + reverse_rank)
            new_order = prev_order + increment
            adjusted_ranks.append(new_order)
            prev_order = new_order

    return np.array(adjusted_ranks), n


def AICc(loglik, k, n):
    """Corrected Akaike Information Criterion.

    Parameters
    ----------
    loglik : float
        Log-likelihood of the fitted model.
    k : int
        Number of model parameters.
    n : int
        Sample size.

    Returns
    -------
    float
        AICc value. Lower is better.
    """
    aic = 2 * k - 2 * loglik
    if n - k - 1 > 0:
        correction = (2 * k * (k + 1)) / (n - k - 1)
    else:
        correction = 0
    return aic + correction


def BIC(loglik, k, n):
    """Bayesian Information Criterion.

    Parameters
    ----------
    loglik : float
        Log-likelihood of the fitted model.
    k : int
        Number of model parameters.
    n : int
        Sample size.

    Returns
    -------
    float
        BIC value. Lower is better.
    """
    return k * np.log(n) - 2 * loglik


def anderson_darling(failures, fitted_cdf_func):
    """Compute the Anderson-Darling statistic.

    Parameters
    ----------
    failures : array-like
        Sorted failure times.
    fitted_cdf_func : callable
        CDF function of the fitted distribution.

    Returns
    -------
    float
        Anderson-Darling statistic. Lower is better.
    """
    failures = np.sort(np.asarray(failures, dtype=float))
    n = len(failures)
    if n == 0:
        return np.inf

    F = fitted_cdf_func(failures)
    F = np.clip(F, 1e-15, 1 - 1e-15)

    i = np.arange(1, n + 1)
    S = np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    AD = -n - S / n
    return AD


def negative_log_likelihood(params, dist_class, failures, right_censored=None):
    """Generic negative log-likelihood for censored data.

    Parameters
    ----------
    params : tuple
        Distribution parameters.
    dist_class : class
        Distribution class with _from_params classmethod.
    failures : ndarray
        Failure times.
    right_censored : ndarray, optional
        Suspension (right-censored) times.

    Returns
    -------
    float
        Negative log-likelihood value.
    """
    try:
        dist = dist_class._from_params(params)
    except (ValueError, RuntimeError):
        return np.inf

    pdf_vals = dist._pdf(failures)
    pdf_vals = np.clip(pdf_vals, 1e-300, None)
    LL = np.sum(np.log(pdf_vals))

    if right_censored is not None and len(right_censored) > 0:
        sf_vals = dist._sf(right_censored)
        sf_vals = np.clip(sf_vals, 1e-300, None)
        LL += np.sum(np.log(sf_vals))

    if np.isnan(LL) or np.isinf(LL):
        return np.inf
    return -LL


def xy_transform(dist_name):
    """Return linearizing transform functions for probability plotting.

    Parameters
    ----------
    dist_name : str
        Distribution name (e.g., 'Weibull', 'Normal', 'Lognormal', etc.)

    Returns
    -------
    x_transform : callable
        Transform for x-axis.
    y_transform : callable
        Transform for y-axis (applied to CDF values F).
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    """
    if dist_name in ('Weibull', 'Weibull_2P', 'Weibull_3P'):
        return (
            np.log,
            lambda F: np.log(-np.log(1 - F)),
            'ln(t)',
            'ln(ln(1/(1-F)))'
        )
    elif dist_name in ('Normal', 'Normal_2P'):
        return (
            lambda x: x,
            lambda F: stats.norm.ppf(F),
            't',
            'Standard Normal Quantile'
        )
    elif dist_name in ('Lognormal', 'Lognormal_2P', 'Lognormal_3P'):
        return (
            np.log,
            lambda F: stats.norm.ppf(F),
            'ln(t)',
            'Standard Normal Quantile'
        )
    elif dist_name in ('Exponential', 'Exponential_1P', 'Exponential_2P'):
        return (
            lambda x: x,
            lambda F: -np.log(1 - F),
            't',
            '-ln(1-F)'
        )
    elif dist_name in ('Gamma', 'Gamma_2P', 'Gamma_3P'):
        return (
            np.log,
            lambda F: stats.norm.ppf(F),
            'ln(t)',
            'Standard Normal Quantile'
        )
    elif dist_name in ('Loglogistic', 'Loglogistic_2P', 'Loglogistic_3P'):
        return (
            np.log,
            lambda F: np.log(F / (1 - F)),
            'ln(t)',
            'ln(F/(1-F))'
        )
    elif dist_name in ('Beta', 'Beta_2P'):
        return (
            lambda x: x,
            lambda F: np.log(F / (1 - F)),
            't',
            'ln(F/(1-F))'
        )
    elif dist_name in ('Gumbel', 'Gumbel_2P'):
        return (
            lambda x: x,
            lambda F: -np.log(-np.log(F)),
            't',
            '-ln(-ln(F))'
        )
    else:
        return (
            lambda x: x,
            lambda F: F,
            't',
            'F(t)'
        )


def generate_X_array(dist, xvals=None, num_points=200):
    """Auto-generate x values for plotting a distribution.

    Parameters
    ----------
    dist : Distribution
        A distribution object with quantile method.
    xvals : array-like, optional
        User-supplied x values. If provided, returned as-is.
    num_points : int
        Number of points to generate.

    Returns
    -------
    ndarray
        Array of x values for plotting.
    """
    if xvals is not None:
        return np.asarray(xvals, dtype=float)

    try:
        q_low = dist.quantile(0.001)
        q_high = dist.quantile(0.999)
    except Exception:
        q_low = 0
        q_high = 100

    if hasattr(dist, 'gamma') and dist.gamma > 0:
        q_low = max(q_low, dist.gamma + 1e-10)

    if q_low >= q_high:
        q_high = q_low + 10

    return np.linspace(q_low, q_high, num_points)


# Color palette for plots
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a'
]
