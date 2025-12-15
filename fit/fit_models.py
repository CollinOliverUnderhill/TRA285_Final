import numpy as np
from scipy.optimize import curve_fit
from kneed import KneeLocator

from config import DENSE_N
from sliding_stats import cv_uncertainty_rapidcode


def exp_model(x, a, b, c):
    """Exponential decay model: a * exp(b * x) + c."""
    return a * np.exp(b * x) + c


def hyper_model(x, a, c):
    """Hyperbolic model: a / x + c."""
    return a / x + c


def fit_models_and_knees(x_um: np.ndarray,
                         cv: np.ndarray,
                         n_windows: np.ndarray):
    """
    Fit exponential and hyperbolic models to CV(x), find knee points,
    and evaluate fits using reduced chi-square.

    Returns a dictionary containing:
        - x_dense_um, cv_exp_dense, cv_hyp_dense
        - knee_exp_um, knee_hyp_um
        - chi2_red_exp, chi2_red_hyper
        - exp_params, hyp_params
    """
    result = {
        "x_dense_um": None,
        "cv_exp_dense": None,
        "cv_hyp_dense": None,
        "knee_exp_um": None,
        "knee_hyp_um": None,
        "chi2_red_exp": np.nan,
        "chi2_red_hyper": np.nan,
        "exp_params": None,
        "hyp_params": None,
    }

    mask = np.isfinite(x_um) & np.isfinite(cv) & (cv > 0) & (n_windows > 0)
    x = x_um[mask]
    y = cv[mask]
    n = n_windows[mask].astype(float)

    if x.size < 4:
        # Not enough points to fit
        return result

    x_dense = np.linspace(x.min(), x.max(), DENSE_N)
    result["x_dense_um"] = x_dense

    # === Fit exponential model ===
    popt_exp = None
    try:
        c0 = float(y[-3:].mean())
        a0 = float(max(y[0] - c0, 1e-6))
        b0 = -1.0 / (x.max() - x.min() + 1e-9)
        popt_exp, _ = curve_fit(exp_model, x, y, p0=[a0, b0, c0], maxfev=10000)
        result["exp_params"] = popt_exp
        y_exp_dense = exp_model(x_dense, *popt_exp)
        result["cv_exp_dense"] = y_exp_dense

        # Knee of exponential curve
        try:
            kl = KneeLocator(x_dense, y_exp_dense,
                             curve="convex",
                             direction="decreasing",
                             online=False)
            if kl.knee is not None:
                result["knee_exp_um"] = float(kl.knee)
        except Exception:
            pass
    except Exception:
        popt_exp = None

    # === Fit hyperbolic model (1/x) ===
    popt_hyp = None
    try:
        c0_2 = float(y[-3:].mean())
        a0_2 = float(max((y[0] - c0_2) * x[0], 1e-6))
        popt_hyp, _ = curve_fit(hyper_model, x, y, p0=[a0_2, c0_2], maxfev=10000)
        result["hyp_params"] = popt_hyp
        y_hyp_dense = hyper_model(x_dense, *popt_hyp)
        result["cv_hyp_dense"] = y_hyp_dense

        # Knee of hyperbolic curve
        try:
            kl2 = KneeLocator(x_dense, y_hyp_dense,
                              curve="convex",
                              direction="decreasing",
                              online=False)
            if kl2.knee is not None:
                result["knee_hyp_um"] = float(kl2.knee)
        except Exception:
            pass
    except Exception:
        popt_hyp = None

    # === Evaluation using χ² with CV uncertainty ===
    delta_cv = np.array(
        [cv_uncertainty_rapidcode(ci, int(ni)) for ci, ni in zip(y, n)],
        dtype=float
    )
    valid_err = np.isfinite(delta_cv) & (delta_cv > 0)
    if np.any(valid_err):
        xv = x[valid_err]
        yv = y[valid_err]
        ev = delta_cv[valid_err]

        if popt_exp is not None:
            yfit_exp = exp_model(xv, *popt_exp)
            chi2 = np.sum(((yv - yfit_exp) / ev) ** 2)
            dof = max(len(yv) - len(popt_exp), 1)
            result["chi2_red_exp"] = chi2 / dof

        if popt_hyp is not None:
            yfit_hyp = hyper_model(xv, *popt_hyp)
            chi2 = np.sum(((yv - yfit_hyp) / ev) ** 2)
            dof = max(len(yv) - len(popt_hyp), 1)
            result["chi2_red_hyper"] = chi2 / dof

    return result
