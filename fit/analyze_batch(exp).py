import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from kneed import KneeLocator

# ===== Global parameters =====
base_directory = Path.cwd()

# Folder with 10 binary images
IMG_DIR = base_directory / "Batch2_NE_1000X_Binary"

# Output folder (all results go here)
OUT_DIR = base_directory /"output"
OUT_DIR.mkdir(exist_ok=True)

# Sliding-window parameters
STEP_RATIO = 0.25       # step = STEP_RATIO * window_width
N_STEPS    = 40         # number of window widths
MIN_X      = None       # None -> max(10px, span/100)
MAX_X      = None       # None -> span/2

# Calibration (micrometers per pixel)  180 px = 10 µm
UM_PER_PX = 10.0 / 180.0

# Dense grid for plotting fitted curves
DENSE_N = 800


def imread_gray01(path: Path) -> np.ndarray:
    """Read image and convert to grayscale in [0,1]."""
    img = plt.imread(path)
    if img.ndim == 3:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img.astype(float)

    gray = gray.astype(float)
    vmin, vmax = float(gray.min()), float(gray.max())
    if vmax > vmin:
        gray = (gray - vmin) / (vmax - vmin)
    else:
        gray = np.zeros_like(gray, dtype=float)
    return gray


def build_fibre_mask_from_binary(gray01: np.ndarray) -> np.ndarray:
    """Convert normalized grayscale image to binary mask (1 = fibre, 0 = background)."""
    return (gray01 < 0.5).astype(np.uint8)


def sliding_area_fraction_stats(mask: np.ndarray,
                                win_px: float,
                                step_px: float):

    H, W = mask.shape
    win_px = max(1, int(round(win_px)))
    step_px = max(1, int(round(step_px)))

    # If window wider than image, just one global window
    if win_px >= W:
        fibre_cols = (mask == 1).sum(axis=0)
        frac = float(fibre_cols.sum()) / float(H * W)
        fracs = np.array([frac], dtype=float)
        mean = frac
        std = 0.0
        cv = np.nan
        return fracs, mean, std, cv, len(fracs)

    fibre_cols = (mask == 1).sum(axis=0).astype(np.int64)

    # Prefix sum for fast window sums
    prefix = np.zeros(W + 1, dtype=np.int64)
    np.cumsum(fibre_cols, out=prefix[1:])

    fracs = []
    x = 0
    last = W - win_px
    while x <= last:
        fibre_in = prefix[x + win_px] - prefix[x]
        fracs.append(fibre_in / float(H * win_px))
        x += step_px

    # Ensure we include last window touching the right edge
    if len(fracs) == 0 or (x - step_px) != last:
        fibre_in = prefix[W] - prefix[last]
        fracs.append(fibre_in / float(H * win_px))

    fracs = np.array(fracs, dtype=float)
    mean = fracs.mean()
    std = fracs.std(ddof=0)
    cv = (std / mean) if mean > 0 else np.nan
    return fracs, mean, std, cv, len(fracs)


def cv_uncertainty_rapidcode(cv: float, n: int) -> float:

    if n <= 0 or cv <= 0:
        return np.nan
    return cv / np.sqrt(2.0 * n)


def exp_model(x, a, b, c):
    """Exponential decay model: a * exp(b * x) + c."""
    return a * np.exp(b * x) + c


def hyper_model(x, a, c):
    """Hyperbolic model: a / x + c."""
    return a / x + c


def fit_models_and_knees(x_um: np.ndarray,
                         cv: np.ndarray,
                         n_windows: np.ndarray):

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

    # --- Fit exponential model ---
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

    # --- Fit hyperbolic model ---
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

    # --- Evaluation using χ² with CV uncertainty ---
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


def analyze_image(img_path: Path):
    """
    Run full analysis for a single image:
        - compute CV vs window width
        - compute dCV/dx (raw)
        - fit exp & 1/x models and find knees
        - save per-image CSV and plots
        - return summary info for thresholds_summary.csv
    """
    stem = img_path.stem

    # --- Build binary mask ---
    gray01 = imread_gray01(img_path)
    mask = build_fibre_mask_from_binary(gray01)
    H, W = mask.shape
    span = float(W)

    # Window-width range in pixels
    min_w_px = MIN_X if MIN_X is not None else max(10.0, span / 100.0)
    max_w_px = MAX_X if MAX_X is not None else max(min_w_px * 2.0, span / 2.0)

    x_widths_px = np.linspace(min_w_px, max_w_px, N_STEPS)
    x_widths_um = x_widths_px * UM_PER_PX

    stds = []
    cvs = []
    n_windows_list = []

    # --- Sliding-window statistics for each window width ---
    for w_px in x_widths_px:
        step_px = max(1.0, STEP_RATIO * w_px)
        fracs, mean, std, cv, n_win = sliding_area_fraction_stats(mask, w_px, step_px)
        stds.append(std)
        cvs.append(cv)
        n_windows_list.append(n_win)

    stds = np.array(stds, dtype=float)
    cvs = np.array(cvs, dtype=float)
    n_windows = np.array(n_windows_list, dtype=int)

    # Raw derivative dCV/dx in µm
    dcv_dx_um = np.gradient(cvs, x_widths_um)

    # CV uncertainty for each point (for plotting and χ²)
    delta_cv = np.array(
        [cv_uncertainty_rapidcode(cv, int(n)) for cv, n in zip(cvs, n_windows)],
        dtype=float
    )

    # --- Fit smooth models + knees + evaluation ---
    fit_res = fit_models_and_knees(x_widths_um, cvs, n_windows)

    x_dense_um = fit_res["x_dense_um"]
    cv_exp_dense = fit_res["cv_exp_dense"]
    cv_hyp_dense = fit_res["cv_hyp_dense"]
    knee_exp_um = fit_res["knee_exp_um"]
    knee_hyp_um = fit_res["knee_hyp_um"]
    chi2_red_exp = fit_res["chi2_red_exp"]
    chi2_red_hyp = fit_res["chi2_red_hyper"]
    exp_params = fit_res["exp_params"]
    hyp_params = fit_res["hyp_params"]

    # --- Save per-image CSV with basic data ---
    stats_df = pd.DataFrame({
        "x_width_px": x_widths_px,
        "x_width_um": x_widths_um,
        "std_area_fraction": stds,
        "cv_area_fraction": cvs,
        "n_windows": n_windows,
        "cv_uncertainty": delta_cv,
        "dcv_dx_per_um": dcv_dx_um,
    })
    stats_csv = OUT_DIR / f"{stem}_stats_vs_x.csv"
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")

    # --- Plot: raw CV with error bars ---
    fig, ax = plt.subplots()
    ax.errorbar(x_widths_um, cvs, yerr=delta_cv, fmt="o", capsize=3, label="CV ± δCV")
    ax.set_xlabel("Window width x [µm]")
    ax.set_ylabel("CV")
    ax.set_title(f"CV vs window width (raw) - {stem}")
    ax.grid(True)
    ax.legend()
    fig.savefig(OUT_DIR / f"{stem}_cv_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot: raw derivative dCV/dx ---
    fig, ax = plt.subplots()
    ax.plot(x_widths_um, dcv_dx_um, "o-")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Window width x [µm]")
    ax.set_ylabel("d(CV)/dx [per µm]")
    ax.set_title(f"dCV/dx vs window width (raw) - {stem}")
    ax.grid(True)
    fig.savefig(OUT_DIR / f"{stem}_dcv_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot: fitted curves + knees (supervisor-style figure) ---
    if x_dense_um is not None and cv_exp_dense is not None and cv_hyp_dense is not None:
        fig, ax = plt.subplots()
        ax.plot(x_widths_um, cvs, "o", alpha=0.5, label="raw CV")

        # Exponential fit
        if exp_params is not None:
            a, b, c = exp_params
            ax.plot(x_dense_um, cv_exp_dense, "-",
                    label=f"exp: {a:.2f}*exp({b:.3f}x)+{c:.3f}")

        # Hyperbolic fit
        if hyp_params is not None:
            a2, c2 = hyp_params
            ax.plot(x_dense_um, cv_hyp_dense, "-",
                    label=f"1/x: {a2:.2f}/x+{c2:.3f}")

        # Knees
        if knee_exp_um is not None:
            ax.axvline(knee_exp_um, color="C3", linestyle="--",
                       label=f"exp knee ≈ {knee_exp_um:.2f} µm")
        if knee_hyp_um is not None:
            ax.axvline(knee_hyp_um, color="C4", linestyle="--",
                       label=f"1/x knee ≈ {knee_hyp_um:.2f} µm")

        ax.set_xlabel("Window width x [µm]")
        ax.set_ylabel("CV")
        ax.set_title(f"CV w.r.t. window width (fits & knees) - {stem}")
        ax.grid(True)
        ax.legend()
        fig.savefig(OUT_DIR / f"{stem}_cv_fits_knees.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[{stem}] exp_knee={knee_exp_um}, hyper_knee={knee_hyp_um}, "
          f"chi2_exp={chi2_red_exp:.3g}, chi2_hyp={chi2_red_hyp:.3g}")

    # Summary row for this image
    return {
        "image_name": img_path.name,
        "knee_um_exp": knee_exp_um,
        "knee_um_hyper": knee_hyp_um,
        "chi2_red_exp": chi2_red_exp,
        "chi2_red_hyper": chi2_red_hyp,
    }


def main():
    # Find all PNG images in the folder
    image_paths = sorted(IMG_DIR.glob("*.png"))

    if not image_paths:
        print(f"No PNG images found in {IMG_DIR}")
        return

    summary_rows = []
    for img_path in image_paths:
        row = analyze_image(img_path)
        summary_rows.append(row)

    # Save summary CSV with thresholds and evaluation metrics
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "thresholds_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved thresholds summary: {summary_csv}")


if __name__ == "__main__":
    main()
