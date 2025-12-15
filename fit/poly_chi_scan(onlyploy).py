#!/usr/bin/env python
# poly_chi_scan.py
#
# For each image:
#   1) Compute CV vs window width
#   2) Fit polynomials of several degrees
#   3) Compute reduced chi-square for each degree
#   4) Save (degree, chi2) CSV and a figure in output/chifigure/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    IMG_DIR,
    OUT_DIR,
    STEP_RATIO,
    N_STEPS,
    MIN_X,
    MAX_X,
    UM_PER_PX,
)
from image_utils import imread_gray01, build_fibre_mask_from_binary
from sliding_stats import sliding_area_fraction_stats, cv_uncertainty_rapidcode


# Degrees of the polynomial to scan
POLY_DEGREES = [1, 2, 3, 4, 5, 6]

# Use weighted fit with w = 1 / delta_cv
USE_WEIGHTED_FIT = True

# Output folder for chi-square plots and CSVs
CHI_DIR = OUT_DIR / "chifigure"
CHI_DIR.mkdir(exist_ok=True)


def compute_cv_curve(img_path: Path):
    gray01 = imread_gray01(img_path)
    mask = build_fibre_mask_from_binary(gray01)
    H, W = mask.shape
    span = float(W)

    # make it identical to analyze_image()
    if MIN_X is not None:
        min_w_px = MIN_X
    else:
        min_w_px = 1.0

    if MAX_X is not None:
        max_w_px = MAX_X
    else:
        max_w_px = span

    x_widths_px = np.linspace(min_w_px, max_w_px, N_STEPS)
    x_widths_um = x_widths_px * UM_PER_PX

    cvs = []
    n_windows_list = []

    for w_px in x_widths_px:
        step_px = max(1.0, STEP_RATIO * w_px)
        _, _, _, cv, n_win = sliding_area_fraction_stats(mask, w_px, step_px)
        cvs.append(cv)
        n_windows_list.append(n_win)

    cvs = np.array(cvs, dtype=float)
    n_windows = np.array(n_windows_list, dtype=int)

    return x_widths_um, cvs, n_windows



def chi2_for_degree(x_um: np.ndarray,
                    cv: np.ndarray,
                    n_windows: np.ndarray,
                    degree: int,
                    use_weighted: bool = True) -> float:
    """
    Fit a polynomial of given degree and compute reduced chi-square.
    Returns np.nan if fit is not possible.
    """
    mask = np.isfinite(x_um) & np.isfinite(cv) & (cv > 0) & (n_windows > 0)
    x = x_um[mask]
    y = cv[mask]
    n = n_windows[mask].astype(float)

    min_points = degree + 2
    if x.size < min_points:
        return np.nan

    delta_cv = np.array(
        [cv_uncertainty_rapidcode(ci, int(ni)) for ci, ni in zip(y, n)],
        dtype=float
    )
    valid_err = np.isfinite(delta_cv) & (delta_cv > 0)

    if use_weighted and np.any(valid_err):
        weights = np.zeros_like(delta_cv)
        weights[valid_err] = 1.0 / delta_cv[valid_err]
    else:
        weights = None

    try:
        if weights is not None:
            coeffs = np.polyfit(x, y, degree, w=weights)
        else:
            coeffs = np.polyfit(x, y, degree)
    except Exception:
        return np.nan

    if not np.any(valid_err):
        return np.nan

    xv = x[valid_err]
    yv = y[valid_err]
    ev = delta_cv[valid_err]

    yfit = np.polyval(coeffs, xv)
    chi2 = np.sum(((yv - yfit) / ev) ** 2)
    dof = max(len(yv) - (degree + 1), 1)
    chi2_red = chi2 / dof

    return float(chi2_red)


def process_all_images():
    """
    For each image in IMG_DIR, generate:
      - CSV: poly_degree vs chi2_red_poly
      - PNG figure: degree vs chi2 plot
    """
    image_paths = sorted(IMG_DIR.glob("*.png"))

    if not image_paths:
        print(f"[Warning] No PNG images found in {IMG_DIR}")
        return

    print(f"[Info] Found {len(image_paths)} images.")

    for img_path in image_paths:
        stem = img_path.stem
        print(f"[Process] {stem} ...")

        x_um, cvs, n_windows = compute_cv_curve(img_path)

        degrees = []
        chi2_values = []

        for deg in POLY_DEGREES:
            chi2_red = chi2_for_degree(
                x_um, cvs, n_windows,
                degree=deg,
                use_weighted=USE_WEIGHTED_FIT
            )
            degrees.append(deg)
            chi2_values.append(chi2_red)

        degrees = np.array(degrees, dtype=int)
        chi2_values = np.array(chi2_values, dtype=float)

        # Save CSV
        df = pd.DataFrame({
            "poly_degree": degrees,
            "chi2_red_poly": chi2_values,
        })
        csv_path = CHI_DIR / f"{stem}_chi_poly_vs_degree.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Save figure
        fig, ax = plt.subplots()
        ax.plot(degrees, chi2_values, "o-")
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("Reduced chi-square")
        ax.set_title(f"Chi-square vs polynomial degree - {stem}")
        ax.grid(True)

        fig_path = CHI_DIR / f"{stem}_chi_poly_vs_degree.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"    Saved: {fig_path.name}, {csv_path.name}")

    print(f"[Done] All figures and CSVs saved in: {CHI_DIR}")


if __name__ == "__main__":
    process_all_images()
