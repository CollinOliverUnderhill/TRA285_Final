# analyze_batch.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# # Method 2: polynomial fitting + knee detection
# from fit_models_poly import fit_models_and_knees

# Method 3: sg smoother
from fit_models_sg import fit_models_and_knees


def analyze_image(img_path: Path):
    stem = img_path.stem

    # --- Build binary mask ---
    gray01 = imread_gray01(img_path)
    mask = build_fibre_mask_from_binary(gray01)
    H, W = mask.shape
    span = float(W)

    # Window-width range in pixels (same logic as original code)
    min_w_px = MIN_X if MIN_X is not None else 1.0
    max_w_px = MAX_X if MAX_X is not None else span

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

    fit_res = fit_models_and_knees(x_widths_um, cvs, n_windows)

    x_dense_um   = fit_res["x_dense_um"]
    cv_exp_dense = fit_res["cv_exp_dense"]
    cv_hyp_dense = fit_res["cv_hyp_dense"]
    knee_exp_um  = fit_res["knee_exp_um"]
    knee_hyp_um  = fit_res["knee_hyp_um"]
    chi2_red_exp   = fit_res["chi2_red_exp"]
    chi2_red_hyp   = fit_res["chi2_red_hyper"]
    exp_params   = fit_res["exp_params"]
    hyp_params   = fit_res["hyp_params"]

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
      
    # --- Plot: fitted curves and knees (多项式或 SG 拟合结果) ---
    if x_dense_um is not None and cv_exp_dense is not None:
        fig, ax = plt.subplots()
        ax.plot(x_widths_um, cvs, "o", alpha=0.5, label="raw CV")

        if exp_params is not None:
            params = np.asarray(exp_params)

            # 情况 1：多项式拟合（exp_params 是系数）
            if params.size > 2:
                deg = len(params) - 1
                label = f"poly fit (deg={deg})"

            # 情况 2：SG 平滑（exp_params = [win_len, poly_order]）
            elif params.size == 2:
                win_len, poly_order = params
                label = f"SG smooth (win={int(win_len)}, poly={int(poly_order)})"

            else:
                label = "fitted curve"

            ax.plot(x_dense_um, cv_exp_dense, "-", label=label)

        # knee
        if knee_exp_um is not None:
            ax.axvline(
                knee_exp_um,
                color="C3",
                linestyle="--",
                label=f"poly knee ≈ {knee_exp_um:.2f} µm"
            )

        ax.set_xlabel("Window width x [µm]")
        ax.set_ylabel("CV")
        ax.set_title(f"CV w.r.t. window width (poly fit & knee) - {stem}")
        ax.grid(True)
        ax.legend()
        fig.savefig(OUT_DIR / f"{stem}_cv_fits_knees.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[{stem}] poly_knee={knee_exp_um}, chi2_poly={chi2_red_exp:.3g}")

    # Summary row for this image
    return {
        "image_name": img_path.name,
        "knee_um_exp": knee_exp_um,
        "knee_um_hyper": knee_hyp_um,
        "chi2_red_exp/chi_poly": chi2_red_exp,  
        # "chi2_red_hyper": chi2_red_hyper,
    }

def main():
    """Batch-processing entry point for all PNG images in IMG_DIR."""
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
