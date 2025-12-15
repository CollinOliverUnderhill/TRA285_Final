from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import IMG_DIR, OUT_DIR, UM_PER_PX
from image_utils import imread_gray01, build_fibre_mask_from_binary


# ---------------- 1. Helpers: S2(r) and L(z) ----------------

def compute_S2_radial(mask: np.ndarray, r_values: np.ndarray) -> np.ndarray:
    """
    Estimate the two-point probability function S2(r) for phase=1
    via FFT-based autocorrelation + radial averaging.

    mask : 2D binary array (0/1), phase of interest = 1
    r_values : 1D array of radii (in pixels, increasing)

    Returns: S2(r) with same length as r_values.
    """
    mask = mask.astype(float)
    H, W = mask.shape

    # Autocorrelation of indicator function I(x)
    F = np.fft.fft2(mask)
    auto = np.fft.ifft2(F * np.conj(F)).real / (H * W)

    # Shift so that zero-lag is at the centre
    auto_shift = np.fft.fftshift(auto)

    # Distance map from image centre
    yy, xx = np.indices((H, W))
    cy, cx = H // 2, W // 2
    dy = yy - cy
    dx = xx - cx
    dist = np.sqrt(dx * dx + dy * dy)

    dist_flat = dist.ravel()
    auto_flat = auto_shift.ravel()

    r_values = np.asarray(r_values, dtype=float)
    n_r = r_values.size

    # Bin edges between radii (simple midpoint scheme)
    edges = np.empty(n_r + 1, dtype=float)
    edges[1:-1] = 0.5 * (r_values[:-1] + r_values[1:])
    # First and last edge
    first_width = edges[1] - r_values[0]
    last_width = r_values[-1] - edges[-2]
    edges[0] = max(0.0, r_values[0] - first_width)
    edges[-1] = r_values[-1] + last_width

    # Digitize distances into radial bins
    bin_idx = np.digitize(dist_flat, edges) - 1  # 0..n_r-1

    S2 = np.zeros_like(r_values, dtype=float)
    for k in range(n_r):
        m = bin_idx == k
        if np.any(m):
            S2[k] = auto_flat[m].mean()
        else:
            S2[k] = np.nan
    return S2


def compute_lineal_L(mask: np.ndarray, z_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizontal lineal-path function L(z) for phase=1.

    Definition: L(z) = probability that a horizontal segment of length z
    lies fully inside phase=1.

    mask : 2D binary array (0/1), phase of interest = 1
    z_values : 1D array of segment lengths (floats, in pixels)

    Returns
    -------
    L_vals : array of same length as z_values
    z_px   : integer pixel lengths actually used for each entry
    """
    mask = mask.astype(bool)
    H, W = mask.shape

    z_vals = np.asarray(z_values, dtype=float)
    # Round to nearest pixel, clip to [1, W]
    z_px = np.rint(z_vals).astype(int)
    z_px = np.clip(z_px, 1, W)

    # Unique pixel lengths to avoid recomputation
    z_unique, inv = np.unique(z_px, return_inverse=True)
    n_z = z_unique.size

    counts = np.zeros(n_z, dtype=np.int64)
    # Total possible segments of length z in the whole image
    total_segments = np.array(
        [H * max(W - int(z) + 1, 0) for z in z_unique],
        dtype=np.int64,
    )

    # For each row, find contiguous runs of ones and accumulate
    for y in range(H):
        row = mask[y, :]
        if not row.any():
            continue

        # Pad with zeros on both sides to detect transitions
        padded = np.concatenate([[0], row.view(np.int8), [0]])
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]   # indices where segment starts
        ends = np.where(diff == -1)[0]    # index after the last 1
        lengths = ends - starts           # length of each 1-segment

        for L in lengths:
            # This segment contributes (L - z + 1) valid segments for
            # every z <= L
            idx_valid = np.where(z_unique <= L)[0]
            if idx_valid.size == 0:
                continue
            z_sub = z_unique[idx_valid]
            counts[idx_valid] += (L - z_sub + 1)

    # Avoid division by zero
    L_unique = np.zeros_like(z_unique, dtype=float)
    nonzero = total_segments > 0
    L_unique[nonzero] = counts[nonzero] / total_segments[nonzero].astype(float)

    # Map back to original order of z_values
    L_vals = L_unique[inv]
    return L_vals, z_px


# ---------------- 2. Energy and SA (Yeong–Torquato style) ----------------

def energy_S2_L(
    S2_cur: np.ndarray,
    S2_target: np.ndarray,
    L_cur: np.ndarray,
    L_target: np.ndarray,
    w_S2: float = 1.0,
    w_L: float = 1.0,
) -> float:
    """
    Quadratic mismatch between current and target descriptors.
    """
    mask_S2 = np.isfinite(S2_cur) & np.isfinite(S2_target)
    mask_L = np.isfinite(L_cur) & np.isfinite(L_target)

    eS = 0.0
    eL = 0.0

    if mask_S2.any() and w_S2 != 0.0:
        dS = S2_cur[mask_S2] - S2_target[mask_S2]
        eS = w_S2 * float(np.mean(dS ** 2))

    if mask_L.any() and w_L != 0.0:
        dL = L_cur[mask_L] - L_target[mask_L]
        eL = w_L * float(np.mean(dL ** 2))

    return eS + eL


def sa_yt_hybrid(
    mask_init: np.ndarray,
    r_values: np.ndarray,
    z_values: np.ndarray,
    S2_target: np.ndarray,
    L_target: np.ndarray,
    max_steps: int = 2000,
    theta0: float = 1.0,
    alpha: float = 0.995,
    w_S2: float = 1.0,
    w_L: float = 1.0,
    print_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Yeong–Torquato pixel-exchange simulated annealing with hybrid
    descriptors S2(r) + L(z).

    mask_init : initial binary image (0/1) with desired volume fraction
    r_values  : radii (pixels) for S2
    z_values  : segment lengths (pixels) for L
    S2_target : target S2(r)
    L_target  : target L(z)

    Returns
    -------
    best_mask : reconstructed binary image
    loss_hist : array of energy values vs SA step
    best_E    : best (lowest) energy achieved
    """
    rng = np.random.default_rng(1234)

    mask = mask_init.copy().astype(np.uint8)
    H, W = mask.shape

    # Initial descriptors and energy
    S2_cur = compute_S2_radial(mask, r_values)
    L_cur, _ = compute_lineal_L(mask, z_values)
    E_cur = energy_S2_L(S2_cur, S2_target, L_cur, L_target, w_S2, w_L)

    best_mask = mask.copy()
    best_E = E_cur

    theta = float(theta0)
    loss_hist: List[float] = [E_cur]

    print(f"[YT-SA] initial energy = {E_cur:.4e}")

    for step in range(1, max_steps + 1):
        # --- propose pixel exchange (1 <-> 0), keeps volume fraction ---
        # try a few times to find opposite phases
        for _ in range(16):
            y1 = rng.integers(0, H)
            x1 = rng.integers(0, W)
            y2 = rng.integers(0, H)
            x2 = rng.integers(0, W)
            if mask[y1, x1] != mask[y2, x2]:
                break
        else:
            # failed to find (highly unlikely if vf is not 0 or 1)
            loss_hist.append(E_cur)
            theta *= alpha
            continue

        # Swap phases
        mask[y1, x1], mask[y2, x2] = mask[y2, x2], mask[y1, x1]

        # Recompute descriptors (brute-force; can be optimised further)
        S2_new = compute_S2_radial(mask, r_values)
        L_new, _ = compute_lineal_L(mask, z_values)
        E_new = energy_S2_L(S2_new, S2_target, L_new, L_target, w_S2, w_L)

        dE = E_new - E_cur

        # Metropolis acceptance
        accept = False
        if dE <= 0.0:
            accept = True
        else:
            if theta > 0.0:
                if rng.random() < np.exp(-dE / theta):
                    accept = True

        if accept:
            S2_cur = S2_new
            L_cur = L_new
            E_cur = E_new

            if E_cur < best_E:
                best_E = E_cur
                best_mask = mask.copy()
        else:
            # revert swap
            mask[y1, x1], mask[y2, x2] = mask[y2, x2], mask[y1, x1]

        loss_hist.append(E_cur)
        theta *= alpha

        if (step % print_every) == 0:
            print(
                f"[YT-SA] step {step:5d}, E = {E_cur:.4e}, "
                f"best = {best_E:.4e}, theta = {theta:.3e}"
            )

    print(f"[YT-SA] finished: best energy = {best_E:.4e}")
    return best_mask, np.asarray(loss_hist, dtype=float), best_E


# ---------------- 3. Main driver ----------------

def main():
    image_paths = sorted(IMG_DIR.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {IMG_DIR}")

    print(f"[MAIN] Found {len(image_paths)} images in {IMG_DIR}")

    masks: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    vfs: List[float] = []

    heights: List[int] = []
    widths: List[int] = []

    for p in image_paths:
        gray01 = imread_gray01(p)
        mask = build_fibre_mask_from_binary(gray01)

        H, W = mask.shape
        vf = float(mask.mean())

        masks.append(mask)
        shapes.append((H, W))
        vfs.append(vf)
        heights.append(H)
        widths.append(W)

        print(f"  {p.name}: shape={mask.shape}, vf={vf:.4f}")

    # --- target volume fraction (area fraction of fibres) ---
    vf_target = float(np.mean(vfs))
    print(f"[MAIN] Average fibre volume fraction over samples: {vf_target:.4f}")

    # --- define radii r and segment lengths z ---
    min_H = int(min(heights))
    min_W = int(min(widths))
    min_dim = min(min_H, min_W)

    # S2 up to half of the smallest dimension
    r_max = 0.5 * float(min_dim)
    n_r_bins = 40
    r_values = np.linspace(r_max / n_r_bins, r_max, n_r_bins)

    # Horizontal lineal-path up to half width
    z_max = 0.5 * float(min_W)
    n_z_bins = 40
    z_values = np.linspace(1.0, z_max, n_z_bins)

    print(
        f"[MAIN] S2 will be estimated up to r_max = {r_max:.1f} px "
        f"with {n_r_bins} bins."
    )
    print(
        f"[MAIN] L will be estimated up to z_max = {z_max:.1f} px "
        f"with {n_z_bins} bins."
    )

    # --- estimate S2_target and L_target from all masks ---
    S2_list = []
    L_list = []

    for mask in masks:
        S2_i = compute_S2_radial(mask, r_values)
        L_i, _ = compute_lineal_L(mask, z_values)
        S2_list.append(S2_i)
        L_list.append(L_i)

    S2_stack = np.stack(S2_list, axis=0)
    L_stack = np.stack(L_list, axis=0)

    S2_target = np.nanmean(S2_stack, axis=0)
    L_target = np.nanmean(L_stack, axis=0)

    # --- choose reconstruction window size ---
    H_target = int(np.median(heights))
    W_target = int(np.median(widths))
    shape_target = (H_target, W_target)

    print(f"[MAIN] Reconstruction window shape_target = {shape_target}")

    # --- initial random mask with same volume fraction ---
    N_pix = H_target * W_target
    N_fibre = int(round(vf_target * N_pix))

    print(
        f"[MAIN] target vf ≈ {vf_target:.4f} → "
        f"N_fibre ≈ {N_fibre} (out of {N_pix})"
    )

    if N_fibre <= 0 or N_fibre >= N_pix:
        raise RuntimeError("Degenerate volume fraction; cannot build nontrivial mask.")

    rng = np.random.default_rng(2025)
    flat = np.zeros(N_pix, dtype=np.uint8)
    flat[:N_fibre] = 1
    rng.shuffle(flat)
    mask_init = flat.reshape(H_target, W_target)

    print("[MAIN] Starting Yeong–Torquato SA (hybrid S2 + L)...")

    mask_best, loss_hist, best_E = sa_yt_hybrid(
        mask_init=mask_init,
        r_values=r_values,
        z_values=z_values,
        S2_target=S2_target,
        L_target=L_target,
        max_steps=2000,
        theta0=1.0,
        alpha=0.995,
        w_S2=1.0,
        w_L=1.0,
        print_every=200,
    )

    # --- compute descriptors of final best mask ---
    S2_recon = compute_S2_radial(mask_best, r_values)
    L_recon, z_px_used = compute_lineal_L(mask_best, z_values)

    OUT_DIR.mkdir(exist_ok=True)

    # --- save reconstructed image ---
    out_img = OUT_DIR / "yt_hybrid_S2_L_recon.png"
    plt.imsave(
        out_img,
        mask_best.astype(float),
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )

    # --- save descriptors and loss history ---
    np.savetxt(OUT_DIR / "yt_hybrid_r_values.txt", r_values)
    np.savetxt(OUT_DIR / "yt_hybrid_S2_target.txt", S2_target)
    np.savetxt(OUT_DIR / "yt_hybrid_S2_recon.txt", S2_recon)

    np.savetxt(OUT_DIR / "yt_hybrid_z_values.txt", z_values)
    np.savetxt(OUT_DIR / "yt_hybrid_z_px_used.txt", z_px_used)
    np.savetxt(OUT_DIR / "yt_hybrid_L_target.txt", L_target)
    np.savetxt(OUT_DIR / "yt_hybrid_L_recon.txt", L_recon)

    np.savetxt(OUT_DIR / "yt_hybrid_loss_history.txt", loss_hist)

    # --- quick comparison plots ---
    fig, ax = plt.subplots()
    ax.plot(r_values, S2_target, "o-", label="S2 target")
    ax.plot(r_values, S2_recon, "s--", label="S2 recon")
    ax.set_xlabel("r (pixels)")
    ax.set_ylabel("S2(r)")
    ax.set_title("Two-point function S2(r): target vs recon")
    ax.grid(True)
    ax.legend()
    fig.savefig(OUT_DIR / "yt_hybrid_S2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(z_values, L_target, "o-", label="L target")
    ax.plot(z_values, L_recon, "s--", label="L recon")
    ax.set_xlabel("z (pixels)")
    ax.set_ylabel("L(z)")
    ax.set_title("Lineal-path function L(z): target vs recon")
    ax.grid(True)
    ax.legend()
    fig.savefig(OUT_DIR / "yt_hybrid_L_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(loss_hist, "-")
    ax.set_xlabel("SA step")
    ax.set_ylabel("Energy E")
    ax.set_title("Yeong–Torquato SA loss history (S2 + L)")
    ax.grid(True)
    fig.savefig(OUT_DIR / "yt_hybrid_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[MAIN] Hybrid (S2 + L) YT reconstruction finished.")
    print(f"       Image saved to: {out_img}")


if __name__ == "__main__":
    main()