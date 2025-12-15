from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from skimage import morphology, feature  # distance transform + peak_local_max

from config import IMG_DIR, OUT_DIR, UM_PER_PX
from image_utils import imread_gray01, build_fibre_mask_from_binary


# ---------------- 0. Global settings ----------------

# Weight of volume-fraction term in SA energy.
# Energy: L = ||K - K_target||^2 + VF_WEIGHT * (vf_recon - vf_target)^2
VF_WEIGHT = 1.0


# ---------------- 0. Directories (small vs origin) ----------------

# IMG_DIR is assumed to point to the "small" disks (used for centre extraction).
# We derive ORIGIN_DIR (same patches but full-size disks) from the folder name.
if IMG_DIR.name.endswith("_small"):
    ORIGIN_DIR = IMG_DIR.parent / (IMG_DIR.name[:-6] + "_origin")
else:
    ORIGIN_DIR = IMG_DIR.parent / (IMG_DIR.name + "_origin")

print(f"[INFO] IMG_DIR (small for centres)  = {IMG_DIR}")
print(f"[INFO] ORIGIN_DIR (full for vf)     = {ORIGIN_DIR}")

if not ORIGIN_DIR.exists():
    print("[WARN] ORIGIN_DIR does not exist, falling back to IMG_DIR for vf_target.")
    ORIGIN_DIR = IMG_DIR


# ---------------- 0. Fibre geometry: 5 µm, 180 px = 10 µm ----------------

FIBRE_DIAM_UM = 5.0
PX_PER_UM = 1.0 / UM_PER_PX
FIBRE_DIAM_PX = FIBRE_DIAM_UM * PX_PER_UM
FIBRE_RADIUS_PX = 0.5 * FIBRE_DIAM_PX
FIBRE_RADIUS_INT = int(round(FIBRE_RADIUS_PX))

print(f"[INFO] UM_PER_PX = {UM_PER_PX:.6f} µm/px")
print(f"[INFO] PX_PER_UM = {PX_PER_UM:.3f} px/µm")
print(
    f"[INFO] Fibre diameter = {FIBRE_DIAM_UM:.2f} µm "
    f"≈ {FIBRE_DIAM_PX:.1f} px; radius ≈ {FIBRE_RADIUS_PX:.1f} px "
    f"(rounded to {FIBRE_RADIUS_INT} px)"
)


# ---------------- 1. Fibre centres: distance-transform peaks ----------------

def extract_fibre_centres_from_mask(
    mask: np.ndarray,
    radius_px: float,
    area_factor_low: float = 0.2,
    area_factor_high: float = 5.0,
) -> np.ndarray:
    """
    Extract fibre centres using distance transform + local maxima.
    Returns array of shape (N_centres, 2) with [y, x] in pixels.
    """
    fibres = mask.astype(bool)

    # remove small objects
    fibres = morphology.remove_small_objects(fibres, 50)

    # distance transform (inside fibres)
    dist = ndimage.distance_transform_edt(fibres)

    # min distance between peaks ~ 0.6 * radius
    min_dist = max(1, int(radius_px * 0.6))
    # ignore very shallow peaks (too close to boundary/noise)
    thr = 0.3 * radius_px

    # peak_local_max: returns (row, col) = (y, x)
    coords = feature.peak_local_max(
        dist,
        min_distance=min_dist,
        threshold_abs=thr,
        exclude_border=False,
    )

    if coords.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    return coords.astype(np.float64)


# ---------------- 2. Ripley's K estimator (no edge correction) ----------------

def estimate_K_for_centres(
    centres: np.ndarray,
    shape: Tuple[int, int],
    r_values: np.ndarray,
) -> Optional[np.ndarray]:
    """
    K_hat(r) = |W| / [N (N-1)] * sum_{i != j} 1(d_ij <= r)
    """
    N = centres.shape[0]
    if N < 2:
        return None

    H, W = shape
    area = float(H * W)

    coords = centres.astype(np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    iu, ju = np.triu_indices(N, k=1)
    dists = np.sqrt(dist_sq[iu, ju])

    d_sorted = np.sort(dists)
    n_pairs = d_sorted.size
    if n_pairs == 0:
        return None

    K_vals = np.zeros_like(r_values, dtype=np.float64)
    denom = float(N * (N - 1))

    for k, r in enumerate(r_values):
        count = int(np.searchsorted(d_sorted, r, side="right"))
        K_vals[k] = area * (2.0 * count) / denom

    return K_vals


# ---------------- 5. Centres → binary circular mask ----------------

def centres_to_mask(
    centres: np.ndarray,
    shape: Tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """
    Convert centres (possibly outside the window) into a binary mask
    of solid disks of given radius, clipped to the window [0,H)×[0,W).
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for (cy, cx) in centres:
        cyi = int(round(cy))
        cxi = int(round(cx))

        # Define sub-window intersection with the image
        i_min = max(0, cyi - radius_px)
        i_max = min(H, cyi + radius_px + 1)
        j_min = max(0, cxi - radius_px)
        j_max = min(W, cxi + radius_px + 1)

        if i_min >= i_max or j_min >= j_max:
            # Circle does not intersect the window at all
            continue

        sub = mask[i_min:i_max, j_min:j_max]
        hh, ww = sub.shape
        yy, xx = np.indices((hh, ww))
        yy = yy + i_min
        xx = xx + j_min

        circle = (yy - cyi) ** 2 + (xx - cxi) ** 2 <= radius_px ** 2
        sub[circle] = 1

    return mask


# ---------------- 3. RSA init: non-overlapping centres ----------------

def rsa_initial_centres(
    shape: Tuple[int, int],
    N_target: int,
    radius_px: int,
    max_attempts_per_point: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Random sequential adsorption (RSA) of non-overlapping disks.
    Centres are allowed in [-R, H+R]×[-R, W+R], so that disks
    may be partially clipped by the image window.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H, W = shape
    centres: List[Tuple[float, float]] = []

    max_total_attempts = N_target * max_attempts_per_point
    attempts = 0
    min_dist_sq = (2 * radius_px) ** 2

    while len(centres) < N_target and attempts < max_total_attempts:
        attempts += 1
        y = float(rng.uniform(-radius_px, H + radius_px))
        x = float(rng.uniform(-radius_px, W + radius_px))

        ok = True
        for (cy, cx) in centres:
            dy = y - cy
            dx = x - cx
            if dy * dy + dx * dx < min_dist_sq:
                ok = False
                break
        if not ok:
            continue

        centres.append((y, x))

    centres_arr = np.asarray(centres, dtype=np.float64)
    print(
        f"[RSA init] requested N = {N_target}, "
        f"placed N = {centres_arr.shape[0]}, attempts = {attempts}"
    )
    return centres_arr


# ---------------- 4. SA on centres to match K_target (+ vf term) ----------------

def sa_match_K(
    r_values: np.ndarray,
    K_target: np.ndarray,
    shape: Tuple[int, int],
    centres_init: np.ndarray,
    radius_px: int,
    max_steps: int = 2000,
    step_sigma: float = 10.0,
    theta0: float = 1.0,
    alpha: float = 0.995,
    vf_target: Optional[float] = None,
    vf_weight: float = 0.0,
    print_every: int = 200,
) -> Tuple[np.ndarray, List[float]]:
    """
    Simulated annealing on centres to match K_target(r).

    Energy:
        L = sum_r [K(r) - K_target(r)]^2
            + vf_weight * (vf_recon - vf_target)^2   (if enabled)

    where vf_recon is the volume fraction (area fraction) of the
    reconstructed disk mask.
    """
    rng = np.random.default_rng()
    H, W = shape

    centres = centres_init.copy()
    N = centres.shape[0]

    # --- initial K and vf ---
    K_cur = estimate_K_for_centres(centres, shape, r_values)
    if K_cur is None:
        raise RuntimeError("Not enough centres to estimate K in SA.")

    diff = K_cur - K_target
    L_K_cur = float(np.dot(diff, diff))

    use_vf = vf_target is not None and vf_weight > 0.0
    if use_vf:
        mask_cur = centres_to_mask(centres, shape, radius_px)
        vf_cur = float(mask_cur.mean())
        L_vf_cur = (vf_cur - vf_target) ** 2
        L_cur = L_K_cur + vf_weight * L_vf_cur
        print(
            f"[SA] initial: L_K = {L_K_cur:.4e}, vf = {vf_cur:.4f}, "
            f"vf_target = {vf_target:.4f}, L_vf = {L_vf_cur:.4e}, "
            f"vf_weight = {vf_weight:.3g}"
        )
    else:
        L_cur = L_K_cur
        print(f"[SA] initial: L_K = {L_K_cur:.4e} (no vf term)")

    best_centres = centres.copy()
    best_loss = L_cur

    theta = float(theta0)
    loss_hist: List[float] = [L_cur]

    min_dist_sq = (2 * radius_px) ** 2

    print(
        f"[SA] initial loss = {L_cur:.4e}, N = {N}, "
        f"theta0 = {theta0}, step_sigma = {step_sigma}"
    )

    for step in range(1, max_steps + 1):
        idx = rng.integers(0, N)
        old_y, old_x = centres[idx]

        # propose new location
        new_y = old_y + rng.normal(scale=step_sigma)
        new_x = old_x + rng.normal(scale=step_sigma)

        # allow centres in [-R, H+R] × [-R, W+R]
        new_y = float(np.clip(new_y, -radius_px, H + radius_px))
        new_x = float(np.clip(new_x, -radius_px, W + radius_px))

        # non-overlap constraint
        ok = True
        for j in range(N):
            if j == idx:
                continue
            dy = new_y - centres[j, 0]
            dx = new_x - centres[j, 1]
            if dy * dy + dx * dx < min_dist_sq:
                ok = False
                break

        if not ok:
            loss_hist.append(L_cur)
            theta *= alpha
            continue

        centres[idx, 0] = new_y
        centres[idx, 1] = new_x

        K_new = estimate_K_for_centres(centres, shape, r_values)
        if K_new is None:
            # revert
            centres[idx, 0] = old_y
            centres[idx, 1] = old_x
            loss_hist.append(L_cur)
            theta *= alpha
            continue

        diff_new = K_new - K_target
        L_K_new = float(np.dot(diff_new, diff_new))

        if use_vf:
            mask_new = centres_to_mask(centres, shape, radius_px)
            vf_new = float(mask_new.mean())
            L_vf_new = (vf_new - vf_target) ** 2
            L_new = L_K_new + vf_weight * L_vf_new
        else:
            L_new = L_K_new

        dL = L_new - L_cur

        accept = False
        if dL <= 0.0:
            accept = True
        else:
            if theta > 0.0 and rng.random() < np.exp(-dL / theta):
                accept = True

        if accept:
            L_cur = L_new
            K_cur = K_new
            if L_cur < best_loss:
                best_loss = L_cur
                best_centres = centres.copy()
        else:
            # revert
            centres[idx, 0] = old_y
            centres[idx, 1] = old_x

        loss_hist.append(L_cur)
        theta *= alpha

        if (step % print_every) == 0:
            print(
                f"[SA] step {step:5d}, L = {L_cur:.4e}, "
                f"best = {best_loss:.4e}, theta = {theta:.3e}"
            )

    print(f"[SA] finished: best loss = {best_loss:.4e}")
    return best_centres, loss_hist


# ---------------- 6. Main driver ----------------

def main():
    image_paths = sorted(IMG_DIR.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {IMG_DIR}")

    print(f"[MAIN] Found {len(image_paths)} images in {IMG_DIR}")

    masks: List[np.ndarray] = []
    shapes: List[Tuple[int, int]] = []
    vfs_small: List[float] = []
    centres_list: List[np.ndarray] = []

    heights: List[int] = []
    widths: List[int] = []

    # ---- 6.1 Read SMALL images: for centres + small-vf (info only) ----
    for p in image_paths:
        gray01 = imread_gray01(p)
        mask = build_fibre_mask_from_binary(gray01)

        H, W = mask.shape
        vf_small = float(mask.mean())

        masks.append(mask)
        shapes.append((H, W))
        vfs_small.append(vf_small)
        heights.append(H)
        widths.append(W)

        centres = extract_fibre_centres_from_mask(
            mask,
            radius_px=FIBRE_RADIUS_PX,
            area_factor_low=0.2,
            area_factor_high=5.0,
        )
        centres_list.append(centres)

        print(
            f"  [small] {p.name}: shape={mask.shape}, vf_small={vf_small:.4f}, "
            f"N_centres={centres.shape[0]}"
        )

    # ---- 6.2 Read ORIGIN images: for true vf_target ----
    origin_paths = sorted(ORIGIN_DIR.glob("*.png"))
    if not origin_paths:
        print(f"[WARN] No PNG images found in ORIGIN_DIR={ORIGIN_DIR}. "
              f"vf_target will be estimated from IMG_DIR instead.")
        if not vfs_small:
            raise RuntimeError("No images available to estimate vf_target.")
        vf_target = float(np.mean(vfs_small))
    else:
        vfs_origin: List[float] = []
        for p in origin_paths:
            gray01_o = imread_gray01(p)
            mask_o = build_fibre_mask_from_binary(gray01_o)
            vf_o = float(mask_o.mean())
            vfs_origin.append(vf_o)
        vf_target = float(np.mean(vfs_origin))
        print(f"[MAIN] Average fibre volume fraction over ORIGIN samples: {vf_target:.4f}")
        if vfs_small:
            print(f"[MAIN] (For reference) average vf over SMALL samples: {np.mean(vfs_small):.4f}")

    # ---- point density λ_i = N_i / A_i and average to λ_target ----
    lambda_list: List[float] = []
    for centres, (H, W) in zip(centres_list, shapes):
        A = float(H * W)
        N = centres.shape[0]
        if A > 0 and N > 0:
            lambda_list.append(N / A)

    if not lambda_list:
        raise RuntimeError("No valid point density estimates (no centres found).")

    lambda_target = float(np.mean(lambda_list))
    print(
        f"[MAIN] Average point density λ_target = "
        f"{lambda_target:.6g} points/px²"
    )

    # ---------------- Ripley's K settings ----------------
    min_H = int(min(heights))
    min_W = int(min(widths))
    r_max = 0.5 * float(min(min_H, min_W))
    r_min = 60.0  # px
    n_r_bins = 40
    if r_min >= r_max:
        raise RuntimeError(
            f"r_min={r_min} ≥ r_max={r_max}, 请调小 r_min 或检查图像尺寸。"
        )

    r_values = np.linspace(r_min, r_max, n_r_bins)
    r_values_um = r_values * UM_PER_PX  # for plotting on µm axis

    r_max_um = r_max * UM_PER_PX
    print(
        f"[MAIN] Ripley's K will be estimated up to "
        f"r_max = {r_max:.1f} px (~{r_max_um:.2f} µm) "
        f"with {n_r_bins} bins."
    )

    # ---- compute K_i for each image, collect only valid ones ----
    K_list: List[np.ndarray] = []
    K_names: List[str] = []
    for p, centres, shape in zip(image_paths, centres_list, shapes):
        K_i = estimate_K_for_centres(centres, shape, r_values)
        if K_i is None:
            print(
                f"[MAIN]   Warning: not enough centres to estimate K for {p.name}."
            )
            continue
        K_list.append(K_i)
        K_names.append(p.name)

    if not K_list:
        raise RuntimeError(
            "Could not estimate K from any image (too few centres)."
        )

    K_stack = np.stack(K_list, axis=0)
    K_target = np.mean(K_stack, axis=0)
    K_std = np.std(K_stack, axis=0)
    K_min = np.min(K_stack, axis=0)
    K_max = np.max(K_stack, axis=0)

    print(
        f"[MAIN] K_stack shape = {K_stack.shape} "
        f"(N_images_with_valid_K, N_r_bins)"
    )

    OUT_DIR.mkdir(exist_ok=True)

    # ---- save all K_i(r) used in the mean ----
    np.savetxt(
        OUT_DIR / "baseline_ripley_K_all.txt",
        K_stack,
        header=(
            "Each row = one image's K(r); "
            "columns correspond to r values in pixels "
            "saved in baseline_ripley_K_r.txt and "
            "r values in micrometers saved in "
            "baseline_ripley_K_r_um.txt"
        ),
    )
    with open(OUT_DIR / "baseline_ripley_K_all_names.txt", "w", encoding="utf-8") as f:
        for name in K_names:
            f.write(name + "\n")

    # ---- plot all K_i, mean, and variability (std, min-max) ----
    fig, ax = plt.subplots()
    # all individual curves
    for K_i in K_stack:
        ax.plot(r_values_um, K_i, linewidth=0.8, alpha=0.3)
    # mean curve
    ax.plot(r_values_um, K_target, linewidth=2.0, label="mean K(r)")
    # ±1σ band
    ax.fill_between(
        r_values_um, K_target - K_std, K_target + K_std, alpha=0.2, label="±1σ"
    )
    # min-max band
    ax.fill_between(
        r_values_um, K_min, K_max, alpha=0.1, label="min-max"
    )
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("K(r)")
    ax.set_title("Ripley's K of all images")
    ax.grid(True)
    ax.legend()
    fig.savefig(
        OUT_DIR / "baseline_ripley_K_all_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---------------- Reconstruction window ----------------
    H_target = int(np.median(heights))
    W_target = int(np.median(widths))
    shape_target = (H_target, W_target)
    area_target = float(H_target * W_target)

    print(f"[MAIN] Reconstruction window shape_target = {shape_target}")

    # use λ_target * area_target to set N_target (point process constraint)
    N_target = int(round(lambda_target * area_target))
    print(
        f"[MAIN] Using point density: N_target ≈ λ_target * A_target "
        f"= {lambda_target:.6g} * {area_target:.0f} ≈ {N_target}"
    )

    if N_target < 2:
        raise RuntimeError(
            "N_target < 2, cannot build meaningful point process."
        )

    # ---------------- RSA + SA reconstruction ----------------
    centres_init = rsa_initial_centres(
        shape_target,
        N_target=N_target,
        radius_px=FIBRE_RADIUS_INT,
        max_attempts_per_point=5000,
        rng=np.random.default_rng(),
    )

    if centres_init.shape[0] < 2:
        raise RuntimeError(
            "RSA initialisation produced < 2 centres; "
            "try smaller radius or relax constraints."
        )

    centres_best, loss_hist = sa_match_K(
        r_values=r_values,
        K_target=K_target,
        shape=shape_target,
        centres_init=centres_init,
        radius_px=FIBRE_RADIUS_INT,
        max_steps=2000,
        step_sigma=10.0,
        theta0=1.0,
        alpha=0.995,
        vf_target=vf_target,
        vf_weight=VF_WEIGHT,
        print_every=200,
    )

    recon_mask = centres_to_mask(
        centres_best,
        shape=shape_target,
        radius_px=FIBRE_RADIUS_INT,
    )

    # volume fraction of reconstructed mask
    vf_recon = float(recon_mask.mean())
    print(
        f"[MAIN] Reconstructed fibre volume fraction (from mask) = {vf_recon:.4f}"
    )
    print(
        f"[MAIN] Target fibre volume fraction (from ORIGIN masks) "
        f"= {vf_target:.4f}"
    )

    K_recon = estimate_K_for_centres(
        centres_best,
        shape_target,
        r_values,
    )

    out_img = OUT_DIR / "baseline_ripley_circular.png"
    plt.imsave(
        out_img,
        recon_mask.astype(float),
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )

    # save r_values in both px and µm
    np.savetxt(OUT_DIR / "baseline_ripley_K_r.txt", r_values)
    np.savetxt(OUT_DIR / "baseline_ripley_K_r_um.txt", r_values_um)

    np.savetxt(OUT_DIR / "baseline_ripley_K_target.txt", K_target)
    if K_recon is not None:
        np.savetxt(OUT_DIR / "baseline_ripley_K_recon.txt", K_recon)

    # save SA loss history (including vf term if enabled)
    np.savetxt(
        OUT_DIR / "baseline_ripley_loss_history.txt",
        np.asarray(loss_hist, dtype=np.float64),
    )

    # plot target vs recon K  —— 横坐标用 µm
    fig, ax = plt.subplots()
    ax.plot(r_values_um, K_target, "o-", label="target (mean)")
    if K_recon is not None:
        ax.plot(r_values_um, K_recon, "s--", label="recon")
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("K(r)")
    ax.set_title("Ripley's K: target vs reconstructed (circular prior)")
    ax.grid(True)
    ax.legend()
    fig.savefig(
        OUT_DIR / "baseline_ripley_K_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # plot SA loss history
    fig, ax = plt.subplots()
    ax.plot(loss_hist, "-")
    ax.set_xlabel("SA step")
    ax.set_ylabel("Loss L")
    ax.set_title("SA loss history (K-matching + vf term)")
    ax.grid(True)
    fig.savefig(
        OUT_DIR / "baseline_ripley_loss_curve.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("[MAIN] Baseline (Ripley+circle) reconstruction finished.")
    print(f"       Image saved to: {out_img}")


if __name__ == "__main__":
    main()
