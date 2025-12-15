import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from config import IMG_DIR, OUT_DIR
from image_utils import imread_gray01, build_fibre_mask_from_binary


# ---------- 1. Descriptor: 2-point auto-correlation via FFT ----------

def two_point_autocorr(mask: np.ndarray, downsample: int = 4) -> np.ndarray:

    if downsample > 1:
        phi = mask[::downsample, ::downsample].astype(np.float64)
    else:
        phi = mask.astype(np.float64)

    # 2D FFT-based auto-correlation (periodic boundary condition)
    F = np.fft.fft2(phi)
    S2 = np.fft.ifft2(F * np.conj(F)).real
    S2 /= phi.size  # normalize

    S2 = np.fft.fftshift(S2)
    D = S2.ravel().astype(np.float64)
    return D


def loss_from_descriptor(D: np.ndarray, D_des: np.ndarray) -> float:
    """
    L = ||D - D_des||_2^2
    """
    diff = D - D_des
    return float(np.dot(diff, diff))


# ---------- 2. Utilities for SA ----------

def random_init_with_vf(shape: tuple[int, int],
                        vf_target: float,
                        rng: np.random.Generator) -> np.ndarray:

    H, W = shape
    n_pix = H * W
    n1 = int(round(vf_target * n_pix))

    flat = np.zeros(n_pix, dtype=np.uint8)
    flat[:n1] = 1
    rng.shuffle(flat)
    return flat.reshape(H, W)


def propose_swap(mask: np.ndarray, rng: np.random.Generator):

    H, W = mask.shape
    while True:
        i1 = rng.integers(0, H)
        j1 = rng.integers(0, W)
        i2 = rng.integers(0, H)
        j2 = rng.integers(0, W)
        if mask[i1, j1] != mask[i2, j2]:
            return i1, j1, i2, j2


# ---------- 3. Yeong–Torquato SA with given averaged descriptor ----------

def yeong_torquato_sa(D_des: np.ndarray,
                      shape: tuple[int, int],
                      vf_target: float,
                      max_steps: int = 50000,
                      downsample_desc: int = 4,
                      theta0: float = 1.0,
                      alpha: float = 0.995,
                      print_every: int = 1000) -> tuple[np.ndarray, list[float]]:

    rng = np.random.default_rng(0)

    H, W = shape

    # --- initial microstructure with desired volume fraction ---
    M = random_init_with_vf(shape, vf_target, rng)
    D = two_point_autocorr(M, downsample=downsample_desc)
    L = loss_from_descriptor(D, D_des)

    best_mask = M.copy()
    best_loss = L

    theta = float(theta0)
    loss_hist = [L]

    for step in range(1, max_steps + 1):
        # Propose: swap two pixels of different phase
        i1, j1, i2, j2 = propose_swap(M, rng)

        # Apply swap in-place
        M[i1, j1], M[i2, j2] = M[i2, j2], M[i1, j1]

        # Evaluate descriptor and loss
        D_new = two_point_autocorr(M, downsample=downsample_desc)
        L_new = loss_from_descriptor(D_new, D_des)
        dL = L_new - L

        # Metropolis criterion
        accept = False
        if dL <= 0.0:
            accept = True
        else:
            if theta > 0.0:
                p = np.exp(-dL / theta)
                if rng.random() < p:
                    accept = True

        if accept:
            # keep new state
            D = D_new
            L = L_new
            if L < best_loss:
                best_loss = L
                best_mask = M.copy()
        else:
            # reject: swap back
            M[i1, j1], M[i2, j2] = M[i2, j2], M[i1, j1]

        loss_hist.append(L)

        # Cool down
        theta *= alpha

        if (step % print_every) == 0:
            print(
                f"[step {step:7d}] L = {L:.4e}, "
                f"best = {best_loss:.4e}, theta = {theta:.3e}"
            )

    return best_mask, loss_hist


# ---------- 4. Driver: use ALL images in IMG_DIR ----------

def main():
    # Collect all images in IMG_DIR
    image_paths = sorted(IMG_DIR.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {IMG_DIR}")

    print(f"Found {len(image_paths)} images in {IMG_DIR}")

    target_masks = []
    vfs = []

    # Load all masks
    for p in image_paths:
        gray01 = imread_gray01(p)
        mask = build_fibre_mask_from_binary(gray01)
        target_masks.append(mask)
        vfs.append(mask.mean())
        print(f"  {p.name}: shape={mask.shape}, vf={mask.mean():.4f}")

    # Check shapes are identical
    shapes = {m.shape for m in target_masks}
    if len(shapes) != 1:
        raise RuntimeError(f"Images have different shapes: {shapes}")
    shape = target_masks[0].shape
    H, W = shape

    # Average volume fraction
    vf_target = float(np.mean(vfs))
    print(f"Average volume fraction over {len(target_masks)} images: {vf_target:.4f}")

    # Descriptor settings
    downsample_desc = 4  # you can change this (2, 4, 8, ...)

    # Compute average descriptor over all images
    print("Computing average descriptor (this may take a while)...")
    Ds = []
    for idx, m in enumerate(target_masks):
        Dm = two_point_autocorr(m, downsample=downsample_desc)
        Ds.append(Dm)
        print(f"  Descriptor {idx+1}/{len(target_masks)} computed, length={Dm.size}")

    D_stack = np.stack(Ds, axis=0)
    D_des = np.mean(D_stack, axis=0)
    print("Average descriptor ready.")

    # Run Yeong–Torquato SA reconstruction
    print("Starting Yeong–Torquato simulated annealing...")
    recon_mask, loss_hist = yeong_torquato_sa(
        D_des=D_des,
        shape=shape,
        vf_target=vf_target,
        max_steps=50000,         # adjust as needed
        downsample_desc=downsample_desc,
        theta0=1.0,
        alpha=0.995,
        print_every=2000,
    )

    # Ensure output directory exists
    OUT_DIR.mkdir(exist_ok=True)

    # Save reconstruction only (no yt_target.png)
    plt.imsave(
        OUT_DIR / "yt_recon.png",
        recon_mask.astype(float),
        cmap="gray_r",   # background white, fibres/particles black
        vmin=0,
        vmax=1,
    )

    # Save loss history
    np.savetxt(OUT_DIR / "yt_loss_history.txt", np.array(loss_hist))

    # Save loss curve
    fig, ax = plt.subplots()
    ax.plot(loss_hist, "-")
    ax.set_xlabel("SA step")
    ax.set_ylabel("Loss L(M)")
    ax.set_title("Yeong–Torquato SA (average descriptor over 10 images)")
    ax.grid(True)
    fig.savefig(OUT_DIR / "yt_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Reconstruction finished.")
    print(f"  Reconstruction saved to: {OUT_DIR / 'yt_recon.png'}")
    print(f"  Loss history saved to:  {OUT_DIR / 'yt_loss_history.txt'}")
    print(f"  Loss curve saved to:    {OUT_DIR / 'yt_loss_curve.png'}")


if __name__ == "__main__":
    main()
