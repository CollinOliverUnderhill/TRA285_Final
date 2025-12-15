import numpy as np


def sliding_area_fraction_stats(mask: np.ndarray,
                                win_px: float,
                                step_px: float):

    H, W = mask.shape
    win_px = max(1, int(round(win_px)))
    step_px = max(1, int(round(step_px)))

    # If window is wider than the image, fall back to a single global window
    if win_px >= W:
        fibre_cols = (mask == 1).sum(axis=0)
        frac = float(fibre_cols.sum()) / float(H * W)
        fracs = np.array([frac], dtype=float)
        mean = frac
        std = 0.0
        cv = np.nan
        return fracs, mean, std, cv, len(fracs)

    # Count fibre pixels per column
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
    """
    Approximate uncertainty of CV for n independent windows.

    This keeps the same formula as in the original code:
        delta_cv = cv / sqrt(2*n)
    """
    if n <= 0 or cv <= 0:
        return np.nan
    return cv / np.sqrt(2.0 * n)
