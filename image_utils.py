from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def imread_gray01(path: Path) -> np.ndarray:
    """
    Read an image and convert to grayscale normalized in [0, 1].
    This matches the original implementation.
    """
    img = plt.imread(path)
    if img.ndim == 3:
        # Convert RGB to grayscale
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
    """
    Convert normalized grayscale image to binary mask
    (1 = fibre, 0 = background).
    Threshold is kept exactly as in the original code: gray < 0.5.
    """
    return (gray01 < 0.5).astype(np.uint8)
