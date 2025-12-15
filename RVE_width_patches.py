from pathlib import Path

import numpy as np
from skimage.io import imread, imsave

from config import UM_PER_PX

BASE_DIR = Path(__file__).resolve().parent
IMG_DIR = BASE_DIR / "Batch2_NE_1000x_Binary_cen"

# Path
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# Physical width of the RVE (in micrometers)
RVE_WIDTH_UM = 30.81

# UM_PER_PX is the physical length per pixel (µm/px), defined in config
PX_PER_UM = 1.0 / UM_PER_PX

# 将 RVE 的物理宽度转换为像素宽度
# Convert the physical RVE width to pixel width
PATCH_WIDTH_PX = int(round(RVE_WIDTH_UM * PX_PER_UM))

# 输出 patch 的目录（只按宽度 RVE 裁剪）
# Directory where cropped patches will be saved
PATCH_OUT_DIR = OUT_DIR / f"patches_widthRVE_{RVE_WIDTH_UM:.2f}um"
PATCH_OUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_width_non_overlapping(img, patch_w):

    H, W = img.shape[:2]
    n_cols = W // patch_w  # 能放下多少个完整 patch（水平）
                           # how many full patches fit horizontally

    patches = []
    for j in range(n_cols):
        x0 = j * patch_w
        x1 = x0 + patch_w
        patch = img[:, x0:x1, ...]
        patches.append((j, patch))

    return patches

def main():
    print(f"[INFO] BASE_DIR  = {BASE_DIR}")
    print(f"[INFO] IMG_DIR   = {IMG_DIR}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR}")
    print(f"[INFO] UM_PER_PX = {UM_PER_PX:.6f} µm/px")
    print(f"[INFO] PX_PER_UM = {PX_PER_UM:.3f} px/µm")
    print(f"[INFO] RVE width = {RVE_WIDTH_UM:.2f} µm "
          f"→ PATCH_WIDTH_PX = {PATCH_WIDTH_PX} px")
    print(f"[INFO] Output patch directory: {PATCH_OUT_DIR}")

    # 找到所有 PNG 图片
    # Find all PNG images
    image_paths = sorted(IMG_DIR.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {IMG_DIR}")

    total_patches = 0

    for p in image_paths:
        img = imread(p)
        H, W = img.shape[:2]

        if W < PATCH_WIDTH_PX:
            print(f"[WARN] {p.name}: width {W} px < patch width {PATCH_WIDTH_PX} px, skip.")
            continue

        patches = crop_width_non_overlapping(img, PATCH_WIDTH_PX)
        n_cols = len(patches)

        if n_cols == 0:
            print(f"[WARN] {p.name}: no full-width patches produced, skip.")
            continue

        print(f"[INFO] {p.name}: shape={H}x{W}, "
              f"patch_width={PATCH_WIDTH_PX}, n_cols={n_cols}")

        stem = p.stem
        for j, patch in patches:
            out_name = f"{stem}_c{j:02d}.png"
            out_path = PATCH_OUT_DIR / out_name
            imsave(out_path, patch)
            total_patches += 1

    print("[DONE] Cropping finished.")
    print(f"       Total patches saved: {total_patches}")
    print(f"       Patches saved to: {PATCH_OUT_DIR}")


if __name__ == "__main__":
    main()

