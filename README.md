# Microstructure Reconstruction Toolkit

This repository contains Python scripts for reconstructing two-phase microstructures from binary micrographs. Two complementary simulated-annealing (SA) pipelines are provided:

- **Ripley's K + circular fibre prior** (`baseline_ripley_circular.py`): matches the spatial statistics of extracted fibre centres while enforcing circular fibres with a fixed radius.
- **Area-based Yeong–Torquato SA** (`AreabasedMethod/yt_reconstruct.py` and `AreabasedMethod/hybrid_yt_reconstruct.py`): matches two-point statistics of the full binary image (or a hybrid of two-point statistics and centre-based metrics).

Both pipelines read binary input images, reconstruct new microstructures with comparable descriptors, and write images plus diagnostic plots into the `output/` directory.

## Repository layout

- `config.py` – central configuration for input/output paths and physical calibration (µm per pixel).
- `image_utils.py` – helpers to read images and convert them to binary fibre masks.
- `baseline_ripley_circular.py` – Ripley's K-based centre extraction and SA reconstruction with circular fibres.
- `AreabasedMethod/yt_reconstruct.py` – classic Yeong–Torquato SA using two-point auto-correlation of the binary mask.
- `AreabasedMethod/hybrid_yt_reconstruct.py` – hybrid variant combining two-point statistics with centre-based metrics.
- `RVE_width_patches.py` – crops non-overlapping patches whose physical width matches a target representative volume element (RVE).
- `Batch2_*/` – example binary datasets (PNG files). Files ending in `_origin` hold full-resolution disks; `_Binary` folders contain already-binarized inputs.
- `output/` – created automatically; all reconstructions and plots are written here.

## Environment setup

1. Use Python 3.10+.
2. Install the required packages (NumPy, SciPy, Matplotlib, scikit-image):

   ```bash
   pip install numpy scipy matplotlib scikit-image
   ```

## Configure inputs

Edit `config.py` to point `IMG_DIR` to your binary micrograph folder (PNG files). The file also defines:

- `OUT_DIR`: where all outputs are written (defaults to `output/`).
- `UM_PER_PX`: micrometres per pixel for your images; used to interpret physical fibre sizes and RVE widths.
- Sliding-window settings for centre extraction in the Ripley's K pipeline.

## Workflow

### (Optional) Crop RVE-width patches

If you want reconstructions on patches with a fixed physical width, run:

```bash
python RVE_width_patches.py
```

This reads all PNGs in `Batch2_NE_1000x_Binary_cen/`, crops non-overlapping columns whose width matches `RVE_WIDTH_UM`, and writes them to `output/patches_widthRVE_<width>um/`.

### Ripley's K + circular fibre prior

This pipeline matches the distribution of pairwise distances between fibre centres while enforcing circular fibres of radius derived from `UM_PER_PX`.

```bash
python baseline_ripley_circular.py
```

The script performs the following:

1. Loads binary images from `IMG_DIR`, extracts fibre masks, and detects centre candidates via distance-transform peaks.
2. Computes Ripley's K across sliding windows to obtain a target curve.
3. Runs SA on fibre centres with a circular-mask renderer to match the target K and volume fraction.
4. Saves the reconstructed binary image (`baseline_ripley_recon.png`), the K comparison plot, and the SA loss curve into `output/`.

Key parameters such as SA steps, circle radius, and VF weighting are defined near the top of `baseline_ripley_circular.py`.

### Area-based Yeong–Torquato reconstruction

Use this when you want to match two-point statistics of the full binary mask. Two variants are available:

- **Pure two-point**: `AreabasedMethod/yt_reconstruct.py`
- **Hybrid (two-point + centre statistics)**: `AreabasedMethod/hybrid_yt_reconstruct.py`

Run either script directly:

```bash
python AreabasedMethod/yt_reconstruct.py
# or
python AreabasedMethod/hybrid_yt_reconstruct.py
```

Both scripts:

1. Load the binary mask from `IMG_DIR` and compute descriptors (auto-correlation and/or radial S2 + centre spacing).
2. Initialize a random binary field with the target volume fraction.
3. Perform SA by swapping pixels, accepting moves with the Metropolis criterion to minimize descriptor mismatch.
4. Save the best-found reconstruction and loss history plots to `output/`.

The hybrid script lets you combine S2 matching with Ripley's K-style centre matching; adjust weighting terms in the "hybrid loss" section near the top of the file.

### Diffusion model

Both the model and the output are online in HPC Storage. This can be made available on request.

## Tips

- Keep images binary (fibres = black/1, background = white/0); `image_utils.build_fibre_mask_from_binary` uses a fixed threshold of 0.5 on normalized grayscale inputs.
- If you change the physical calibration (`UM_PER_PX`), re-run any scripts that depend on fibre radius or RVE size.
- For reproducibility, both SA pipelines use fixed NumPy random seeds unless you modify the corresponding `default_rng` calls.

## Outputs

All reconstruction images and diagnostic plots are placed in `output/` by default. Check the console logs for exact filenames after each run.
