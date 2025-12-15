from pathlib import Path

# Base directory (current working directory)
BASE_DIR = Path.cwd()

# Input image folder (unchanged)
IMG_DIR = BASE_DIR  / "Batch2_NE_1000X_Binary"

# Output folder (all results go here, unchanged)
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# Sliding-window parameters (unchanged)
STEP_RATIO = 0.25       # step = STEP_RATIO * window_width
N_STEPS    = 40         # number of window widths
MIN_X      = None       # None -> max(10px, span/100)
MAX_X      = None       # None -> span/2

# Calibration (micrometers per pixel)  180 px = 10 µm  (unchanged)
UM_PER_PX = 10.0 / 180.0

# Dense grid for plotting fitted curves (unchanged)
DENSE_N = 800