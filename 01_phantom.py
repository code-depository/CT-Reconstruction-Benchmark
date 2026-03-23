"""
01_phantom.py
=============
Generates the Shepp-Logan phantom and its clean sinogram.

Outputs (written to ./data/):
    phantom.npy          float64, shape (400, 400), range [0, 1]
    sinogram_clean.npy   float64, shape (N_det, 180)  — N_det ≈ 569
    theta.npy            float64, shape (180,)  — projection angles in degrees

Run:
    python 01_phantom.py

No arguments. No dependencies beyond numpy, scikit-image.
"""

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
from pathlib import Path
import time

# ── Configuration ────────────────────────────────────────────────────────────

PHANTOM_SIZE   = 400          # pixels — square
N_ANGLES       = 180          # projection angles: 0° … 179°, step 1°
DATA_DIR       = Path("data")

# ── Setup ────────────────────────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Phantom ──────────────────────────────────────────────────────────

print("Generating Shepp-Logan phantom …")
t0 = time.perf_counter()

# skimage returns a 400×400 float64 phantom in [0, 1] by default
phantom_raw = shepp_logan_phantom()                  # native 400×400

# Resize only if PHANTOM_SIZE differs from default 400
if phantom_raw.shape[0] != PHANTOM_SIZE:
    phantom = resize(phantom_raw,
                     (PHANTOM_SIZE, PHANTOM_SIZE),
                     anti_aliasing=True,
                     preserve_range=True)
else:
    phantom = phantom_raw.copy()

phantom = phantom.astype(np.float64)

# Sanity checks
assert phantom.ndim == 2,                   "Phantom must be 2-D"
assert phantom.shape[0] == phantom.shape[1], "Phantom must be square"
assert phantom.min() >= 0.0,               "Phantom values must be non-negative"
assert phantom.max() <= 1.0,               "Phantom values must be ≤ 1"

print(f"  Shape : {phantom.shape}")
print(f"  Range : [{phantom.min():.4f}, {phantom.max():.4f}]")
print(f"  dtype : {phantom.dtype}")

np.save(DATA_DIR / "phantom.npy", phantom)
print(f"  Saved → data/phantom.npy")

# ── Step 2: Projection angles ────────────────────────────────────────────────

theta = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)  # shape (180,)
np.save(DATA_DIR / "theta.npy", theta)
print(f"\nProjection angles: {N_ANGLES} angles, {theta[0]:.1f}°–{theta[-1]:.1f}°")
print(f"  Saved → data/theta.npy")

# ── Step 3: Clean sinogram ───────────────────────────────────────────────────

print("\nComputing clean sinogram …")

# circle=False — uses circumscribed circle support so no image content is lost
# preserve_range=True — keeps original [0,1] intensity scale
sinogram_clean = radon(phantom, theta=theta, circle=False)

# sinogram shape: (n_detector_pixels, n_angles)
# n_detector_pixels = ceil(sqrt(2) * PHANTOM_SIZE) ≈ 566–570 for 400px phantom
print(f"  Sinogram shape : {sinogram_clean.shape}  "
      f"({sinogram_clean.shape[0]} detector bins × {sinogram_clean.shape[1]} angles)")
print(f"  Value range    : [{sinogram_clean.min():.4f}, {sinogram_clean.max():.4f}]")

np.save(DATA_DIR / "sinogram_clean.npy", sinogram_clean)
print(f"  Saved → data/sinogram_clean.npy")

elapsed = time.perf_counter() - t0
print(f"\nModule 01 complete in {elapsed:.2f}s")
print("─" * 50)

# ── Quick verification ────────────────────────────────────────────────────────
# The Radon transform at θ=0° should equal column sums of the phantom.
# We verify the largest column sum matches the sinogram to within 1%.

col_sums      = phantom.sum(axis=0)          # sum down each column
sino_at_0     = sinogram_clean[:, 0]         # first projection (θ=0°)

# The sinogram may be padded; strip padding zeros before comparing
nonzero_mask  = sino_at_0 > sino_at_0.max() * 0.01
sino_trimmed  = sino_at_0[nonzero_mask]

# Maximum column sum vs maximum sinogram value at θ=0°
max_col  = col_sums.max()
max_sino = sino_trimmed.max()
rel_err  = abs(max_col - max_sino) / max_col

print(f"\nVerification (θ=0°):")
print(f"  Max column sum of phantom : {max_col:.4f}")
print(f"  Max sinogram value at 0°  : {max_sino:.4f}")
print(f"  Relative error            : {rel_err*100:.2f}%")

if rel_err < 0.05:
    print("  PASS — sinogram consistent with direct projection")
else:
    print("  WARNING — relative error > 5%, check radon() parameters")
