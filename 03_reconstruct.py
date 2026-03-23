"""
03_reconstruct.py
=================
Reconstructs images from all noisy sinograms using six algorithms.
Also reconstructs from the clean sinogram (noise-free ceiling).

Algorithms
----------
1. IRT        — Inverse Radon Transform, no filter (blurred baseline)
2. FBP/ramp   — Filtered backprojection, ramp filter
3. FBP/SL     — Filtered backprojection, Shepp-Logan filter
4. FBP/cos    — Filtered backprojection, cosine filter
5. SART       — Simultaneous Algebraic Reconstruction Technique (iterative)

Outputs (written to ./results/recons/)
---------------------------------------
recon_{algo}_{dose_pct}_{seed}.npy    float64, shape (400, 400)
recon_{algo}_clean.npy                clean-sinogram ceiling reconstruction
timing.json                           wall-clock seconds per algorithm (mean over doses/seeds)

Notes
-----
- All reconstructions are clipped to [0, 1] before saving (unphysical negative
  values from ringing artefacts are zeroed; values > 1 are capped).
- SART output is 566×566 (matches detector count); cropped to 400×400 on save.
- Total run time: ~20 min on a standard laptop (SART dominates).

Run
---
    python 03_reconstruct.py

Reads:  data/phantom.npy
        data/theta.npy
        data/sinogram_clean.npy
        data/noisy/sino_d{dose}_{seed}.npy   (30 files from 02_noise.py)
"""

import numpy as np
import json
import time
import warnings
from pathlib import Path

from skimage.transform import radon, iradon, iradon_sart

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = Path("data")
NOISY_DIR   = DATA_DIR / "noisy"
RESULTS_DIR = Path("results") / "recons"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOSE_LEVELS = [100, 75, 50, 25, 10, 1]
SEEDS       = [42, 43, 44, 45, 46]

PHANTOM_SIZE = 400   # target output size (SART returns 566×566; we crop)

# Algorithm identifiers — used in filenames and tables
ALGO_KEYS = ["IRT", "FBP_ramp", "FBP_SL", "FBP_cos", "SART"]

# ── Load shared inputs ────────────────────────────────────────────────────────

print("Loading inputs …")
phantom        = np.load(DATA_DIR / "phantom.npy")          # (400, 400)
theta          = np.load(DATA_DIR / "theta.npy")            # (180,)
sinogram_clean = np.load(DATA_DIR / "sinogram_clean.npy")   # (566, 180)

N = phantom.shape[0]   # 400
print(f"  Phantom     : {phantom.shape}")
print(f"  Sinogram    : {sinogram_clean.shape}")
print(f"  Angles      : {len(theta)}")

# ── Utility: post-processing ──────────────────────────────────────────────────

def postprocess(recon: np.ndarray, target_size: int = PHANTOM_SIZE) -> np.ndarray:
    """
    Crop to target_size × target_size and clip to [0, 1].

    The iradon() family may return an image slightly larger than the input
    phantom (e.g. 566×566 for SART with circle=False). Centre-crop to
    target_size so all reconstructions share the same shape for metric
    computation.

    Clipping to [0,1] removes unphysical negative values (ringing artefacts
    in FBP) and super-unit values. This is standard practice in CT
    reconstruction evaluation — see Willemink & Noël (2019).
    """
    h, w = recon.shape
    if h != target_size or w != target_size:
        r0 = (h - target_size) // 2
        c0 = (w - target_size) // 2
        recon = recon[r0:r0 + target_size, c0:c0 + target_size]
    return np.clip(recon, 0.0, 1.0)


# ── Algorithm implementations ─────────────────────────────────────────────────

def reconstruct_irt(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Unfiltered backprojection — theoretical baseline, known to produce blurred images."""
    return iradon(sinogram, theta=theta, filter_name=None, circle=False)


def reconstruct_fbp_ramp(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """FBP with ramp filter — maximum resolution, noise-amplifying."""
    return iradon(sinogram, theta=theta, filter_name='ramp', circle=False)


def reconstruct_fbp_sl(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """FBP with Shepp-Logan filter — balanced resolution/noise, clinical standard."""
    return iradon(sinogram, theta=theta, filter_name='shepp-logan', circle=False)


def reconstruct_fbp_cos(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """FBP with cosine filter — aggressive noise suppression at cost of resolution."""
    return iradon(sinogram, theta=theta, filter_name='cosine', circle=False)


def reconstruct_sart(sinogram: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    SART — Simultaneous Algebraic Reconstruction Technique.

    Uses skimage's iradon_sart with default parameters:
    - relaxation = 0.15  (step size per iteration)
    - 1 pass through all angles per iteration
    Returns a 566×566 array; caller crops to PHANTOM_SIZE.

    Reference: Andersen & Kak (1984), Ultrason. Imaging 6:81-94.
    """
    return iradon_sart(sinogram, theta=theta)


# Map algorithm key → reconstruction function
ALGO_FNS = {
    "IRT":      reconstruct_irt,
    "FBP_ramp": reconstruct_fbp_ramp,
    "FBP_SL":   reconstruct_fbp_sl,
    "FBP_cos":  reconstruct_fbp_cos,
    "SART":     reconstruct_sart,
}

# ── Clean-sinogram ceiling reconstructions ────────────────────────────────────

print("\nReconstructing from CLEAN sinogram (algorithm ceilings) …")

ceiling_times = {}
for algo in ALGO_KEYS:
    fn = ALGO_FNS[algo]
    t0 = time.perf_counter()
    recon_raw = fn(sinogram_clean, theta)
    elapsed   = time.perf_counter() - t0

    recon = postprocess(recon_raw)
    out_path = RESULTS_DIR / f"recon_{algo}_clean.npy"
    np.save(out_path, recon)
    ceiling_times[algo] = round(elapsed, 3)

    print(f"  {algo:10s}  {elapsed:6.1f}s  shape_before={recon_raw.shape}  "
          f"saved → {out_path.name}")

# ── Noisy reconstructions ─────────────────────────────────────────────────────

total_runs    = len(ALGO_KEYS) * len(DOSE_LEVELS) * len(SEEDS)
completed     = 0
algo_times    = {a: [] for a in ALGO_KEYS}   # collect per-algo timing

print(f"\nReconstructing {total_runs} noisy cases "
      f"({len(ALGO_KEYS)} algos × {len(DOSE_LEVELS)} doses × {len(SEEDS)} seeds) …")
print("This will take approximately 25–35 minutes (SART is slow).\n")

grand_t0 = time.perf_counter()

for algo in ALGO_KEYS:
    fn = ALGO_FNS[algo]
    print(f"── {algo} ─────────────────────────────────────────")

    for dose_pct in DOSE_LEVELS:
        dose_times = []

        for seed in SEEDS:
            sino_path = NOISY_DIR / f"sino_d{dose_pct:03d}_{seed}.npy"
            sinogram  = np.load(sino_path)

            t0        = time.perf_counter()
            recon_raw = fn(sinogram, theta)
            elapsed   = time.perf_counter() - t0

            recon    = postprocess(recon_raw)
            out_path = RESULTS_DIR / f"recon_{algo}_d{dose_pct:03d}_{seed}.npy"
            np.save(out_path, recon)

            dose_times.append(elapsed)
            algo_times[algo].append(elapsed)
            completed += 1

        mean_t = sum(dose_times) / len(dose_times)
        print(f"  dose={dose_pct:3d}%  mean_time={mean_t:.2f}s  "
              f"[{completed}/{total_runs} runs done]")

    algo_mean = sum(algo_times[algo]) / len(algo_times[algo])
    print(f"  → {algo} mean per reconstruction: {algo_mean:.2f}s\n")

# ── Save timing summary ───────────────────────────────────────────────────────

timing_summary = {
    algo: {
        "mean_s":   round(sum(algo_times[algo]) / len(algo_times[algo]), 3),
        "min_s":    round(min(algo_times[algo]), 3),
        "max_s":    round(max(algo_times[algo]), 3),
        "ceiling_s": ceiling_times.get(algo, None),
        "n_runs":   len(algo_times[algo]),
    }
    for algo in ALGO_KEYS
}

timing_path = Path("results") / "timing.json"
with open(timing_path, "w") as f:
    json.dump(timing_summary, f, indent=2)

grand_elapsed = time.perf_counter() - grand_t0
print(f"All {total_runs} reconstructions complete in "
      f"{grand_elapsed/60:.1f} min")
print(f"Timing summary saved → results/timing.json")
print("─" * 50)

# ── Verification ──────────────────────────────────────────────────────────────

print("\nVerification — checking saved files …")
expected_noisy  = len(ALGO_KEYS) * len(DOSE_LEVELS) * len(SEEDS)
expected_clean  = len(ALGO_KEYS)
found_noisy     = len(list(RESULTS_DIR.glob("recon_*_d*_*.npy")))
found_clean     = len(list(RESULTS_DIR.glob("recon_*_clean.npy")))

print(f"  Noisy recons : expected {expected_noisy}, found {found_noisy}  "
      f"{'PASS' if found_noisy == expected_noisy else 'FAIL'}")
print(f"  Clean recons : expected {expected_clean}, found {found_clean}  "
      f"{'PASS' if found_clean == expected_clean else 'FAIL'}")

# Spot-check: SART full-dose seed 42 should be 400×400 and in [0,1]
spot = np.load(RESULTS_DIR / "recon_SART_d100_42.npy")
print(f"\n  Spot-check recon_SART_d100_42.npy:")
print(f"    Shape : {spot.shape}  (expected (400, 400))")
print(f"    Range : [{spot.min():.4f}, {spot.max():.4f}]  (expected [0, 1])")
print(f"    {'PASS' if spot.shape == (400, 400) and spot.min() >= 0 and spot.max() <= 1 else 'FAIL'}")

print("\nModule 03 complete.")
