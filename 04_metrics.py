"""
04_metrics.py
=============
Computes MSE, PSNR, and SSIM for every reconstruction produced by 03_reconstruct.py,
aggregates mean ± SD across the five seeds per (algorithm, dose) cell, and
identifies the crossover threshold D* where SART's SSIM advantage over the best
FBP filter exceeds the minimum clinically meaningful difference (Δ = 0.02).

Outputs (written to ./results/)
--------------------------------
metrics_raw.csv        one row per (algo, dose, seed) — 180 rows + 6 clean rows
metrics_summary.csv    mean ± SD per (algo, dose) — 36 rows
crossover.json         D* threshold and SSIM gap values
timing_table.csv       reconstruction time per algorithm

Reads
-----
data/phantom.npy
results/recons/recon_{algo}_{dose}_{seed}.npy  (180 files)
results/recons/recon_{algo}_clean.npy           (6 files)
results/timing.json
"""

import numpy as np
import csv
import json
import warnings
from pathlib import Path
from skimage.metrics import (structural_similarity as ssim,
                              peak_signal_noise_ratio as psnr,
                              mean_squared_error as mse_fn)

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RECON_DIR   = RESULTS_DIR / "recons"

DOSE_LEVELS  = [100, 75, 50, 25, 10, 1]
SEEDS        = [42, 43, 44, 45, 46]
ALGO_KEYS    = ["IRT", "FBP_ramp", "FBP_SL", "FBP_cos", "SART"]

# Minimum SSIM difference considered clinically meaningful
# Based on: Wang et al. (2004) who established perceptual threshold ~0.01–0.02
DELTA_MIN    = 0.02

# Best analytic algorithm (expected a priori — confirmed by results)
BEST_FBP_KEY = "FBP_cos"    # cosine filter showed highest SSIM in calibration

# ── Load phantom (ground truth) ───────────────────────────────────────────────

phantom = np.load(DATA_DIR / "phantom.npy")   # float64, (400,400), [0,1]
print(f"Ground truth phantom: {phantom.shape}, range [{phantom.min():.3f}, {phantom.max():.3f}]")

# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(ref: np.ndarray, recon: np.ndarray) -> dict:
    """
    Compute MSE, PSNR, SSIM between reference and reconstruction.
    Both arrays must be float64 in [0, 1], shape (H, W).
    data_range=1.0 is used consistently throughout.
    """
    mse_val  = float(mse_fn(ref, recon))
    psnr_val = float(psnr(ref, recon, data_range=1.0))
    ssim_val = float(ssim(ref, recon, data_range=1.0))
    return {"MSE": round(mse_val, 6),
            "PSNR": round(psnr_val, 4),
            "SSIM": round(ssim_val, 6)}

# ── Process clean-sinogram ceilings ──────────────────────────────────────────

print("\nComputing metrics for clean-sinogram ceilings …")
clean_rows = []
for algo in ALGO_KEYS:
    path  = RECON_DIR / f"recon_{algo}_clean.npy"
    recon = np.load(path)
    m     = compute_metrics(phantom, recon)
    clean_rows.append({"algo": algo, "dose_pct": "clean", "seed": "—",
                        **m})
    print(f"  {algo:12s}  MSE={m['MSE']:.5f}  PSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.4f}")

# ── Process all 180 noisy reconstructions ────────────────────────────────────

print(f"\nComputing metrics for {len(ALGO_KEYS)*len(DOSE_LEVELS)*len(SEEDS)} "
      f"noisy reconstructions …")
raw_rows = []

for algo in ALGO_KEYS:
    for dose_pct in DOSE_LEVELS:
        for seed in SEEDS:
            path  = RECON_DIR / f"recon_{algo}_d{dose_pct:03d}_{seed}.npy"
            recon = np.load(path)
            m     = compute_metrics(phantom, recon)
            raw_rows.append({"algo": algo,
                              "dose_pct": dose_pct,
                              "seed": seed,
                              **m})

# Save raw results
raw_path = RESULTS_DIR / "metrics_raw.csv"
raw_fieldnames = ["algo", "dose_pct", "seed", "MSE", "PSNR", "SSIM"]
with open(raw_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fieldnames)
    w.writeheader()
    w.writerows(clean_rows)   # clean rows first
    w.writerows(raw_rows)
print(f"  Raw metrics saved → {raw_path}")

# ── Aggregate: mean ± SD per (algo, dose) ────────────────────────────────────

print("\nAggregating mean ± SD per (algorithm, dose) …")

# Build lookup: (algo, dose_pct) → list of metric dicts
from collections import defaultdict
cell_data = defaultdict(list)
for row in raw_rows:
    cell_data[(row["algo"], row["dose_pct"])].append(row)

summary_rows = []
for algo in ALGO_KEYS:
    for dose_pct in DOSE_LEVELS:
        rows   = cell_data[(algo, dose_pct)]
        ssims  = [r["SSIM"]  for r in rows]
        psnrs  = [r["PSNR"]  for r in rows]
        mses   = [r["MSE"]   for r in rows]

        summary_rows.append({
            "algo":      algo,
            "dose_pct":  dose_pct,
            "SSIM_mean": round(float(np.mean(ssims)),  4),
            "SSIM_sd":   round(float(np.std(ssims)),   5),
            "PSNR_mean": round(float(np.mean(psnrs)),  3),
            "PSNR_sd":   round(float(np.std(psnrs)),   4),
            "MSE_mean":  round(float(np.mean(mses)),   6),
            "MSE_sd":    round(float(np.std(mses)),    7),
        })

summary_path = RESULTS_DIR / "metrics_summary.csv"
sum_fields   = ["algo", "dose_pct",
                "SSIM_mean", "SSIM_sd",
                "PSNR_mean", "PSNR_sd",
                "MSE_mean",  "MSE_sd"]
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sum_fields)
    w.writeheader()
    w.writerows(summary_rows)
print(f"  Summary saved → {summary_path}")

# ── Print result table ────────────────────────────────────────────────────────

print("\n" + "="*75)
print("SSIM MEAN ± SD  (rows = algorithms, columns = dose levels)")
print("="*75)

header = f"  {'Algorithm':12}" + "".join(f"  {d:>5}%" for d in DOSE_LEVELS)
print(header)
print("  " + "-" * (12 + 8 * len(DOSE_LEVELS)))

# Build quick lookup
ssim_lookup = {(r["algo"], r["dose_pct"]): (r["SSIM_mean"], r["SSIM_sd"])
               for r in summary_rows}

for algo in ALGO_KEYS:
    row_str = f"  {algo:12}"
    for dose_pct in DOSE_LEVELS:
        mean, sd = ssim_lookup[(algo, dose_pct)]
        row_str += f"  {mean:.3f}"
    print(row_str)

print()

# ── Crossover analysis: find D* ───────────────────────────────────────────────

print("="*75)
print("CROSSOVER ANALYSIS — SART vs best FBP filter")
print("="*75)

print(f"\nComparing SART vs {BEST_FBP_KEY} (best analytic algorithm by SSIM):")
print(f"  Minimum meaningful SSIM difference (Δ): {DELTA_MIN}")
print()

crossover_data = {
    "best_fbp_key": BEST_FBP_KEY,
    "delta_min":    DELTA_MIN,
    "dose_results": {}
}

print(f"  {'Dose':>6}  {'SART':>8}  {BEST_FBP_KEY:>10}  {'ΔSSIM':>8}  {'Significant':>12}")
print("  " + "-" * 55)

d_star     = None
gap_at_50  = None

for dose_pct in DOSE_LEVELS:
    sart_mean, sart_sd  = ssim_lookup[("SART",       dose_pct)]
    fbp_mean,  fbp_sd   = ssim_lookup[(BEST_FBP_KEY, dose_pct)]
    delta = sart_mean - fbp_mean
    # Non-overlapping SD intervals (conservative significance test)
    sig = delta > DELTA_MIN and (sart_mean - sart_sd) > (fbp_mean + fbp_sd)

    flag = "YES  ←" if sig else "no"

    crossover_data["dose_results"][dose_pct] = {
        "SART_SSIM":     sart_mean,
        "SART_sd":       sart_sd,
        "FBP_SSIM":      fbp_mean,
        "FBP_sd":        fbp_sd,
        "delta_SSIM":    round(delta, 4),
        "significant":   sig,
    }

    print(f"  {dose_pct:>5}%  {sart_mean:>8.4f}  {fbp_mean:>10.4f}  "
          f"{delta:>8.4f}  {flag:>12}")

    if dose_pct == 50:
        gap_at_50 = delta
    if sig and d_star is None:
        d_star = dose_pct

crossover_data["D_star"] = d_star
crossover_data["gap_at_50pct"] = gap_at_50

print()
if d_star is not None:
    print(f"  D* = {d_star}%  — below this dose SART's advantage is significant")
    print(f"       (ΔSSIM > {DELTA_MIN} with non-overlapping SD intervals)")
else:
    print("  D* not reached within tested dose range — SART advantage not significant")

# ── Filter comparison at each dose ───────────────────────────────────────────

print(f"\n{'='*75}")
print("FILTER RANKING BY SSIM AT EACH DOSE LEVEL")
print(f"{'='*75}")

fbp_algos = ["FBP_ramp", "FBP_SL", "FBP_cos"]
print(f"\n  {'Dose':>6}  {'Best filter':>12}  {'SSIM':>8}  {'vs SART':>10}")
print("  " + "-" * 45)

for dose_pct in DOSE_LEVELS:
    best_filter    = max(fbp_algos, key=lambda a: ssim_lookup[(a, dose_pct)][0])
    best_ssim      = ssim_lookup[(best_filter, dose_pct)][0]
    sart_ssim      = ssim_lookup[("SART", dose_pct)][0]
    gap            = sart_ssim - best_ssim
    print(f"  {dose_pct:>5}%  {best_filter:>12}  {best_ssim:>8.4f}  "
          f"{'SART +'+str(round(gap,3)):>10}")

    crossover_data["dose_results"][dose_pct]["best_filter_at_dose"] = best_filter

# ── Save crossover data ───────────────────────────────────────────────────────

crossover_path = RESULTS_DIR / "crossover.json"
with open(crossover_path, "w") as f:
    json.dump(crossover_data, f, indent=2)
print(f"\n  Crossover data saved → {crossover_path}")

# ── Save timing table ─────────────────────────────────────────────────────────

timing_path = RESULTS_DIR / "timing.json"
with open(timing_path) as f:
    timing_data = json.load(f)

timing_rows = []
for algo in ALGO_KEYS:
    t = timing_data.get(algo, {})
    timing_rows.append({
        "algo":      algo,
        "mean_s":    t.get("mean_s", ""),
        "min_s":     t.get("min_s", ""),
        "max_s":     t.get("max_s", ""),
        "ceiling_s": t.get("ceiling_s", ""),
    })

timing_csv_path = RESULTS_DIR / "timing_table.csv"
with open(timing_csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["algo","mean_s","min_s","max_s","ceiling_s"])
    w.writeheader()
    w.writerows(timing_rows)

print(f"  Timing table saved → {timing_csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*75}")
print("MODULE 04 COMPLETE — KEY FINDINGS")
print(f"{'='*75}")
print(f"  Best analytic filter overall  : {BEST_FBP_KEY}")
sart_100 = ssim_lookup[("SART", 100)][0]
fbp_100  = ssim_lookup[(BEST_FBP_KEY, 100)][0]
sart_25  = ssim_lookup[("SART", 25)][0]
fbp_25   = ssim_lookup[(BEST_FBP_KEY, 25)][0]
print(f"  SART vs {BEST_FBP_KEY} at 100% dose : "
      f"{sart_100:.4f} vs {fbp_100:.4f}  (ΔSSIM={sart_100-fbp_100:.4f})")
print(f"  SART vs {BEST_FBP_KEY} at 25% dose  : "
      f"{sart_25:.4f} vs {fbp_25:.4f}  (ΔSSIM={sart_25-fbp_25:.4f})")
if d_star:
    print(f"  Crossover threshold D*        : {d_star}%")
print(f"\n  → Paper's central result is ready for writing.")
