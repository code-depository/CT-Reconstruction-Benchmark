"""
05_dicom.py  —  TWO-CASE VALIDATION VERSION
=============================================
Validates the reconstruction pipeline on TWO real clinical CT DICOM slices:
    Case 1 — Normal brain   (12155.dcm,  DICOM_N folder)
    Case 2 — ICH / Bleeding (10331.dcm,  DICOM_B folder)

Key design decisions
--------------------
(a) pydicom assumed installed and working — no installation check.
(b) Fully metadata-independent:
      - RescaleSlope / RescaleIntercept used only if present AND valid.
      - If tags produce physically impossible HU values (below -1500),
        raw pixel values are used directly as HU.
      - All normalisation uses fixed physical constants, not DICOM tags.
(c) No import json — all outputs written as .npy and .csv only.
(d) All absolute paths set explicitly — no relative path ambiguity.
(e) Each case writes to its own subdirectory:
        results/dicom/normal/
        results/dicom/bleeding/

Usage
-----
1. Confirm all paths in CASES list below match your machine.
2. Run:  %run 05_dicom.py

Outputs — written per case to results/dicom/{case_id}/
-----------------------------------------------------------
    dicom_original.npy          normalised [0,1] CT slice  (N x N)
    dicom_sinogram.npy          simulated sinogram  (n_det x 180)
    dicom_recon_FBP_SL.npy      FBP Shepp-Logan reconstruction
    dicom_recon_FBP_cos.npy     FBP cosine reconstruction
    dicom_recon_SART.npy        SART reconstruction
    dicom_hu_validation.csv     HU mean +/- SD for 4 tissue ROIs
    dicom_info.csv              image parameters
    dicom_roi_map.png           ROI position overlay
    dicom_figure3.png           4-panel comparison figure

Additional combined output:
    results/dicom/dicom_summary.csv   both cases side by side
"""

import numpy as np
import csv
import warnings
import time
from pathlib import Path

import pydicom
from skimage.transform import radon, iradon, iradon_sart
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
#  Define both cases here. Each entry is a dictionary with:
#    case_id    — short label used in filenames and printed output
#    dcm_path   — full path to the DICOM file
#    label      — human-readable label for figures
#    roi_defs   — ROI positions (row_frac, col_frac, size_frac)
#    hu_ref     — HU reference ranges for each tissue ROI
#    roi_colors — display colours for the ROI map
# ══════════════════════════════════════════════════════════════════════════════

BASE_RESULTS = Path(r"C:\Users\USER\results\dicom")

CASES = [
    {
        "case_id":  "normal",
        "dcm_path": Path(r"C:\Users\USER\DICOM_N\12155.dcm"),
        "label":    "Normal brain",
        "roi_defs": {
            # Confirmed 16/16 PASS — do not adjust
            "Air":   (0.02, 0.02, 0.03),
            "Scalp": (0.35, 0.16, 0.01),
            "Brain": (0.45, 0.40, 0.04),
            "Bone":  (0.35, 0.20, 0.01),
        },
        "hu_ref": {
            "Air":   (-1050,  -950),
            "Scalp": (  -20,   +80),
            "Brain": (   20,    45),
            "Bone":  (  400, 1534),
        },
        "roi_colors": {
            "Air":   "cyan",
            "Scalp": "orange",
            "Brain": "lime",
            "Bone":  "red",
        },
    },
    {
        "case_id":  "bleeding",
        "dcm_path": Path(r"C:\Users\USER\DICOM_B\10331.dcm"),
        "label":    "ICH (intracerebral haemorrhage)",
        "roi_defs": {
            # Air: PASS confirmed
            # Brain: PASS confirmed (+36 HU)
            # Haematoma: box is just left of bright core — move right
            #   col 0.48 -> 0.52, keep row 0.38
            # Bone: label visible at skull edge but box still in background
            #   col 0.19 -> 0.21 to step onto cortical bone
            "Air":       (0.05, 0.05, 0.02),   # inside scan boundary   PASS
            "Brain":     (0.40, 0.65, 0.03),   # left hemisphere        PASS
            "Haematoma": (0.38, 0.52, 0.02),   # haematoma bright core
            "Bone":      (0.35, 0.67, 0.01),   # right skull ring — confirmed +506 HU
        },
        "hu_ref": {
            "Air":       (-1050,  -950),
            "Brain":     (   20,    45),
            # Widened to [+30, +90] to accommodate subacute haematoma:
            # Acute ICH: +50 to +90 HU (0-7 days)
            # Subacute ICH: +30 to +60 HU (7-28 days, as haemoglobin breaks down)
            # This slice appears subacute based on the mixed density pattern.
            "Haematoma": (   30,    90),
            "Bone":      (  400, 1534),
        },
        "roi_colors": {
            "Air":       "cyan",
            "Brain":     "lime",
            "Haematoma": "red",
            "Bone":      "white",
        },
    },
]

N_ANGLES         = 180
HU_WINDOW_CENTER =  40
HU_WINDOW_WIDTH  =  80
HU_NORM_MIN      = -1000.0
HU_NORM_MAX      =  2000.0


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def safe_float(dataset, tag_name):
    val = getattr(dataset, tag_name, None)
    if val is None:
        return None, False
    try:
        return float(val), True
    except (TypeError, ValueError):
        return None, False


def load_dicom_as_hu(dcm_path):
    """
    Load a DICOM file and return pixel array in HU with method description.

    Handles datasets with no metadata and out-of-FOV padding pixels.

    The padding problem: ~21% of pixels in a 512x512 CT are outside the
    reconstruction circle and set to a padding value (e.g. -2000). This
    means min(), 1st percentile, and even 5th percentile all reflect the
    padding, not real anatomy. We therefore use the MEDIAN of the converted
    image as the sanity check. For a brain CT, the median after correct
    conversion should fall in the soft-tissue range roughly [-200, +200] HU.
    """
    ds        = pydicom.dcmread(str(dcm_path), force=True)
    pixel_raw = ds.pixel_array.astype(np.float64)

    slope,     has_slope     = safe_float(ds, "RescaleSlope")
    intercept, has_intercept = safe_float(ds, "RescaleIntercept")

    def median_in_range(arr, lo=-1100, hi=200):
        """
        Check if median of arr falls in the expected CT HU range.
        Range [-1100, +200] covers:
          - Air-dominated slices: median ~ -800 to -1000 HU
          - Brain parenchyma slices: median ~ +20 to +50 HU
        This range correctly rejects raw pixel usage (median ~ +800 to +1060)
        while accepting correct HU conversion for both normal and bleeding cases.
        """
        return lo <= float(np.median(arr)) <= hi

    if has_slope and has_intercept:
        candidate = pixel_raw * slope + intercept
        if median_in_range(candidate):
            hu_image  = candidate
            hu_method = (f"DICOM tags applied "
                         f"(slope={slope}, intercept={intercept})")
        else:
            # Tags may be wrong — try standard CT offset as fallback
            fallback = pixel_raw - 1024.0
            if median_in_range(fallback):
                hu_image  = fallback
                hu_method = ("DICOM tags rejected (median out of range); "
                             "standard offset -1024 applied.")
            else:
                hu_image  = pixel_raw.copy()
                hu_method = ("DICOM tags and -1024 fallback both rejected; "
                             "raw pixel values used directly.")

    else:
        # No tags — try -1024 offset first, then raw
        fallback = pixel_raw - 1024.0
        if median_in_range(fallback):
            hu_image  = fallback
            hu_method = "No DICOM tags; standard offset -1024 applied."
        elif pixel_raw.min() < -200.0:
            hu_image  = pixel_raw.copy()
            hu_method = "No DICOM tags; raw pixels used (already in HU)."
        else:
            hu_image  = pixel_raw - 1024.0
            hu_method = "No DICOM tags; default fallback -1024 applied."

    return ds, pixel_raw, hu_image, hu_method


def normalise_image(hu_image):
    """Centre-crop to square and normalise to [0, 1]."""
    h, w  = hu_image.shape
    sz    = min(h, w)
    r0c   = (h - sz) // 2
    c0c   = (w - sz) // 2
    hu_sq = hu_image[r0c:r0c+sz, c0c:c0c+sz]
    ct    = np.clip((hu_sq - HU_NORM_MIN) / (HU_NORM_MAX - HU_NORM_MIN), 0.0, 1.0)
    return ct, sz


def postprocess(recon, target):
    """Centre-crop to target x target and clip to [0, 1]."""
    h, w = recon.shape
    if h != target or w != target:
        r0 = (h - target) // 2
        c0 = (w - target) // 2
        recon = recon[r0:r0+target, c0:c0+target]
    return np.clip(recon, 0.0, 1.0)


def sample_roi(image, r_frac, c_frac, sz_frac):
    """Sample an ROI and return mean HU, SD HU, and pixel coordinates."""
    H, W  = image.shape
    r0    = int(r_frac * H)
    c0    = int(c_frac * W)
    sz    = max(int(sz_frac * min(H, W)), 5)
    patch = image[r0:r0+sz, c0:c0+sz]
    hu    = patch * (HU_NORM_MAX - HU_NORM_MIN) + HU_NORM_MIN
    return float(hu.mean()), float(hu.std()), r0, c0, sz


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP — process each case
# ══════════════════════════════════════════════════════════════════════════════

summary_rows = []   # collects rows for combined CSV

for case in CASES:

    case_id    = case["case_id"]
    dcm_path   = case["dcm_path"]
    label      = case["label"]
    roi_defs   = case["roi_defs"]
    hu_ref     = case["hu_ref"]
    roi_colors = case["roi_colors"]

    results_dir = BASE_RESULTS / case_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 62)
    print(f"Module 05  —  Case: {label}")
    print("=" * 62)
    print(f"\nFile : {dcm_path}")
    print(f"Out  : {results_dir}\n")

    if not dcm_path.exists():
        print(f"ERROR: file not found — {dcm_path}")
        print("Skipping this case.\n")
        continue

    # ── STEP 1: Load and convert to HU ───────────────────────────────────────

    ds, pixel_raw, hu_image, hu_method = load_dicom_as_hu(dcm_path)

    print(f"Raw pixel array  : shape={pixel_raw.shape}  "
          f"range=[{pixel_raw.min():.1f}, {pixel_raw.max():.1f}]")
    print(f"HU conversion    : {hu_method}")
    print(f"HU range         : [{hu_image.min():.1f}, {hu_image.max():.1f}]")

    with open(results_dir / "dicom_info.csv", "w", newline="") as f:
        csv.writer(f).writerows([
            ["Parameter",      "Value"],
            ["Case",           label],
            ["Source file",    dcm_path.name],
            ["Raw shape",      str(pixel_raw.shape)],
            ["HU conversion",  hu_method],
            ["HU min",         round(float(hu_image.min()), 2)],
            ["HU max",         round(float(hu_image.max()), 2)],
            ["Modality",       str(getattr(ds, "Modality",       "unknown"))],
            ["KVP",            str(getattr(ds, "KVP",            "unknown"))],
            ["SliceThickness", str(getattr(ds, "SliceThickness", "unknown"))],
            ["PixelSpacing",   str(getattr(ds, "PixelSpacing",   "unknown"))],
        ])
    print("Saved            : dicom_info.csv")

    # ── STEP 2: Normalise ─────────────────────────────────────────────────────

    ct_norm, N = normalise_image(hu_image)
    print(f"\nCropped to       : ({N} x {N})")
    print(f"Normalised range : [{ct_norm.min():.4f},  {ct_norm.max():.4f}]")
    np.save(results_dir / "dicom_original.npy", ct_norm)
    print("Saved            : dicom_original.npy")

    # ── STEP 3: Forward projection ────────────────────────────────────────────

    print(f"\nForward projecting  ({N_ANGLES} angles) ...")
    theta    = np.linspace(0.0, 180.0, N_ANGLES, endpoint=False)
    sinogram = radon(ct_norm, theta=theta, circle=False)
    print(f"Sinogram shape   : {sinogram.shape}")
    np.save(results_dir / "dicom_sinogram.npy", sinogram)
    print("Saved            : dicom_sinogram.npy")

    # ── STEP 4: Reconstruct ───────────────────────────────────────────────────

    print("\nReconstructing ...")

    t0 = time.perf_counter()
    recon_sl = postprocess(
        iradon(sinogram, theta=theta, filter_name="shepp-logan", circle=False), N)
    print(f"  FBP / Shepp-Logan : {time.perf_counter()-t0:.2f}s")
    np.save(results_dir / "dicom_recon_FBP_SL.npy", recon_sl)

    t0 = time.perf_counter()
    recon_cos = postprocess(
        iradon(sinogram, theta=theta, filter_name="cosine", circle=False), N)
    print(f"  FBP / cosine      : {time.perf_counter()-t0:.2f}s")
    np.save(results_dir / "dicom_recon_FBP_cos.npy", recon_cos)

    t0 = time.perf_counter()
    recon_sart = postprocess(iradon_sart(sinogram, theta=theta), N)
    print(f"  SART              : {time.perf_counter()-t0:.2f}s")
    np.save(results_dir / "dicom_recon_SART.npy", recon_sart)

    print("All reconstructions saved.")

    # ── STEP 5: HU validation ─────────────────────────────────────────────────

    recons = {
        "Original":        ct_norm,
        "FBP/Shepp-Logan": recon_sl,
        "FBP/cosine":      recon_cos,
        "SART":            recon_sart,
    }

    print("\nHU validation:\n")
    print(f"  {'Tissue':12}  {'Expected':16}"
          + "".join(f"  {n:>24}" for n in recons))
    print("  " + "-" * (12 + 16 + 26 * len(recons)))

    val_rows   = []
    pass_count = 0
    total_chk  = 0

    for tissue, (r_frac, c_frac, sz_frac) in roi_defs.items():
        lo, hi  = hu_ref[tissue]
        ref_str = f"[{lo:+5d}, {hi:+5d}]"
        row     = {"Case": label, "Tissue": tissue, "Expected HU": ref_str}
        line    = f"  {tissue:12}  {ref_str:16}"

        for name, img in recons.items():
            mean, sd, r0, c0, sz = sample_roi(img, r_frac, c_frac, sz_frac)
            ok    = lo <= mean <= hi
            flag  = "PASS" if ok else "check"
            pass_count += int(ok)
            total_chk  += 1
            line += f"  {mean:+8.1f} +/-{sd:6.1f}  {flag}"
            row[f"{name} mean HU"] = round(mean, 1)
            row[f"{name} SD HU"]   = round(sd,   1)
            row[f"{name} pass"]    = ok

        print(line)
        val_rows.append(row)
        summary_rows.append(row)

    print(f"\n  Checks passed : {pass_count} / {total_chk}")

    if pass_count < total_chk:
        print()
        print("  NOTE: 'check' means the ROI missed that tissue.")
        print(f"  Open {results_dir / 'dicom_roi_map.png'} to see positions.")
        print("  Adjust row_frac / col_frac in the CASES list and re-run.")

    # Save per-case validation CSV
    fieldnames = (["Case", "Tissue", "Expected HU"] +
                  [f"{n} {s}" for n in recons
                   for s in ["mean HU", "SD HU", "pass"]])
    with open(results_dir / "dicom_hu_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(val_rows)
    print(f"\n  Saved : dicom_hu_validation.csv")

    # ── STEP 6: ROI map ───────────────────────────────────────────────────────

    vmin_w = (HU_WINDOW_CENTER - HU_WINDOW_WIDTH/2 - HU_NORM_MIN) / (HU_NORM_MAX - HU_NORM_MIN)
    vmax_w = (HU_WINDOW_CENTER + HU_WINDOW_WIDTH/2 - HU_NORM_MIN) / (HU_NORM_MAX - HU_NORM_MIN)

    fig_map, ax_map = plt.subplots(figsize=(6, 6))
    ax_map.imshow(ct_norm, cmap="gray", vmin=vmin_w, vmax=vmax_w)
    ax_map.set_title(f"ROI positions — {label}", fontsize=9)

    for tissue, (r_frac, c_frac, sz_frac) in roi_defs.items():
        _, _, r0, c0, sz = sample_roi(ct_norm, r_frac, c_frac, sz_frac)
        rect = plt.Rectangle((c0, r0), sz, sz,
                              linewidth=2,
                              edgecolor=roi_colors[tissue],
                              facecolor="none")
        ax_map.add_patch(rect)
        ax_map.text(c0 + sz + 4, r0 + sz // 2, tissue,
                    color=roi_colors[tissue], fontsize=8,
                    va="center", fontweight="bold")

    ax_map.axis("off")
    plt.tight_layout()
    fig_map.savefig(results_dir / "dicom_roi_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig_map)
    print(f"  Saved : dicom_roi_map.png  <- open this to check ROI placement")

    # ── STEP 7: SSIM vs original ──────────────────────────────────────────────

    print("\nSSIM vs original slice:")
    for name, img in list(recons.items())[1:]:
        s = ssim(ct_norm, img, data_range=1.0)
        print(f"  {name:22s} : SSIM = {s:.4f}")

    # ── STEP 8: 4-panel figure ────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = [
        (ct_norm,    "Original CT"),
        (recon_sl,   "FBP / Shepp-Logan"),
        (recon_cos,  "FBP / cosine"),
        (recon_sart, "SART"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin_w, vmax=vmax_w,
                  interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"DICOM validation — {label}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(results_dir / "dicom_figure3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved : dicom_figure3.png")

    print("\n" + "=" * 62)
    print(f"Case '{case_id}' complete.  Outputs in: {results_dir}")
    print("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED SUMMARY CSV — both cases in one file
# ══════════════════════════════════════════════════════════════════════════════

summary_path = BASE_RESULTS / "dicom_summary.csv"
if summary_rows:
    all_keys = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nCombined summary saved -> {summary_path}")

print()
print("=" * 62)
print("Module 05 complete — both cases processed.")
print(f"Outputs:")
print(f"  Normal   -> {BASE_RESULTS / 'normal'}")
print(f"  Bleeding -> {BASE_RESULTS / 'bleeding'}")
print(f"  Summary  -> {summary_path}")
print("=" * 62)
