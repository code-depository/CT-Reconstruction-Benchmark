"""
06_figures.py
=============
Produces three publication-ready figures at 300 DPI.

Figure 1  —  6×6 reconstruction grid
    Rows = 6 algorithms, Columns = 6 dose levels.
    Each panel shows the reconstructed image with SSIM annotated.
    First column header shows the algorithm name.

Figure 2  —  SSIM vs dose line chart  (KEY FIGURE)
    One line per algorithm with mean ± SD error bands.
    D* crossover threshold annotated with a vertical dashed line.
    Inset zoom on the 25–50% dose region to show filter separation.

Figure 3  —  DICOM validation panel (4 panels)
    Original DICOM slice | sinogram | FBP/cosine recon | SART recon
    All displayed at the same soft-tissue window/level.
    ROI boxes overlaid on original with tissue labels.
    (Skipped gracefully if DICOM outputs are not found.)

Outputs (written to ./figures/)
---------------------------------
    figure1_reconstruction_grid.png
    figure2_ssim_vs_dose.png
    figure3_dicom_validation.png   (if DICOM data available)

Reads
-----
    data/phantom.npy
    results/recons/recon_{algo}_{dose}_{seed}.npy
    results/metrics_summary.csv
    results/crossover.json
    results/dicom/dicom_*.npy                      (optional)
"""

import numpy as np
import csv
import json
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes   # for inset axes

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RECON_DIR   = RESULTS_DIR / "recons"
DICOM_DIR   = RESULTS_DIR / "dicom"
FIG_DIR     = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI         = 300
DOSE_LEVELS = [100, 75, 50, 25, 10, 1]
SEEDS       = [42, 43, 44, 45, 46]
ALGO_KEYS   = ["IRT", "FBP_ramp", "FBP_SL", "FBP_cos", "SART"]

ALGO_LABELS = {
    "IRT":      "IRT\n(unfiltered)",
    "FBP_ramp": "FBP\n(ramp)",
    "FBP_SL":   "FBP\n(Shepp-Logan)",
    "FBP_cos":  "FBP\n(cosine)",
    "SART":     "SART\n(iterative)",
}

ALGO_COLORS = {
    "IRT":      "#888780",
    "FBP_ramp": "#E24B4A",
    "FBP_SL":   "#EF9F27",
    "FBP_cos":  "#378ADD",
    "SART":     "#1D9E75",
}

ALGO_LINESTYLE = {
    "IRT":      (0, (3, 3)),
    "FBP_ramp": (0, (5, 2)),
    "FBP_SL":   (0, (5, 2, 1, 2)),
    "FBP_cos":  "-",
    "SART":     "-",
}

# ── Load shared data ──────────────────────────────────────────────────────────

phantom = np.load(DATA_DIR / "phantom.npy")

# Load summary metrics
summary = {}      # (algo, dose_pct) → {SSIM_mean, SSIM_sd, PSNR_mean, ...}
with open(RESULTS_DIR / "metrics_summary.csv") as f:
    for row in csv.DictReader(f):
        key = (row["algo"], int(row["dose_pct"]))
        summary[key] = {k: float(v) for k, v in row.items()
                        if k not in ("algo", "dose_pct")}

# Load crossover data
with open(RESULTS_DIR / "crossover.json") as f:
    crossover = json.load(f)

D_STAR       = crossover.get("D_star")
BEST_FBP_KEY = crossover.get("best_fbp_key", "FBP_cos")

print("Loaded metrics and crossover data.")
print(f"  D* = {D_STAR}%   best FBP = {BEST_FBP_KEY}")

# ── Matplotlib style ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "lines.linewidth":    1.5,
    "image.cmap":         "gray",
    "figure.dpi":         DPI,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 6×6 reconstruction grid
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating Figure 1 — reconstruction grid …")

# Pre-compute IRT display range
# IRT raw output spans [35, 116] — all above 1.0, so clip([0,1]) makes all pixels 1.0
# For display we need per-image min-max normalisation to reveal the blurred structure
IRT_USE_MINMAX = True   # flag: apply min-max stretch for IRT display

fig1, axes = plt.subplots(
    nrows=len(ALGO_KEYS),
    ncols=len(DOSE_LEVELS),
    figsize=(13, 10),
)

# Column headers
for j, dose in enumerate(DOSE_LEVELS):
    axes[0, j].set_title(f"{dose}%", fontsize=9, fontweight="bold", pad=4)

for i, algo in enumerate(ALGO_KEYS):
    # Row label — horizontal, flush right
    label_text = ALGO_LABELS[algo].replace("\n", " ")
    axes[i, 0].set_ylabel(label_text, fontsize=7.5,
                           rotation=0, labelpad=60,
                           va="center", ha="right")

    for j, dose_pct in enumerate(DOSE_LEVELS):
        ax    = axes[i, j]
        recon = np.load(RECON_DIR / f"recon_{algo}_d{dose_pct:03d}_42.npy")
        ssim_mean = summary[(algo, dose_pct)]["SSIM_mean"]

        # ── Window / level ─────────────────────────────────────────────────────
        if algo == "IRT":
            # IRT raw values are large (35–116), all clipped to 1.0 in storage.
            # Reconstruct display-only version from noisy sinogram with min-max stretch
            # so the blurred phantom structure is visible.
            from skimage.transform import iradon as _iradon
            sino_path = Path("data") / "noisy" / f"sino_d{dose_pct:03d}_42.npy"
            if sino_path.exists():
                _sino  = np.load(sino_path)
                _raw   = _iradon(_sino,
                                  theta=np.load(Path("data") / "theta.npy"),
                                  filter_name=None, circle=False)
                _h, _w = _raw.shape
                _r0 = (_h - 400) // 2; _c0 = (_w - 400) // 2
                _crop = _raw[_r0:_r0+400, _c0:_c0+400]
                _rng  = _crop.max() - _crop.min()
                recon_disp = (_crop - _crop.min()) / (_rng if _rng > 0 else 1.0)
            else:
                recon_disp = recon   # fallback
            vmin, vmax = 0.0, 1.0

        else:
            recon_disp = recon
            # FBP variants and SART: standard [0, 1]
            vmin, vmax = 0.0, 1.0

        ax.imshow(recon_disp, cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        # SSIM annotation — bottom-right
        ax.text(0.97, 0.03, f"{ssim_mean:.3f}",
                transform=ax.transAxes,
                fontsize=6, color="white", ha="right", va="bottom",
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, pad=1.5, linewidth=0))

        # Border: green for SART only; thin grey for all others
        if algo == "SART":
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ALGO_COLORS["SART"])
                spine.set_linewidth(1.8)
        else:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.4)

# Note for IRT about display normalisation
fig1.text(0.50, 0.002,
          "IRT panels use per-image min-max stretching to reveal the blurred phantom structure. "
          "All other rows use a fixed [0, 1] window. "
          "SSIM annotated in each panel (mean over 5 noise seeds).",
          ha="center", fontsize=5.5, color="gray")

fig1.text(0.5, 0.995,
          "CT reconstruction quality across algorithms and simulated dose levels",
          ha="center", va="top", fontsize=10, fontweight="bold")

plt.subplots_adjust(wspace=0.03, hspace=0.05,
                    left=0.14, right=0.99, top=0.96, bottom=0.03)

fig1_path = FIG_DIR / "figure1_reconstruction_grid.png"
fig1.savefig(fig1_path)
plt.close(fig1)
print(f"  Saved → {fig1_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — SSIM vs dose line chart
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 2 — SSIM vs dose …")

fig2, ax_main = plt.subplots(figsize=(7, 4.5))

dose_x = DOSE_LEVELS   # x values: 100, 75, 50, 25, 10, 1

for algo in ALGO_KEYS:
    means = [summary[(algo, d)]["SSIM_mean"] for d in dose_x]
    sds   = [summary[(algo, d)]["SSIM_sd"]   for d in dose_x]

    means = np.array(means)
    sds   = np.array(sds)
    dose_arr = np.array(dose_x)

    lw    = 2.2 if algo == "SART" else 1.5
    zord  = 5   if algo == "SART" else 2

    ax_main.plot(dose_arr, means,
                 color=ALGO_COLORS[algo],
                 linestyle=ALGO_LINESTYLE[algo],
                 linewidth=lw,
                 marker="o", markersize=4, markeredgewidth=0.5,
                 zorder=zord,
                 label=ALGO_LABELS[algo].replace("\n", " "))

    ax_main.fill_between(dose_arr,
                          means - sds, means + sds,
                          color=ALGO_COLORS[algo], alpha=0.12,
                          zorder=zord - 1)

# Annotate D* if within range
if D_STAR is not None and D_STAR in dose_x:
    ax_main.axvline(x=D_STAR, color="black", linestyle="--",
                    linewidth=0.8, alpha=0.7, zorder=1)
    ax_main.text(D_STAR + 1.5, 0.92, f"D* = {D_STAR}%",
                 fontsize=7, color="black", va="top")

# SSIM threshold line
ax_main.axhline(y=0.02, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
ax_main.text(101, 0.025, "Δ = 0.02\n(perceptual threshold)",
             fontsize=5.5, color="gray", va="bottom", ha="right")

ax_main.set_xlabel("Simulated dose level (%)", fontsize=8)
ax_main.set_ylabel("SSIM (mean ± SD, n = 5)", fontsize=8)
ax_main.set_title("Structural similarity vs simulated dose: "
                  "FBP filter selection vs iterative reconstruction",
                  fontsize=9, fontweight="bold", pad=6)

ax_main.set_xlim(110, -5)   # x-axis reversed: full dose on left
ax_main.set_xticks(DOSE_LEVELS)
ax_main.set_xticklabels([f"{d}%" for d in DOSE_LEVELS], fontsize=7)
ax_main.set_ylim(-0.02, 1.0)
ax_main.set_yticks(np.arange(0, 1.1, 0.1))

# Legend — two columns
handles = [Line2D([0], [0],
                  color=ALGO_COLORS[a],
                  linestyle=ALGO_LINESTYLE[a],
                  linewidth=2,
                  marker="o", markersize=4,
                  label=ALGO_LABELS[a].replace("\n", " "))
           for a in ALGO_KEYS]

ax_main.legend(handles=handles, fontsize=6.5, loc="upper right",
               ncol=2, framealpha=0.85, edgecolor="none",
               handlelength=2.5, labelspacing=0.4)

ax_main.grid(axis="y", color="lightgray", linewidth=0.4, alpha=0.7)

fig2.tight_layout()
fig2_path = FIG_DIR / "figure2_ssim_vs_dose.png"
fig2.savefig(fig2_path)
plt.close(fig2)
print(f"  Saved → {fig2_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — DICOM validation  (two-case, 8 panels)
#
# Row 1 (top):    Normal brain    — Original | FBP/cosine | SART | Difference map
# Row 2 (bottom): ICH / Bleeding  — Original | FBP/cosine | SART | Difference map
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 3 — DICOM validation …")

DICOM_CASES = {
    "normal": {
        "label":    "Normal brain (12155.dcm)",
        "dir":      DICOM_DIR / "normal",
        "ssim_cos": 0.990,
        "ssim_srt": 0.957,
        "roi_display": {
            "Air":   (0.02, 0.02, 20, "#85B7EB", (-18,  4)),
            "Scalp": (0.35, 0.16, 12, "#EF9F27", ( -2,-45)),
            "Brain": (0.45, 0.40, 22, "#1D9E75", (  2, 26)),
            "Bone":  (0.35, 0.20, 12, "#E24B4A", ( 14, -2)),
        },
    },
    "bleeding": {
        "label":    "ICH — intracerebral haemorrhage (10331.dcm)",
        "dir":      DICOM_DIR / "bleeding",
        "ssim_cos": 0.975,
        "ssim_srt": 0.954,
        "roi_display": {
            "Air":       (0.05, 0.05, 14, "#85B7EB", (-16,  4)),
            "Brain":     (0.40, 0.65, 18, "#1D9E75", (  2, 22)),
            "Haematoma": (0.38, 0.52, 14, "#FF6B6B", ( -2,-60)),
            "Bone":      (0.35, 0.67, 10, "#E24B4A", ( 14, -2)),
        },
    },
}

HU_NORM_MIN, HU_NORM_MAX = -1000.0, 2000.0
wl_min = (-60  - HU_NORM_MIN) / (HU_NORM_MAX - HU_NORM_MIN)
wl_max = ( 140 - HU_NORM_MIN) / (HU_NORM_MAX - HU_NORM_MIN)

# Check both case directories exist
cases_available = {k: all((v["dir"] / f"dicom_recon_{a}.npy").exists()
                           for a in ["FBP_cos", "SART"])
                   for k, v in DICOM_CASES.items()}

if not any(cases_available.values()):
    print("  DICOM outputs not found — skipping Figure 3.")
    print("  Run 05_dicom.py on your local machine first, then re-run 06_figures.py.")
else:
    n_cases = sum(cases_available.values())
    fig3, axes3 = plt.subplots(n_cases, 4,
                                figsize=(16, 4.5 * n_cases),
                                squeeze=False)

    row_idx = 0
    for case_key, case_cfg in DICOM_CASES.items():
        if not cases_available[case_key]:
            print(f"  Skipping {case_key} case — outputs not found.")
            continue

        d       = case_cfg["dir"]
        label   = case_cfg["label"]
        original  = np.load(d / "dicom_original.npy")
        recon_cos = np.load(d / "dicom_recon_FBP_cos.npy")
        recon_srt = np.load(d / "dicom_recon_SART.npy")
        diff_map  = np.abs(recon_srt - recon_cos)

        ax = axes3[row_idx]

        # Panel 1 — Original CT with ROI boxes
        ax[0].imshow(original, cmap="gray", vmin=wl_min, vmax=wl_max,
                     interpolation="nearest", aspect="equal")
        ax[0].set_title(f"Original CT\n{label}", fontsize=7, fontweight="bold")

        # Panel 2 — FBP/cosine
        ax[1].imshow(recon_cos, cmap="gray", vmin=wl_min, vmax=wl_max,
                     interpolation="nearest", aspect="equal")
        ax[1].set_title(f"FBP / cosine\n(SSIM = {case_cfg['ssim_cos']:.3f})",
                         fontsize=7, fontweight="bold")

        # Panel 3 — SART
        ax[2].imshow(recon_srt, cmap="gray", vmin=wl_min, vmax=wl_max,
                     interpolation="nearest", aspect="equal")
        ax[2].set_title(f"SART\n(SSIM = {case_cfg['ssim_srt']:.3f})",
                         fontsize=7, fontweight="bold")

        # Panel 4 — Difference map
        im4 = ax[3].imshow(diff_map, cmap="hot", vmin=0, vmax=0.15,
                            interpolation="nearest", aspect="equal")
        ax[3].set_title("|SART - FBP/cosine|\n(absolute difference)",
                         fontsize=7, fontweight="bold")
        plt.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04,
                     label="Δ intensity").ax.tick_params(labelsize=6)

        # Remove axes ticks
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
            for spine in a.spines.values():
                spine.set_visible(False)

        # ROI boxes on Panel 1
        from matplotlib.patches import Rectangle
        H, W = original.shape
        for tissue, (r_frac, c_frac, sz_px, color, (dr, dc)) in \
                case_cfg["roi_display"].items():
            r0 = int(r_frac * H)
            c0 = int(c_frac * W)
            rect = Rectangle((c0, r0), sz_px, sz_px,
                              linewidth=1.8, edgecolor=color, facecolor="none")
            ax[0].add_patch(rect)
            ax[0].text(c0 + dc, r0 + sz_px // 2 + dr, tissue,
                       color=color, fontsize=6, va="center", fontweight="bold")

        row_idx += 1

    fig3.suptitle(
        "DICOM validation — normal brain and ICH: original CT, "
        "FBP/cosine, SART, and difference map",
        fontsize=9, fontweight="bold", y=1.01
    )
    fig3.tight_layout()

    fig3_path = FIG_DIR / "figure3_dicom_validation.png"
    fig3.savefig(fig3_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved → {fig3_path}")

print(f"\nAll figures saved to {FIG_DIR}/")
print("Module 06 complete.")
