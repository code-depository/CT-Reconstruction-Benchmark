"""
07_tables.py
============
Produces three publication-ready tables from the metrics and validation data.

Table 1  — SSIM matrix (primary result)
    Rows = algorithms, Columns = dose levels.
    Cells = mean ± SD. Best value per column marked with *.

Table 2  — Full-dose comparison (MSE, PSNR, SSIM, time)
    One row per algorithm at 100% dose + clean sinogram ceiling.

Table 3  — DICOM HU validation
    4 tissue ROIs × Original + FBP/cosine + SART reconstructions.
    Reads dicom_hu_validation.csv written by 05_dicom.py.
    Column names in that file use spaces and capital T:
        "Tissue", "Expected HU",
        "Original mean HU", "Original SD HU",
        "FBP/cosine mean HU", "FBP/cosine SD HU",
        "SART mean HU", "SART SD HU"

Outputs (written to ./tables/)
---------------------------------
    table1_ssim_matrix.csv / .tex
    table2_full_dose.csv   / .tex
    table3_hu_validation.csv / .tex

Reads
-----
    results/metrics_summary.csv
    results/metrics_raw.csv
    results/timing.json
    results/dicom/dicom_hu_validation.csv
"""

import csv
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\USER\results")
TABLES_DIR  = Path(r"C:\Users\USER\tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DOSE_LEVELS = [100, 75, 50, 25, 10, 1]
ALGO_KEYS   = ["IRT", "FBP_ramp", "FBP_SL", "FBP_cos", "SART"]

ALGO_LABELS = {
    "IRT":      "IRT (unfiltered)",
    "FBP_ramp": "FBP — ramp",
    "FBP_SL":   "FBP — Shepp-Logan",
    "FBP_cos":  "FBP — cosine",
    "SART":     "SART (iterative)",
}

# ── Load data ─────────────────────────────────────────────────────────────────

summary = {}
with open(RESULTS_DIR / "metrics_summary.csv") as f:
    for row in csv.DictReader(f):
        summary[(row["algo"], int(row["dose_pct"]))] = row

clean_metrics = {}
with open(RESULTS_DIR / "metrics_raw.csv") as f:
    for row in csv.DictReader(f):
        if row["dose_pct"] == "clean":
            clean_metrics[row["algo"]] = row

with open(RESULTS_DIR / "timing.json") as f:
    timing = json.load(f)

print("Data loaded. Generating tables …\n")


# ── Helper: LaTeX table wrapper ───────────────────────────────────────────────

def latex_table(caption, label, header_row, data_rows, col_format=None):
    n_cols = len(header_row)
    if col_format is None:
        col_format = "l" + "r" * (n_cols - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\small",
        r"\begin{tabular}{" + col_format + "}",
        r"\toprule",
        " & ".join(header_row) + r" \\",
        r"\midrule",
    ]
    for row in data_rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def bold(s):
    return r"\textbf{" + str(s) + "}"


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — SSIM matrix
# ═══════════════════════════════════════════════════════════════════════════════

print("Table 1 — SSIM matrix …")

best_ssim = {}
for d in DOSE_LEVELS:
    best_ssim[d] = max(float(summary[(a, d)]["SSIM_mean"]) for a in ALGO_KEYS)

# CSV
t1_csv_path = TABLES_DIR / "table1_ssim_matrix.csv"
with open(t1_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Algorithm"] + [f"{d}%" for d in DOSE_LEVELS])
    for algo in ALGO_KEYS:
        row = [ALGO_LABELS[algo]]
        for d in DOSE_LEVELS:
            mean = float(summary[(algo, d)]["SSIM_mean"])
            sd   = float(summary[(algo, d)]["SSIM_sd"])
            marker = " *" if abs(mean - best_ssim[d]) < 0.0005 else ""
            row.append(f"{mean:.4f} ± {sd:.4f}{marker}")
        w.writerow(row)
print(f"  Saved → {t1_csv_path}")

# LaTeX
t1_header = [r"\textbf{Algorithm}"] + \
            [r"\textbf{" + f"{d}\\%" + "}" for d in DOSE_LEVELS]
t1_data = []
for algo in ALGO_KEYS:
    row = [ALGO_LABELS[algo]]
    for d in DOSE_LEVELS:
        mean = float(summary[(algo, d)]["SSIM_mean"])
        sd   = float(summary[(algo, d)]["SSIM_sd"])
        cell = f"{mean:.4f} $\\pm$ {sd:.4f}"
        if abs(mean - best_ssim[d]) < 0.0005:
            cell = bold(cell)
        row.append(cell)
    t1_data.append(row)

t1_tex = latex_table(
    caption=("SSIM (mean $\\pm$ SD, $n=5$) for six reconstruction algorithms "
             "across six simulated dose levels. Bold = highest SSIM per dose. "
             "$I_0 = 5 \\times 10^4$ at full dose."),
    label="tab:ssim_matrix",
    header_row=t1_header,
    data_rows=t1_data,
    col_format="l" + "r" * len(DOSE_LEVELS),
)
(TABLES_DIR / "table1_ssim_matrix.tex").write_text(t1_tex)
print(f"  Saved → {TABLES_DIR / 'table1_ssim_matrix.tex'}")

# Print to terminal
print(f"\n  {'Algorithm':22}" + "".join(f"  {d:>5}%" for d in DOSE_LEVELS))
print("  " + "-" * (22 + 8 * len(DOSE_LEVELS)))
for algo in ALGO_KEYS:
    row_str = f"  {ALGO_LABELS[algo]:22}"
    for d in DOSE_LEVELS:
        mean = float(summary[(algo, d)]["SSIM_mean"])
        marker = "*" if abs(mean - best_ssim[d]) < 0.0005 else " "
        row_str += f"  {mean:.4f}{marker}"
    print(row_str)
print("  (* = best at that dose level)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Full-dose comparison
# ═══════════════════════════════════════════════════════════════════════════════

print("Table 2 — Full-dose comparison (MSE, PSNR, SSIM, time) …")

t2_rows_csv = []
t2_rows_tex = []

for algo in ALGO_KEYS:
    s100      = summary[(algo, 100)]
    ssim_str  = f"{float(s100['SSIM_mean']):.4f} ± {float(s100['SSIM_sd']):.4f}"
    psnr_str  = f"{float(s100['PSNR_mean']):.2f} ± {float(s100['PSNR_sd']):.3f}"
    cm        = clean_metrics.get(algo, {})
    ssim_clean = f"{float(cm.get('SSIM', 0)):.4f}" if cm else "—"
    t_mean    = timing.get(algo, {}).get("mean_s", "—")
    t_str     = f"{t_mean:.3f}" if isinstance(t_mean, (int, float)) else t_mean

    t2_rows_csv.append({
        "Algorithm":          ALGO_LABELS[algo],
        "SSIM (100% dose)":   ssim_str,
        "PSNR (100% dose)":   psnr_str,
        "SSIM (clean sino)":  ssim_clean,
        "Time (s)":           t_str,
    })

    ssim_tex = f"{float(s100['SSIM_mean']):.4f} $\\pm$ {float(s100['SSIM_sd']):.4f}"
    psnr_tex = f"{float(s100['PSNR_mean']):.2f} $\\pm$ {float(s100['PSNR_sd']):.3f}"
    t2_rows_tex.append([ALGO_LABELS[algo], ssim_tex, psnr_tex, ssim_clean, t_str])

t2_csv_path = TABLES_DIR / "table2_full_dose.csv"
with open(t2_csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(t2_rows_csv[0].keys()))
    w.writeheader()
    w.writerows(t2_rows_csv)
print(f"  Saved → {t2_csv_path}")

t2_header = [r"\textbf{Algorithm}",
             r"\textbf{SSIM (100\%)}",
             r"\textbf{PSNR (dB)}",
             r"\textbf{SSIM (clean)}",
             r"\textbf{Time (s)}"]
t2_tex = latex_table(
    caption=("Reconstruction quality at full dose ($I_0 = 5 \\times 10^4$). "
             "SSIM and PSNR: mean $\\pm$ SD over five noise realisations. "
             "SSIM (clean): noise-free sinogram ceiling. "
             "Time: mean wall-clock seconds per reconstruction."),
    label="tab:full_dose",
    header_row=t2_header,
    data_rows=t2_rows_tex,
    col_format="lrrrr",
)
(TABLES_DIR / "table2_full_dose.tex").write_text(t2_tex)
print(f"  Saved → {TABLES_DIR / 'table2_full_dose.tex'}")

print(f"\n  {'Algorithm':22}  {'SSIM(100%)':>12}  {'PSNR':>8}  "
      f"{'SSIM(clean)':>12}  {'Time(s)':>8}")
print("  " + "-" * 68)
for row in t2_rows_csv:
    print(f"  {row['Algorithm']:22}  "
          f"{row['SSIM (100% dose)'].split()[0]:>12}  "
          f"{row['PSNR (100% dose)'].split()[0]:>8}  "
          f"{row['SSIM (clean sino)']:>12}  "
          f"{row['Time (s)']:>8}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — DICOM HU validation (two cases: normal + bleeding)
# ═══════════════════════════════════════════════════════════════════════════════

print("Table 3 — DICOM HU validation …")

# Read from the combined summary CSV written by 05_dicom.py
# This file has one row per (case, tissue) covering both normal and bleeding
dicom_val_path = RESULTS_DIR / "dicom" / "dicom_summary.csv"

# Fallback: try the normal-only path if summary not yet generated
if not dicom_val_path.exists():
    dicom_val_path = RESULTS_DIR / "dicom" / "normal" / "dicom_hu_validation.csv"

# Reference ranges
HU_REFERENCE_DISPLAY = {
    "Air":       "[$-$1050, $-$950]",
    "Scalp":     "[$-$20, $+$80]",
    "Brain":     "[$+$20, $+$45]",
    "Bone":      "[$+$400, $+$1534]",
    "Haematoma": "[$+$30, $+$90]",
}

t3_header_tex = [r"\textbf{Case}", r"\textbf{Tissue}", r"\textbf{Expected HU}",
                 r"\textbf{Original}", r"\textbf{FBP / cosine}",
                 r"\textbf{SART}"]

if not dicom_val_path.exists():
    print(f"  WARNING: {dicom_val_path} not found.")
    print("  Run 05_dicom.py first, then re-run 07_tables.py.")

    t3_csv_path = TABLES_DIR / "table3_hu_validation.csv"
    with open(t3_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Tissue", "Expected HU",
                    "Original mean±SD", "FBP/cosine mean±SD", "SART mean±SD"])
        for tissue, ref in HU_REFERENCE_DISPLAY.items():
            w.writerow(["—", tissue, ref, "—", "—", "—"])
    print(f"  Placeholder saved → {t3_csv_path}")

else:
    val_rows = []
    with open(dicom_val_path) as f:
        for row in csv.DictReader(f):
            val_rows.append(row)

    if not val_rows:
        print("  ERROR: validation CSV is empty.")
    else:
        print(f"  Loaded {len(val_rows)} rows.")
        print(f"  Columns: {list(val_rows[0].keys())}")

    t3_csv_path   = TABLES_DIR / "table3_hu_validation.csv"
    t3_header_csv = ["Case", "Tissue", "Expected HU",
                     "Original mean±SD HU",
                     "FBP/cosine mean±SD HU",
                     "SART mean±SD HU"]

    t3_rows_tex = []
    with open(t3_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(t3_header_csv)

        for row in val_rows:
            tissue    = row.get("Tissue",             "—")
            case_lbl  = row.get("Case",               "—")
            orig_mean = row.get("Original mean HU",   "—")
            orig_sd   = row.get("Original SD HU",     "—")
            fbp_mean  = row.get("FBP/cosine mean HU", "—")
            fbp_sd    = row.get("FBP/cosine SD HU",   "—")
            sart_mean = row.get("SART mean HU",       "—")
            sart_sd   = row.get("SART SD HU",         "—")
            ref       = row.get("Expected HU",        "—")

            orig = f"{orig_mean} ± {orig_sd}"
            fbp  = f"{fbp_mean} ± {fbp_sd}"
            sart = f"{sart_mean} ± {sart_sd}"

            w.writerow([case_lbl, tissue, ref, orig, fbp, sart])
            t3_rows_tex.append([case_lbl, tissue, ref,
                                 f"{orig} HU", f"{fbp} HU", f"{sart} HU"])

    print(f"  Saved → {t3_csv_path}")

    # Print to terminal
    print(f"\n  {'Case':32}  {'Tissue':10}  {'Expected HU':16}"
          + "".join(f"  {n:24}" for n in ["Original", "FBP/cosine", "SART"]))
    print("  " + "-" * (32 + 10 + 16 + 3 * 26))
    prev_case = None
    for row in val_rows:
        tissue    = row["Tissue"]
        case_lbl  = row.get("Case", "—")
        if case_lbl != prev_case:
            if prev_case is not None:
                print()
            prev_case = case_lbl
        orig      = (f"{row.get('Original mean HU','—'):>8} "
                     f"± {row.get('Original SD HU','—'):>6}")
        fbp       = (f"{row.get('FBP/cosine mean HU','—'):>8} "
                     f"± {row.get('FBP/cosine SD HU','—'):>6}")
        sart      = (f"{row.get('SART mean HU','—'):>8} "
                     f"± {row.get('SART SD HU','—'):>6}")
        ref       = row.get("Expected HU", "—")
        print(f"  {case_lbl:32}  {tissue:10}  {ref:16}  {orig:24}  {fbp:24}  {sart:24}")

    # LaTeX
    t3_tex = latex_table(
        caption=("HU validation on two clinical brain CT slices from the Kaggle "
                 "brain stroke CT dataset: a normal case (12155.dcm) and an "
                 "intracerebral haemorrhage case (10331.dcm). "
                 "Values are mean $\\pm$ SD HU within confirmed tissue ROIs. "
                 "Normal case: FBP/cosine SSIM = 0.990, SART SSIM = 0.957. "
                 "ICH case: FBP/cosine SSIM = 0.975, SART SSIM = 0.954."),
        label="tab:hu_validation",
        header_row=t3_header_tex,
        data_rows=t3_rows_tex,
        col_format="llrrrr",
    )
    (TABLES_DIR / "table3_hu_validation.tex").write_text(t3_tex)
    print(f"\n  Saved → {TABLES_DIR / 'table3_hu_validation.tex'}")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nAll tables saved to {TABLES_DIR}")
print("\nFiles generated:")
for p in sorted(TABLES_DIR.iterdir()):
    print(f"  {p.name}")

print("\n" + "=" * 60)
print("FULL PIPELINE COMPLETE")
print("=" * 60)
print("\nPublication outputs ready:")
print("  figures/figure1_reconstruction_grid.png")
print("  figures/figure2_ssim_vs_dose.png")
print("  figures/figure3_dicom_validation.png")
print("  tables/table1_ssim_matrix.tex  (.csv)")
print("  tables/table2_full_dose.tex    (.csv)")
print("  tables/table3_hu_validation.tex (.csv)")

print("\nCentral results for abstract:")
with open(RESULTS_DIR / "crossover.json") as f:
    cx = json.load(f)
best_fbp = cx["best_fbp_key"]
print(f"  Best analytic filter : {best_fbp}")
for d in sorted(cx["dose_results"].keys(), key=int, reverse=True):
    info  = cx["dose_results"][d]
    sart  = info.get("SART_SSIM", 0)
    fbp   = info.get("FBP_SSIM",  0)
    delta = info.get("delta_SSIM", 0)
    sig   = info.get("significant", False)
    tag   = "  ← SIGNIFICANT" if sig else ""
    print(f"  Dose {d:>3}%: SART={sart:.4f}  {best_fbp}={fbp:.4f}  "
          f"ΔSSIM={delta:.4f}{tag}")
