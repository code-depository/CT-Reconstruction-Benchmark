================================================================================
  Filter selection versus iterative reconstruction in CT:
  a noise-dose benchmark using open-source Python and clinical DICOM validation
================================================================================

  Dr. Arijit Roy
  ORCiD: https://orcid.org/0000-0002-9618-6641
  email: arijitroy@live.com


  GitHub repository: https://github.com/code-depository/CT-Reconstruction-Benchmark

────────────────────────────────────────────────────────────────────────────────
 OVERVIEW
────────────────────────────────────────────────────────────────────────────────

This repository contains the complete, reproducible Python pipeline described
in the manuscript. It benchmarks five CT reconstruction algorithms across six
simulated dose levels in three experimental arms:

  Arm 1 — Parallel-beam phantom study
           Five algorithms (IRT, FBP/ramp, FBP/Shepp-Logan, FBP/cosine, SART)
           evaluated on a 400×400 Shepp-Logan phantom at dose levels
           100%, 75%, 50%, 25%, 10%, and 1%, with five independent Poisson
           noise realisations per cell (150 reconstructions total).

  Arm 2 — Clinical DICOM validation
           Two axial brain CT slices from the Kaggle Brain Stroke CT Dataset:
           a normal brain case (12155.dcm) and an intracerebral haemorrhage
           (ICH) case (10331.dcm). HU accuracy validated across four tissue
           ROIs per case (32/32 checks passed).

  Arm 3 — Fan-beam geometry benchmark
           SART repeated under fan-flat geometry using ASTRA Toolbox 2.4.1
           (SID = 570 px, SDD = 1040 px, 736 detectors, 360 angles).
           Confirms that SART superiority and D* = 100% are geometry-independent.

KEY FINDINGS
  - FBP/cosine outperforms FBP/Shepp-Logan on SSIM at all six dose levels
    (0.761 vs 0.580 at full dose). This contradicts the conventional default.
  - SART outperforms FBP/cosine at every dose level tested (Δ-SSIM = 0.134
    at full dose, widening to 0.419 at 10% dose). D* = 100%.
  - Finding confirmed under fan-beam geometry (Δ-SSIM = 0.309 at full dose).
  - Fan-beam SART at 25% dose (SSIM = 0.543) exceeds FBP/cosine at 25% dose
    (SSIM = 0.448): quantitative basis for dose reduction without quality loss.
  - 32/32 HU checks passed across normal brain and ICH clinical DICOM cases.


────────────────────────────────────────────────────────────────────────────────
 REPOSITORY STRUCTURE
────────────────────────────────────────────────────────────────────────────────

  ct-reconstruction-benchmark/
  │
  ├── 01_phantom.py          Generate Shepp-Logan phantom and clean sinogram
  ├── 02_noise.py            Inject Poisson noise at six dose levels × five seeds
  ├── 03_reconstruct.py      Reconstruct with five algorithms (150 images)
  ├── 04_metrics.py          Compute SSIM / PSNR / MSE; identify D*
  ├── 05_dicom.py            Two-case clinical DICOM validation (Arm 2)
  ├── 06_figures.py          Generate Figures 1–3 at 300 DPI
  ├── 07_tables.py           Generate Tables 1–3 as CSV and LaTeX
  ├── 08_fanbeam.py          Fan-beam benchmark using ASTRA Toolbox (Arm 3)
  │
  ├── data/                  Created by 01_phantom.py (phantom + sinograms)
  │   └── noisy/             Created by 02_noise.py  (30 noisy sinograms)
  │
  ├── results/               Created by 03_reconstruct.py and 04_metrics.py
  │   ├── recons/            180 reconstruction .npy files + 5 clean ceilings
  │   ├── dicom/             DICOM validation outputs (05_dicom.py)
  │   │   ├── normal/        Normal brain case outputs
  │   │   └── bleeding/      ICH case outputs
  │   └── fanbeam/           Fan-beam results (08_fanbeam.py)
  │
  ├── figures/               All publication figures (PNG, 300 DPI)
  ├── tables/                All publication tables (CSV + LaTeX)
  │
  └── README.txt             This file


────────────────────────────────────────────────────────────────────────────────
 REQUIREMENTS
────────────────────────────────────────────────────────────────────────────────

  Python   : 3.10 or later (tested on 3.12)
  Platform : Windows, Linux, or macOS

  Core dependencies (Modules 01–07):
  ┌─────────────────────┬───────────┬─────────────────────────────────────────┐
  │ Package             │ Version   │ Purpose                                 │
  ├─────────────────────┼───────────┼─────────────────────────────────────────┤
  │ numpy               │ ≥ 1.24    │ Array operations, random seeds          │
  │ scipy               │ ≥ 1.10    │ Interpolation (fan-beam rebinning)      │
  │ scikit-image        │ ≥ 0.21    │ Radon transform, FBP, SART, SSIM       │
  │ matplotlib          │ ≥ 3.7     │ Figure generation                       │
  │ pydicom             │ ≥ 2.3     │ DICOM file reading (Module 05)          │
  └─────────────────────┴───────────┴─────────────────────────────────────────┘

  Additional dependency (Module 08 — fan-beam only):
  ┌─────────────────────┬───────────┬─────────────────────────────────────────┐
  │ Package             │ Version   │ Purpose                                 │
  ├─────────────────────┼───────────┼─────────────────────────────────────────┤
  │ astra-toolbox       │ 2.4.1     │ Fan-beam forward projection and SART    │
  └─────────────────────┴───────────┴─────────────────────────────────────────┘

  Install core dependencies:
      pip install numpy scipy scikit-image matplotlib pydicom

  Install ASTRA Toolbox (conda recommended, requires conda or miniconda):
      conda install -c astra-toolbox -c nvidia astra-toolbox

  Note: ASTRA Toolbox requires Anaconda/Miniconda. Module 08 will not run
  without it. Modules 01–07 run with pip-only environments.

  Note: GPU (CUDA) is NOT required. All computations run on CPU.
  ASTRA GPU algorithms (FBP_CUDA) are not used in this pipeline.


────────────────────────────────────────────────────────────────────────────────
 QUICK START
────────────────────────────────────────────────────────────────────────────────

  Step 1 — Clone the repository
      git clone https://github.com/[username]/ct-reconstruction-benchmark.git
      cd ct-reconstruction-benchmark

  Step 2 — Install dependencies (see REQUIREMENTS above)

  Step 3 — Run modules in order
      python 01_phantom.py
      python 02_noise.py
      python 03_reconstruct.py        # ~20 min on a standard laptop
      python 04_metrics.py
      python 05_dicom.py              # requires DICOM files — see DATA below
      python 06_figures.py
      python 07_tables.py
      python 08_fanbeam.py            # ~3–4 min; requires ASTRA Toolbox

  Each module prints a summary to stdout and writes its outputs to the
  appropriate subdirectory. Modules read only from subdirectories written
  by earlier modules, so sequential execution is all that is required.

  To skip DICOM validation (Module 05), run modules 01–04 and 06–08.
  Modules 06 and 07 will skip Figure 3 and Table 3 gracefully if DICOM
  outputs are absent.


────────────────────────────────────────────────────────────────────────────────
 MODULE REFERENCE
────────────────────────────────────────────────────────────────────────────────

  01_phantom.py
    Generates the 400×400 Shepp-Logan phantom using skimage.data.shepp_logan_
    phantom() and computes the clean parallel-beam sinogram (180 angles, 0°–179°)
    using skimage.transform.radon().
    Outputs : data/phantom.npy  |  data/sinogram_clean.npy  |  data/theta.npy
    Runtime : < 5 seconds

  02_noise.py
    Applies physically correct Poisson noise via the transmission domain:
    sinogram → transmission (exp(-s)) → Poisson draw → back to log-attenuation.
    Six dose levels × five seeds = 30 noisy sinogram files.
    I0 values: 5e4 (100%), 3e4 (75%), 1.5e4 (50%), 5e3 (25%), 1.5e3 (10%),
    2e2 (1%). SCALE = 5.0 maps the sinogram maximum to this log-attenuation.
    Outputs : data/noisy/sino_d{dose}_{seed}.npy  (30 files)
              data/noise_manifest.csv
    Runtime : < 1 minute

  03_reconstruct.py
    Applies five algorithms to all 30 noisy sinograms plus the clean sinogram:
      IRT       — skimage iradon, filter=None
      FBP/ramp  — skimage iradon, filter='ramp'
      FBP/SL    — skimage iradon, filter='shepp-logan'
      FBP/cos   — skimage iradon, filter='cosine'
      SART      — skimage iradon_sart, relaxation=0.15, 1 iteration
    All reconstructions cropped to 400×400 and clipped to [0, 1].
    Outputs : results/recons/recon_{algo}_{dose}_{seed}.npy  (180 files)
              results/recons/recon_{algo}_clean.npy           (5 files)
              results/timing.json
    Runtime : ~20 minutes (SART dominates: ~6.7 s per reconstruction)

  04_metrics.py
    Computes SSIM (data_range=1.0), PSNR, and MSE against the ground-truth
    phantom for all 180 reconstructions. Aggregates mean ± SD across five seeds
    per (algorithm, dose) cell. Identifies D* as the lowest dose at which
    SSIM(SART) − SSIM(FBP/cosine) > 0.02 and SD bands do not overlap.
    Outputs : results/metrics_raw.csv      (180 rows)
              results/metrics_summary.csv   (30 rows: 5 algos × 6 doses)
              results/crossover.json        (D*, per-dose gap values)
              results/timing_table.csv
    Runtime : < 1 minute

  05_dicom.py
    Loads two clinical brain CT DICOM slices using pydicom. Applies HU
    conversion (RescaleSlope + RescaleIntercept, verified by median plausibility
    check). Simulates a parallel-beam sinogram from each slice, reconstructs
    with FBP/Shepp-Logan, FBP/cosine, and SART, and validates HU accuracy
    in four tissue ROIs per case. See DATA section below for file locations.
    Outputs : results/dicom/normal/  and  results/dicom/bleeding/
              (per case: original.npy, recon files, HU CSV, ROI map, figure)
    Runtime : < 2 minutes

  06_figures.py
    Produces Figures 1–3 for the manuscript at 300 DPI:
      Figure 1 — 5×6 reconstruction grid (algorithm × dose)
      Figure 2 — SSIM vs dose line chart with ±SD bands and D* annotation
      Figure 3 — Two-case DICOM validation panel (8 panels)
    Outputs : figures/figure1_reconstruction_grid.png
              figures/figure2_ssim_vs_dose.png
              figures/figure3_dicom_validation.png
    Runtime : < 1 minute

  07_tables.py
    Produces Tables 1–3 as both CSV and LaTeX:
      Table 1 — SSIM matrix (5 algorithms × 6 dose levels, mean ± SD)
      Table 2 — Full-dose comparison (SSIM, PSNR, SSIM ceiling, time)
      Table 3 — HU validation (8 rows: 4 tissues × 2 cases)
    Outputs : tables/table{1,2,3}_*.csv  and  tables/table{1,2,3}_*.tex
    Runtime : < 10 seconds

  08_fanbeam.py
    Fan-beam benchmark (Arm 3). Requires ASTRA Toolbox (see REQUIREMENTS).
    Implements fan-flat geometry using ASTRA's CPU fan-beam projector.
    Fan-beam SART uses ASTRA's native iterative algorithm (no geometric
    approximation). Fan-beam FBP uses fan-to-parallel rebinning followed by
    skimage iradon (CPU-compatible workaround; ASTRA FBP requires GPU).
    Runs 3 algorithms × 6 doses × 5 seeds = 90 reconstructions.
    Produces Figure 4 (fan-beam vs parallel-beam SSIM comparison).
    Outputs : results/fanbeam/  (metrics CSV, crossover JSON, recon .npy files)
              figures/figure_fanbeam_ssim_comparison.png
              figures/figure_fanbeam_recon_grid.png
    Runtime : ~3–4 minutes on CPU


────────────────────────────────────────────────────────────────────────────────
 DATA
────────────────────────────────────────────────────────────────────────────────

  PHANTOM DATA
  Generated automatically by 01_phantom.py. No external download required.

  CLINICAL DICOM DATA (required for Module 05 only)
  Source  : Kaggle Brain Stroke CT Image Dataset
  Author  : Ozgur Aslan
  URL     : https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset
  Licence : Community Data Licence Agreement — Permissive (CDLA-Permissive-1.0)
  Files used:
    12155.dcm  — normal brain case     → place in:  DICOM_N/12155.dcm
    10331.dcm  — ICH / bleeding case   → place in:  DICOM_B/10331.dcm

  IMPORTANT: The DICOM files are NOT included in this repository. Download them
  from Kaggle (free account required) and place them in the paths above before
  running Module 05. All other modules (01–04, 06–08) run without DICOM data.

  Path configuration in 05_dicom.py:
    DICOM_N_PATH = Path(r"C:\Users\USER\DICOM_N\12155.dcm")   # ← edit this
    DICOM_B_PATH = Path(r"C:\Users\USER\DICOM_B\10331.dcm")   # ← edit this

  Update these two lines to match your local directory structure.


────────────────────────────────────────────────────────────────────────────────
 REPRODUCIBILITY
────────────────────────────────────────────────────────────────────────────────

  All random seeds are fixed (NumPy seeds 42–46) and are set explicitly at the
  start of each module. Given the same software versions, running the pipeline
  in order will reproduce all numerical results in the manuscript exactly.

  Tested environment:
    OS       : Windows 11 (64-bit)
    Python   : 3.12.0  (Anaconda distribution)
    NumPy    : 2.4.2
    SciPy    : 1.17.0
    scikit-image : 0.26.0
    Matplotlib   : 3.10.8
    pydicom      : 2.4.3
    ASTRA Toolbox: 2.4.1  (conda, CPU mode)

  Cross-platform note: Modules 01–07 are platform-independent. Module 08
  (ASTRA) has been tested on Windows with Anaconda. Linux is supported by
  ASTRA; macOS support depends on the ASTRA build available on your platform.


────────────────────────────────────────────────────────────────────────────────
 EXPECTED OUTPUTS — KEY NUMBERS
────────────────────────────────────────────────────────────────────────────────

  After running all modules, the following values should be reproduced:

  SSIM at full dose (100%), parallel-beam:
    IRT (unfiltered)   : 0.142
    FBP / ramp         : 0.486
    FBP / Shepp-Logan  : 0.580
    FBP / cosine       : 0.761      ← best analytic filter
    SART (iterative)   : 0.895 *    ← highest overall

  D* (crossover threshold) : 100%  under both parallel-beam and fan-beam

  Δ-SSIM (SART vs FBP/cosine):
    Parallel-beam : 0.134 (full dose) → 0.419 (10% dose)
    Fan-beam      : 0.309 (full dose) → 0.188 (10% dose)

  DICOM validation:
    Normal brain (12155.dcm) : 16/16 HU checks passed
    ICH case     (10331.dcm) : 16/16 HU checks passed
    Combined                 : 32/32 passed

  Fan-beam SSIM at full dose:
    FBP / ramp    : 0.204
    FBP / cosine  : 0.375
    SART          : 0.684


────────────────────────────────────────────────────────────────────────────────
 CITATION
────────────────────────────────────────────────────────────────────────────────

  If you use this code or pipeline in your research, please cite:

  [Author Name] (2026) "Filter selection versus iterative reconstruction in CT:
  a noise-dose benchmark using open-source Python and clinical DICOM validation."
  Physica Medica — European Journal of Medical Physics. [DOI to be added upon
  acceptance]

  BibTeX:
    @article{author2026ct,
      author  = {[Author Name]},
      title   = {Filter selection versus iterative reconstruction in {CT}:
                 a noise-dose benchmark using open-source {Python} and
                 clinical {DICOM} validation},
      journal = {Physica Medica},
      year    = {2026},
      note    = {DOI to be added upon acceptance}
    }

  Please also cite the ASTRA Toolbox if using Module 08:
    van Aarle W, et al. (2016) "Fast and flexible X-ray tomography using the
    ASTRA toolbox." Optics Express, 24(22), 25129–25147.
    DOI: 10.1364/OE.24.025129

  And the SSIM metric:
    Wang Z, Bovik AC, Sheikh HR, Simoncelli EP. (2004) "Image quality
    assessment: from error visibility to structural similarity."
    IEEE Transactions on Image Processing, 13(4), 600–612.
    DOI: 10.1109/TIP.2003.819861


────────────────────────────────────────────────────────────────────────────────
 LICENCE
────────────────────────────────────────────────────────────────────────────────

  Code: MIT Licence. See LICENCE.txt.

  The DICOM dataset (Kaggle Brain Stroke CT) is distributed under the Community
  Data Licence Agreement — Permissive (CDLA-Permissive-1.0). It is NOT included
  in this repository. Users must download it independently from Kaggle.


────────────────────────────────────────────────────────────────────────────────
 CONTACT
────────────────────────────────────────────────────────────────────────────────
 Dr. Arijit Roy
 ORCiD: https://orcid.org/0000-0002-9618-6641
 email: arijitroy@live.com


  GitHub repository: https://github.com/code-depository/CT-Reconstruction-Benchmark


  Bug reports and questions: please open a GitHub Issue.

================================================================================
