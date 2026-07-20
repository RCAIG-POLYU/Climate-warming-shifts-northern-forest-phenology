# Climate warming shifts northern forest phenology

R / Python code for stage-resolved forest phenology analysis using GOSIF solar-induced fluorescence (SIF), partial least squares path modeling (PLS-PM), pixel-wise first-difference regression, and CMIP6-based future projections.

## Overview

This repository supports a multi-stage phenological framework in which the growing season is partitioned into:

| Stage | Definition |
|-------|------------|
| **L1** | Green-up (SOS → PPOS) |
| **L2** | Plateau (PPOS → APOS) |
| **L3** | Senescence (APOS → EOS) |

Key analyses include:

- **Phenology extraction** from daily GOSIF (dynamic threshold, HANTS, HPLM)
- **Amplitude-threshold sensitivity** (SOS/EOS at 10% and 30%; main analysis uses 20%)
- **Time-allocation classification** of AI and PR (constant vs changing; tested separately)
- **PLS-PM** linking stage-specific climate, phenological carry-over, and stage duration
- **First-difference regression** quantifying drivers of interannual variability in L1 and total season length
- **Dominant ΔR² climate-block maps / trajectory / chord diagrams** for ΔL_total
- **Paired-window validation** at 20 mixed 20×20 sites
- **Random forest** projecting future growing-season length from yearly climate

## Repository structure

```
.
├── R/                    # Shared config, paths, I/O helpers
├── phenology/            # Extraction, allocation class, amplitude sensitivity
├── plspm/                # Pixel-wise PLS-PM and dominant-factor extraction
├── cmip6/                # Difference regression and variance-share processing
├── random-forest/        # RF training, future projection, EEMD trends
├── visualization/        # Hovmöller, histograms, polar maps, ΔR² trajectory/chord
├── validation/           # Paired-site difference-regression validation
└── README.md
```

## Requirements

- R >= 4.1
- Python >= 3.9 (for sensitivity tables, ΔR² trajectory, paired validation)
- R packages: [terra](https://cran.r-project.org/package=terra), [ranger](https://cran.r-project.org/package=ranger), [plspm](https://cran.r-project.org/package=plspm), [dplyr](https://cran.r-project.org/package=dplyr), [ggplot2](https://cran.r-project.org/package=ggplot2)
- Optional R: `Rlibeemd`, `segmented`, `lmtest`, `lubridate`, `tidyr`, `zoo`, `patchwork`, `minpack.lm`, `tidyterra`, `sf`, `rnaturalearth`, `scales`, `circlize`
- Optional Python: `numpy`, `pandas`, `rasterio`, `scipy`, `matplotlib`

```r
install.packages(c(
  "terra", "ranger", "plspm", "dplyr", "ggplot2",
  "lubridate", "tidyr", "zoo", "patchwork", "segmented", "lmtest",
  "minpack.lm", "tidyterra", "sf", "rnaturalearth", "scales", "circlize"
))
```

```bash
pip install numpy pandas rasterio scipy matplotlib
```

## Data layout

Set `GOSIF_DATA_ROOT` to your local data directory (default: `data/`). Expected structure:

```
data/
├── phenology/            # Greenup / Plateau / Senescence / LGS by forest type
├── gosif_sg/             # daily smoothed GOSIF tiles
├── stage_climate/
├── climate/
├── cmip6/                # SSP futures
└── results/              # script outputs
```

GeoTIFF naming examples:

- `Greenup_Duration_days_{year}_{ForestType}.tif`
- `{ClimateVar}_{Stage}_{year}_{ForestType}.tif`

## Usage

```r
Sys.setenv(GOSIF_DATA_ROOT = "F:/GO_SIF")  # your data root
setwd("path/to/this/repository")
```

```bash
# Phenology (main 20% amplitude)
Rscript phenology/dynamic_threshold_extract.R

# Amplitude sensitivity (10% / 30%)
Rscript phenology/amplitude_sensitivity_extract.R
python phenology/classify_allocation_sensitivity.py
python phenology/build_supplementary_table1.py

# AI / PR allocation classification (separate tests; P > 0.05 = constant)
Rscript phenology/classify_time_allocation.R

# Dominant ΔR² trajectory + chord (R2(L_total) > 0.30; strict argmax)
python visualization/plot_dominant_deltar2_trajectory.py
Rscript visualization/plot_dominant_transition_chord.R

# Paired 20-site validation
python validation/paired_diffreg_validation.py
```

Scripts use `if (sys.nframe() == 0L)` guards so that sourcing loads functions without automatically running the full pipeline.

## Script reference

| Script | Description |
|--------|-------------|
| `phenology/dynamic_threshold_extract.R` | Dynamic-threshold SOS/EOS (**20%**) and PPOS/APOS (90%) |
| `phenology/amplitude_sensitivity_extract.R` | SOS/EOS sensitivity re-extraction at **10%** and **30%** |
| `phenology/classify_allocation_sensitivity.py` | Separate AI / PR constant-allocation stats for Supp. Table 1 |
| `phenology/build_supplementary_table1.py` | Build Supplementary Table 1 (10% vs 30%) |
| `phenology/hants_extract.R` | HANTS harmonic fitting |
| `phenology/hplm_extract.R` | HPLM Logistic–Weibull fitting |
| `phenology/classify_time_allocation.R` | Constant vs linear vs breakpoint (**AI and PR separately**) |
| `plspm/plspm_pixelwise_bootstrap.R` | Pixel-wise three-stage PLS-PM |
| `plspm/extract_dominant_path_factor.R` | Dominant carry-over vs climate pathway |
| `plspm/extract_dominant_climate_weight.R` | Dominant climate variable from outer weights |
| `cmip6/diffreg_l1_carryover.R` | `dL1 ~ dCLI_L1 + lag(dL3) + lag(dCLI_L3)` |
| `cmip6/diffreg_ltotal_stage_climate.R` | `dL_total ~ dCLI_L1 + dCLI_L2 + dCLI_L3` |
| `cmip6/extract_dominant_share_l1.R` | Dominant variance share for L1 drivers |
| `random-forest/rf_train_predict_yearly.R` | RF training + CMIP6 future projection |
| `random-forest/eemd_trend_analysis.R` | EEMD trends by latitude band and SSP |
| `visualization/hovmoller_plot.R` | Latitude–year Hovmöller diagrams |
| `visualization/stage_trend_histogram.R` | Sen's slope distribution histograms |
| `visualization/polar_coefficient_map.R` | Arctic polar coefficient map |
| `visualization/plot_dominant_deltar2_trajectory.py` | Dominant climate-block share trajectory |
| `visualization/plot_dominant_transition_chord.R` | Hist → SSP chord diagram (paired pixels) |
| `visualization/stage_colors.csv` | Shared L1/L2/L3 colour palette |
| `validation/paired_diffreg_validation.py` | 20 paired windows; lag(dL3)→dL1 and ΔL_total dominance |

## Important methodological notes (recent updates)

1. **AI vs PR constant allocation** must be classified **separately** on each metric’s annual time series (`P > 0.05` = constant). Do not reuse the PR *P*-value for AI (or vice versa). Main 20% natural-forest reference: AI ≈ 86%, PR ≈ 93%.
2. **Amplitude sensitivity** varies only SOS/EOS amplitude (10% and 30%); PPOS/APOS stay at 90%. The main analysis remains at 20%.
3. **Dominant climate block for ΔL_total** uses `R²(L_total) > 0.30` and **strict `argmax(ΔR²_L1, L2, L3)`** (no tie exclusion). Trajectory and chord figures use the same rule.

## Forest types

- `NaturalForest`
- `PlantedForest`

## Notes

- Large raster operations are memory-intensive; adjust `terraOptions(memfrac = ...)` as needed.
- Output paths default under `{GOSIF_DATA_ROOT}/results/` (or manuscript-specific folders under `12_other/` / `9_model/Results/` when those exist).
- GeoTIFF inputs are not included in this repository.

## Citation

If you use this code, please cite the associated manuscript (to be added).

## License

MIT License — see `LICENSE`.
