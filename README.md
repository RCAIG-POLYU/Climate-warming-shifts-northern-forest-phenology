# Climate warming shifts northern forest phenology

R code for stage-resolved forest phenology analysis using GOSIF solar-induced fluorescence (SIF), partial least squares path modeling (PLS-PM), pixel-wise first-difference regression, and CMIP6-based future projections.

## Overview

This repository supports a multi-stage phenological framework in which the growing season is partitioned into:

| Stage | Definition |
|-------|------------|
| **L1** | Green-up (SOS → PPOS) |
| **L2** | Plateau (PPOS → APOS) |
| **L3** | Senescence (APOS → EOS) |

Key analyses include:

- **Phenology extraction** from daily GOSIF (dynamic threshold, HANTS, HPLM)
- **PLS-PM** linking stage-specific climate, phenological carry-over, and stage duration
- **First-difference regression** quantifying drivers of interannual variability in L1 and total season length
- **Random forest** projecting future growing-season length from yearly climate
- **Post-processing** of dominant climate/pathway factors and visualization

## Repository structure

```
.
├── R/
│   ├── config.R          # Shared constants and data-root helper
│   └── paths.R           # Phenology / climate path builders
├── phenology/            # Dynamic threshold, HANTS, HPLM extraction; allocation class
├── plspm/                # Pixel-wise PLS-PM and dominant-factor extraction
├── cmip6/                # Difference regression and variance-share processing
├── random-forest/        # RF training, future projection, EEMD trends
├── visualization/        # Hovmöller, histograms, polar coefficient maps
└── README.md
```

## Requirements

- R >= 4.1
- [terra](https://cran.r-project.org/package=terra)
- [ranger](https://cran.r-project.org/package=ranger)
- [plspm](https://cran.r-project.org/package=plspm)
- [dplyr](https://cran.r-project.org/package=dplyr)
- [ggplot2](https://cran.r-project.org/package=ggplot2)
- Optional: `Rlibeemd`, `segmented`, `lmtest`, `lubridate`, `tidyr`, `zoo`, `patchwork`, `minpack.lm`, `tidyterra`, `sf`, `rnaturalearth`, `scales`

Install dependencies:

```r
install.packages(c(
  "terra", "ranger", "plspm", "dplyr", "ggplot2",
  "lubridate", "tidyr", "zoo", "patchwork", "segmented", "lmtest",
  "minpack.lm", "tidyterra", "sf", "rnaturalearth", "scales"
))
install.packages("Rlibeemd")  # if available for your platform
```

## Data layout

Set the environment variable `GOSIF_DATA_ROOT` to your local data directory (default: `data/`). Expected structure:

```
data/
├── phenology/
│   ├── LGS/
│   ├── Greenup_Duration/
│   ├── Plateau_Duration/
│   └── Senescence_Duration/
├── stage_climate/
├── climate/              # yearly climate by forest type
├── analysis/ai_pr/       # AI and PR mean rasters per forest type
├── gosif_sg/             # daily smoothed GOSIF stacks (subfolders per region)
├── cmip6/
│   ├── SSP1_2.6/
│   │   ├── yearly_climate_naturalforest/
│   │   └── future_phenology/
│   ├── SSP2_4.5/
│   └── SSP5_8.5/
└── results/              # created by scripts
```

GeoTIFF naming follows consistent patterns, e.g.:

- `Greenup_Duration_days_{year}_{ForestType}.tif`
- `{ClimateVar}_{Stage}_{year}_{ForestType}.tif`
- Future phenology: `{Stage}_{year}.tif` under `future_phenology/{ForestType}/`

## Usage

From the repository root:

```r
Sys.setenv(GOSIF_DATA_ROOT = "path/to/your/data")
setwd("path/to/this/repository")

source("phenology/dynamic_threshold_extract.R")
source("random-forest/rf_train_predict_yearly.R")
```

Train RF, then project futures:

```bash
Rscript random-forest/rf_train_predict_yearly.R          # RF_MODE=train (default)
RF_MODE=predict RF_SSP=SSP2_4.5 Rscript random-forest/rf_train_predict_yearly.R
```

Each script can also be run with `Rscript` from the repository root.

Scripts use `if (sys.nframe() == 0L)` guards so that sourcing loads functions without automatically running the full pipeline.

## Script reference

| Script | Description |
|--------|-------------|
| `phenology/dynamic_threshold_extract.R` | Dynamic-threshold SOS/EOS (20%) and PPOS/APOS (90%) |
| `phenology/hants_extract.R` | HANTS harmonic fitting for SOS/PPOS/APOS/EOS |
| `phenology/hplm_extract.R` | HPLM Logistic–Weibull fitting; batch over regions |
| `phenology/classify_time_allocation.R` | Constant vs linear vs breakpoint allocation |
| `plspm/plspm_pixelwise_bootstrap.R` | Pixel-wise three-stage PLS-PM with bootstrap paths |
| `plspm/extract_dominant_path_factor.R` | Dominant carry-over vs climate pathway |
| `plspm/extract_dominant_climate_weight.R` | Dominant climate variable from outer weights |
| `cmip6/diffreg_l1_carryover.R` | `dL1 ~ dCLI_L1 + lag(dL3) + lag(dCLI_L3)` |
| `cmip6/diffreg_ltotal_stage_climate.R` | `dL_total ~ dCLI_L1 + dCLI_L2 + dCLI_L3` |
| `cmip6/extract_dominant_share_l1.R` | Dominant variance share for L1 drivers |
| `random-forest/rf_train_predict_yearly.R` | RF training (hold-out) + CMIP6 future L1/L2/L3 projection |
| `random-forest/eemd_trend_analysis.R` | EEMD trends by latitude band and SSP |
| `visualization/hovmoller_plot.R` | Latitude–year Hovmöller diagrams |
| `visualization/stage_trend_histogram.R` | Sen's slope distribution histograms |
| `visualization/polar_coefficient_map.R` | Arctic polar map with zero-centered coefficient scale |

## Forest types

- `NaturalForest`
- `PlantedForest`

## Notes

- Large raster operations are memory-intensive; adjust `terra` options (`terraOptions(memfrac = ...)`) as needed.
- PLS-PM pixel-wise fitting is computationally expensive; use a spatial subset (`extent`) for testing.
- Output paths are written under `{GOSIF_DATA_ROOT}/results/` by default.
- GeoTIFF inputs are not included in this repository.

## Citation

If you use this code, please cite the associated manuscript (to be added).

## License

MIT License — see `LICENSE` if provided; otherwise contact the repository owner.
