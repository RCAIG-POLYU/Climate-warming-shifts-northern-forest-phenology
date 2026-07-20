# Amplitude-threshold sensitivity extraction (SOS/EOS at 10% and 30%; PPOS/APOS fixed at 90%).
# Main analysis uses 20% (see phenology/dynamic_threshold_extract.R).
# Does not modify the main phenology products.

suppressPackageStartupMessages({
  library(lubridate)
  library(terra)
  source("R/config.R")
  source("R/phenology_io.R")
})

RATIO_PLAT <- 0.90
SENS_THRESHOLDS <- c(0.10, 0.30)
SENS_LABELS <- c("10pct", "30pct")

extract_phenology_tile <- function(daily_sg_dir, ratio_green, years = YEARS_HIST,
                                   daily_dates = seq(ymd("2000-02-26"), ymd("2024-12-31"), by = "day")) {
  daily_sg_files <- sort(list.files(daily_sg_dir, pattern = "gosif_daily_.*\\.tif$", full.names = TRUE))
  if (length(daily_sg_files) == 0) stop("No gosif_daily_*.tif in: ", daily_sg_dir)

  gosif_stack_daily <- rast(daily_sg_files)
  daily_matrix <- as.matrix(values(gosif_stack_daily))
  template <- rast(daily_sg_files[1])
  n_pixels <- nrow(daily_matrix)
  n_years <- length(years)

  SOS_matrix <- PPOS_matrix <- APOS_matrix <- EOS_matrix <-
    matrix(NA_real_, nrow = n_pixels, ncol = n_years)
  Greenup_Dur <- Plateau_Dur <- Senesc_Dur <-
    matrix(NA_real_, nrow = n_pixels, ncol = n_years)

  for (i in seq_along(years)) {
    yr <- years[i]
    year_days_idx <- which(year(daily_dates) == yr)
    if (length(year_days_idx) == 0) next
    year_matrix <- daily_matrix[, year_days_idx, drop = FALSE]

    annual_max <- apply(year_matrix, 1, max, na.rm = TRUE)
    annual_min <- apply(year_matrix, 1, min, na.rm = TRUE)
    invalid <- is.na(annual_max) | is.na(annual_min) |
      annual_max <= 0.1 | (annual_max - annual_min) < 0.05
    annual_max[invalid] <- NA
    annual_min[invalid] <- NA
    range_val <- annual_max - annual_min
    range_val[range_val == 0] <- NA

    year_ratio_amplitude <- (year_matrix - annual_min) / range_val
    year_ratio_max <- year_matrix / annual_max

    sos_idx <- apply(year_ratio_amplitude >= ratio_green, 1, function(x) which(x)[1])
    eos_idx <- apply(year_ratio_amplitude >= ratio_green, 1, function(x) {
      h <- which(x); if (length(h)) rev(h)[1] else NA_integer_
    })
    ppos_idx <- apply(year_ratio_max >= RATIO_PLAT, 1, function(x) which(x)[1])
    apos_idx <- apply(year_ratio_max >= RATIO_PLAT, 1, function(x) {
      h <- which(x); if (length(h)) rev(h)[1] else NA_integer_
    })

    SOS_global <- year_days_idx[sos_idx]
    EOS_global <- year_days_idx[eos_idx]
    PPOS_global <- year_days_idx[ppos_idx]
    APOS_global <- year_days_idx[apos_idx]

    doy_this_year <- yday(daily_dates[year_days_idx])
    SOS_doy <- doy_this_year[sos_idx]
    EOS_doy <- doy_this_year[eos_idx]

    sos_ok <- !is.na(SOS_doy) & SOS_doy >= 70 & SOS_doy <= 200
    eos_ok <- !is.na(EOS_doy) & EOS_doy >= 220 & EOS_doy <= 310
    order_ok <- !is.na(SOS_global) & !is.na(PPOS_global) &
      !is.na(APOS_global) & !is.na(EOS_global) &
      SOS_global < PPOS_global & PPOS_global <= APOS_global & APOS_global < EOS_global
    valid <- sos_ok & eos_ok & order_ok

    SOS_matrix[, i] <- ifelse(valid, SOS_global, NA)
    EOS_matrix[, i] <- ifelse(valid, EOS_global, NA)
    PPOS_matrix[, i] <- ifelse(valid, PPOS_global, NA)
    APOS_matrix[, i] <- ifelse(valid, APOS_global, NA)
    Greenup_Dur[, i] <- ifelse(valid, PPOS_global - SOS_global + 1, NA)
    Plateau_Dur[, i] <- ifelse(valid, APOS_global - PPOS_global + 1, NA)
    Senesc_Dur[, i] <- ifelse(valid, EOS_global - APOS_global + 1, NA)
  }

  list(
    template = template,
    SOS_DOY = matrix(as.integer(yday(daily_dates[SOS_matrix])), nrow = n_pixels, ncol = n_years),
    PPOS_DOY = matrix(as.integer(yday(daily_dates[PPOS_matrix])), nrow = n_pixels, ncol = n_years),
    APOS_DOY = matrix(as.integer(yday(daily_dates[APOS_matrix])), nrow = n_pixels, ncol = n_years),
    EOS_DOY = matrix(as.integer(yday(daily_dates[EOS_matrix])), nrow = n_pixels, ncol = n_years),
    Greenup_Dur = Greenup_Dur,
    Plateau_Dur = Plateau_Dur,
    Senesc_Dur = Senesc_Dur
  )
}

save_phenology_tif_enhanced <- function(mat, prefix, output_dir, template, years_vec) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  for (y in seq_along(years_vec)) {
    r <- template
    values(r) <- mat[, y]
    writeRaster(
      r, file.path(output_dir, sprintf("%s_%d.tif", prefix, years_vec[y])),
      overwrite = TRUE, gdal = c("COMPRESS=LZW")
    )
  }
  mean_vec <- rowMeans(mat, na.rm = TRUE)
  mean_vec[is.nan(mean_vec)] <- NA
  r_mean <- template
  values(r_mean) <- mean_vec
  writeRaster(
    r_mean, file.path(output_dir, sprintf("%s_2000_2024_mean.tif", prefix)),
    overwrite = TRUE, gdal = c("COMPRESS=LZW")
  )
}

merge_tile_folder <- function(parent_tiles_dir, merged_dir) {
  dir.create(merged_dir, showWarnings = FALSE, recursive = TRUE)
  subfolders <- list.dirs(parent_tiles_dir, recursive = FALSE, full.names = TRUE)
  all_files <- unique(unlist(lapply(subfolders, function(sf) {
    list.files(sf, pattern = "\\.tif$", full.names = FALSE)
  })))
  for (fname in all_files) {
    file_paths <- file.path(subfolders, fname)
    file_paths <- file_paths[file.exists(file_paths)]
    if (!length(file_paths)) next
    out_file <- file.path(merged_dir, fname)
    if (length(file_paths) == 1) {
      file.copy(file_paths[1], out_file, overwrite = TRUE)
      next
    }
    src <- sprc(lapply(file_paths, rast))
    merge(src, filename = out_file, overwrite = TRUE)
  }
}

classify_by_forest <- function(phenology_dir, output_dir, forest_type_file) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  forest_global <- rast(forest_type_file)
  tif_files <- list.files(phenology_dir, pattern = "\\.tif$", full.names = TRUE)
  for (file in tif_files) {
    r <- rast(file)
    f3 <- mask(resample(crop(forest_global, r), r), r)
    forest_vec <- as.integer(round(values(f3)))
    pheno_vec <- values(r)
    r_plant <- r; r_natural <- r
    values(r_plant) <- NA; values(r_natural) <- NA
    idx_planted <- which(forest_vec == 2)
    idx_natural <- which(forest_vec == 1)
    if (length(idx_planted)) values(r_plant)[idx_planted] <- pheno_vec[idx_planted]
    if (length(idx_natural)) values(r_natural)[idx_natural] <- pheno_vec[idx_natural]
    writeRaster(r_plant, file.path(output_dir, sub("\\.tif$", "_PlantedForest.tif", basename(file))),
                overwrite = TRUE, gdal = c("COMPRESS=LZW"))
    writeRaster(r_natural, file.path(output_dir, sub("\\.tif$", "_NaturalForest.tif", basename(file))),
                overwrite = TRUE, gdal = c("COMPRESS=LZW"))
  }
}

organize_forest_outputs <- function(forest_root_dir) {
  mapping <- list(
    SOS = "^SOS_DOY_.*\\.tif$",
    EOS = "^EOS_DOY_.*\\.tif$",
    Greenup_Duration = "^Greenup_Duration_days_.*\\.tif$",
    Senescence_Duration = "^Senescence_Duration_days_.*\\.tif$",
    Plateau_Duration = "^Plateau_Duration_days_.*\\.tif$"
  )
  for (sub in names(mapping)) {
    out_sub <- file.path(forest_root_dir, sub)
    dir.create(out_sub, showWarnings = FALSE, recursive = TRUE)
    hits <- list.files(forest_root_dir, pattern = mapping[[sub]], full.names = TRUE)
    for (src in hits) file.copy(src, file.path(out_sub, basename(src)), overwrite = TRUE)
  }
}

run_amplitude_sensitivity <- function(
  parent_input_dir,
  output_root,
  forest_type_file,
  thresholds = SENS_THRESHOLDS,
  threshold_labels = SENS_LABELS,
  tile_filter = NULL,
  years = YEARS_HIST
) {
  subfolders <- list.dirs(parent_input_dir, recursive = FALSE, full.names = TRUE)
  if (!is.null(tile_filter)) {
    subfolders <- subfolders[basename(subfolders) %in% tile_filter]
  }
  message("Tiles to process: ", length(subfolders))

  for (k in seq_along(thresholds)) {
    thr <- thresholds[k]
    thr_lab <- threshold_labels[k]
    message("\n================ Threshold ", thr * 100, "% ================")
    thr_root <- file.path(output_root, paste0("threshold_", thr_lab))
    tiles_dir <- file.path(thr_root, "1_tiles")
    merged_dir <- file.path(thr_root, "2_merged")
    forest_root <- file.path(thr_root, "3_forest_types")

    for (folder in subfolders) {
      folder_name <- basename(folder)
      out_tile <- file.path(tiles_dir, folder_name)
      marker <- file.path(out_tile, "DONE.txt")
      if (file.exists(marker)) {
        message("Skip existing tile: ", folder_name)
        next
      }
      message("Processing tile: ", folder_name)
      dir.create(out_tile, showWarnings = FALSE, recursive = TRUE)
      res <- extract_phenology_tile(folder, thr, years = years)
      save_phenology_tif_enhanced(res$SOS_DOY, "SOS_DOY", out_tile, res$template, years)
      save_phenology_tif_enhanced(res$EOS_DOY, "EOS_DOY", out_tile, res$template, years)
      save_phenology_tif_enhanced(res$PPOS_DOY, "PPOS_DOY", out_tile, res$template, years)
      save_phenology_tif_enhanced(res$APOS_DOY, "APOS_DOY", out_tile, res$template, years)
      save_phenology_tif_enhanced(res$Greenup_Dur, "Greenup_Duration_days", out_tile, res$template, years)
      save_phenology_tif_enhanced(res$Plateau_Dur, "Plateau_Duration_days", out_tile, res$template, years)
      save_phenology_tif_enhanced(res$Senesc_Dur, "Senescence_Duration_days", out_tile, res$template, years)
      writeLines(sprintf("threshold=%s; tile=%s; finished=%s", thr, folder_name, Sys.time()), marker)
      rm(res); gc()
    }

    message("Merging tiles for threshold ", thr)
    merge_tile_folder(tiles_dir, merged_dir)
    message("Classifying by forest type for threshold ", thr)
    classify_by_forest(merged_dir, forest_root, forest_type_file)
    organize_forest_outputs(forest_root)
  }
  message("Amplitude sensitivity extraction complete.")
}

if (sys.nframe() == 0L) {
  root <- get_data_root()
  args <- commandArgs(trailingOnly = TRUE)
  tile_filter <- if (length(args)) args else NULL
  run_amplitude_sensitivity(
    parent_input_dir = file.path(root, "gosif_sg"),
    output_root = file.path(root, "results", "phenology_sensitivity_amplitude"),
    forest_type_file = Sys.getenv("FOREST_TYPE_TIF", unset = file.path(root, "forest", "planted_natural_forest.tif")),
    tile_filter = tile_filter
  )
}
