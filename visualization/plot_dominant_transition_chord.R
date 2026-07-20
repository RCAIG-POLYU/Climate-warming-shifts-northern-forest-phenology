# Filter: R2(L_total) > 0.30; strict argmax(DeltaR2_L1, L2, L3), no tie exclusion.
# Run: Rscript plot_dominant_transition_chord.R
suppressPackageStartupMessages({  library(terra)
  library(dplyr)
  library(circlize)
})

data_root <- Sys.getenv("GOSIF_DATA_ROOT", unset = "F:/GO_SIF")
HERE <- Sys.getenv(
  "FIGURES_SHARE_LTOTAL_DIR",
  unset = file.path(dirname(normalizePath(".", winslash = "/")), "visualization")
)
# Prefer script directory when run via Rscript from repo
args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
if (length(file_arg)) {
  HERE <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
}
ROOT <- file.path(data_root, "9_model", "Results")
CMIP6 <- file.path(data_root, "10_CMIP6_Climate")

forest_types <- c("NaturalForest", "PlantedForest")
r2_min <- 0.30
link_transparency <- 0.30

stage_labels <- c("CLI_L1", "CLI_L2", "CLI_L3")
layer_names <- c(
  "DeltaR2_CLI_L1_on_Ltotal",
  "DeltaR2_CLI_L2_on_Ltotal",
  "DeltaR2_CLI_L3_on_Ltotal"
)

# Shared palette: stage_colors.csv (same hex as trajectory figure)
pal <- read.csv(file.path(HERE, "stage_colors.csv"), stringsAsFactors = FALSE)
C_STAGE <- setNames(pal$hex, paste0("CLI_", pal$stage))

periods <- list(
  list(
    id = "Hist", label = "2000-2024",
    dir = Sys.getenv("LTOTAL_HIST_DIR", unset = file.path(ROOT, "Historical_LM_L1_Ltotal_only"))
  ),
  list(
    id = "SSP126", label = "SSP126",
    dir = Sys.getenv("LTOTAL_SSP126_DIR", unset = file.path(CMIP6, "SSP1_2.6", "9_Future_LM_L1_Ltotal_only"))
  ),
  list(
    id = "SSP245", label = "SSP245",
    dir = Sys.getenv("LTOTAL_SSP245_DIR", unset = file.path(CMIP6, "SSP2_4.5", "9_Future_LM_L1_Ltotal_only"))
  ),
  list(
    id = "SSP585", label = "SSP585",
    dir = Sys.getenv("LTOTAL_SSP585_DIR", unset = file.path(CMIP6, "SSP5_8.5", "9_Future_LM_L1_Ltotal_only"))
  )
)

sector_grid_col <- function(sector_order) {
  stage <- sub("^.*_(CLI_L[123])$", "\\1", sector_order)
  unname(C_STAGE[stage])
}

read_period_matrix <- function(base_dir, forest) {
  r2 <- rast(file.path(base_dir, forest, "R2_Ltotal.tif"))
  mats <- lapply(layer_names, function(nm) {
    as.vector(values(rast(file.path(base_dir, forest, paste0(nm, ".tif")))))
  })
  m <- do.call(cbind, mats)
  r2v <- as.vector(values(r2))
  ok <- is.finite(r2v) & r2v > r2_min & apply(m, 1, function(x) all(is.finite(x) & x >= 0))
  list(m = m, ok = ok)
}

dominant_idx <- function(m) {
  # Strict argmax; ties -> first block (L1 > L2 > L3), same as numpy.argmax
  max.col(m, ties.method = "first")
}

build_flow_matrix <- function(forest) {
  message("\n=== ", forest, " ===")
  mats <- lapply(periods, function(p) {
    message("  read ", p$id)
    read_period_matrix(p$dir, forest)
  })

  ok_all <- Reduce(`&`, lapply(mats, `[[`, "ok"))
  n_pix <- sum(ok_all)
  message("  intersection valid pixels: ", format(n_pix, big.mark = ","))

  dom <- lapply(mats, function(x) dominant_idx(x$m[ok_all, , drop = FALSE]))
  names(dom) <- vapply(periods, `[[`, "", "id")

  hist_idx <- dom$Hist
  fut_ids <- c("SSP126", "SSP245", "SSP585")

  sectors <- c(
    paste0("Hist_", stage_labels),
    as.vector(outer(fut_ids, stage_labels, paste, sep = "_"))
  )
  mat <- matrix(0, nrow = 3, ncol = 9, dimnames = list(stage_labels, sectors[4:12]))

  for (fi in seq_along(fut_ids)) {
    fidx <- dom[[fut_ids[fi]]]
    for (hi in 1:3) {
      for (fj in 1:3) {
        cnt <- sum(hist_idx == hi & fidx == fj)
        col_nm <- paste0(fut_ids[fi], "_", stage_labels[fj])
        mat[hi, col_nm] <- cnt
      }
    }
  }

  rownames(mat) <- paste0("Hist_", stage_labels)
  list(mat = mat, n_pix = n_pix)
}

link_col_matrix <- function(mat) {
  link_col <- matrix(NA_character_, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
  for (hi in seq_len(nrow(mat))) {
    stage <- sub("^Hist_", "", rownames(mat)[hi])
    link_col[hi, ] <- C_STAGE[stage]
  }
  link_col
}

plot_chord <- function(mat, forest, n_pix, out_file) {
  sector_order <- c(
    paste0("Hist_", stage_labels),
    as.vector(outer(c("SSP126", "SSP245", "SSP585"), stage_labels, paste, sep = "_"))
  )

  cols <- sector_grid_col(sector_order)
  names(cols) <- sector_order
  mat2 <- mat[, sector_order[4:12], drop = FALSE]
  link_col <- link_col_matrix(mat2)

  png(out_file, width = 2600, height = 2600, res = 300)
  circos.clear()
  circos.par(
    start.degree = 90,
    gap.after = c(2, 2, 28, 2, 2, 28, 2, 2, 28, 2, 2, 28),
    track.margin = c(0.01, 0.01),
    points.overflow.warning = FALSE
  )

  chordDiagram(
    mat2,
    order = sector_order,
    grid.col = cols,
    col = link_col,
    transparency = link_transparency,
    directional = 1,
    direction.type = c("arrows", "diffHeight"),
    diffHeight = 0.04,
    link.arr.type = "big.arrow",
    annotationTrack = "grid",
    annotationTrackHeight = 0.035,
    preAllocateTracks = 1
  )

  circos.track(track.index = 1, panel.fun = function(x, y) {
    sector <- CELL_META$sector.index
    xlim <- CELL_META$xlim
    ylim <- CELL_META$ylim
    stage <- sub("^.*_(CLI_L[123])$", "\\1", sector)
    if (stage == "CLI_L2") {
      lab <- sub("^(Hist|SSP126|SSP245|SSP585)_.*$", "\\1", sector)
      if (lab == "Hist") lab <- "2000-2024"
      circos.text(
        mean(xlim), ylim[1] + mm_y(2),
        lab,
        facing = "bending.outside", niceFacing = TRUE, adj = c(0.5, 0.5), cex = 1.45,
        font = 2
      )
    }
  }, bg.border = NA)

  title(
    main = paste0("Dominant climate-block transitions - ", gsub("Forest", " forest", forest)),
    sub = paste0(
      "Hist -> SSP126 / SSP245 / SSP585; argmax(DeltaR2); ",
      "n = ", format(n_pix, big.mark = ","), " paired pixels; R2(L_total) > ", r2_min
    ),
    cex.main = 1.15, cex.sub = 0.75
  )

  legend(
    "topleft",
    legend = c("L1 (green-up)", "L2 (plateau)", "L3 (senescence)"),
    fill = unname(C_STAGE),
    border = NA, bty = "n", cex = 1.35, inset = c(0.02, 0.08),
    title = "Dominant block"
  )

  circos.clear()
  dev.off()
  message("Saved: ", out_file)
}

for (forest in forest_types) {
  res <- build_flow_matrix(forest)

  flow_long <- as.data.frame(as.table(res$mat), stringsAsFactors = FALSE) %>%
    rename(Hist = Var1, Future = Var2, n = Freq) %>%
    mutate(
      forest = forest,
      pct = round(100 * n / res$n_pix, 3)
    )
  write.csv(
    flow_long,
    file.path(HERE, paste0("Chord_flow_Hist_to_SSP_", forest, "_R2gt030.csv")),
    row.names = FALSE
  )

  plot_chord(
    res$mat,
    forest,
    res$n_pix,
    file.path(HERE, paste0("Fig_Ltotal_DominantTransition_Chord_", forest, "_R2gt030.png"))
  )
}

message("\nDone.")
