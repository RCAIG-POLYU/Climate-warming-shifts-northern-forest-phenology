# Classify temporal stability of stage time allocation (constant / linear / nonlinear).
# AI and PR are tested SEPARATELY on their own annual time series (P > 0.05 = constant).
# Matches F:/GO_SIF/6_Analysis/AI_PR and the sensitivity Supplementary Table 1 logic.

suppressPackageStartupMessages({
  library(terra)
  library(segmented)
  library(lmtest)
  source("R/config.R")
})

classify_pixel <- function(ts, years) {
  valid <- !is.na(ts)
  if (sum(valid) < 15) return(c(class = NA_integer_, pvalue = NA_real_))
  y <- ts[valid]; x <- years[valid]
  lm_mod <- lm(y ~ x)
  p_lin <- pf(summary(lm_mod)$fstatistic[1], summary(lm_mod)$fstatistic[2],
              summary(lm_mod)$fstatistic[3], lower.tail = FALSE)
  if (p_lin > 0.05) return(c(class = 1L, pvalue = p_lin))
  seg_mod <- try(segmented(lm_mod, seg.Z = ~x), silent = TRUE)
  if (inherits(seg_mod, "try-error")) return(c(class = 2L, pvalue = p_lin))
  p_seg <- davies.test(lm_mod, seg.Z = ~x)$p.value
  c(class = ifelse(p_seg <= 0.05, 3L, 2L), pvalue = p_lin)
}

#' Build AI and PR annual stacks from stage-duration rasters.
build_ai_pr_stacks <- function(greenup_dir, senescence_dir, plateau_dir, forest, years = YEARS_HIST) {
  vgu <- rast(file.path(greenup_dir, paste0("Greenup_Duration_days_", years, "_", forest, ".tif")))
  vss <- rast(file.path(senescence_dir, paste0("Senescence_Duration_days_", years, "_", forest, ".tif")))
  plt <- rast(file.path(plateau_dir, paste0("Plateau_Duration_days_", years, "_", forest, ".tif")))
  ai <- ifel(vss == 0, NA, vgu / vss)
  pr <- ifel((vgu + vss) == 0, NA, plt / (vgu + vss))
  list(AI = ai, PR = pr)
}

#' Classify one metric stack (AI or PR). Class 1 = constant (P > 0.05).
run_classify_allocation <- function(metric_stack, years = YEARS_HIST) {
  out <- app(metric_stack, function(v) classify_pixel(v, years))
  names(out) <- c("Allocation_class", "Linear_pvalue")
  out
}

#' Classify both AI and PR; write rasters and print constant proportions.
run_classify_ai_pr <- function(greenup_dir, senescence_dir, plateau_dir, forest,
                              out_dir, years = YEARS_HIST) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  stacks <- build_ai_pr_stacks(greenup_dir, senescence_dir, plateau_dir, forest, years)

  for (metric in c("AI", "PR")) {
    message("Classifying ", metric, " for ", forest, " ...")
    res <- run_classify_allocation(stacks[[metric]], years)
    writeRaster(
      res[[1]],
      file.path(out_dir, sprintf("classification_%s_%s.tif", tolower(metric), forest)),
      overwrite = TRUE,
      wopt = list(datatype = "INT1U", NAflag = 255)
    )
    writeRaster(
      res[[2]],
      file.path(out_dir, sprintf("P_value_%s_%s.tif", metric, forest)),
      overwrite = TRUE,
      wopt = list(datatype = "FLT4S", NAflag = -9999)
    )
    fr <- freq(res[[1]])
    total <- sum(fr$count, na.rm = TRUE)
    pct1 <- if (any(fr$value == 1)) fr$count[fr$value == 1] / total * 100 else NA_real_
    message(sprintf("  %s constant (class 1): %.2f%%", metric, pct1))
  }
  invisible
}

if (sys.nframe() == 0L) {
  root <- get_data_root()
  pheno <- file.path(root, "phenology")
  out_dir <- file.path(root, "results", "ai_pr_allocation")
  for (forest in FOREST_TYPES) {
    run_classify_ai_pr(
      file.path(pheno, "Greenup_Duration"),
      file.path(pheno, "Senescence_Duration"),
      file.path(pheno, "Plateau_Duration"),
      forest,
      out_dir
    )
  }
}
