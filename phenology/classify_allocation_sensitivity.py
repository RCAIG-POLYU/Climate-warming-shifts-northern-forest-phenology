"""
Classify constant vs optimal time allocation for amplitude sensitivity (10% / 30%).

Logic follows F:/Codes/判断最优_恒定时间分配.R and
F:/GO_SIF/6_Analysis/AI_PR/ (separate yearly-trend tests for AI and PR):
  - AI time series = Greenup / Senescence
  - PR time series = Plateau / (Greenup + Senescence)
  - Constant allocation: linear trend P > 0.05 (tested separately per metric)
  - Optimal allocation: linear trend P <= 0.05
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
from scipy import stats

import os

# Default: {GOSIF_DATA_ROOT}/results/phenology_sensitivity_amplitude
_DATA_ROOT = Path(os.environ.get("GOSIF_DATA_ROOT", r"F:/GO_SIF"))
ROOT = Path(
    os.environ.get(
        "SENSITIVITY_ROOT",
        str(_DATA_ROOT / "results" / "phenology_sensitivity_amplitude"),
    )
)
# Local working copy used during manuscript revision (override via SENSITIVITY_ROOT)
if not ROOT.exists():
    _fallback = _DATA_ROOT / "12_other" / "phenology_sensitivity_amplitude"
    if _fallback.exists():
        ROOT = _fallback

THRESHOLDS = [("10pct", "10%"), ("30pct", "30%")]
FOREST_TYPES = ["NaturalForest", "PlantedForest"]
YEARS = np.arange(2000, 2025, dtype=np.float64)
P_THRESHOLD = 0.05
MIN_VALID_YEARS = 15


def read_stack(forest_root: Path, subdir: str, prefix: str, forest_type: str) -> tuple[np.ndarray, dict]:
    files = [forest_root / subdir / f"{prefix}_{y}_{forest_type}.tif" for y in range(2000, 2025)]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} files, e.g. {missing[0]}")

    with rasterio.open(files[0]) as src:
        shape = (src.height, src.width)
        profile = src.profile.copy()

    stack = np.full((len(files), shape[0], shape[1]), np.nan, dtype=np.float32)
    for i, fp in enumerate(files):
        with rasterio.open(fp) as src:
            stack[i] = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                stack[i][stack[i] == nodata] = np.nan
    return stack, profile


def linear_trend_pvalues(ts: np.ndarray) -> np.ndarray:
    """ts: (n_year, n_pixel). Return two-sided p-values for linear slope (lm y ~ year)."""
    x = YEARS
    x_mean = x.mean()
    x_center = x - x_mean
    ssx = np.sum(x_center**2)

    y = ts
    valid = np.isfinite(y)
    n_valid = valid.sum(axis=0)

    y_mean = np.nanmean(y, axis=0)
    y_center = y - y_mean
    y_center[~valid] = 0.0

    slope = np.sum(x_center[:, None] * y_center, axis=0) / ssx
    resid = y - (y_mean + slope * x_center[:, None])
    resid[~valid] = np.nan
    sse = np.nansum(resid**2, axis=0)
    mse = sse / np.maximum(n_valid - 2, 1)
    se_slope = np.sqrt(mse / ssx)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = slope / se_slope
        p_vals = 2 * stats.t.sf(np.abs(t_stat), df=np.maximum(n_valid - 2, 1))

    p_vals[n_valid < MIN_VALID_YEARS] = np.nan
    return p_vals


def safe_stats(vals: np.ndarray) -> tuple[int, float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0, np.nan, np.nan
    return int(vals.size), float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)


def summarize_group(
    threshold_label: str,
    forest_type: str,
    metric_name: str,
    group_name: str,
    metric_vals: np.ndarray,
    mask: np.ndarray,
    total_n: int,
) -> dict:
    n, mean_v, sd_v = safe_stats(metric_vals[mask])
    return {
        "threshold": threshold_label,
        "forest_type": forest_type,
        "metric": metric_name,
        "allocation_group": group_name,
        "n_pixels": n,
        "pct_pixels": n / total_n * 100 if total_n else np.nan,
        "mean_value": mean_v,
        "sd_value": sd_v,
    }


def masks_from_pvalues(p_vals: np.ndarray, ai_mean: np.ndarray, pr_mean: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(p_vals)
    is_constant = valid & (p_vals > P_THRESHOLD)
    is_optimal = valid & (p_vals <= P_THRESHOLD)
    return {
        "valid": valid,
        "constant": is_constant,
        "optimal": is_optimal,
        "total_n": int(valid.sum()),
    }


def process_one(thr_dir: str, thr_label: str, forest_type: str) -> tuple[list[dict], dict]:
    forest_root = ROOT / f"threshold_{thr_dir}" / "3_forest_types"
    g_stack, profile = read_stack(forest_root, "Greenup_Duration", "Greenup_Duration_days", forest_type)
    s_stack, _ = read_stack(forest_root, "Senescence_Duration", "Senescence_Duration_days", forest_type)
    p_stack, _ = read_stack(forest_root, "Plateau_Duration", "Plateau_Duration_days", forest_type)

    with np.errstate(divide="ignore", invalid="ignore"):
        ai_stack = g_stack / s_stack
        ai_stack[s_stack == 0] = np.nan
        pr_stack = p_stack / (g_stack + s_stack)
        pr_stack[(g_stack + s_stack) == 0] = np.nan

    ai_mean = np.nanmean(ai_stack, axis=0)
    pr_mean = np.nanmean(pr_stack, axis=0)
    h, w = ai_mean.shape

    p_ai = linear_trend_pvalues(ai_stack.reshape(ai_stack.shape[0], -1)).reshape(h, w)
    p_pr = linear_trend_pvalues(pr_stack.reshape(pr_stack.shape[0], -1)).reshape(h, w)

    ai_masks = masks_from_pvalues(p_ai, ai_mean, pr_mean)
    pr_masks = masks_from_pvalues(p_pr, ai_mean, pr_mean)

    rows: list[dict] = []
    for metric_name, arr, masks in [
        ("AI", ai_mean, ai_masks),
        ("PR", pr_mean, pr_masks),
    ]:
        flat = arr.reshape(-1)
        for group_name, key in [
            ("constant_P_gt_0.05", "constant"),
            ("optimal_P_le_0.05", "optimal"),
            ("all_valid_pixels", "valid"),
        ]:
            rows.append(
                summarize_group(
                    thr_label,
                    forest_type,
                    metric_name,
                    group_name,
                    flat,
                    masks[key].reshape(-1),
                    masks["total_n"],
                )
            )

    class_summary = {
        "threshold": thr_label,
        "forest_type": forest_type,
        "ai_valid_pixels": ai_masks["total_n"],
        "pr_valid_pixels": pr_masks["total_n"],
        "ai_pct_constant": int(ai_masks["constant"].sum()) / ai_masks["total_n"] * 100 if ai_masks["total_n"] else np.nan,
        "ai_pct_optimal": int(ai_masks["optimal"].sum()) / ai_masks["total_n"] * 100 if ai_masks["total_n"] else np.nan,
        "pr_pct_constant": int(pr_masks["constant"].sum()) / pr_masks["total_n"] * 100 if pr_masks["total_n"] else np.nan,
        "pr_pct_optimal": int(pr_masks["optimal"].sum()) / pr_masks["total_n"] * 100 if pr_masks["total_n"] else np.nan,
    }

    out_dir = ROOT / f"threshold_{thr_dir}" / "4_allocation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, p_vals, masks in [("AI", p_ai, ai_masks), ("PR", p_pr, pr_masks)]:
        class_arr = np.full((h, w), 255, dtype=np.uint8)
        class_arr[masks["constant"]] = 1
        class_arr[masks["optimal"]] = 2

        prof = profile.copy()
        prof.update(dtype="uint8", nodata=255, count=1, compress="lzw")
        with rasterio.open(out_dir / f"classification_{metric}_{forest_type}.tif", "w", **prof) as dst:
            dst.write(class_arr, 1)

        prof = profile.copy()
        prof.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
        with rasterio.open(out_dir / f"P_value_{metric}_{forest_type}.tif", "w", **prof) as dst:
            out_p = p_vals.astype(np.float32)
            out_p[~np.isfinite(out_p)] = -9999.0
            dst.write(out_p, 1)

    prof = profile.copy()
    prof.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
    with rasterio.open(out_dir / f"AI_mean_{forest_type}.tif", "w", **prof) as dst:
        out = ai_mean.astype(np.float32)
        out[~np.isfinite(out)] = -9999.0
        dst.write(out, 1)
    with rasterio.open(out_dir / f"PR_mean_{forest_type}.tif", "w", **prof) as dst:
        out = pr_mean.astype(np.float32)
        out[~np.isfinite(out)] = -9999.0
        dst.write(out, 1)

    return rows, class_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean_sd(groups: list[tuple[float, float, int]]) -> tuple[float, float]:
    n_group = sum(g[2] for g in groups)
    mean_w = sum(g[0] * g[2] for g in groups) / n_group
    var_w = sum(((g[2] - 1) * g[1] ** 2 + g[2] * (g[0] - mean_w) ** 2) for g in groups) / max(n_group - 1, 1)
    return mean_w, float(np.sqrt(var_w))


def append_allforest_metrics(all_metrics: list[dict], thr_label: str) -> list[dict]:
    combined_rows: list[dict] = []
    for metric in ["AI", "PR"]:
        for group in ["constant_P_gt_0.05", "optimal_P_le_0.05", "all_valid_pixels"]:
            vals = []
            for ft in ["NaturalForest", "PlantedForest"]:
                m = next(
                    r
                    for r in all_metrics
                    if r["threshold"] == thr_label
                    and r["forest_type"] == ft
                    and r["metric"] == metric
                    and r["allocation_group"] == group
                )
                vals.append((m["mean_value"], m["sd_value"], m["n_pixels"]))
            mean_w, sd_w = weighted_mean_sd(vals)
            n_group = sum(v[2] for v in vals)
            all_valid = next(
                r
                for r in all_metrics
                if r["threshold"] == thr_label
                and r["forest_type"] == "NaturalForest"
                and r["metric"] == metric
                and r["allocation_group"] == "all_valid_pixels"
            )
            all_valid_pf = next(
                r
                for r in all_metrics
                if r["threshold"] == thr_label
                and r["forest_type"] == "PlantedForest"
                and r["metric"] == metric
                and r["allocation_group"] == "all_valid_pixels"
            )
            total_n = all_valid["n_pixels"] + all_valid_pf["n_pixels"]
            combined_rows.append(
                {
                    "threshold": thr_label,
                    "forest_type": "AllForest",
                    "metric": metric,
                    "allocation_group": group,
                    "n_pixels": n_group,
                    "pct_pixels": n_group / total_n * 100 if total_n else np.nan,
                    "mean_value": mean_w,
                    "sd_value": sd_w,
                }
            )
    return combined_rows


def append_allforest_class(all_class: list[dict], thr_label: str) -> dict:
    nf = next(r for r in all_class if r["threshold"] == thr_label and r["forest_type"] == "NaturalForest")
    pf = next(r for r in all_class if r["threshold"] == thr_label and r["forest_type"] == "PlantedForest")

    def wmean(field_n: str, field_pct: str) -> float:
        total = nf[field_n] + pf[field_n]
        return (nf[field_n] * nf[field_pct] + pf[field_n] * pf[field_pct]) / total

    return {
        "threshold": thr_label,
        "forest_type": "AllForest",
        "ai_valid_pixels": nf["ai_valid_pixels"] + pf["ai_valid_pixels"],
        "pr_valid_pixels": nf["pr_valid_pixels"] + pf["pr_valid_pixels"],
        "ai_pct_constant": wmean("ai_valid_pixels", "ai_pct_constant"),
        "ai_pct_optimal": wmean("ai_valid_pixels", "ai_pct_optimal"),
        "pr_pct_constant": wmean("pr_valid_pixels", "pr_pct_constant"),
        "pr_pct_optimal": wmean("pr_valid_pixels", "pr_pct_optimal"),
    }


def main() -> None:
    all_metrics: list[dict] = []
    all_class: list[dict] = []

    for thr_dir, thr_label in THRESHOLDS:
        for forest_type in FOREST_TYPES:
            print(f"Processing {thr_label} / {forest_type} ...")
            rows, class_summary = process_one(thr_dir, thr_label, forest_type)
            all_metrics.extend(rows)
            all_class.append(class_summary)

    for thr_label in ["10%", "30%"]:
        all_metrics.extend(append_allforest_metrics(all_metrics, thr_label))
        all_class.append(append_allforest_class(all_class, thr_label))

    metrics_file = ROOT / "allocation_summary_by_group_10_30pct.csv"
    class_file = ROOT / "allocation_class_proportions_10_30pct.csv"
    write_csv(metrics_file, all_metrics)
    write_csv(class_file, all_class)

    compare_rows = []
    for thr_label in ["10%", "30%"]:
        for forest_type in FOREST_TYPES + ["AllForest"]:
            for metric in ["AI", "PR"]:
                for group in ["constant_P_gt_0.05", "optimal_P_le_0.05", "all_valid_pixels"]:
                    m = next(
                        r
                        for r in all_metrics
                        if r["threshold"] == thr_label
                        and r["forest_type"] == forest_type
                        and r["metric"] == metric
                        and r["allocation_group"] == group
                    )
                    compare_rows.append(
                        {
                            "threshold": thr_label,
                            "forest_type": forest_type,
                            "metric": metric,
                            "group": group.replace("_P_gt_0.05", "").replace("_P_le_0.05", "").replace("_valid_pixels", ""),
                            "n_pixels": m["n_pixels"],
                            "pct_pixels": round(m["pct_pixels"], 2) if np.isfinite(m["pct_pixels"]) else "",
                            "mean": round(m["mean_value"], 4),
                            "sd": round(m["sd_value"], 4),
                        }
                    )

    compare_file = ROOT / "allocation_summary_table_10_30pct.csv"
    write_csv(compare_file, compare_rows)

    robust_rows = []
    for forest_type in FOREST_TYPES + ["AllForest"]:
        for metric in ["AI", "PR"]:
            m10 = next(
                r
                for r in all_metrics
                if r["threshold"] == "10%"
                and r["forest_type"] == forest_type
                and r["metric"] == metric
                and r["allocation_group"] == "all_valid_pixels"
            )
            m30 = next(
                r
                for r in all_metrics
                if r["threshold"] == "30%"
                and r["forest_type"] == forest_type
                and r["metric"] == metric
                and r["allocation_group"] == "all_valid_pixels"
            )
            robust_rows.append(
                {
                    "forest_type": forest_type,
                    "metric": metric,
                    "mean_10pct": round(m10["mean_value"], 4),
                    "mean_30pct": round(m30["mean_value"], 4),
                    "delta_mean_30_minus_10": round(m30["mean_value"] - m10["mean_value"], 4),
                    "pct_change_mean": round((m30["mean_value"] - m10["mean_value"]) / m10["mean_value"] * 100, 2)
                    if m10["mean_value"]
                    else np.nan,
                    "sd_10pct": round(m10["sd_value"], 4),
                    "sd_30pct": round(m30["sd_value"], 4),
                }
            )

    robust_file = ROOT / "AI_PR_robustness_10_vs_30pct.csv"
    write_csv(robust_file, robust_rows)

    pub_rows = []
    for thr_label in ["10%", "30%"]:
        for forest_type in ["NaturalForest", "PlantedForest", "AllForest"]:
            cls = next(r for r in all_class if r["threshold"] == thr_label and r["forest_type"] == forest_type)
            for metric in ["AI", "PR"]:
                const = next(
                    r
                    for r in all_metrics
                    if r["threshold"] == thr_label
                    and r["forest_type"] == forest_type
                    and r["metric"] == metric
                    and r["allocation_group"] == "constant_P_gt_0.05"
                )
                opt = next(
                    r
                    for r in all_metrics
                    if r["threshold"] == thr_label
                    and r["forest_type"] == forest_type
                    and r["metric"] == metric
                    and r["allocation_group"] == "optimal_P_le_0.05"
                )
                allpix = next(
                    r
                    for r in all_metrics
                    if r["threshold"] == thr_label
                    and r["forest_type"] == forest_type
                    and r["metric"] == metric
                    and r["allocation_group"] == "all_valid_pixels"
                )
                pct_const = cls["ai_pct_constant"] if metric == "AI" else cls["pr_pct_constant"]
                pct_opt = cls["ai_pct_optimal"] if metric == "AI" else cls["pr_pct_optimal"]
                n_valid = cls["ai_valid_pixels"] if metric == "AI" else cls["pr_valid_pixels"]
                pub_rows.append(
                    {
                        "threshold": thr_label,
                        "forest_type": forest_type,
                        "metric": metric,
                        "n_valid_pixels": n_valid,
                        "pct_constant_allocation": round(pct_const, 2),
                        "pct_optimal_allocation": round(pct_opt, 2),
                        "constant_mean": round(const["mean_value"], 3),
                        "constant_sd": round(const["sd_value"], 3),
                        "optimal_mean": round(opt["mean_value"], 3),
                        "optimal_sd": round(opt["sd_value"], 3),
                        "domain_mean": round(allpix["mean_value"], 3),
                        "domain_sd": round(allpix["sd_value"], 3),
                    }
                )

    pub_file = ROOT / "Table_amplitude_sensitivity_allocation_AI_PR.csv"
    write_csv(pub_file, pub_rows)

    print(f"\nSaved:\n  {metrics_file}\n  {class_file}\n  {compare_file}\n  {robust_file}\n  {pub_file}")
    for row in all_class:
        print(row)


if __name__ == "__main__":
    main()
