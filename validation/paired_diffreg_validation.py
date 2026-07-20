# -*- coding: utf-8 -*-
"""
Paired-site v4 — validate TWO Results claims from difference-regression outputs
at 20 mixed 20x20 windows (matched local climate).

Filtering aligned with main study (Figures_Share_Ltotal/):
  - plot_dominant_deltar2_trajectory.py : R2_Ltotal > 0.30; strict argmax(DeltaR2)
  - plot_dominant_transition_chord.R    : same dominant rule on paired pixels
  - Part 1 (Coef_L3prev_to_L1)          : R2_L1 > 0.30

PART 1  lag(dL3) -> dL1  [Coef_L3prev_to_L1]
PART 2  dL_total: dominant-pixel share by L1/L2/L3 climate blocks (strict argmax DeltaR2)

Run: python paired_diffreg_validation.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import rasterio
from scipy import stats

# Paths & constants (match Figures_Share_Ltotal main figures)
_DATA_ROOT = Path(os.environ.get("GOSIF_DATA_ROOT", r"F:/GO_SIF"))
_REPO_VIS = Path(__file__).resolve().parent.parent / "visualization"
STAGE_COLORS_CSV = Path(
    os.environ.get("STAGE_COLORS_CSV", str(_REPO_VIS / "stage_colors.csv"))
)
FOREST_TIF = Path(
    os.environ.get("FOREST_TYPE_TIF", r"F:/Forest/planted_natural forest.tif")
)
SITES_CSV = Path(
    os.environ.get(
        "PAIRED_SITES_CSV",
        str(_DATA_ROOT / "12_other" / "paired-site comparison" / "20 sites" / "mixed_sites_20x20.csv"),
    )
)
OUT_DIR = Path(
    os.environ.get(
        "PAIRED_OUT_DIR",
        str(
            _DATA_ROOT
            / "12_other"
            / "paired-site comparison"
            / "20 sites"
            / "discussion"
            / "v4"
        ),
    )
)

WIN = 20
LAT_MIN = 30.0
R2_MIN_L1 = 0.30
R2_MIN_LTOTAL = 0.30
COEF_CLIP = 2.0

PERIOD_ORDER = ["2000-2024", "SSP126", "SSP245", "SSP585"]
PERIOD_LABEL = {p: p for p in PERIOD_ORDER}
HIST = "2000-2024"
FUTURE_PERIODS = ["SSP126", "SSP245", "SSP585"]

_CMIP6 = _DATA_ROOT / "10_CMIP6_Climate"
PERIOD_DIRS = {
    "2000-2024": Path(
        os.environ.get(
            "LTOTAL_HIST_DIR",
            str(_DATA_ROOT / "9_model" / "Results" / "Historical_LM_L1_Ltotal_only"),
        )
    ),
    "SSP126": Path(
        os.environ.get(
            "LTOTAL_SSP126_DIR",
            str(_CMIP6 / "SSP1_2.6" / "9_Future_LM_L1_Ltotal_only"),
        )
    ),
    "SSP245": Path(
        os.environ.get(
            "LTOTAL_SSP245_DIR",
            str(_CMIP6 / "SSP2_4.5" / "9_Future_LM_L1_Ltotal_only"),
        )
    ),
    "SSP585": Path(
        os.environ.get(
            "LTOTAL_SSP585_DIR",
            str(_CMIP6 / "SSP5_8.5" / "9_Future_LM_L1_Ltotal_only"),
        )
    ),
}

LTOTAL_SHARE_KEYS = [
    "Share_CLI_L1_on_Ltotal",
    "Share_CLI_L2_on_Ltotal",
    "Share_CLI_L3_on_Ltotal",
]
LTOTAL_DR2_KEYS = [
    "DeltaR2_CLI_L1_on_Ltotal",
    "DeltaR2_CLI_L2_on_Ltotal",
    "DeltaR2_CLI_L3_on_Ltotal",
]

C_NAT, C_PLA = "#3C5488", "#E64B35"
_palette = pd.read_csv(STAGE_COLORS_CSV)
C_STAGE = dict(zip(_palette["stage"], _palette["hex"]))
C_L1, C_L2, C_L3 = C_STAGE["L1"], C_STAGE["L2"], C_STAGE["L3"]
C_MULTI = "#5BA8A0"
C_BUBBLE_PCT = "#9E9E9E"  # negative-pixel % bubbles: one colour, size encodes value
PAL_SCENARIO = {
    "2000-2024": "#8C8C8C",
    "SSP126": "#4DBBD5",
    "SSP245": "#3C5488",
    "SSP585": "#E64B35",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FILTER_NOTE = (
    f"R$^2_{{L1}}$ > {R2_MIN_L1:g} (Part 1); "
    f"R$^2_{{Ltotal}}$ > {R2_MIN_LTOTAL:g}, strict argmax($\\Delta R^2$) (Part 2)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def save_fig(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", stem)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", colors="black", width=0.8, length=3)


def sample_vals(ds, lons, lats):
    if len(lons) == 0:
        return np.array([], dtype=float)
    v = np.array([x[0] for x in ds.sample(zip(lons, lats))], dtype=float)
    if ds.nodata is not None:
        v[v == ds.nodata] = np.nan
    return v


def window_coords(transform, row_c, col_c, half):
    rows = np.arange(row_c - half, row_c - half + WIN)
    cols = np.arange(col_c - half, col_c - half + WIN)
    lons, lats = [], []
    for r in rows:
        for c in cols:
            lons.append(transform.c + c * transform.a + transform.a / 2)
            lats.append(transform.f + r * transform.e + transform.e / 2)
    return np.array(lons), np.array(lats)


def dominant_dr2(d1, d2, d3):
    """Strict argmax(DeltaR2_L1, L2, L3); ties -> first block (same as main trajectory/chord)."""
    arr = np.column_stack([d1, d2, d3])
    if arr.size == 0:
        return np.array([], dtype=int)
    bad = ~np.all(np.isfinite(arr), axis=1) | np.any(arr < 0, axis=1)
    codes = np.argmax(arr, axis=1) + 1
    codes[bad] = 0
    return codes


def wilcoxon_sites(hist_vals, fut_vals):
    d = np.asarray(fut_vals, float) - np.asarray(hist_vals, float)
    d = d[np.isfinite(d)]
    if len(d) < 8:
        return np.nan
    try:
        _, p = stats.wilcoxon(d, alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def summarize_coef(coef):
    coef = coef[np.isfinite(coef)]
    coef = coef[np.abs(coef) <= COEF_CLIP]
    n = len(coef)
    if n == 0:
        return dict(n=0, mean=np.nan, median=np.nan, pct_neg=np.nan, pct_pos=np.nan)
    return dict(
        n=n,
        mean=float(np.mean(coef)),
        median=float(np.median(coef)),
        pct_neg=100.0 * np.mean(coef < 0),
        pct_pos=100.0 * np.mean(coef > 0),
    )


def summarize_ltotal(sh1, sh2, sh3, d1, d2, d3, dom):
    m = np.isfinite(sh1) & np.isfinite(sh2) & np.isfinite(sh3)
    sh1, sh2, sh3 = sh1[m], sh2[m], sh3[m]
    d1, d2, d3 = d1[m], d2[m], d3[m]
    dom = dom[m]
    n = len(sh1)
    if n == 0:
        return {}
    tot_sh = sh1 + sh2 + sh3
    tot_sh = np.where(tot_sh > 0, tot_sh, np.nan)
    dom_ok = dom > 0
    n_dom = int(dom_ok.sum())
    pct = lambda code: (100.0 * np.mean(dom[dom_ok] == code) if n_dom else np.nan)
    dsum = d1 + d2 + d3
    dsum = np.where(dsum > 0, dsum, np.nan)
    return {
        "n_lt": n,
        "n_dom": n_dom,
        "mean_share_L1": float(np.nanmean(sh1)),
        "mean_share_L2": float(np.nanmean(sh2)),
        "mean_share_L3": float(np.nanmean(sh3)),
        "spring_frac": float(np.nanmean(sh1 / tot_sh)),
        "multistage_frac": float(np.nanmean((sh2 + sh3) / tot_sh)),
        "mean_frac_L1": float(np.nanmean(d1 / dsum)),
        "mean_frac_L2": float(np.nanmean(d2 / dsum)),
        "mean_frac_L3": float(np.nanmean(d3 / dsum)),
        "mean_dr2_L1": float(np.nanmean(d1)),
        "mean_dr2_L2": float(np.nanmean(d2)),
        "mean_dr2_L3": float(np.nanmean(d3)),
        "pct_dom_L1": pct(1),
        "pct_dom_L2": pct(2),
        "pct_dom_L3": pct(3),
        "pct_multistage_dom": (
            100.0 * np.mean((dom[dom_ok] == 2) | (dom[dom_ok] == 3)) if n_dom else np.nan
        ),
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_all():
    sites = pd.read_csv(SITES_CSV)
    half = WIN // 2
    rows = []

    with rasterio.open(FOREST_TIF) as fsrc:
        transform = fsrc.transform
        row30 = int((transform.f - LAT_MIN) / abs(transform.e))
        forest_crop = fsrc.read(1, window=rasterio.windows.Window(0, 0, fsrc.width, row30))

    for period, base in PERIOD_DIRS.items():
        print(f"Extracting {period} ...")
        for forest in ["NaturalForest", "PlantedForest"]:
            ft = "natural" if forest == "NaturalForest" else "planted"
            paths = {
                "coef": base / forest / "Coef_L3prev_to_L1.tif",
                "r2_l1": base / forest / "R2_L1.tif",
                "r2_lt": base / forest / "R2_Ltotal.tif",
            }
            shares = {s: base / forest / f"{s}.tif" for s in LTOTAL_SHARE_KEYS}
            dr2s = {s: base / forest / f"{s}.tif" for s in LTOTAL_DR2_KEYS}
            for p in list(paths.values()) + list(shares.values()) + list(dr2s.values()):
                if not p.exists():
                    raise FileNotFoundError(p)

            with rasterio.open(paths["coef"]) as ds_c, \
                 rasterio.open(paths["r2_l1"]) as ds_r1, \
                 rasterio.open(paths["r2_lt"]) as ds_rlt, \
                 rasterio.open(shares[LTOTAL_SHARE_KEYS[0]]) as ds_s1, \
                 rasterio.open(shares[LTOTAL_SHARE_KEYS[1]]) as ds_s2, \
                 rasterio.open(shares[LTOTAL_SHARE_KEYS[2]]) as ds_s3, \
                 rasterio.open(dr2s[LTOTAL_DR2_KEYS[0]]) as ds_d1, \
                 rasterio.open(dr2s[LTOTAL_DR2_KEYS[1]]) as ds_d2, \
                 rasterio.open(dr2s[LTOTAL_DR2_KEYS[2]]) as ds_d3:

                for _, site in sites.iterrows():
                    sid = int(site["site_id"])
                    rc, cc = int(site["row"]), int(site["col"])
                    lons, lats = window_coords(transform, rc, cc, half)
                    r0, c0 = rc - half, cc - half
                    patch = forest_crop[r0:r0 + WIN, c0:c0 + WIN].ravel()
                    code = 1 if ft == "natural" else 2
                    mask = patch == code
                    lon, lat = lons[mask], lats[mask]

                    coef = sample_vals(ds_c, lon, lat)
                    r1 = sample_vals(ds_r1, lon, lat)
                    rlt = sample_vals(ds_rlt, lon, lat)
                    s1 = sample_vals(ds_s1, lon, lat)
                    s2 = sample_vals(ds_s2, lon, lat)
                    s3 = sample_vals(ds_s3, lon, lat)
                    d1 = sample_vals(ds_d1, lon, lat)
                    d2 = sample_vals(ds_d2, lon, lat)
                    d3 = sample_vals(ds_d3, lon, lat)

                    # Part 1: R2_L1 > 0.30 (strict, as in R scripts)
                    m1 = np.isfinite(coef) & np.isfinite(r1) & (r1 > R2_MIN_L1)
                    cstat = summarize_coef(coef[m1])

                    # Part 2: R2_Ltotal > 0.30; Share in [0, 1.5]; DeltaR2 >= 0
                    m2 = (
                        np.isfinite(s1) & np.isfinite(s2) & np.isfinite(s3)
                        & np.isfinite(d1) & np.isfinite(d2) & np.isfinite(d3)
                        & np.isfinite(rlt) & (rlt > R2_MIN_LTOTAL)
                    )
                    for sh in (s1, s2, s3):
                        m2 &= np.isfinite(sh) & (sh >= 0) & (sh <= 1.5)
                    dom = dominant_dr2(d1[m2], d2[m2], d3[m2])
                    lstat = summarize_ltotal(s1[m2], s2[m2], s3[m2], d1[m2], d2[m2], d3[m2], dom)

                    rec = {
                        "site_id": sid, "period": period, "forest_type": ft,
                        "lon": site["lon"], "lat": site["lat"],
                        "n_forest_pixels": int(mask.sum()),
                        **{f"coef_{k}": v for k, v in cstat.items()},
                        **{k: v for k, v in lstat.items()},
                    }
                    rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "paired_sites_diffreg_metrics.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_part1(df):
    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(PERIOD_ORDER))

    for ax, metric, ylab, title in [
        (ax1, "coef_mean", "Mean coefficient (d d$^{-1}$)", "a  Magnitude"),
        (ax2, "coef_pct_neg", "Negative pixels (%)", "b  Sign structure"),
    ]:
        for ft, col, ls in [("natural", C_NAT, "-"), ("planted", C_PLA, "--")]:
            m, e = [], []
            for p in PERIOD_ORDER:
                sub = df[(df.period == p) & (df.forest_type == ft)]
                m.append(sub[metric].mean())
                e.append(sem(sub[metric]))
            ax.plot(x, m, f"o{ls}", color=col, lw=1.8, ms=5, label=ft.capitalize())
            ax.fill_between(x, np.array(m) - e, np.array(m) + e, color=col, alpha=0.12)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([PERIOD_LABEL[p] for p in PERIOD_ORDER], rotation=15, ha="right")
        ax.set_ylabel(ylab, fontweight="bold", fontsize=10)
        ax.set_title(title, fontweight="bold", loc="left", fontsize=11)
        ax.grid(axis="y", color="#eee", lw=0.5)
    ax1.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "PART 1  lag($\\Delta$L3) $\\rightarrow$ $\\Delta$L1  [Coef_L3prev_to_L1]\n"
        f"20 paired windows; {FILTER_NOTE.split(';')[0]}",
        fontsize=11, fontweight="bold", y=1.10,
    )
    fig.tight_layout()
    save_fig(fig, "Part1_carryover_coef_trajectory")

    # paired site lines: 2000-2024 -> each SSP
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    sub_h = df[df.period == HIST]
    for ax, fut in zip(axes, FUTURE_PERIODS):
        sub_f = df[df.period == fut]
        for ft, col in [("natural", C_NAT), ("planted", C_PLA)]:
            h = sub_h[sub_h.forest_type == ft].sort_values("site_id")["coef_mean"].values
            f = sub_f[sub_f.forest_type == ft].sort_values("site_id")["coef_mean"].values
            for i in range(len(h)):
                ax.plot([0, 1], [h[i], f[i]], "-", color="#ccc", lw=0.8, zorder=1)
            ax.scatter(np.zeros(len(h)), h, s=30, c=col, edgecolors="white", lw=0.4, zorder=3)
            ax.scatter(np.ones(len(f)), f, s=30, c=col, marker="s", edgecolors="white", lw=0.4, zorder=3)
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2000-2024", fut])
        ax.set_title(fut, fontweight="bold")
    axes[0].set_ylabel("Site-mean coefficient (d d$^{-1}$)", fontweight="bold")
    fig.suptitle("2000-2024 $\\rightarrow$ each SSP (per site)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "Part1_carryover_coef_site_slopes_allSSP")


def plot_part2(df):
    fig = plt.figure(figsize=(11, 4.8))
    gs = gridspec.GridSpec(1, 2, wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(PERIOD_ORDER))

    for ft, lstyle in [("natural", "-"), ("planted", "--")]:
        for key, c in zip(
            ["pct_dom_L1", "pct_dom_L2", "pct_dom_L3"],
            [C_L1, C_L2, C_L3],
        ):
            m = [df[(df.period == p) & (df.forest_type == ft)][key].mean() for p in PERIOD_ORDER]
            lab = key.replace("pct_dom_", "").upper() + f" ({ft[0].upper()})"
            ax1.plot(x, m, f"o{lstyle}", color=c, lw=1.5, ms=4, alpha=0.85, label=lab)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PERIOD_LABEL[p] for p in PERIOD_ORDER], rotation=15, ha="right")
    ax1.set_ylabel("Dominant pixels (%)", fontweight="bold")
    ax1.set_title("a  Dominant block ($\\Delta R^2$) on $\\Delta L_{total}$", fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    ax1.grid(axis="y", color="#eee", lw=0.5)

    for ft, col, ls in [("natural", C_NAT, "-"), ("planted", C_PLA, "--")]:
        spr = [df[(df.period == p) & (df.forest_type == ft)]["spring_frac"].mean() for p in PERIOD_ORDER]
        mul = [df[(df.period == p) & (df.forest_type == ft)]["multistage_frac"].mean() for p in PERIOD_ORDER]
        ax2.plot(x, spr, f"o{ls}", color=col, lw=1.8, ms=5, label=f"Spring (L1) — {ft}")
        ax2.plot(x, mul, f"s{ls}", color=col, lw=1.5, ms=4.5, alpha=0.8, label=f"L2+L3 — {ft}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([PERIOD_LABEL[p] for p in PERIOD_ORDER], rotation=15, ha="right")
    ax2.set_ylabel("Mean relative Share", fontweight="bold")
    ax2.set_title("b  Spring vs multi-stage Share", fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=7)
    ax2.grid(axis="y", color="#eee", lw=0.5)

    fig.suptitle(
        "PART 2  $\\Delta L_{total}$ regulation: spring $\\rightarrow$ multi-stage\n"
        f"20 paired windows; {FILTER_NOTE.split(';', 1)[1].strip()}",
        fontsize=11, fontweight="bold", y=1.10,
    )
    fig.tight_layout()
    save_fig(fig, "Part2_Ltotal_regulation_shift")

    fig, axes = plt.subplots(1, 4, figsize=(12, 4.2), sharey=True)
    for ax, period in zip(axes, PERIOD_ORDER):
        sub = df[df.period == period]
        nat = [sub[sub.forest_type == "natural"][f"pct_dom_{s}"].mean() for s in ["L1", "L2", "L3"]]
        pla = [sub[sub.forest_type == "planted"][f"pct_dom_{s}"].mean() for s in ["L1", "L2", "L3"]]
        xp = np.array([0, 1])
        for vals, xpos in [(nat, 0), (pla, 1)]:
            bottom = 0.0
            for v, c, lab in zip(vals, [C_L1, C_L2, C_L3], ["L1", "L2", "L3"]):
                ax.bar(xpos, v, 0.55, bottom=bottom, color=c, edgecolor="white", label=lab if xpos == 0 else "")
                bottom += v
        ax.set_xticks(xp)
        ax.set_xticklabels(["Natural", "Planted"])
        ax.set_title(period, fontweight="bold", fontsize=10)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Mean dominant pixels (%)", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Dominant $\\Delta R^2$ block — all periods", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "Part2_dominant_stack_all_periods")


def plot_part2_dr2(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(PERIOD_ORDER))
    for ft, lstyle in [("natural", "-"), ("planted", "--")]:
        for key, c, lab in [
            ("mean_dr2_L1", C_L1, "L1"),
            ("mean_dr2_L2", C_L2, "L2"),
            ("mean_dr2_L3", C_L3, "L3"),
        ]:
            m = [df[(df.period == p) & (df.forest_type == ft)][key].mean() for p in PERIOD_ORDER]
            ax.plot(x, m, f"o{lstyle}", color=c, lw=1.5, ms=4.5, label=f"{lab} ({ft[0].upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels([PERIOD_LABEL[p] for p in PERIOD_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Mean block $\\Delta R^2$ on $\\Delta L_{total}$", fontweight="bold")
    ax.set_title(f"Block $\\Delta R^2$ trajectory ({FILTER_NOTE.split(';', 1)[1].strip()})", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(axis="y", color="#eee", lw=0.5)
    fig.tight_layout()
    save_fig(fig, "Part2_DeltaR2_trajectory")


def plot_part1_direction_subjournal(df):
    """Improved Part 1 figure: forest-specific carry-over coefficient direction."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    x = np.arange(len(PERIOD_ORDER))

    for ax, ft, col, title in [
        (axes[0], "natural", C_NAT, "Natural forest"),
        (axes[1], "planted", C_PLA, "Planted forest"),
    ]:
        sub = df[df.forest_type == ft]
        wide_coef = sub.pivot(index="site_id", columns="period", values="coef_mean").reindex(columns=PERIOD_ORDER)
        wide_neg = sub.pivot(index="site_id", columns="period", values="coef_pct_neg").reindex(columns=PERIOD_ORDER)

        for _, row in wide_coef.iterrows():
            ax.plot(x, row.values, color="#BDBDBD", lw=0.7, alpha=0.55, zorder=1)

        mean_coef = wide_coef.mean(axis=0).values
        se_coef = wide_coef.sem(axis=0).values
        ax.errorbar(
            x, mean_coef, yerr=se_coef, fmt="o-", color=col, lw=2.2, ms=6.5,
            capsize=3, markeredgecolor="white", markeredgewidth=0.7, zorder=3,
            label=title,
        )

        # Small top strip: mean percentage of pixels with negative coefficient.
        ymax = 0.08
        for xi, period in zip(x, PERIOD_ORDER):
            neg = wide_neg[period].mean()
            ax.scatter(
                xi, ymax, s=18 + neg * 2.4, color=C_BUBBLE_PCT,
                edgecolor="white", linewidth=0.5, alpha=0.9, zorder=4,
            )
            ax.text(xi, ymax + 0.035, f"{neg:.0f}%", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(0, color="#4D4D4D", lw=0.9, ls="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(PERIOD_ORDER, rotation=25, ha="right")
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_ylim(-0.36, 0.16)
        style_axis(ax)

    axes[0].set_ylabel("Site-mean coefficient (d d$^{-1}$)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "Part1_carryover_direction_subjournal")


def ternary_xy(l1, l2, l3):
    """Barycentric coordinates for L1/L2/L3 shares."""
    total = l1 + l2 + l3
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    a, b, c = l1 / total, l2 / total, l3 / total
    x = b + 0.5 * c
    y = (np.sqrt(3) / 2.0) * c
    return x, y


def draw_ternary_frame(ax):
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#222222", lw=0.9)
    # Medians meet at the equal-share centroid (L1=L2=L3=1/3).
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    opposite_midpoints = np.array([
        [(1 + 0.5) / 2, (0 + np.sqrt(3) / 2) / 2],
        [(0 + 0.5) / 2, (0 + np.sqrt(3) / 2) / 2],
        [0.5, 0],
    ])
    for v, m in zip(vertices, opposite_midpoints):
        ax.plot([v[0], m[0]], [v[1], m[1]], color="#D9D9D9", lw=0.7, ls="--", zorder=0)
    cx, cy = ternary_xy(1, 1, 1)
    ax.scatter(cx, cy, s=24, facecolor="white", edgecolor="#555555", lw=0.8, zorder=2)
    ax.text(cx, cy + 0.035, "1/3", ha="center", va="bottom", fontsize=7, color="#555555")
    # grid lines at 25, 50, 75%.
    for t in [0.25, 0.50, 0.75]:
        ax.plot([t, 0.5 + 0.5 * t], [0, (np.sqrt(3) / 2) * (1 - t)], color="#E6E6E6", lw=0.6)
        ax.plot([1 - t, 0.5 * (1 - t)], [0, (np.sqrt(3) / 2) * t], color="#E6E6E6", lw=0.6)
        ax.plot([0.5 * t, 1 - 0.5 * t], [(np.sqrt(3) / 2) * t, (np.sqrt(3) / 2) * t], color="#E6E6E6", lw=0.6)
    ax.text(-0.04, -0.045, "L1\nspring", ha="right", va="top", fontsize=9, fontweight="bold", color=C_L1)
    ax.text(1.04, -0.045, "L2", ha="left", va="top", fontsize=9, fontweight="bold", color=C_L2)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, "L3", ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_L3)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.10, np.sqrt(3) / 2 + 0.12)
    ax.axis("off")


def plot_part2_share_ternary(df):
    """Improved Part 2 figure: ternary Share composition of L1/L2/L3 climate blocks."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))

    for ax, ft, title in [
        (axes[0], "natural", "Natural forest"),
        (axes[1], "planted", "Planted forest"),
    ]:
        draw_ternary_frame(ax)
        sub = df[df.forest_type == ft].copy()
        coords = {}

        for period in PERIOD_ORDER:
            psub = sub[sub.period == period]
            xs, ys = [], []
            for _, row in psub.iterrows():
                x, y = ternary_xy(row.mean_share_L1, row.mean_share_L2, row.mean_share_L3)
                xs.append(x)
                ys.append(y)
            ax.scatter(
                xs, ys, s=22, color=PAL_SCENARIO[period], alpha=0.38,
                edgecolors="white", linewidths=0.35, label=period if ft == "natural" else None,
            )

            cx, cy = ternary_xy(
                psub["mean_share_L1"].mean(),
                psub["mean_share_L2"].mean(),
                psub["mean_share_L3"].mean(),
            )
            coords[period] = (cx, cy)
            ax.scatter(
                cx, cy, s=105, color=PAL_SCENARIO[period], marker="D",
                edgecolors="black", linewidths=0.8, zorder=5,
            )

        ax.set_title(title, fontweight="bold", fontsize=10.5)

    fig.suptitle(
        "Relative climate-block Share on $\\Delta L_{total}$\n"
        "Each point = one paired window; diamonds = 20-window mean; dashed medians meet at equal contribution; $R^2_{Ltotal}$ > 0.30",
        fontsize=11.5, fontweight="bold", y=1.03,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PAL_SCENARIO[p],
                   markeredgecolor="white", markersize=7, label=p)
        for p in PERIOD_ORDER
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.035))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.20, wspace=0.20)
    save_fig(fig, "Part2_share_ternary_subjournal")


def plot_part2_share_dotmatrix(df):
    """Improved Part 2 figure: direct Share matrix for three climate stages."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9), sharey=True)
    row_defs = [
        ("mean_share_L1", "L1 spring", C_L1),
        ("mean_share_L2", "L2", C_L2),
        ("mean_share_L3", "L3", C_L3),
        ("multi_share", "L2+L3", C_MULTI),
    ]
    y_pos = np.arange(len(row_defs))[::-1]
    x_pos = np.arange(len(PERIOD_ORDER))

    plot_df = df.copy()
    plot_df["multi_share"] = plot_df["mean_share_L2"] + plot_df["mean_share_L3"]
    max_share = np.nanmax([plot_df[k].max() for k, _, _ in row_defs])

    for ax, ft, title in [
        (axes[0], "natural", "Natural forest"),
        (axes[1], "planted", "Planted forest"),
    ]:
        sub = plot_df[plot_df.forest_type == ft]
        for yi, (key, label, color) in zip(y_pos, row_defs):
            for xi, period in zip(x_pos, PERIOD_ORDER):
                vals = sub[sub.period == period][key].astype(float).values
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                m = float(np.mean(vals))
                e = sem(vals)
                size = 150 + 1050 * (m / max_share)

                # Site-level distribution inside the cell.
                jitter = np.linspace(-0.18, 0.18, len(vals))
                ax.scatter(
                    np.full(len(vals), xi) + jitter,
                    np.full(len(vals), yi - 0.27),
                    s=10 + 45 * (vals / max_share),
                    color=color, alpha=0.23, edgecolors="none", zorder=1,
                )
                ax.scatter(
                    xi, yi, s=size, color=color, alpha=0.86,
                    edgecolors="white", linewidths=1.0, zorder=3,
                )
                ax.errorbar(
                    xi, yi, xerr=0.0, yerr=0.0, fmt="none", zorder=2
                )
                ax.text(
                    xi, yi, f"{m:.2f}", ha="center", va="center",
                    fontsize=8.2, fontweight="bold",
                    color="white" if key != "mean_share_L2" else "#233044",
                    zorder=4,
                )
                ax.text(
                    xi, yi + 0.30, f"±{e:.02f}", ha="center", va="bottom",
                    fontsize=6.8, color="#555555",
                )

        for xi, period in zip(x_pos, PERIOD_ORDER):
            ax.axvline(xi, color="#F2F2F2", lw=0.7, zorder=0)
            ax.text(
                xi, len(row_defs) - 0.15, period, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=PAL_SCENARIO[period],
            )

        ax.set_xlim(-0.55, len(PERIOD_ORDER) - 0.45)
        ax.set_ylim(-0.75, len(row_defs) - 0.05)
        ax.set_xticks([])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([lab for _, lab, _ in row_defs], fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10.5, pad=12)
        ax.grid(axis="y", color="#EFEFEF", lw=0.6)
        style_axis(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(
        "Relative climate-block Share on $\\Delta L_{total}$\n"
        "Bubble area = 20-window mean Share; small dots = sites; text = mean ± SEM; $R^2_{Ltotal}$ > 0.30",
        fontsize=11.5, fontweight="bold", y=1.03,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                   markeredgecolor="white", markersize=9, label=lab)
        for _, lab, c in row_defs
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    save_fig(fig, "Part2_share_dotmatrix_subjournal")


def plot_part2_share_change_heatmap(df):
    """Part 2 figure: future-minus-historical Share changes, emphasizing differences."""
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.8), sharey=True)
    plot_df = df.copy()
    share_sum = plot_df["mean_share_L1"] + plot_df["mean_share_L2"] + plot_df["mean_share_L3"]
    plot_df["frac_L1"] = plot_df["mean_share_L1"] / share_sum
    plot_df["frac_L2"] = plot_df["mean_share_L2"] / share_sum
    plot_df["frac_L3"] = plot_df["mean_share_L3"] / share_sum
    plot_df["frac_multi"] = (plot_df["mean_share_L2"] + plot_df["mean_share_L3"]) / share_sum
    metrics = [
        ("frac_L1", "L1 spring", C_L1),
        ("frac_L2", "L2", C_L2),
        ("frac_L3", "L3", C_L3),
        ("frac_multi", "L2+L3", C_MULTI),
    ]
    futs = FUTURE_PERIODS

    # Build site-paired future minus historical deltas in percentage points.
    records = []
    for ft in ["natural", "planted"]:
        for site_id in sorted(plot_df.site_id.unique()):
            hist = plot_df[(plot_df.forest_type == ft) & (plot_df.site_id == site_id) & (plot_df.period == HIST)]
            if hist.empty:
                continue
            hist = hist.iloc[0]
            for fut in futs:
                row = plot_df[(plot_df.forest_type == ft) & (plot_df.site_id == site_id) & (plot_df.period == fut)]
                if row.empty:
                    continue
                row = row.iloc[0]
                for key, lab, _ in metrics:
                    records.append({
                        "forest_type": ft,
                        "site_id": site_id,
                        "period": fut,
                        "metric": key,
                        "label": lab,
                        "delta_pp": 100.0 * (row[key] - hist[key]),
                    })
    delta = pd.DataFrame(records)
    vmax = max(6.0, np.nanpercentile(np.abs(delta["delta_pp"]), 95))

    for ax, ft, title in [
        (axes[0], "natural", "Natural forest"),
        (axes[1], "planted", "Planted forest"),
    ]:
        sub = delta[delta.forest_type == ft]
        mat = np.full((len(metrics), len(futs)), np.nan)
        sem_mat = np.full_like(mat, np.nan)
        for i, (_, lab, _) in enumerate(metrics):
            for j, fut in enumerate(futs):
                vals = sub[(sub.label == lab) & (sub.period == fut)]["delta_pp"].values
                mat[i, j] = np.nanmean(vals)
                sem_mat[i, j] = sem(vals)

        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                color = "white" if abs(val) > 0.55 * vmax else "black"
                ax.text(
                    j, i, f"{val:+.1f}\n±{sem_mat[i, j]:.1f}",
                    ha="center", va="center", fontsize=8.3,
                    fontweight="bold", color=color,
                )

        ax.set_xticks(np.arange(len(futs)))
        ax.set_xticklabels(futs, fontweight="bold")
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_yticklabels([lab for _, lab, _ in metrics], fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10.5, pad=10)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(futs), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(metrics), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

        # Colored row labels make stage identity readable.
        for tick, (_, _, col) in zip(ax.get_yticklabels(), metrics):
            tick.set_color(col)
            tick.set_fontweight("bold")

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.065, pad=0.18, aspect=42)
    cbar.set_label("Change in relative contribution vs 2000-2024 (percentage points)", fontweight="bold")
    fig.suptitle(
        "Future shift in relative climate-block contribution on $\\Delta L_{total}$\n"
        "Share composition is normalized within L1+L2+L3; cells show site-paired SSP minus 2000-2024 mean ± SEM; $R^2_{Ltotal}$ > 0.30",
        fontsize=11.5, fontweight="bold", y=1.03,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.24, wspace=0.06)
    save_fig(fig, "Part2_share_change_heatmap_subjournal")


def plot_part2_dominant_stacked_subjournal(df):
    """Part 2: 100% stacked bars of dominant-pixel share (strict argmax DeltaR2)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), sharey=True)
    stage_cols = [C_L1, C_L2, C_L3]
    stage_labs = ["L1", "L2", "L3"]
    y = np.arange(len(PERIOD_ORDER))

    for ax, ft, title in [
        (axes[0], "natural", "Natural forest"),
        (axes[1], "planted", "Planted forest"),
    ]:
        sub = df[df.forest_type == ft]
        vals = []
        for period in PERIOD_ORDER:
            psub = sub[sub.period == period]
            means = [psub[f"pct_dom_{s}"].mean() / 100.0 for s in stage_labs]
            vals.append(means)
        vals = np.asarray(vals)

        left = np.zeros(len(PERIOD_ORDER))
        for i, (col, lab) in enumerate(zip(stage_cols, stage_labs)):
            ax.barh(
                y, vals[:, i], left=left, height=0.58,
                color=col, edgecolor="white", linewidth=1.1,
                label=lab if ft == "natural" else None,
            )
            left += vals[:, i]

        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, len(PERIOD_ORDER) - 0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(PERIOD_ORDER, fontsize=9)
        ax.invert_yaxis()
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.set_xlabel("Relative contribution (%)", fontweight="bold")
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax.grid(axis="x", color="#EDEDED", lw=0.55)
        style_axis(ax)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)
        for tick, period in zip(ax.get_yticklabels(), PERIOD_ORDER):
            tick.set_color(PAL_SCENARIO[period])
            tick.set_fontweight("bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white", label=l)
        for c, l in zip(stage_cols, stage_labs)
    ]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.00))
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    save_fig(fig, "Part2_dominant_stacked_subjournal")


def plot_supplementary_fig19(df):
    """Combined Supp Fig 19: Part 1 coef trajectories + Part 2 dominant stacked bars."""
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 0.95], hspace=0.42, wspace=0.28)

    x = np.arange(len(PERIOD_ORDER))
    for col, (ft, col_ft, title) in enumerate([
        ("natural", C_NAT, "Natural forest"),
        ("planted", C_PLA, "Planted forest"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        sub = df[df.forest_type == ft]
        wide_coef = sub.pivot(index="site_id", columns="period", values="coef_mean").reindex(columns=PERIOD_ORDER)
        wide_neg = sub.pivot(index="site_id", columns="period", values="coef_pct_neg").reindex(columns=PERIOD_ORDER)
        for _, row in wide_coef.iterrows():
            ax.plot(x, row.values, color="#BDBDBD", lw=0.7, alpha=0.55, zorder=1)
        mean_coef = wide_coef.mean(axis=0).values
        se_coef = wide_coef.sem(axis=0).values
        ax.errorbar(
            x, mean_coef, yerr=se_coef, fmt="o-", color=col_ft, lw=2.0, ms=5.5,
            capsize=3, markeredgecolor="white", markeredgewidth=0.6, zorder=3,
        )
        ymax = 0.08
        for xi, period in zip(x, PERIOD_ORDER):
            neg = wide_neg[period].mean()
            ax.scatter(
                xi, ymax, s=18 + neg * 2.4, color=C_BUBBLE_PCT,
                edgecolor="white", linewidth=0.5, alpha=0.9, zorder=4,
            )
            ax.text(xi, ymax + 0.032, f"{neg:.0f}%", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="#4D4D4D", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(PERIOD_ORDER, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylim(-0.36, 0.16)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        style_axis(ax)
        if col == 0:
            ax.set_ylabel("Site-mean coefficient (d d$^{-1}$)", fontweight="bold", fontsize=9)

    stage_cols = [C_L1, C_L2, C_L3]
    stage_labs = ["L1", "L2", "L3"]
    y = np.arange(len(PERIOD_ORDER))
    for col, (ft, title) in enumerate([("natural", "Natural forest"), ("planted", "Planted forest")]):
        ax = fig.add_subplot(gs[1, col])
        sub = df[df.forest_type == ft]
        vals = []
        for period in PERIOD_ORDER:
            psub = sub[sub.period == period]
            vals.append([psub[f"pct_dom_{s}"].mean() / 100.0 for s in stage_labs])
        vals = np.asarray(vals)
        left = np.zeros(len(PERIOD_ORDER))
        for i, (c, lab) in enumerate(zip(stage_cols, stage_labs)):
            ax.barh(y, vals[:, i], left=left, height=0.55, color=c, edgecolor="white", linewidth=1.0,
                    label=lab if col == 0 else None)
            left += vals[:, i]
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.55, len(PERIOD_ORDER) - 0.45)
        ax.set_yticks(y)
        ax.set_yticklabels(PERIOD_ORDER, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.set_xlabel("Relative contribution (%)", fontweight="bold", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.grid(axis="x", color="#EDEDED", lw=0.5)
        style_axis(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.text(0.02, 0.97, "a", fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.48, "b", fontsize=14, fontweight="bold", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white", label=l)
               for c, l in zip(stage_cols, stage_labs)]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.08, hspace=0.40, wspace=0.28)
    save_fig(fig, "SupplementaryFig19_paired_validation")


# ---------------------------------------------------------------------------
# Stats & discussion snippets
# ---------------------------------------------------------------------------
def write_stats(df):
    lines = [
        "Paired-site v4 — difference-regression validation (20 windows)",
        "",
        "Filter (from main study scripts):",
        f"  Part 1: R2_L1 > {R2_MIN_L1}  [Coef_L3prev_to_L1]",
        f"  Part 2: R2_Ltotal > {R2_MIN_LTOTAL}; strict argmax(DeltaR2) dominant-pixel share",
        "  Source: Figures_Share_Ltotal/plot_dominant_deltar2_trajectory.py, plot_dominant_transition_chord.R",
        "",
        "======== PART 1: Coef_L3prev_to_L1 (lag dL3 -> dL1) ========",
        "Claim: 2000-2024 mixed +/- ; all SSPs predominantly negative",
        "",
    ]
    for ft in ["natural", "planted"]:
        lines.append(f"--- {ft} ---")
        for p in PERIOD_ORDER:
            sub = df[(df.period == p) & (df.forest_type == ft)]
            lines.append(
                f"  {p}: mean={sub['coef_mean'].mean():+.3f}, "
                f"pct_neg={sub['coef_pct_neg'].mean():.1f}%, "
                f"pct_pos={sub['coef_pct_pos'].mean():.1f}%, "
                f"n_pixels_mean={sub['coef_n'].mean():.0f}"
            )
        h = df[(df.period == HIST) & (df.forest_type == ft)].sort_values("site_id")["coef_mean"]
        for fut in FUTURE_PERIODS:
            f = df[(df.period == fut) & (df.forest_type == ft)].sort_values("site_id")["coef_mean"]
            pn = df[(df.period == HIST) & (df.forest_type == ft)].sort_values("site_id")["coef_pct_neg"]
            fn = df[(df.period == fut) & (df.forest_type == ft)].sort_values("site_id")["coef_pct_neg"]
            lines.append(f"  vs {fut}: Wilcoxon mean coef p={wilcoxon_sites(h, f):.4f}; pct_neg p={wilcoxon_sites(pn, fn):.4f}; neg sites {(f < 0).sum()}/20")
        lines.append("")

    lines.extend([
        "======== PART 2: dL_total dominant-block share (strict argmax DeltaR2) ========",
        "Claim: 2000-2024 spring-dominated; L2+L3 gain under SSPs",
        "",
    ])
    for ft in ["natural", "planted"]:
        lines.append(f"--- {ft} ---")
        for p in PERIOD_ORDER:
            sub = df[(df.period == p) & (df.forest_type == ft)]
            lines.append(
                f"  {p}: dom_L1={sub['pct_dom_L1'].mean():.1f}%, "
                f"dom_L2={sub['pct_dom_L2'].mean():.1f}%, dom_L3={sub['pct_dom_L3'].mean():.1f}%; "
                f"multistage_dom={sub['pct_multistage_dom'].mean():.1f}%; "
                f"spring_share={sub['spring_frac'].mean():.3f}, multistage_share={sub['multistage_frac'].mean():.3f}"
            )
        for fut in FUTURE_PERIODS:
            h_dom = df[(df.period == HIST) & (df.forest_type == ft)].sort_values("site_id")["pct_dom_L1"]
            f_dom = df[(df.period == fut) & (df.forest_type == ft)].sort_values("site_id")["pct_dom_L1"]
            h_msd = df[(df.period == HIST) & (df.forest_type == ft)].sort_values("site_id")["pct_multistage_dom"]
            f_msd = df[(df.period == fut) & (df.forest_type == ft)].sort_values("site_id")["pct_multistage_dom"]
            h_mul = df[(df.period == HIST) & (df.forest_type == ft)].sort_values("site_id")["multistage_frac"]
            f_mul = df[(df.period == fut) & (df.forest_type == ft)].sort_values("site_id")["multistage_frac"]
            lines.append(
                f"  vs {fut}: Wilcoxon dom_L1 p={wilcoxon_sites(h_dom, f_dom):.4f}; "
                f"multistage_dom p={wilcoxon_sites(h_msd, f_msd):.4f}; "
                f"multistage_share p={wilcoxon_sites(h_mul, f_mul):.4f}"
            )
        lines.append("")

    (OUT_DIR / "paired_diffreg_v4_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def write_discussion_snippets(df):
    def pool_mean(period, col):
        return df[df.period == period][col].mean()

    h_coef = pool_mean(HIST, "coef_mean")
    h_neg = pool_mean(HIST, "coef_pct_neg")
    h_l1 = pool_mean(HIST, "pct_dom_L1")
    h_msd = pool_mean(HIST, "pct_multistage_dom")
    h_ms = pool_mean(HIST, "multistage_frac")

    fut_lines = []
    for fut in FUTURE_PERIODS:
        f_coef = pool_mean(fut, "coef_mean")
        f_neg = pool_mean(fut, "coef_pct_neg")
        f_msd = pool_mean(fut, "pct_multistage_dom")
        f_ms = pool_mean(fut, "multistage_frac")
        neg_sites = (
            (df[df.period == fut].groupby("site_id")["coef_mean"].mean() < 0).sum()
        )
        fut_lines.append(
            f"  {fut}: coef mean {f_coef:+.3f}, {f_neg:.0f}% negative pixels, "
            f"{neg_sites}/20 sites negative; multistage_dom {f_msd:.0f}%, multistage_share {f_ms:.2f}"
        )

    text = f"""Discussion support (paired 20 windows, v4 filters: R2_L1/Ltotal > 0.30):

PART 1 — autumn-to-spring carry-over becomes negative:
2000-2024: lag(dL3) on dL1 mixed in sign (mean {h_coef:+.3f}; {h_neg:.0f}% negative pixels).
Under all SSPs:
{chr(10).join(fut_lines)}

PART 2 — dL_total dominant climate block (strict argmax DeltaR2, R2_Ltotal > 0.30):
2000-2024: L1 dominant {h_l1:.0f}%; multistage_dom {h_msd:.0f}%.
Future SSPs show higher L2+L3 dominant share (see Part2_dominant_stacked_subjournal).
"""
    (OUT_DIR / "discussion_snippets_v4.txt").write_text(text, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Paired-site v4 extraction ===")
    print(f"Filters: R2_L1 > {R2_MIN_L1}, R2_Ltotal > {R2_MIN_LTOTAL}, strict argmax DeltaR2")
    df = extract_all()
    print(f"Rows: {len(df)}  -> paired_sites_diffreg_metrics.csv")

    print("\n=== Figures ===")
    plot_part1_direction_subjournal(df)
    plot_part2_dominant_stacked_subjournal(df)
    plot_supplementary_fig19(df)

    write_stats(df)
    write_discussion_snippets(df)
    print("\nDone. Output:", OUT_DIR)


if __name__ == "__main__":
    main()
