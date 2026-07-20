"""
Dominant climate block per pixel (argmax DeltaR2 on Delta L_total).
Computes scenario-level dominant-pixel shares and draws the compact trajectory figure.

Filter: R2_Ltotal > 0.30; strict argmax(DeltaR2_L1, L2, L3), no tie exclusion.

Run:
  python plot_dominant_deltar2_trajectory.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import os

HERE = Path(__file__).resolve().parent
_DATA_ROOT = Path(os.environ.get("GOSIF_DATA_ROOT", r"F:/GO_SIF"))
# DiffReg L_total outputs (historical + SSP futures)
_HIST = Path(
    os.environ.get(
        "LTOTAL_HIST_DIR",
        str(_DATA_ROOT / "9_model" / "Results" / "Historical_LM_L1_Ltotal_only"),
    )
)
_CMIP6 = _DATA_ROOT / "10_CMIP6_Climate"
ROOT = _HIST.parent  # for optional local CSV/PNG next to script outputs

SCENARIOS = [
    ("2000-2024", _HIST),
    ("SSP126", Path(os.environ.get("LTOTAL_SSP126_DIR", str(_CMIP6 / "SSP1_2.6" / "9_Future_LM_L1_Ltotal_only")))),
    ("SSP245", Path(os.environ.get("LTOTAL_SSP245_DIR", str(_CMIP6 / "SSP2_4.5" / "9_Future_LM_L1_Ltotal_only")))),
    ("SSP585", Path(os.environ.get("LTOTAL_SSP585_DIR", str(_CMIP6 / "SSP5_8.5" / "9_Future_LM_L1_Ltotal_only")))),
]
FORESTS = [("NaturalForest", "Natural forest"), ("PlantedForest", "Planted forest")]
DELTAR2_KEYS = [
    "DeltaR2_CLI_L1_on_Ltotal",
    "DeltaR2_CLI_L2_on_Ltotal",
    "DeltaR2_CLI_L3_on_Ltotal",
]
STAGES = ["L1", "L2", "L3"]
_palette = pd.read_csv(HERE / "stage_colors.csv")
C_STAGE = dict(zip(_palette["stage"], _palette["hex"]))
STAGE_LABEL = {
    "L1": "L1 (green-up)",
    "L2": "L2 (plateau)",
    "L3": "L3 (senescence)",
}
STAGE_MARKER = {"L1": "o", "L2": "s", "L3": "^"}
R2_MIN = 0.30


def read_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr.reshape(-1)


def classify_dominant(forest_key: str, sc_dir: Path) -> dict:
    r2 = read_raster(sc_dir / forest_key / "R2_Ltotal.tif")
    d = np.stack([read_raster(sc_dir / forest_key / f"{k}.tif") for k in DELTAR2_KEYS], axis=1)
    ok = np.isfinite(r2) & (r2 > R2_MIN) & np.all(np.isfinite(d), axis=1) & np.all(d >= 0, axis=1)
    dom = np.argmax(d[ok], axis=1) + 1

    n_all = int(ok.sum())
    pcts = {}
    for k, st in enumerate(STAGES, start=1):
        pcts[st] = 100.0 * (dom == k).sum() / n_all if n_all else np.nan
    pcts["L2+L3"] = pcts["L2"] + pcts["L3"]

    return {
        "n_r2_ok": n_all,
        "pct_L1": pcts["L1"],
        "pct_L2": pcts["L2"],
        "pct_L3": pcts["L3"],
        "pct_L2_L3": pcts["L2+L3"],
    }


def build_stats_table() -> pd.DataFrame:
    rows = []
    for sc_label, sc_dir in SCENARIOS:
        for forest_key, forest_label in FORESTS:
            s = classify_dominant(forest_key, sc_dir)
            rows.append({"scenario": sc_label, "forest": forest_label, **s})
            print(
                f"{forest_label:16} {sc_label:10} n={s['n_r2_ok']:,} | "
                f"L1={s['pct_L1']:.1f}% L2={s['pct_L2']:.1f}% L3={s['pct_L3']:.1f}% "
                f"(sum={s['pct_L1'] + s['pct_L2'] + s['pct_L3']:.1f}%)"
            )
    return pd.DataFrame(rows)


def style_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scenario_labels() -> list[str]:
    return [s[0] for s in SCENARIOS]


def _forest_sub(stats: pd.DataFrame, flabel: str) -> pd.DataFrame:
    return stats[stats.forest == flabel].set_index("scenario").loc[_scenario_labels()]


def plot_trajectory(stats: pd.DataFrame) -> Path:
    scenarios = _scenario_labels()
    xs = np.arange(len(scenarios))

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.0), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (_, flabel) in zip(axes, FORESTS):
        ax.set_facecolor("white")
        sub = _forest_sub(stats, flabel)

        for st in STAGES:
            ys = sub[f"pct_{st}"].values
            color = C_STAGE[st]
            ax.plot(xs, ys, color=color, linewidth=1.6, zorder=3, solid_capstyle="round")
            ax.scatter(
                xs, ys, s=52, facecolors="white", edgecolors=color,
                linewidths=1.4, marker=STAGE_MARKER[st], zorder=5,
            )

        l23 = sub["pct_L2_L3"].values
        ax.plot(
            xs, l23, color="#5BA8A0", linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=2, label="L2+L3" if flabel == "Natural forest" else None,
        )
        ax.scatter(
            xs, l23, s=24, facecolors="white", edgecolors="#5BA8A0",
            linewidths=0.9, marker="D", zorder=4,
        )

        ax.set_xlim(-0.25, len(scenarios) - 0.75)
        ax.set_ylim(12, 68)
        ax.set_yticks(np.arange(15, 66, 10))
        ax.set_title(flabel, fontweight="bold", fontsize=9.5, pad=4)
        style_spines(ax)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(scenarios, fontsize=8.2)
    axes[-1].set_xlabel("Scenario", fontweight="bold", fontsize=8.5)
    axes[0].set_ylabel("Dominant-pixel share (%)", fontweight="bold", fontsize=8.5)

    stage_handles = [
        plt.Line2D([0], [0], color=C_STAGE[s], marker=STAGE_MARKER[s], lw=1.6,
                   markerfacecolor="white", markeredgecolor=C_STAGE[s],
                   markersize=6.2, markeredgewidth=1.2, label=STAGE_LABEL[s])
        for s in STAGES
    ]
    l23_handle = plt.Line2D(
        [0], [0], color="#5BA8A0", linestyle=(0, (4, 3)), lw=1.2,
        marker="D", markerfacecolor="white", markeredgecolor="#5BA8A0",
        markersize=4.8, label="L2+L3 (combined)",
    )
    fig.legend(
        handles=stage_handles + [l23_handle], ncol=4, frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=7.8,
        handlelength=1.8, columnspacing=0.9,
    )
    fig.subplots_adjust(hspace=0.28, top=0.88, bottom=0.12, left=0.14, right=0.98)

    fp = HERE / "Fig_Ltotal_Dominant_DeltaR2_trajectory_R2gt030.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fp}")
    return fp


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    stats = build_stats_table()
    stats.to_csv(HERE / "Dominant_DeltaR2_pixel_pct_R2gt030.csv", index=False)
    plot_trajectory(stats)


if __name__ == "__main__":
    main()
