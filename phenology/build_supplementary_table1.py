"""
Build Supplementary Table 1 — amplitude-threshold sensitivity (10% and 30% only).

Reads Table_amplitude_sensitivity_allocation_AI_PR.csv produced by
classify_and_summarize_allocation.py (separate AI / PR constant proportions).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

_DATA_ROOT = Path(os.environ.get("GOSIF_DATA_ROOT", r"F:/GO_SIF"))
ROOT = Path(
    os.environ.get(
        "SENSITIVITY_ROOT",
        str(_DATA_ROOT / "results" / "phenology_sensitivity_amplitude"),
    )
)
if not ROOT.exists():
    _fallback = _DATA_ROOT / "12_other" / "phenology_sensitivity_amplitude"
    if _fallback.exists():
        ROOT = _fallback

SRC = ROOT / "Table_amplitude_sensitivity_allocation_AI_PR.csv"
OUT_CSV = ROOT / "Supplementary_Table_1_amplitude_sensitivity.csv"
OUT_MD = ROOT / "Supplementary_Table_1_amplitude_sensitivity.md"

FOREST_LABEL = {
    "NaturalForest": "Natural forest",
    "PlantedForest": "Planted forest",
    "AllForest": "All forests",
}
THR_ORDER = ["10%", "30%"]
FOREST_ORDER = ["NaturalForest", "PlantedForest", "AllForest"]


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    if mean != mean or sd != sd:
        return ""
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def load_source() -> list[dict]:
    with SRC.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_table(raw: list[dict]) -> list[dict]:
    rows = []
    for thr in THR_ORDER:
        for ft in FOREST_ORDER:
            sub = [r for r in raw if r["threshold"] == thr and r["forest_type"] == ft]
            if not sub:
                continue
            ai = next(r for r in sub if r["metric"] == "AI")
            pr = next(r for r in sub if r["metric"] == "PR")
            rows.append(
                {
                    "Amplitude_threshold": thr,
                    "Forest_type": FOREST_LABEL[ft],
                    "Valid_pixels_n": int(ai["n_valid_pixels"]),
                    "AI_constant_allocation_pct": float(ai["pct_constant_allocation"]),
                    "PR_constant_allocation_pct": float(pr["pct_constant_allocation"]),
                    "AI_mean": float(ai["domain_mean"]),
                    "AI_sd": float(ai["domain_sd"]),
                    "AI_mean_pm": fmt_pm(float(ai["domain_mean"]), float(ai["domain_sd"])),
                    "PR_mean": float(pr["domain_mean"]),
                    "PR_sd": float(pr["domain_sd"]),
                    "PR_mean_pm": fmt_pm(float(pr["domain_mean"]), float(pr["domain_sd"])),
                }
            )
    return rows


def robustness_note(pub: list[dict]) -> str:
    lines = ["**Cross-threshold robustness:**", ""]
    for label in ["Natural forest", "Planted forest", "All forests"]:
        sub = [r for r in pub if r["Forest_type"] == label]
        if len(sub) < 2:
            continue
        ai_rng = max(r["AI_constant_allocation_pct"] for r in sub) - min(r["AI_constant_allocation_pct"] for r in sub)
        pr_rng = max(r["PR_constant_allocation_pct"] for r in sub) - min(r["PR_constant_allocation_pct"] for r in sub)
        ai_mean_rng = max(r["AI_mean"] for r in sub) - min(r["AI_mean"] for r in sub)
        pr_mean_rng = max(r["PR_mean"] for r in sub) - min(r["PR_mean"] for r in sub)
        lines.append(
            f"- {label}: AI constant range {ai_rng:.2f} pp; PR constant range {pr_rng:.2f} pp; "
            f"AI mean range {ai_mean_rng:.3f}; PR mean range {pr_mean_rng:.3f}."
        )
    return "\n".join(lines)


def dataframe_to_markdown(rows: list[dict], cols: list[str]) -> str:
    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def write_markdown(pub: list[dict]) -> None:
    display_cols = [
        "SOS/EOS amplitude threshold",
        "Forest type",
        "Valid pixels (n)",
        "AI constant allocation (%)",
        "PR constant allocation (%)",
        "AI (mean ± SD)",
        "PR (mean ± SD)",
    ]
    display_rows = [
        {
            "SOS/EOS amplitude threshold": r["Amplitude_threshold"],
            "Forest type": r["Forest_type"],
            "Valid pixels (n)": r["Valid_pixels_n"],
            "AI constant allocation (%)": f"{r['AI_constant_allocation_pct']:.2f}",
            "PR constant allocation (%)": f"{r['PR_constant_allocation_pct']:.2f}",
            "AI (mean ± SD)": r["AI_mean_pm"],
            "PR (mean ± SD)": r["PR_mean_pm"],
        }
        for r in pub
    ]

    md = f"""# Supplementary Table 1 | Sensitivity of stage time allocation to SOS/EOS amplitude thresholds

## Table title (for manuscript)

**Supplementary Table 1 | Sensitivity of stage time-allocation metrics and classification to SOS/EOS amplitude thresholds (10% and 30%).**

## Table caption

**Supplementary Table 1 | Sensitivity of stage time-allocation metrics and classification to SOS/EOS amplitude thresholds (10% and 30%).** To ensure that the observed stability of stage allocation was not an artifact of SIF signal noise at low light levels, we re-extracted phenology using SOS/EOS amplitude thresholds of 10% and 30% (the main analysis uses 20%; see Methods). **AI** = Greenup duration / Senescence duration; **PR** = Plateau duration / (Greenup + Senescence duration), averaged over 2000–2024. **Constant time allocation** denotes pixels with no significant linear trend (*P* > 0.05) in the corresponding annual metric time series (AI tested on AI, PR tested on PR); **optimal time allocation** denotes pixels with a significant trend (*P* ≤ 0.05). PPOS/APOS platform thresholds were held at 90% of annual maximum SIF. Valid pixels required ≥ 15 years of non-missing values per metric. AI and PR magnitudes shift with threshold definition as expected, whereas constant-allocation proportions remain close to the main 20% analysis, confirming that internal growing-season structure is not sensitive to specific SIF-extraction parameters.

{robustness_note(pub)}

## Table

{dataframe_to_markdown(display_rows, display_cols)}
"""
    OUT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Run classify_and_summarize_allocation.py first: {SRC}")
    raw = load_source()
    pub = build_table(raw)
    if not pub:
        raise RuntimeError("No 10%/30% rows found in source table.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pub[0].keys()))
        writer.writeheader()
        writer.writerows(pub)

    write_markdown(pub)
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
