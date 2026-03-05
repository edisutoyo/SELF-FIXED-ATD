#!/usr/bin/env python3
# coding: utf-8
"""
Generate 3 boxplot PDFs untuk RQ3 dengan transformasi log10(1+x) pada data,
dan sumbu Y tetap linear:

1) rq3_box_lines_changed.pdf   -> log10(1 + Lines_Changed_Total) (self vs non-self)
2) rq3_box_commits_total.pdf   -> log10(1 + Num_Commits_Total)  (self vs non-self)
3) rq3_box_devs_total.pdf      -> log10(1 + Num_Devs_Total)     (self vs non-self)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ====== BIGGER FONTS (GLOBAL) ======
plt.rcParams.update({
    "font.size": 14,          # base
    "axes.labelsize": 12,     # y-label
    "axes.titlesize": 12,     # title (if you use it)
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ===================== CONFIG =====================

DATA_CSV = Path(
    "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/ISSUES_with_LevelItem_LevelProcess.csv"
)

# kolom untuk identitas self-fixed
COL_SELF_RAW = "Is Self-Fixed (Intro=Payment)"

# nama kolom metrik yang mau diplot
COL_LINES   = "Lines_Changed_Total"
COL_COMMITS = "Num_Commits_Total"
COL_DEVS    = "Num_Devs_Total"

# apakah akan melakukan transformasi log10(1 + x) pada data?
LOG_TRANSFORM = True

# style umum boxplot
SHOW_FLIERS = False   # tidak tampilkan outlier sebagai titik
SHOW_MEANS  = True    # tampilkan mean (marker 'D')

# ===================== UTIL: konversi self-fixed ke 0/1 =====================

def to_self_flag(series: pd.Series) -> pd.Series:
    """
    Konversi kolom self-fixed mentah ke flag 0/1.
    """
    s = series.copy()

    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        return s.astype(int)

    s = s.astype(str).str.strip().str.upper()

    mapping = {
        "1": 1, "0": 0,
        "TRUE": 1, "FALSE": 0,
        "YES": 1, "NO": 0,
        "Y": 1, "N": 0,
        "SELF-FIXED": 1,
        "NON-SELF-FIXED": 0,
    }
    return s.map(mapping).astype(int)

# ===================== LOAD DATA =====================

print(f"[INFO] Load data: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

if "Self_Flag" not in df.columns:
    if COL_SELF_RAW not in df.columns:
        raise ValueError(
            f"Kolom Self_Flag dan {COL_SELF_RAW} tidak ditemukan di CSV."
        )
    df["Self_Flag"] = to_self_flag(df[COL_SELF_RAW])

df = df[df["Self_Flag"].isin([0, 1])].copy()

print(
    f"[INFO] N total: {len(df)}, "
    f"self-fixed={len(df[df['Self_Flag']==1])}, "
    f"non-self-fixed={len(df[df['Self_Flag']==0])}"
)

# ===================== FUNGSI HELPER UNTUK PLOT =====================

def make_two_group_boxplot(
    data: pd.DataFrame,
    metric_col: str,
    base_ylabel: str,
    output_pdf: Path,
):
    """
    Buat boxplot 2 grup (Self-fixed vs Non-self-fixed) untuk satu metrik,
    dengan transformasi log10(1 + x) dan axis linear + custom style.
    """
    if metric_col not in data.columns:
        raise ValueError(f"Kolom {metric_col} tidak ada di DataFrame.")

    vals_self_raw = data.loc[data["Self_Flag"] == 1, metric_col].dropna().values
    vals_non_raw  = data.loc[data["Self_Flag"] == 0, metric_col].dropna().values

    print(f"[INFO] Plot {metric_col}: n_self={len(vals_self_raw)}, n_non={len(vals_non_raw)}")

    # Transformasi log10(1 + x) bila diaktifkan
    if LOG_TRANSFORM:
        vals_self = np.log10(1.0 + vals_self_raw)
        vals_non  = np.log10(1.0 + vals_non_raw)
        ylabel = f"log10(1 + {base_ylabel})"
    else:
        vals_self = vals_self_raw
        vals_non  = vals_non_raw
        ylabel = base_ylabel

    fig, ax = plt.subplots(figsize=(4.0, 4.5))

    # NOTE: patch_artist=True supaya facecolor dipakai
    bp = ax.boxplot(
        [vals_self, vals_non],
        labels=["Self-fixed", "Non-self-fixed"],
        showfliers=SHOW_FLIERS,
        showmeans=SHOW_MEANS,
        meanline=False,
        patch_artist=True,
    )

    # ====== STYLE ======
    # Warna box (2 grup → pakai 2 warna pertama)
    colors = ["#01BFC4", "#F9766D", "#999999"]  # teal, salmon, grey
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)

    # median & mean -> lebih menonjol
    for med in bp["medians"]:
        med.set(color="black", linewidth=2.4)
    for mean in bp["means"]:
        mean.set(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
        )

    ax.set_ylabel(ylabel)
    ax.set_title("")  # bisa diisi jika perlu
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    # ====== END STYLE ======

    fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved boxplot to {output_pdf}")


# ===================== GENERATE 3 BOXPLOTS =====================

OUT_LINES   = DATA_CSV.with_name("rq3_box_lines_changed.pdf")
OUT_COMMITS = DATA_CSV.with_name("rq3_box_commits_total.pdf")
OUT_DEVS    = DATA_CSV.with_name("rq3_box_devs_total.pdf")

# 1) Lines_Changed_Total
make_two_group_boxplot(
    data=df,
    metric_col=COL_LINES,
    base_ylabel="total lines changed (intro–payment)",
    output_pdf=OUT_LINES,
)

# 2) Num_Commits_Total
make_two_group_boxplot(
    data=df,
    metric_col=COL_COMMITS,
    base_ylabel="total commits (intro–payment)",
    output_pdf=OUT_COMMITS,
)

# 3) Num_Devs_Total
make_two_group_boxplot(
    data=df,
    metric_col=COL_DEVS,
    base_ylabel="total developers involved",
    output_pdf=OUT_DEVS,
)

print("[DONE] All RQ3 boxplots generated.")
