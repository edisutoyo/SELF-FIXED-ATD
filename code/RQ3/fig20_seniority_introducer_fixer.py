#!/usr/bin/env python3
# coding: utf-8
"""
Analisis senioritas developer untuk self-fixed vs non–self-fixed ATD.

Pertanyaan:
- Apakah self-fixed ATD cenderung diperkenalkan/diperbaiki oleh developer yang lebih senior?

Input:
- ISSUES_CSV    : 2-UPGRADED-ATD-FINAL-DATASET-1091-TRACED.csv
- DEV_STATS_CSV : dev_stats.csv (output script DEV-SENIORITY)

Output:
- Statistik deskriptif senioritas (tahun) introducer & fixer
  untuk self-fixed vs non-self-fixed.
- Mann–Whitney U test dan Cliff's delta sebagai effect size.
- File merged: ATD_with_seniority.csv
- Figure: boxplot side-by-side (intro vs payment seniority), disimpan sebagai PDF.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path
import matplotlib.pyplot as plt   # <=== tambahan untuk figure

# ===================== CONFIG: SESUAIKAN DI SINI =====================


# ====== BIGGER FONTS (GLOBAL) ======
plt.rcParams.update({
    "font.size": 12,          # base
    "axes.labelsize": 12,     # y-label
    "axes.titlesize": 12,     # title (if you use it)
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

ISSUES_CSV    = Path(
    "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/"
    "ATD-FINAL-DATASET-TRACED.csv"
)
DEV_STATS_CSV = Path(
    "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/"
    "DEV-STATS/dev_stats.csv"
)

# Lokasi output figure (boleh diubah)
FIG_OUT = Path(
    "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/"
    "DEV-STATS/seniority_boxplots_self_vs_nonself.pdf"
)

# Nama kolom di ISSUES_CSV
COL_KEY              = "Key"
COL_PROJECT          = "Project"   # akan kita isi sendiri bila belum ada
COL_SELF_FLAG        = "Is Self-Fixed (Intro=Payment)"
COL_INTRO_DATE       = "Intro Date"
COL_PAYMENT_DATE     = "Payment Date"
COL_INTRO_AUTHOR_NAME   = "Intro Author Name"
COL_PAYMENT_AUTHOR_NAME = "Payment Author Name"

# Nama kolom di dev_stats.csv
COL_DEV_PROJECT          = "Project"
COL_DEV_AUTHOR_KEY       = "Author_Key"
COL_DEV_FIRST_COMMIT     = "First_Commit_Date"
COL_DEV_LAST_COMMIT      = "Last_Commit_Date"
COL_DEV_TOTAL_COMMITS    = "Total_Commits"
COL_DEV_TOTAL_FILES      = "Total_Files_Touched"

# ========== MAPPING PREFIX JIRA -> PROJECT (disesuaikan proyekmu) ==========

PROJECT_JIRA_PREFIXES = {
    "AMQ":      ["AMQ", "ACTIVEMQ"],
    "CAMEL":    ["CAMEL"],
    "SPARK":    ["SPARK"],
    "KAFKA":    ["KAFKA"],
    "CASSANDRA":["CASSANDRA"],
    "DRILL":    ["DRILL"],
    "SOLR":     ["SOLR"],
    "LUCENE":   ["LUCENE"],
    "GEODE":    ["GEODE"],
    "NETBEANS": ["NETBEANS"],
}

# Build reverse map: prefix -> canonical project (AMQ, CAMEL, ...)
JIRA_PREFIX_TO_PROJECT = {}
for proj, prefixes in PROJECT_JIRA_PREFIXES.items():
    for p in prefixes:
        JIRA_PREFIX_TO_PROJECT[p.upper()] = proj


# ===================== HELPER FUNCTIONS =====================

def make_author_key(name: str) -> str:
    """Sama dengan fungsi di script DEV-SENIORITY: lower-case + strip."""
    return (name or "").strip().lower()


def infer_project_from_key(issue_key: str):
    """Ambil prefix sebelum '-' dari Key lalu map ke Project."""
    if not isinstance(issue_key, str):
        return None
    prefix = issue_key.split("-")[0].upper()
    # kalau tidak ketemu di mapping, fallback ke prefix itu sendiri
    return JIRA_PREFIX_TO_PROJECT.get(prefix, prefix)


def cliffs_delta(x: pd.Series, y: pd.Series):
    """
    Hitung Cliff's delta dan kategorinya.
    x = nilai self-fixed, y = nilai non-self.
    """
    x = x.dropna().to_numpy()
    y = y.dropna().to_numpy()
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return np.nan, "undefined"

    more = 0
    less = 0
    for xv in x:
        more += np.sum(xv > y)
        less += np.sum(xv < y)

    d = (more - less) / (m * n)
    ad = abs(d)
    if ad < 0.147:
        size = "negligible"
    elif ad < 0.33:
        size = "small"
    elif ad < 0.474:
        size = "medium"
    else:
        size = "large"
    return d, size


def style_boxplot(bp):
    """
    Terapkan style:
    - warna box (teal, salmon, grey)
    - median garis tebal hitam
    - mean marker diamond putih dengan edge hitam
    """
    colors = ["#01BFC4", "#F9766D", "#999999"]  # teal, salmon, grey
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)      # isi box
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")  # garis tepi box
        patch.set_linewidth(1.0)

    # median & mean -> lebih menonjol
    for med in bp["medians"]:
        med.set(color="black", linewidth=2.4)
    if "means" in bp:  # jaga-jaga kalau showmeans=False
        for mean in bp["means"]:
            mean.set(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=6,
            )



# ===================== MAIN =====================

def main():
    # ---------- Load data ----------
    print(f"[*] Loading issues from: {ISSUES_CSV}")
    df_issues = pd.read_csv(ISSUES_CSV)

    print(f"[*] Loading dev stats from: {DEV_STATS_CSV}")
    df_dev = pd.read_csv(DEV_STATS_CSV)

    # ---------- Pastikan kolom Project di issues ----------
    if COL_PROJECT not in df_issues.columns:
        print(f"[!] Kolom '{COL_PROJECT}' tidak ada di issues; infer dari '{COL_KEY}' via prefix.")
        if COL_KEY not in df_issues.columns:
            raise KeyError(f"Tidak menemukan kolom '{COL_KEY}' di ISSUES_CSV.")
        df_issues[COL_PROJECT] = df_issues[COL_KEY].map(infer_project_from_key)

    # ---------- Parse tanggal ----------
    # Issues: Intro Date & Payment Date (tz-aware string -> naive datetime)
    df_issues[COL_INTRO_DATE] = pd.to_datetime(
        df_issues[COL_INTRO_DATE], utc=True, errors="coerce"
    ).dt.tz_localize(None)

    df_issues[COL_PAYMENT_DATE] = pd.to_datetime(
        df_issues[COL_PAYMENT_DATE], utc=True, errors="coerce"
    ).dt.tz_localize(None)

    # dev_stats: First_Commit_Date & Last_Commit_Date
    for col in [COL_DEV_FIRST_COMMIT, COL_DEV_LAST_COMMIT]:
        df_dev[col] = pd.to_datetime(
            df_dev[col], utc=True, errors="coerce"
        ).dt.tz_localize(None)

    # ---------- Normalisasi Self_Flag ----------
    df_issues["Is_Self_Fixed_Flag"] = (
        df_issues[COL_SELF_FLAG].fillna(False).astype(bool)
    )

    # ---------- Author_Key intro & payment ----------
    df_issues["Intro_Author_Key"]   = df_issues[COL_INTRO_AUTHOR_NAME].astype(str).map(make_author_key)
    df_issues["Payment_Author_Key"] = df_issues[COL_PAYMENT_AUTHOR_NAME].astype(str).map(make_author_key)

    # Pastikan Project berupa string di kedua dataframe
    df_dev[COL_DEV_PROJECT] = df_dev[COL_DEV_PROJECT].astype(str)
    df_issues[COL_PROJECT]  = df_issues[COL_PROJECT].astype(str)

    # ---------- Merge senioritas INTRODUCER ----------
    dev_intro = df_dev.rename(columns={
        COL_DEV_AUTHOR_KEY:    "Intro_Author_Key",
        COL_DEV_FIRST_COMMIT:  "Intro_First_Commit_Date",
        COL_DEV_TOTAL_COMMITS: "Intro_Total_Commits",
        COL_DEV_TOTAL_FILES:   "Intro_Total_Files_Touched",
    })

    df = df_issues.merge(
        dev_intro[[COL_DEV_PROJECT, "Intro_Author_Key",
                   "Intro_First_Commit_Date", "Intro_Total_Commits", "Intro_Total_Files_Touched"]],
        left_on=[COL_PROJECT, "Intro_Author_Key"],
        right_on=[COL_DEV_PROJECT, "Intro_Author_Key"],
        how="left",
    )

    # ---------- Merge senioritas FIXER ----------
    dev_fix = df_dev.rename(columns={
        COL_DEV_AUTHOR_KEY:    "Payment_Author_Key",
        COL_DEV_FIRST_COMMIT:  "Payment_First_Commit_Date",
        COL_DEV_TOTAL_COMMITS: "Payment_Total_Commits",
        COL_DEV_TOTAL_FILES:   "Payment_Total_Files_Touched",
    })

    df = df.merge(
        dev_fix[[COL_DEV_PROJECT, "Payment_Author_Key",
                 "Payment_First_Commit_Date", "Payment_Total_Commits", "Payment_Total_Files_Touched"]],
        left_on=[COL_PROJECT, "Payment_Author_Key"],
        right_on=[COL_DEV_PROJECT, "Payment_Author_Key"],
        how="left",
        suffixes=("", "_fixproj"),
    )

    # ---------- Hitung senioritas (hari & tahun) ----------
    df["Intro_Seniority_Days"] = (
        df[COL_INTRO_DATE] - df["Intro_First_Commit_Date"]
    ).dt.days
    df["Intro_Seniority_Years"] = df["Intro_Seniority_Days"] / 365.25

    df["Payment_Seniority_Days"] = (
        df[COL_PAYMENT_DATE] - df["Payment_First_Commit_Date"]
    ).dt.days
    df["Payment_Seniority_Years"] = df["Payment_Seniority_Days"] / 365.25

    # Buang nilai aneh (negatif)
    df.loc[df["Intro_Seniority_Days"]   < 0, ["Intro_Seniority_Days", "Intro_Seniority_Years"]]   = np.nan
    df.loc[df["Payment_Seniority_Days"] < 0, ["Payment_Seniority_Days", "Payment_Seniority_Years"]] = np.nan

    # ---------- Pisahkan self-fixed vs non-self ----------
    df_self = df[df["Is_Self_Fixed_Flag"] == True].copy()
    df_non  = df[df["Is_Self_Fixed_Flag"] == False].copy()

    print("\n[*] Jumlah ATD items:")
    print(f"    Self-fixed     : {len(df_self)}")
    print(f"    Non self-fixed : {len(df_non)}")

    # ---------- Fungsi bantu: deskriptif + Mann–Whitney + Cliff's delta ----------
    def describe_and_test(col, label):
        s_self = df_self[col].dropna()
        s_non  = df_non[col].dropna()

        print(f"\n=== {label} Seniority ({col}) ===")
        print(f"  N self-fixed     : {len(s_self)}")
        print(f"  N non self-fixed : {len(s_non)}")

        if len(s_self) < 5 or len(s_non) < 5:
            print("  [!] Terlalu sedikit data untuk uji statistik yang kuat.")
            return

        print("  Self-fixed:")
        print(f"    mean  = {s_self.mean():.2f} years")
        print(f"    median= {s_self.median():.2f} years")
        print(f"    Q1-Q3 = ({s_self.quantile(0.25):.2f}, {s_self.quantile(0.75):.2f})")

        print("  Non self-fixed:")
        print(f"    mean  = {s_non.mean():.2f} years")
        print(f"    median= {s_non.median():.2f} years")
        print(f"    Q1-Q3 = ({s_non.quantile(0.25):.2f}, {s_non.quantile(0.75):.2f})")

        # Mann–Whitney U (non-parametrik, dua sisi)
        u_stat, p_val = mannwhitneyu(s_self, s_non, alternative="two-sided")
        print(f"  Mann–Whitney U test: U = {u_stat:.1f}, p = {p_val:.3e}")

        # Cliff's delta
        d, size = cliffs_delta(s_self, s_non)
        print(f"  Cliff's delta: d = {d:.3f} ({size} effect)")

        return s_self, s_non  # kembalikan untuk dipakai di boxplot

    # ---------- Analisis: INTRO & PAYMENT seniority ----------
    s_self_intro, s_non_intro = describe_and_test("Intro_Seniority_Years",   label="INTRODUCER")
    s_self_pay,   s_non_pay   = describe_and_test("Payment_Seniority_Years", label="FIXER / PAYMENT")

    # ---------- BUAT FIGURE: BOXplot side-by-side ----------
    if s_self_intro is not None and s_self_pay is not None:
        FIG_OUT.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # (a) Introducer seniority
        bp_intro = axes[0].boxplot(
            [s_self_intro, s_non_intro],
            tick_labels=["Self-fixed", "Non self-fixed"],  # ganti labels→tick_labels (hilangkan warning)
            showmeans=True,
            showfliers=False,
            patch_artist=True,  # <<< PENTING: supaya boxes jadi Patch, bisa diwarnai
        )
        style_boxplot(bp_intro)
        axes[0].set_title("Introducer seniority")
        axes[0].set_ylabel("Seniority (years)")

        # (b) Fixer / Payment seniority
        bp_fix = axes[1].boxplot(
            [s_self_pay, s_non_pay],
            tick_labels=["Self-fixed", "Non self-fixed"],
            showmeans=True,
            showfliers=False,
            patch_artist=True,  # <<< PENTING
        )
        style_boxplot(bp_fix)
        axes[1].set_title("Fixer seniority")

        fig.suptitle("", y=1.02)
        plt.tight_layout()

        fig.savefig(FIG_OUT, bbox_inches="tight")
        plt.close(fig)

        print(f"\n[+] Saved boxplot figure to: {FIG_OUT}")


    # ---------- Simpan hasil merge ----------
    out_merged = ISSUES_CSV.parent / "ATD_with_seniority.csv"
    df.to_csv(out_merged, index=False)
    print(f"[+] Saved merged dataset with seniority to: {out_merged}")


if __name__ == "__main__":
    main()
