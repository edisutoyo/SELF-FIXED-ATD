#!/usr/bin/env python3
# coding: utf-8
"""
KM RQ3 for non–self-fixed ATD using role involvement ratios (IIR / FIR / OIR)
Semua ratio berbasis total kontribusi (C_tot = intro + fixer + others):

    IIR = C_intro  / C_tot
    FIR = C_fix    / C_tot
    OIR = C_others / C_tot

Pilih metrik yang mau dianalisis lewat METRIC = "IIR" / "FIR" / "OIR".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test, logrank_test
from scipy.stats import kruskal
from itertools import combinations

# ========= CONFIG: sesuaikan jika perlu =========
ISSUE_CSV    = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/ATD-FINAL-DATASET-TRACED.csv"

# Dataset per-file terbaru yang punya intro/fixer/others
PER_FILE_CSV = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/intro_fixer_others_per_file_shares-FINAL.csv" # intro_fixer_others_per_file_shares-FINAL.csv

# METRIC bisa: "IIR", "FIR", atau "OIR"
METRIC = "IIR"   # ganti ke "FIR" atau "OIR" kalau mau analisis yang lain

COL_ISSUE_ID   = "Key"
COL_INTRO_DT   = "Intro Date"       # momen introduksi ATD
COL_FIX_DT     = "Payment Date"     # NaN/blank jika belum fixed (censored)
COL_SELF_FLAG  = "Is Self-Fixed (Intro=Payment)"  # bool/1/0/yes/no

TIMEZONE       = None  # contoh: "Europe/Amsterdam" jika datetime naive perlu dilokal-kan
ZOOM_DAYS      = 30

OUT_PREFIX     = f"KM_{METRIC}_non_self_fixed"  # prefix nama file output

# (opsional) warna mendekati ggplot2 / konsisten 3 grup
COL_LOW  = "#F8766D"  # Low group
COL_MID  = "#7CAE00"  # Mid group
COL_HIGH = "#00BFC4"  # High group
# ================================================

# === Global style: bigger fonts ===
plt.rcParams.update({
    "font.size": 14,          # baseline font
    "axes.labelsize": 16,     # axis labels
    "axes.titlesize": 18,     # title
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14
})
LINEWIDTH = 3.0               # tebal garis kurva


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "t", "yes", "y"])


def to_datetime(series: pd.Series, tz=None) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if tz:
        try:
            s = s.dt.tz_convert(tz)
        except Exception:
            s2 = pd.to_datetime(series, errors="coerce")
            s = s2.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    return s


# ============================================================
# 1) Hitung C_intro, C_fix, C_others, C_tot per issue + ratio
# ============================================================
pf = pd.read_csv(PER_FILE_CSV, dtype={COL_ISSUE_ID: str}).fillna(0)

need_cols = {
    COL_ISSUE_ID,
    "n_with_endpoints_by_intro",
    "n_with_endpoints_by_pay",
    "n_with_endpoints_by_others",
    "n_with_endpoints_total",
}
missing_pf = need_cols - set(pf.columns)
if missing_pf:
    raise ValueError(f"Kolom hilang di PER_FILE_CSV: {missing_pf}")

agg = pf.groupby(COL_ISSUE_ID, as_index=False).agg(
    C_intro  = ("n_with_endpoints_by_intro",  "sum"),
    C_fix    = ("n_with_endpoints_by_pay",    "sum"),
    C_others = ("n_with_endpoints_by_others", "sum"),
    C_tot    = ("n_with_endpoints_total",     "sum"),
)

# pastikan numerik
for c in ["C_intro", "C_fix", "C_others", "C_tot"]:
    agg[c] = agg[c].astype(float)

# --- definisi ratio berbasis total ---
agg["IIR"] = np.where(agg["C_tot"] > 0, agg["C_intro"]  / agg["C_tot"], np.nan)
agg["FIR"] = np.where(agg["C_tot"] > 0, agg["C_fix"]    / agg["C_tot"], np.nan)
agg["OIR"] = np.where(agg["C_tot"] > 0, agg["C_others"] / agg["C_tot"], np.nan)

# ============================================================
# 2) Load issue-level data & gabungkan dengan ratio yang dipilih
# ============================================================
df = pd.read_csv(ISSUE_CSV, dtype={COL_ISSUE_ID: str})

required_cols = [COL_ISSUE_ID, COL_INTRO_DT, COL_FIX_DT, COL_SELF_FLAG]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Kolom hilang di ISSUE_CSV: {missing}")

df["_intro_dt"]   = to_datetime(df[COL_INTRO_DT], tz=TIMEZONE)
df["_fix_dt"]     = to_datetime(df[COL_FIX_DT], tz=TIMEZONE)
df["_self_fixed"] = to_bool(df[COL_SELF_FLAG])

# hanya baris dengan intro date valid
df = df[~df["_intro_dt"].isna()].copy()

RATIO_COL = METRIC  # "IIR" / "FIR" / "OIR"

if RATIO_COL not in ["IIR", "FIR", "OIR"]:
    raise ValueError("METRIC harus salah satu dari: 'IIR', 'FIR', 'OIR'")

# merge ratio
df = df.merge(agg[[COL_ISSUE_ID, RATIO_COL]], on=COL_ISSUE_ID, how="left")

# fokus hanya non-self-fixed
df = df[df["_self_fixed"] == False].copy()

# buang baris tanpa ratio
# df = df.dropna(subset=[RATIO_COL]).copy()
# df[RATIO_COL] = df[RATIO_COL].astype(float)
# fokus hanya non-self-fixed
df = df[df["_self_fixed"] == False].copy()

# --- HITUNG ISSUE YANG TIDAK MASUK GRUP (TIDAK PUNYA RATIO) ---
mask_no_ratio = df[RATIO_COL].isna()
n_no_ratio = mask_no_ratio.sum()

print(f"\nTotal non–self-fixed issues tanpa nilai {RATIO_COL} (tidak masuk Low/Mid/High): {n_no_ratio}")

if n_no_ratio > 0:
    cols_to_save = [COL_ISSUE_ID, COL_INTRO_DT, COL_FIX_DT, RATIO_COL]
    # jaga-jaga kalau ada kolom yang tidak ada di df
    cols_to_save = [c for c in cols_to_save if c in df.columns]

    df_no_ratio = df.loc[mask_no_ratio, cols_to_save].copy()
    out_csv_no_ratio = f"{OUT_PREFIX}_issues_without_{RATIO_COL}.csv"
    df_no_ratio.to_csv(out_csv_no_ratio, index=False)
    print(f"Daftar issue tanpa nilai {RATIO_COL} disimpan ke: {out_csv_no_ratio}")
# ----------------------------------------------------------------

# buang baris tanpa ratio (supaya analisis lanjut hanya pakai yang punya ratio lengkap)
df = df.dropna(subset=[RATIO_COL]).copy()
df[RATIO_COL] = df[RATIO_COL].astype(float)




# ============================================================
# 3) Durasi & event (time-to-fix) untuk KM
# ============================================================
df["_event"] = (~df["_fix_dt"].isna()).astype(int)

# horizon sensor konservatif
max_obs = pd.concat([df["_intro_dt"], df["_fix_dt"]]).max()

def _dur_days(row):
    end = row["_fix_dt"] if row["_event"] == 1 else max_obs
    return (end - row["_intro_dt"]).total_seconds() / 86400.0

df["_duration_days"] = df.apply(_dur_days, axis=1)
df.loc[df["_duration_days"] < 1e-9, "_duration_days"] = 1e-9
df["_duration_days"] = df["_duration_days"].astype(float)

# ============================================================
# 4) Bagi menjadi Low / Mid / High ratio pakai Q1 & Q3
# ============================================================
q1, q3 = df[RATIO_COL].quantile([0.25, 0.75])
print(f"{RATIO_COL} Q1 = {q1:.3f}, Q3 = {q3:.3f}")

# ---------- Boxplot ratio untuk penentuan Q1 & Q3 ----------
values = df[RATIO_COL].dropna()

fig_box, ax_box = plt.subplots(figsize=(6, 4.5))

meanprops = dict(
    marker="D",
    markerfacecolor="white",
    markeredgecolor="black",
    markersize=8,
    linestyle="none",
)

bp = ax_box.boxplot(
    values,
    vert=True,
    showmeans=True,
    meanline=False,
    widths=0.1,
    patch_artist=True,
    meanprops=meanprops,
    showfliers=False,  # kalau mau lihat outlier, ganti True
)

# Styling boxplot
for box in bp["boxes"]:
    box.set(
        facecolor="#286AAA",
        alpha=0.8,
        linewidth=2.0
    )

for median in bp["medians"]:
    median.set(linewidth=2.5)

for whisker in bp["whiskers"]:
    whisker.set(linewidth=2.0)

for cap in bp["caps"]:
    cap.set(linewidth=2.0)

for flier in bp["fliers"]:
    flier.set(marker="o", markersize=5, alpha=0.5)

ylabel_map = {
    "IIR": "Introducer Involvement Ratio (IIR)",
    "FIR": "Fixer Involvement Ratio (FIR)",
    "OIR": "Others Involvement Ratio (OIR)",
}
ax_box.set_ylabel(ylabel_map[RATIO_COL])
ax_box.set_xticks([1])
ax_box.set_xticklabels([RATIO_COL])
ax_box.set_ylim(0, 1)

ax_box.axhline(q1, color="red", linestyle="--", linewidth=1, label=f"$q_{{25}}$ = {q1:.2f}")
ax_box.axhline(q3, color="green", linestyle="--", linewidth=1, label=f"$q_{{75}}$ = {q3:.2f}")

ax_box.grid(axis="y", alpha=0.3)
ax_box.legend(loc="best")
ax_box.set_title("")

fig_box.tight_layout()
fig_box.savefig(f"{OUT_PREFIX}_{RATIO_COL}_boxplot.pdf", bbox_inches="tight")

# label grup: Low / Mid / High + nama metric
def ratio_group(x):
    if x <= q1:
        return f"Low {RATIO_COL}"
    elif x >= q3:
        return f"High {RATIO_COL}"
    else:
        return f"Mid {RATIO_COL}"

GROUP_COL = f"_{RATIO_COL}_group"
df[GROUP_COL] = df[RATIO_COL].apply(ratio_group)

print("\nCounts per group:")
print(df[GROUP_COL].value_counts())

# ============================================================
# 5) Fit KM untuk tiap grup & plot CUMULATIVE FIXED
# ============================================================
def plot_cumulative_fixed(ax, kmf, color=None):
    surv = kmf.survival_function_
    cum_fixed = (1.0 - surv) * 100.0
    ax.plot(
        cum_fixed.index.values,
        cum_fixed.values,
        label=kmf._label,
        color=color,
        linewidth=LINEWIDTH,
    )

def finalize_axes(ax, title=None, xlim=None):
    ax.set_xlabel("Days since introduction")
    ax.set_ylabel("Cumulative fixed (%)")
    if xlim is not None:
        ax.set_xlim(0, xlim)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend()

groups_def = {
    f"Low {RATIO_COL}":  {"label": f"Low {RATIO_COL}",  "color": COL_LOW},
    f"Mid {RATIO_COL}":  {"label": f"Mid {RATIO_COL}",  "color": COL_MID},
    f"High {RATIO_COL}": {"label": f"High {RATIO_COL}", "color": COL_HIGH},
}

kmf_dict = {}
for g, meta in groups_def.items():
    mask = df[GROUP_COL] == g
    n_g = mask.sum()
    if n_g == 0:
        print(f"[WARN] Group {g} kosong, skip KM fitting.")
        continue

    T = df.loc[mask, "_duration_days"]
    E = df.loc[mask, "_event"]

    kmf = KaplanMeierFitter(label=meta["label"])
    kmf.fit(T, event_observed=E)
    kmf_dict[g] = kmf

if len(kmf_dict) == 0:
    raise RuntimeError("Tidak ada grup dengan data untuk KM fitting.")

# ---------- 5a) FULL horizon ----------
fig_full, ax_full = plt.subplots(figsize=(9, 6))
for g, kmf in kmf_dict.items():
    color = groups_def[g]["color"]
    plot_cumulative_fixed(ax_full, kmf, color=color)

title_map = {
    "IIR": "Kaplan–Meier Cumulative fixed by IIR (non–self-fixed)",
    "FIR": "Kaplan–Meier Cumulative fixed by FIR (non–self-fixed)",
    "OIR": "Kaplan–Meier Cumulative fixed by OIR (non–self-fixed)",
}
finalize_axes(ax_full, title=title_map[RATIO_COL])

add_at_risk_counts(*kmf_dict.values(), ax=ax_full, rows_to_show=["At risk", "Events"])

fig_full.tight_layout()
fig_full.savefig(f"{OUT_PREFIX}_full.pdf", bbox_inches="tight")

# ---------- 5b) ZOOM ≤ ZOOM_DAYS ----------
fig_zoom, ax_zoom = plt.subplots(figsize=(9, 6))
for g, kmf in kmf_dict.items():
    color = groups_def[g]["color"]
    plot_cumulative_fixed(ax_zoom, kmf, color=color)

title_zoom_map = {
    "IIR": f"Kaplan–Meier Cumulative fixed by IIR (≤ {ZOOM_DAYS} days, non–self-fixed)",
    "FIR": f"Kaplan–Meier Cumulative fixed by FIR (≤ {ZOOM_DAYS} days, non–self-fixed)",
    "OIR": f"Kaplan–Meier Cumulative fixed by OIR (≤ {ZOOM_DAYS} days, non–self-fixed)",
}
finalize_axes(
    ax_zoom,
    title=title_zoom_map[RATIO_COL],
    xlim=ZOOM_DAYS
)

xt = np.arange(0, ZOOM_DAYS + 1, 7)
ax_zoom.set_xticks(xt)
ax_zoom.set_xticklabels([str(int(x)) for x in xt])

add_at_risk_counts(*kmf_dict.values(), ax=ax_zoom, rows_to_show=["At risk", "Events"])

fig_zoom.tight_layout()
fig_zoom.savefig(f"{OUT_PREFIX}_{ZOOM_DAYS}days-week.pdf", bbox_inches="tight")

print("\nSaved:",
      f"{OUT_PREFIX}_{RATIO_COL}_boxplot.pdf,",
      f"{OUT_PREFIX}_full.pdf,",
      f"{OUT_PREFIX}_{ZOOM_DAYS}days-week.pdf")

# ============================================================
# 6) Log-rank tests
# ============================================================
present_groups = [g for g in groups_def.keys()
                  if (df[GROUP_COL] == g).sum() > 0]

if len(present_groups) >= 2:
    results_global = multivariate_logrank_test(
        event_durations=df["_duration_days"],
        groups=df[GROUP_COL],
        event_observed=df["_event"]
    )

    print(f"\n=== Global log-rank test (Low vs Mid vs High {RATIO_COL}) ===")
    print(results_global.summary)

    pair_results = []
    for g1, g2 in combinations(present_groups, 2):
        d1 = df[df[GROUP_COL] == g1]
        d2 = df[df[GROUP_COL] == g2]

        res = logrank_test(
            d1["_duration_days"], d2["_duration_days"],
            event_observed_A=d1["_event"],
            event_observed_B=d2["_event"]
        )
        pair_results.append((g1, g2, res.test_statistic, res.p_value))

    print("\n=== Pairwise log-rank tests (raw p-values) ===")
    for g1, g2, stat, p in pair_results:
        print(f"{g1} vs {g2}: chi2={stat:.3f}, p={p:.3g}")

    print("\n=== Pairwise log-rank tests (Bonferroni-adjusted p) ===")
    m = len(pair_results)
    for g1, g2, stat, p in pair_results:
        p_adj = min(p * m, 1.0)
        print(f"{g1} vs {g2}: chi2={stat:.3f}, p_raw={p:.3g}, p_adj={p_adj:.3g}")
else:
    print(f"\n[WARN] Kurang dari 2 grup {RATIO_COL} yang punya data, log-rank test dilewati.")

# ============================================================
# 7) Kruskal–Wallis test pada time-to-fix (TTF) per group
# ============================================================
arrays = []
names  = []
for g in [f"Low {RATIO_COL}", f"Mid {RATIO_COL}", f"High {RATIO_COL}"]:
    T_g = df.loc[df[GROUP_COL] == g, "_duration_days"].dropna()
    if len(T_g) > 0:
        arrays.append(T_g)
        names.append(g)

if len(arrays) >= 2:
    H_stat, p_kw = kruskal(*arrays)
    print(f"\n=== Kruskal–Wallis test on time-to-fix by {RATIO_COL} group ===")
    print("Groups:", ", ".join(names))
    print(f"H = {H_stat:.3f}, p = {p_kw:.3g}")
else:
    print(f"\n[WARN] Kurang dari 2 grup {RATIO_COL} dengan data untuk Kruskal–Wallis, test dilewati.")

print(f"\n=== Ringkasan per grup {RATIO_COL} ===")
print(df.groupby(GROUP_COL)[RATIO_COL].agg(["count", "min", "median", "max"]))

print(f"\n=== Rata-rata durasi (hari) per grup {RATIO_COL} ===")
print(df.groupby(GROUP_COL)["_duration_days"].describe())
