#!/usr/bin/env python3
# coding: utf-8
"""
Collect per-file commits between Intro Hash and Payment Hash untuk SELF-FIXED issues,
dan bandingkan kontribusi:
- Self-fixer (introducer == fixer)
- Others (developer lain)

Output:
- CSV per file dengan count & ratio commit per role
- Boxplot COUNT & RATIO (INTRO+MID+PAYMENT) -> Self-fixer vs Others
- Boxplot RATIO MID-only                   -> Self-fixer vs Others
- Statistical tests (MOST APPROPRIATE):
    * One-sample Wilcoxon signed-rank test on self_share vs 0.5 (one-sided, greater)
    * Effect size: matched-pairs rank-biserial correlation (RBC)
    * Dominance rate: Pr(self_share > 0.5)
    * Sign test (binomial test) as robust alternative
Notes:
- Do NOT run two-sample tests Self vs Others for ratios, because Others = 1 - Self by construction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional



# ====== BIGGER FONTS (GLOBAL) ======
plt.rcParams.update({
    "font.size": 14,          # base
    "axes.labelsize": 12,     # y-label
    "axes.titlesize": 12,     # title (if you use it)
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# --- Stats imports (SciPy) ---
try:
    from scipy.stats import wilcoxon, rankdata, binomtest
except Exception as e:
    raise SystemExit(
        "This script requires SciPy for Wilcoxon/rankdata/binomtest.\n"
        "Install with: pip install scipy\n"
        f"Original import error: {e}"
    )

# ====== CONFIG: ubah jika perlu ======
COMMITS_CSV = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/SELF-FIXED_commits_between_periods_autorepo-ONLY-INTRO-ALL-AUTHORS-SELF-FIXED-ATD-FINAL-DATASET-TRACED-ALL-EXTS.csv"

OUT_CSV           = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/rq2/SF/rq2_self_fixer_vs_others_per_file_WITH_ENDPOINTS.csv"  #OUTPUT
OUT_PDF_CNT       = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/rq2/SF/rq2_box_counts_per_file_SELF-FIXER_vs_OTHERS_WITH_ENDPOINTS.pdf"
OUT_PDF_RATIO     = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/rq2/SF/rq2_box_RATIO_per_file_SELF-FIXER_vs_OTHERS_WITH_ENDPOINTS.pdf"
OUT_PDF_RATIO_MID = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/rq2/SF/rq2_box_RATIO_per_file_SELF-FIXER_vs_OTHERS_MID_ONLY.pdf"

# NEW: stats output (CSV + TXT)
OUT_STATS_CSV     = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/rq2/SF/rq2_self_fixer_vs_0p5_stats.csv"
OUT_STATS_TXT     = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/rq2/SF/rq2_self_fixer_vs_0p5_stats.txt"

# >>> Hanya gunakan pasangan Key/File yang memiliki MID?
FILTER_REQUIRE_MID = True

# >>> FILTER ekstensi file (True/False) <<<
FILTER_BY_EXT = False
ALLOWED_EXTS = {
    ".java", ".scala", ".kt", ".kts", ".py", ".rb", ".go", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".php", ".ts", ".tsx", ".jsx", ".rs", ".swift",
    ".erl", ".ex", ".exs", ".xml"
}
# =======================================

# --- GLOBAL FONT/THEME ---
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# ==== THEME: dua warna konsisten ====
COL_SELF  = "#01BFC4"   # teal  (Self-fixer)
COL_OTHER = "#999999"   # grey  (Others)

def boxplot_multi_series(ax, data_list, tick_labels, y_label, title, ylim=None):
    bp = ax.boxplot(
        data_list,
        tick_labels=tick_labels,
        showmeans=True,
        patch_artist=True
    )
    fills = [COL_SELF, COL_OTHER]

    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=fills[i % len(fills)], edgecolor="black", alpha=0.9, linewidth=1.4)
    for w in bp['whiskers']:
        w.set(color="black", linewidth=1.4)
    for c in bp['caps']:
        c.set(color="black", linewidth=1.4)
    for med in bp['medians']:
        med.set(color="black", linewidth=2.4)
    for mean in bp['means']:
        mean.set(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=8)
    for i, fl in enumerate(bp['fliers']):
        fl.set(marker='o', alpha=0.55, markersize=4.5,
               markerfacecolor=fills[i % len(fills)], markeredgecolor="black")

    ax.set_ylabel(y_label, fontsize=15)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.6)

def norm_str(x: Optional[str]) -> str:
    return ("" if x is None else str(x)).strip().lower()

def norm_role(r):
    rr = norm_str(r)
    if rr in {"intro", "inro", "introduction"}: return "INTRO"
    if rr in {"payment", "pay"}:                return "PAYMENT"
    if rr in {"mid", "between"}:                return "MID"
    return str(r)

def get_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def author_id(row):
    for c in ["Author_Email", "Author_Name", "Committer_Email", "Committer_Name"]:
        if c in row and str(row[c]).strip():
            return norm_str(row[c])
    return ""

# =========================
# Statistical test helpers
# =========================
def _safe_float_array(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]

def one_sample_wilcoxon_vs_mu0(x: np.ndarray, mu0: float = 0.5, alternative: str = "greater"):
    """
    One-sample Wilcoxon signed-rank test on d = x - mu0.
    Returns dict: n_total, n_used, W, p, RBC, median, q25, q75, pr_gt_mu0, sign_p
    """
    x = _safe_float_array(x)
    n_total = int(x.size)

    if n_total == 0:
        return {
            "n_total": 0, "n_used": 0, "W": np.nan, "p": np.nan, "RBC": np.nan,
            "median": np.nan, "q25": np.nan, "q75": np.nan, "pr_gt_mu0": np.nan, "sign_p": np.nan
        }

    d = x - mu0

    # Wilcoxon discards exact zero differences; do the same for consistent n_used
    d_nz = d[d != 0.0]
    n_used = int(d_nz.size)

    if n_used == 0:
        # All equal to mu0
        pr_gt = float(np.mean(x > mu0))
        return {
            "n_total": n_total, "n_used": 0, "W": 0.0, "p": 1.0, "RBC": 0.0,
            "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)),
            "pr_gt_mu0": pr_gt,
            "sign_p": 1.0
        }

    res = wilcoxon(d_nz, alternative=alternative, zero_method="wilcox")
    W = float(res.statistic)
    p = float(res.pvalue)

    # Effect size: matched-pairs rank-biserial correlation (RBC)
    ranks = rankdata(np.abs(d_nz), method="average")
    W_pos = float(ranks[d_nz > 0].sum())
    W_neg = float(ranks[d_nz < 0].sum())
    RBC = (W_pos - W_neg) / (W_pos + W_neg)

    # Dominance rate (intuitive)
    pr_gt = float(np.mean(x > mu0))

    # Sign test (binomial): count(d > 0) among non-zero diffs
    n_pos = int(np.sum(d_nz > 0))
    sign_p = float(binomtest(n_pos, n_used, p=0.5, alternative=alternative).pvalue)

    return {
        "n_total": n_total,
        "n_used": n_used,
        "W": W,
        "p": p,
        "RBC": float(RBC),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "pr_gt_mu0": pr_gt,
        "sign_p": sign_p
    }

def format_p(p: float) -> str:
    # Avoid printing p=0 due to underflow; show a conventional bound
    if np.isnan(p):
        return "nan"
    if p == 0.0:
        return "< 2.2e-16"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.6f}"

# ----- Load -----
df = pd.read_csv(COMMITS_CSV, dtype=str).fillna("")

if "Role" not in df.columns:
    raise SystemExit("Column 'Role' is required in commits CSV.")
df["Role_std"] = df["Role"].apply(norm_role)

# ----- Kolom File_Path yang robust -----
if "File_Path" in df.columns:
    df["File_Path_std"] = df["File_Path"].astype(str)
elif "Tracked_File" in df.columns:
    df["File_Path_std"] = df["Tracked_File"].astype(str)
elif "Touched_Files" in df.columns:
    df["File_Path_std"] = df["Touched_Files"].apply(
        lambda s: (s.split(";")[0] if isinstance(s, str) and s else "")
    )
else:
    raise SystemExit("Tidak menemukan kolom path file (File_Path/Tracked_File/Touched_Files).")

# ----- FILTER berdasarkan ekstensi file (opsional) -----
if FILTER_BY_EXT:
    df["File_Ext"] = df["File_Path_std"].astype(str).apply(get_ext)
    before_ext = len(df)
    df = df[df["File_Ext"].isin(ALLOWED_EXTS)].copy()
    print(f"Extension filter ON: kept {len(df)} / {before_ext} rows for extensions {sorted(ALLOWED_EXTS)}")
else:
    print("Extension filter OFF: all file types are included.")

# ----- Identitas author -----
df["author_id"] = df.apply(author_id, axis=1)

# ----- Pastikan kolom hash ada untuk dedup per file -----
commit_col = "Commit_Hash" if "Commit_Hash" in df.columns else None
if commit_col is None:
    if "Authored_Datetime" in df.columns and "Commit_Summary" in df.columns:
        df["Commit_Hash"] = df["Authored_Datetime"].astype(str) + "||" + df["Commit_Summary"].astype(str)
    else:
        df["Commit_Hash"] = df.reset_index().index.astype(str)
    commit_col = "Commit_Hash"

# ====== Ambil author INTRO & PAYMENT per (Key, File_Path) ======
intro_auth = (
    df[df["Role_std"] == "INTRO"]
    .groupby(["Key", "File_Path_std"], as_index=False)["author_id"]
    .apply(lambda s: next((x for x in s if x), ""))
    .rename(columns={"author_id": "intro_author"})
)

pay_auth = (
    df[df["Role_std"] == "PAYMENT"]
    .groupby(["Key", "File_Path_std"], as_index=False)["author_id"]
    .apply(lambda s: next((x for x in s if x), ""))
    .rename(columns={"author_id": "pay_author"})
)

authors = pd.merge(intro_auth, pay_auth, on=["Key", "File_Path_std"], how="outer")

for col in ["intro_author", "pay_author"]:
    if col not in authors.columns:
        authors[col] = ""

authors["intro_author_norm"] = authors["intro_author"].apply(norm_str)
authors["pay_author_norm"]   = authors["pay_author"].apply(norm_str)

authors["self_author"] = authors["intro_author_norm"]
mask = (authors["self_author"] == "") & (authors["pay_author_norm"] != "")
authors.loc[mask, "self_author"] = authors.loc[mask, "pay_author_norm"]

# ================= MID-only =================
mids = df[df["Role_std"] == "MID"].copy()
mids = mids.drop_duplicates(subset=["Key", "File_Path_std", commit_col])

mids = pd.merge(
    mids,
    authors[["Key", "File_Path_std", "self_author"]],
    on=["Key", "File_Path_std"],
    how="left"
)

mids["self_author"] = mids["self_author"].fillna("")
mids["is_self"]   = (mids["author_id"] == mids["self_author"])
mids["is_other"]  = ~mids["is_self"]

per_file_mid = mids.groupby(["Key", "File_Path_std"]).agg(
    n_mid_total     = ("author_id", "size"),
    n_mid_by_self   = ("is_self", "sum"),
    n_mid_by_other  = ("is_other", "sum"),
).reset_index()

if not authors.empty:
    per_file_mid = (
        authors[["Key", "File_Path_std", "self_author"]]
        .merge(per_file_mid, on=["Key", "File_Path_std"], how="left")
        .fillna({"n_mid_total": 0, "n_mid_by_self": 0, "n_mid_by_other": 0})
    )
    per_file_mid[["n_mid_total", "n_mid_by_self", "n_mid_by_other"]] = (
        per_file_mid[["n_mid_total", "n_mid_by_self", "n_mid_by_other"]].astype(int)
    )

# ========== INTRO + MID + PAYMENT (rentang penuh) ==========
range_df = df[df["Role_std"].isin(["INTRO", "MID", "PAYMENT"])].copy()
range_df = range_df.drop_duplicates(subset=["Key", "File_Path_std", commit_col])

range_df = pd.merge(
    range_df,
    authors[["Key", "File_Path_std", "self_author"]],
    on=["Key", "File_Path_std"],
    how="left"
)

range_df["self_author"] = range_df["self_author"].fillna("")
range_df["is_self_author"]  = (range_df["author_id"] == range_df["self_author"])
range_df["is_other_author"] = ~range_df["is_self_author"]

per_file_range = range_df.groupby(["Key", "File_Path_std"]).agg(
    n_with_endpoints_total     = ("author_id", "size"),
    n_with_endpoints_by_self   = ("is_self_author", "sum"),
    n_with_endpoints_by_other  = ("is_other_author", "sum"),
).reset_index()

# ====== Gabungkan MID + full range & hitung rasio ======
per_file = pd.merge(
    per_file_mid,
    per_file_range,
    on=["Key", "File_Path_std"],
    how="outer"
).fillna(0)

for col in [
    "n_mid_total", "n_mid_by_self", "n_mid_by_other",
    "n_with_endpoints_total",
    "n_with_endpoints_by_self", "n_with_endpoints_by_other",
]:
    per_file[col] = per_file[col].astype(int)

# --- Rasio MID (KEEP ZEROS; do NOT filter >0 for tests) ---
per_file["n_mid_by_self/n_mid_total"] = np.where(
    per_file["n_mid_total"] > 0,
    per_file["n_mid_by_self"] / per_file["n_mid_total"],
    np.nan
)
per_file["n_mid_by_other/n_mid_total"] = np.where(
    per_file["n_mid_total"] > 0,
    per_file["n_mid_by_other"] / per_file["n_mid_total"],
    np.nan
)

# --- Rasio seluruh rentang (INTRO+MID+PAYMENT) ---
per_file["n_with_endpoints_by_self/n_with_endpoints_total"] = np.where(
    per_file["n_with_endpoints_total"] > 0,
    per_file["n_with_endpoints_by_self"] / per_file["n_with_endpoints_total"],
    np.nan
)
per_file["n_with_endpoints_by_other/n_with_endpoints_total"] = np.where(
    per_file["n_with_endpoints_total"] > 0,
    per_file["n_with_endpoints_by_other"] / per_file["n_with_endpoints_total"],
    np.nan
)

# ----- Simpan CSV -----
per_file = per_file.rename(columns={"File_Path_std": "File_Path"})
cols_order = [
    "Key", "File_Path", "self_author",
    "n_mid_total", "n_mid_by_self", "n_mid_by_other",
    "n_with_endpoints_total",
    "n_with_endpoints_by_self", "n_with_endpoints_by_other",
    "n_mid_by_self/n_mid_total",
    "n_mid_by_other/n_mid_total",
    "n_with_endpoints_by_self/n_with_endpoints_total",
    "n_with_endpoints_by_other/n_with_endpoints_total",
]
per_file = per_file[cols_order]

if FILTER_REQUIRE_MID:
    before = len(per_file)
    per_file = per_file[per_file["n_mid_total"] > 0].copy()
    print(f"Filter MID only: kept {len(per_file)} / {before} rows (removed {before - len(per_file)} without MID).")

per_file.to_csv(OUT_CSV, index=False)
print(f"Saved per-file summary WITH endpoints: {OUT_CSV}")

# ================== BOXPLOT COUNT (INTRO+MID+PAYMENT) ==================
if len(per_file) > 0:
    self_counts   = per_file["n_with_endpoints_by_self"].values
    others_counts = per_file["n_with_endpoints_by_other"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_multi_series(
        ax,
        [self_counts, others_counts],
        tick_labels=["Self-fixer", "Others"],
        y_label="Commits per file",
        title="",
        ylim=(0, None)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_CNT, bbox_inches="tight")
    print(f"Saved boxplot COUNT (Self-fixer vs Others): {OUT_PDF_CNT}")
else:
    print("Tidak ada data per-file untuk boxplot COUNT.")

# ================== BOXPLOT RATIO (INTRO+MID+PAYMENT) ==================
rs = pd.to_numeric(
    per_file["n_with_endpoints_by_self/n_with_endpoints_total"],
    errors="coerce"
).dropna()

# IMPORTANT: others is deterministically 1 - self
ratio_self  = rs.to_numpy()
ratio_other = (1.0 - ratio_self)

if ratio_self.size:
    fig, ax = plt.subplots(figsize=(7, 4))
    boxplot_multi_series(
        ax,
        [ratio_self, ratio_other],
        tick_labels=["Self-fixer", "Others"],
        y_label="Share of commits per file",
        title="",
        ylim=(0, 1)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_RATIO, bbox_inches="tight")
    print(f"Saved ratio boxplot (Self-fixer vs Others): {OUT_PDF_RATIO}")
else:
    print("Tidak ada data rasio untuk boxplot RATIO (INTRO+MID+PAYMENT).")

# ================== BOXPLOT RATIO (MID ONLY) ==================
rs_mid = pd.to_numeric(
    per_file["n_mid_by_self/n_mid_total"],
    errors="coerce"
).dropna()

ro_mid = 1.0 - rs_mid  # deterministic complement

# For the plot, you previously filtered >0; keep that if you want the same look,
# but do NOT use that filtered data for statistical tests.
ratio_self_mid_plot  = rs_mid[rs_mid > 0].to_numpy()
ratio_other_mid_plot = ro_mid[ro_mid > 0].to_numpy()

if ratio_self_mid_plot.size or ratio_other_mid_plot.size:
    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot_multi_series(
        ax,
        [ratio_self_mid_plot, ratio_other_mid_plot],
        tick_labels=["Self-fixer", "Others"],
        y_label="Share of commits per file (MID only)",
        title="Share of commits during MID phase by role",
        ylim=(0, 1)
    )
    plt.tight_layout()
    plt.savefig(OUT_PDF_RATIO_MID, bbox_inches="tight")
    print(f"Saved MID ratio boxplot (Self-fixer vs Others): {OUT_PDF_RATIO_MID}")
else:
    print("Tidak ada data rasio untuk boxplot RATIO MID.")

# ==========================
# STATISTICAL TESTS (MOST APPROPRIATE)
# ==========================
# Test target:
#   H0: median(self_share) = 0.5
#   H1: median(self_share) > 0.5
#
# We run:
#   (A) Full range self_share (INTRO + MID + PAYMENT)
#   (B) MID-only self_share (recommended as less “by definition”)

full_x = pd.to_numeric(
    per_file["n_with_endpoints_by_self/n_with_endpoints_total"],
    errors="coerce"
).to_numpy()

mid_x = pd.to_numeric(
    per_file["n_mid_by_self/n_mid_total"],
    errors="coerce"
).to_numpy()

res_full = one_sample_wilcoxon_vs_mu0(full_x, mu0=0.5, alternative="greater")
res_mid  = one_sample_wilcoxon_vs_mu0(mid_x,  mu0=0.5, alternative="greater")

stats_df = pd.DataFrame([
    {"phase": "FULL_RANGE_INTRO_MID_PAYMENT", **res_full},
    {"phase": "MID_ONLY", **res_mid},
])

stats_df.to_csv(OUT_STATS_CSV, index=False)

lines = []
lines.append("=== One-sample tests against 0.5 (H1: self_share > 0.5) ===\n")
for _, r in stats_df.iterrows():
    lines.append(f"[{r['phase']}]")
    lines.append(f"n_total (non-NaN)  : {int(r['n_total'])}")
    lines.append(f"n_used (non-zero d): {int(r['n_used'])}")
    lines.append(f"median [q25,q75]   : {r['median']:.4f} [{r['q25']:.4f}, {r['q75']:.4f}]")
    lines.append(f"Pr(self_share > .5): {r['pr_gt_mu0']:.3f}")
    lines.append(f"Wilcoxon W         : {r['W']:.6g}")
    lines.append(f"Wilcoxon p         : {format_p(r['p'])}")
    lines.append(f"Effect size (RBC)  : {r['RBC']:.4f}")
    lines.append(f"Sign test p        : {format_p(r['sign_p'])}")
    lines.append("")

with open(OUT_STATS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"Saved stats CSV: {OUT_STATS_CSV}")
print(f"Saved stats TXT: {OUT_STATS_TXT}")
