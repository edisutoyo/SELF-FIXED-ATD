#!/usr/bin/env python3
# coding: utf-8
"""
Hitung rasio kontribusi Introducer, Fixer, dan Others per file dan buat boxplot.

Definisi:
- Introducer = author INTRO di level issue (Key), diambil dari commit dengan Role INTRO.
- Fixer      = author PAYMENT di level issue (Key), diambil dari commit dengan Role PAYMENT.
- Others     = semua author lain di antara INTRO dan PAYMENT.

Untuk setiap (Key, File_Path):
  total_commits = n_intro + n_fix + n_others
  Share_Intro   = n_intro   / total_commits
  Share_Fix     = n_fix     / total_commits
  Share_Others  = n_others  / total_commits

Tambahan:
- Has_MID / No_MID: berdasarkan ada/tidaknya commit dengan Role MID di file tersebut.
- FILTER_REQUIRE_MID:
    - True  -> hanya file dengan Has_MID == True yang disertakan.
    - False -> semua file (Has_MID dan No_MID).
- ALLOWED_EXTENSIONS:
    - Filter file berdasarkan ekstensi (case-insensitive).
- FILTER_NON_SELF_FIXED_ONLY:
    - True  -> hanya non–self-fixed (introducer != fixer) yang dianalisis.
    - False -> semua (self-fixed dan non–self-fixed).
"""

import os
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ====== BIGGER FONTS (GLOBAL) ======
plt.rcParams.update({
    "font.size": 14,          # base
    "axes.labelsize": 12,     # y-label
    "axes.titlesize": 12,     # title (if you use it)
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ========== CONFIG: UBAH SESUAI LOKASI FILE ANDA ==========
COMMITS_CSV = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/collect_commits_between_intro_payment_ALL-DEVELOPERS-INTRO-FILES_UPGRADED-ATD-FINAL-DATASET-TRACED-ALL-EXTS.csv"

OUT_CSV = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/NEW/intro_fixer_others_per_file_shares-FINAL.csv"
OUT_PNG = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/NEW/boxplot_intro_fixer_others_per_file-FINAL.png"
OUT_PDF = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/NEW/boxplot_intro_fixer_others_per_file-FINAL.pdf"

# Filter: hanya file yang punya MID?
FILTER_REQUIRE_MID = True

# Filter: hanya NON–SELF-FIXED (introducer != fixer)?
# True  -> hanya non–self-fixed
# False -> semua (self-fixed + non–self-fixed)
FILTER_NON_SELF_FIXED_ONLY = True

# Filter: hanya file dengan ekstensi berikut (case-insensitive).
ALLOWED_EXTENSIONS = {
    ".java", ".scala", ".kt", ".kts", ".py", ".rb", ".go", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".php", ".ts", ".tsx", ".jsx", ".rs", ".swift",
    ".erl", ".ex", ".exs", ".xml"
}

# Kontrol tampilan outlier (fliers) di boxplot
SHOW_FLIERS = False   # set True kalau mau titik outlier terlihat
# ==========================================================

# (optional) sedikit perapian tampilan
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.size": 12,
})


def norm_str(x: Optional[str]) -> str:
    """Trim + lower-case untuk string (dipakai di author_id & Role)."""
    return ("" if x is None else str(x)).strip().lower()


def load_data(csv_path: str) -> pd.DataFrame:
    print(f"[*] Membaca CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"[*] Jumlah baris: {len(df):,}")
    print(f"[*] Kolom yang tersedia: {list(df.columns)}")
    return df


def norm_role(r: str) -> str:
    rr = norm_str(r)
    if rr in {"intro", "inro", "introduction"}:
        return "INTRO"
    if rr in {"payment", "pay"}:
        return "PAYMENT"
    if rr in {"mid", "between"}:
        return "MID"
    return r  # biarkan apa adanya kalau tidak dikenal


def build_file_path_std(df: pd.DataFrame) -> pd.DataFrame:
    """Bangun kolom File_Path_std yang robust dari File_Path / Tracked_File / Touched_Files."""
    if "File_Path" in df.columns:
        df["File_Path_std"] = df["File_Path"].astype(str)
    elif "Tracked_File" in df.columns:
        df["File_Path_std"] = df["Tracked_File"].astype(str)
    elif "Touched_Files" in df.columns:
        df["File_Path_std"] = df["Touched_Files"].apply(
            lambda s: (s.split(";")[0] if isinstance(s, str) and s else "")
        )
    else:
        raise ValueError("Tidak menemukan kolom path file (File_Path/Tracked_File/Touched_Files).")
    return df


def build_author_id(df: pd.DataFrame) -> pd.DataFrame:
    """Ambil identitas author yang dinormalisasi: email > nama > committer."""
    def author_id(row) -> str:
        for c in ["Author_Email", "Author_Name", "Committer_Email", "Committer_Name"]:
            if c in row and str(row[c]).strip():
                return norm_str(row[c])
        return ""
    df["author_id"] = df.apply(author_id, axis=1)
    return df


def filter_by_extension(df: pd.DataFrame) -> pd.DataFrame:
    """Filter baris berdasarkan ekstensi File_Path_std."""
    if not ALLOWED_EXTENSIONS:
        print("[*] ALLOWED_EXTENSIONS kosong → tidak ada filter ekstensi file.")
        return df

    before_rows = len(df)
    df["File_Path_std"] = df["File_Path_std"].astype(str)
    lowers = [ext.lower() for ext in ALLOWED_EXTENSIONS]

    def has_allowed_ext(path: str) -> bool:
        p = path.lower()
        return any(p.endswith(ext) for ext in lowers)

    df = df[df["File_Path_std"].apply(has_allowed_ext)].copy()
    after_rows = len(df)
    print(f"[*] Filter ekstensi diaktifkan: {ALLOWED_EXTENSIONS}")
    print(f"    Sebelum filter ekstensi: {before_rows} baris, sesudah: {after_rows} baris.")
    if df.empty:
        print("[!] Semua baris terbuang setelah filter ekstensi.")
    return df


def infer_intro_and_fixer_by_key(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Untuk setiap Key, ambil:
      - intro_author_id dari commit Role_std == INTRO
      - fix_author_id   dari commit Role_std == PAYMENT
    Dipakai untuk semua file di issue tersebut.
    """
    intro_authors: Dict[str, str] = {}
    fix_authors: Dict[str, str] = {}

    for key, sub in df.groupby("Key"):
        # INTRO
        intro_rows = sub[sub["Role_std"] == "INTRO"]
        ia = None
        if not intro_rows.empty:
            for val in intro_rows["author_id"]:
                if val:
                    ia = val
                    break
        if ia:
            intro_authors[key] = ia

        # PAYMENT
        pay_rows = sub[sub["Role_std"] == "PAYMENT"]
        fa = None
        if not pay_rows.empty:
            for val in pay_rows["author_id"]:
                if val:
                    fa = val
                    break
        if fa:
            fix_authors[key] = fa

    print(f"[*] Ditemukan introducer (Key-level) untuk {len(intro_authors)} Key")
    print(f"[*] Ditemukan fixer      (Key-level) untuk {len(fix_authors)} Key")
    return intro_authors, fix_authors


def compute_per_file_shares(
    df: pd.DataFrame,
    intro_authors: Dict[str, str],
    fix_authors: Dict[str, str]
) -> pd.DataFrame:
    """
    Hitung Share_Intro, Share_Fix, dan Share_Others per (Key, File_Path_std),
    sekaligus flag Has_MID / No_MID.

    Self-fixed (intro_author == fix_author):
      -> semua commit author tersebut dihitung sebagai INTRO,
         Fixer (n_fix) = 0 supaya tidak double-count.
    """
    records = []

    for (key, file_path), sub in df.groupby(["Key", "File_Path_std"]):
        ia = intro_authors.get(key)  # author_id introducer
        fa = fix_authors.get(key)    # author_id fixer

        # Jika introducer atau fixer tidak diketahui, skip
        if not ia or not fa:
            continue

        # Cek apakah ada MID untuk file ini (pakai Role_std yang sudah dinormalisasi)
        has_mid = (sub["Role_std"] == "MID").any()

        same_author = (ia == fa)

        if same_author:
            # Self-fixed: semua commit author itu dihitung INTRO, fixer=0
            n_intro = (sub["author_id"] == ia).sum()
            n_fix = 0
            mask_others = sub["author_id"] != ia
        else:
            n_intro = (sub["author_id"] == ia).sum()
            n_fix = (sub["author_id"] == fa).sum()
            mask_others = (sub["author_id"] != ia) & (sub["author_id"] != fa)

        n_others = mask_others.sum()

        total = n_intro + n_fix + n_others
        if total == 0:
            continue

        share_intro = n_intro / total
        share_fix = n_fix / total
        share_others = n_others / total

        records.append({
            "Key": key,
            "File_Path": file_path,
            "Intro_Author_ID": ia,
            "Fix_Author_ID": fa,
            "Self_Fixed_Same_Author": bool(same_author),
            "Commits_Intro": int(n_intro),
            "Commits_Fix": int(n_fix),
            "Commits_Others": int(n_others),
            "Total_Commits_File": int(total),
            "Share_Intro": share_intro,
            "Share_Fix": share_fix,
            "Share_Others": share_others,
            "Has_MID": bool(has_mid),
            "No_MID": bool(not has_mid),
        })

    shares_df = pd.DataFrame(records)
    print(f"[*] Jumlah (Key, File) dengan data kontribusi (sebelum filter MID/self-fixed): {len(shares_df)}")
    return shares_df


def save_shares_csv(shares_df: pd.DataFrame, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    shares_df.to_csv(out_csv, index=False)
    print(f"[*] Disimpan CSV per-file share ke: {out_csv}")


def plot_boxplot(shares_df: pd.DataFrame, out_png: str, out_pdf: str) -> None:
    """
    Buat boxplot untuk Share_Intro, Share_Fix, dan Share_Others.
    """
    if shares_df.empty:
        print("[!] shares_df kosong, tidak ada yang bisa diplot.")
        return

    # data = [
    #     shares_df["Share_Intro"],
    #     shares_df["Share_Fix"],
    #     shares_df["Share_Others"],
    # ]

    tmp = shares_df[["Share_Intro", "Share_Fix", "Share_Others"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    ratio_intro  = tmp["Share_Intro"].to_numpy()
    ratio_fixer  = tmp["Share_Fix"].to_numpy()
    ratio_others = tmp["Share_Others"].to_numpy()

    data = [ratio_intro, ratio_fixer, ratio_others]


    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        data,
        labels=["Introducer", "Fixer", "Others"],
        showmeans=True,
        showfliers=SHOW_FLIERS,
        patch_artist=True,
    )

    # Warna box
    colors = ["#01BFC4", "#F9766D", "#999999"]  # teal, salmon, grey
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    # median & mean -> lebih menonjol (snipped style)
    for med in bp["medians"]:
        med.set(color="black", linewidth=2.4)
    for mean in bp["means"]:
        mean.set(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
        )

    ax.set_ylabel("Share of commits per file")
    ax.set_title("")  # bisa diisi jika perlu
    ax.set_ylim(0.0, 1.0)

    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"[*] Boxplot disimpan sebagai PNG: {out_png}")
    print(f"[*] Boxplot disimpan sebagai PDF: {out_pdf}")


def main():
    df = load_data(COMMITS_CSV)

    # Pastikan kolom Key dan Role ada
    if "Key" not in df.columns:
        raise ValueError("Kolom 'Key' tidak ditemukan di CSV.")
    if "Role" not in df.columns:
        raise ValueError("Kolom 'Role' tidak ditemukan di CSV.")

    # Normalisasi Role & File_Path_std & author_id
    df["Role_std"] = df["Role"].apply(norm_role)
    df = build_file_path_std(df)
    df = build_author_id(df)

    # Filter ekstensi (jika diset)
    df = filter_by_extension(df)
    if df.empty:
        return

    # Hitung author INTRO & PAYMENT per Key (issue-level)
    intro_authors, fix_authors = infer_intro_and_fixer_by_key(df)

    # Hitung share kontribusi per (Key, File_Path)
    shares_df = compute_per_file_shares(df, intro_authors, fix_authors)
    if shares_df.empty:
        print("[!] Tidak ada data share yang dihasilkan. Cek CSV atau mapping intro/fixer.")
        return

    # ===== Filter NON–SELF-FIXED (introducer != fixer) bila diaktifkan =====
    if FILTER_NON_SELF_FIXED_ONLY:
        before = len(shares_df)
        shares_df = shares_df[~shares_df["Self_Fixed_Same_Author"]].copy()
        after = len(shares_df)
        print("[*] FILTER_NON_SELF_FIXED_ONLY = True → hanya non–self-fixed (introducer != fixer).")
        print(f"    Sebelum filter self-fixed: {before} baris, sesudah: {after} baris.")
        if shares_df.empty:
            print("[!] Semua baris terfilter sebagai self-fixed. Tidak ada non–self-fixed.")
            return
    else:
        print("[*] FILTER_NON_SELF_FIXED_ONLY = False → memakai semua (self-fixed dan non–self-fixed).")
    # =======================================================================

    # Filter: hanya file dengan MID?
    if FILTER_REQUIRE_MID:
        before = len(shares_df)
        shares_df = shares_df[shares_df["Has_MID"]].copy()
        after = len(shares_df)
        print(f"[*] FILTER_REQUIRE_MID = True → hanya file dengan MID.")
        print(f"    Sebelum filter MID: {before} baris, sesudah: {after} baris.")
        if shares_df.empty:
            print("[!] Semua baris terbuang setelah filter MID. Tidak ada yang bisa dianalisis.")
            return
    else:
        print("[*] FILTER_REQUIRE_MID = False → memakai semua file (Has_MID dan No_MID).")

    # Simpan CSV hasil
    save_shares_csv(shares_df, OUT_CSV)

    # Statistik singkat
    print("\n[*] Statistik singkat (setelah semua filter):")
    for col in ["Share_Intro", "Share_Fix", "Share_Others"]:
        print(f"\n{col}:")
        print(shares_df[col].describe())

    print("\n[*] Distribusi Has_MID / No_MID di data akhir:")
    print(shares_df[["Has_MID", "No_MID"]].value_counts(dropna=False))

    # Plot boxplot
    plot_boxplot(shares_df, OUT_PNG, OUT_PDF)


if __name__ == "__main__":
    main()
