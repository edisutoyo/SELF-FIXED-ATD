# km_rq3_for_sample_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

# ========= CONFIG: sesuaikan jika perlu =========
CSV_PATH = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/ATD-FINAL-DATASET-TRACED.csv"

COL_ISSUE_ID   = "Key"
COL_INTRO_DT   = "Intro Date"       # momen introduksi ATD
COL_FIX_DT     = "Payment Date"     # NaN/blank jika belum fixed (censored)
COL_SELF_FLAG  = "Is Self-Fixed (Intro=Payment)"  # bool/1/0/yes/no

TIMEZONE       = None  # contoh: "Europe/Amsterdam" jika datetime naive perlu dilokal-kan
ZOOM_DAYS      = 30

# (opsional) warna mendekati ggplot2 di figure artikel
COL_SF  = "#00BFC4"  # Self-fixed
COL_NSF = "#F8766D"  # Non self-fixed
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
    return s.isin(["1","true","t","yes","y"])

def to_datetime(series: pd.Series, tz=None) -> pd.Series:
    # parse ke UTC (aman), jika diminta convert/lokalize ke tz
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if tz:
        try:
            s = s.dt.tz_convert(tz)
        except Exception:
            s2 = pd.to_datetime(series, errors="coerce")
            s = s2.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    return s

# 1) Load & preprocess
df = pd.read_csv(CSV_PATH)

# Pastikan kolom ada
required_cols = [COL_ISSUE_ID, COL_INTRO_DT, COL_FIX_DT, COL_SELF_FLAG]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Kolom hilang di CSV: {missing}")

df["_intro_dt"]   = to_datetime(df[COL_INTRO_DT], tz=TIMEZONE)
df["_fix_dt"]     = to_datetime(df[COL_FIX_DT], tz=TIMEZONE)
df["_self_fixed"] = to_bool(df[COL_SELF_FLAG])

# Buang baris tanpa intro date
df = df[~df["_intro_dt"].isna()].copy()

# Event observed: 1 jika ada fix date, 0 jika NaT (censored)
df["_event"] = (~df["_fix_dt"].isna()).astype(int)

# Horizon sensor konservatif (maks waktu teramati)
max_obs = pd.concat([df["_intro_dt"], df["_fix_dt"]]).max()

# Durasi (hari): fixed -> fix-intro; censored -> max_obs-intro
def _dur_days(row):
    end = row["_fix_dt"] if row["_event"] == 1 else max_obs
    return (end - row["_intro_dt"]).total_seconds() / 86400.0

df["_duration_days"] = df.apply(_dur_days, axis=1)
df.loc[df["_duration_days"] < 1e-9, "_duration_days"] = 1e-9  # jaga dari 0/negatif

# 2) Split grup
mask_sf  = df["_self_fixed"] == True
mask_nsf = df["_self_fixed"] == False

T_sf,  E_sf  = df.loc[mask_sf,  "_duration_days"], df.loc[mask_sf,  "_event"]
T_nsf, E_nsf = df.loc[mask_nsf, "_duration_days"], df.loc[mask_nsf, "_event"]

kmf_sf  = KaplanMeierFitter(label="Self-fixed")
kmf_nsf = KaplanMeierFitter(label="Non self-fixed")

kmf_sf.fit(T_sf, event_observed=E_sf)
kmf_nsf.fit(T_nsf, event_observed=E_nsf)

# Helper plot cumulative fixed (= 1 - S(t)) sebagai %
# def plot_cumulative_fixed(ax, kmf, color=None):
#     surv = kmf.survival_function_
#     cum_fixed = (1.0 - surv) * 100.0
#     ax.plot(cum_fixed.index.values, cum_fixed.values, label=kmf._label, color=color)

def plot_cumulative_fixed(ax, kmf, color=None):
    surv = kmf.survival_function_
    cum_fixed = (1.0 - surv) * 100.0
    ax.plot(
        cum_fixed.index.values,
        cum_fixed.values,
        label=kmf._label,
        color=color,
        linewidth=LINEWIDTH   # <— garis lebih tebal
    )


def finalize_axes(ax, title=None, xlim=None):
    ax.set_xlabel("Days since introduction")
    ax.set_ylabel("Cumulative fixed (%)")
    if xlim:
        ax.set_xlim(0, xlim)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.4)
    ax.legend()

# 3) FULL figure (satu plot per figure + risk table)
fig_full, ax_full = plt.subplots(figsize=(9, 6))
plot_cumulative_fixed(ax_full, kmf_sf,  color=COL_SF)
plot_cumulative_fixed(ax_full, kmf_nsf, color=COL_NSF)
finalize_axes(ax_full, title="Kaplan-Meier Cumulative fixed (Full)")
plt.title("Kaplan-Meier Cumulative fixed (Full)")
# risk table memakai lifelines (berbasis S(t), tetap valid)
# add_at_risk_counts(kmf_sf, kmf_nsf, ax=ax_full)
add_at_risk_counts(
    kmf_sf, kmf_nsf,
    ax=ax_full,
    rows_to_show=["At risk", "Events"]   # hilangkan "Censored"
)




fig_full.tight_layout()
fig_full.savefig("KM_RQ2_full.png", dpi=150, bbox_inches="tight")
fig_full.savefig("KM_RQ2_full.pdf", bbox_inches="tight")

# 4) ZOOM ≤ 30 hari
fig_zoom, ax_zoom = plt.subplots(figsize=(9, 6))
plot_cumulative_fixed(ax_zoom, kmf_sf,  color=COL_SF)
plot_cumulative_fixed(ax_zoom, kmf_nsf, color=COL_NSF)
finalize_axes(ax_zoom, title=f"KM - Cumulative fixed (≤ {ZOOM_DAYS} days)", xlim=ZOOM_DAYS)
plt.title(f"Kaplan-Meier Cumulative fixed (≤ {ZOOM_DAYS} days)")
# add_at_risk_counts(kmf_sf, kmf_nsf, ax=ax_zoom)

xt = np.arange(0, ZOOM_DAYS + 1, 7)   # 0, 7, 14, 21, 28
ax_zoom.set_xticks(xt)
ax_zoom.set_xticklabels([str(int(x)) for x in xt])  # pastikan label bilangan bulat


add_at_risk_counts(
    kmf_sf, kmf_nsf,
    ax=ax_zoom,
    rows_to_show=["At risk", "Events"]   # hilangkan "Censored"
)
fig_zoom.tight_layout()
fig_zoom.savefig("KM_RQ2_30days-week.png", dpi=150, bbox_inches="tight")
fig_zoom.savefig("KM_RQ2_30days-week.pdf", bbox_inches="tight")

print("Saved: KM_RQ3_full.(png|pdf), KM_RQ3_30days.(png|pdf)")
