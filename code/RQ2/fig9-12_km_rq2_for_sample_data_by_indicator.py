# km_viomod_self_vs_nonself.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

# ========= CONFIG: sesuaikan jika perlu =========
CSV_PATH = "/media/edsu/Drive-D2/Python/SEL-FIXED-ATD/ATD-FINAL-DATASET-TRACED.csv"

COL_ISSUE_ID    = "Key"
COL_INTRO_DT    = "Intro Date"
COL_FIX_DT      = "Payment Date"
COL_SELF_FLAG   = "Is Self-Fixed (Intro=Payment)"
COL_INDICATOR   = "indicator"          # berisi viomod/obstech

TIMEZONE        = None                  # contoh: "Europe/Amsterdam"
ZOOM_DAYS       = 30

# Warna
COL_SF  = "#00BFC4"   # Self-fixed
COL_NSF = "#F8766D"   # Non self-fixed
# ================================================

# === Global style: bigger fonts ===
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14
})
LINEWIDTH = 3.0

def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1","true","t","yes","y"])

def to_datetime(series: pd.Series, tz=None) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if tz:
        try:
            s = s.dt.tz_convert(tz)
        except Exception:
            s2 = pd.to_datetime(series, errors="coerce")
            s = s2.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    return s

# 1) Load & cek kolom
df = pd.read_csv(CSV_PATH)
required = [COL_ISSUE_ID, COL_INTRO_DT, COL_FIX_DT, COL_SELF_FLAG, COL_INDICATOR]
miss = [c for c in required if c not in df.columns]
if miss:
    raise ValueError(f"Kolom hilang di CSV: {miss}")

# Normalisasi indikator & filter VIOMOD saja
label_map = {
    "viomod": "viomod", "vio-mod": "viomod", "vio_mod": "viomod",
    "obstech": "obstech", "obs-tech": "obstech", "obs_tech": "obstech",
}

df["_indicator"] = df[COL_INDICATOR].astype(str).str.strip().str.lower().map(lambda x: label_map.get(x, x))
df = df[df["_indicator"] == "viomod"].copy()

# Parse tanggal & self-fixed flag
df["_intro_dt"]   = to_datetime(df[COL_INTRO_DT], tz=TIMEZONE)
df["_fix_dt"]     = to_datetime(df[COL_FIX_DT], tz=TIMEZONE)
df["_self_fixed"] = to_bool(df[COL_SELF_FLAG])

# Buang baris tanpa intro date
df = df[~df["_intro_dt"].isna()].copy()

# Event: 1 jika fixed, 0 jika censored
df["_event"] = (~df["_fix_dt"].isna()).astype(int)

# Horizon sensor konservatif
max_obs = pd.concat([df["_intro_dt"], df["_fix_dt"]]).max()

# Durasi hari
def _dur_days(row):
    end = row["_fix_dt"] if row["_event"] == 1 else max_obs
    return (end - row["_intro_dt"]).total_seconds() / 86400.0

df["_duration_days"] = df.apply(_dur_days, axis=1)
df.loc[df["_duration_days"] < 1e-9, "_duration_days"] = 1e-9

# 2) Split viomod → self vs non-self
mask_sf  = df["_self_fixed"] == True
mask_nsf = df["_self_fixed"] == False

T_sf,  E_sf  = df.loc[mask_sf,  "_duration_days"], df.loc[mask_sf,  "_event"]
T_nsf, E_nsf = df.loc[mask_nsf, "_duration_days"], df.loc[mask_nsf, "_event"]

if len(T_sf) == 0 or len(T_nsf) == 0:
    print(f"[WARN] Data tidak lengkap. viomod/self-fixed: {len(T_sf)}, viomod/non-self-fixed: {len(T_nsf)}")

kmf_sf  = KaplanMeierFitter(label="VioMod · Self-fixed")
kmf_nsf = KaplanMeierFitter(label="VioMod · Non self-fixed")
if len(T_sf)  > 0: kmf_sf.fit(T_sf,  event_observed=E_sf)
if len(T_nsf) > 0: kmf_nsf.fit(T_nsf, event_observed=E_nsf)

def plot_cumulative_fixed(ax, kmf, color=None):
    surv = kmf.survival_function_
    cum_fixed = (1.0 - surv) * 100.0
    ax.plot(cum_fixed.index.values, cum_fixed.values, label=kmf._label, color=color, linewidth=LINEWIDTH)

def finalize_axes(ax, title=None, xlim=None):
    ax.set_xlabel("Days since introduction")
    ax.set_ylabel("Cumulative fixed (%)")
    if xlim:
        ax.set_xlim(0, xlim)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

# 3) FULL figure + risk table
fig_full, ax_full = plt.subplots(figsize=(9, 6))
if len(T_sf)  > 0: plot_cumulative_fixed(ax_full, kmf_sf,  color=COL_SF)
if len(T_nsf) > 0: plot_cumulative_fixed(ax_full, kmf_nsf, color=COL_NSF)
finalize_axes(ax_full, title="Kaplan-Meier Cumulative fixed — VioMod")

kmfs = []
if len(T_sf)  > 0: kmfs.append(kmf_sf)
if len(T_nsf) > 0: kmfs.append(kmf_nsf)
if kmfs:
    add_at_risk_counts(*kmfs, ax=ax_full, rows_to_show=["At risk", "Events"])

fig_full.tight_layout()
# fig_full.savefig("KM_VIOMOD_self_vs_nonself_full.png", dpi=150, bbox_inches="tight")
fig_full.savefig("KM_VIOMOD_self_vs_nonself_full.pdf", bbox_inches="tight")

# 4) ZOOM ≤ 30 hari + risk table
fig_zoom, ax_zoom = plt.subplots(figsize=(9, 6))
if len(T_sf)  > 0: plot_cumulative_fixed(ax_zoom, kmf_sf,  color=COL_SF)
if len(T_nsf) > 0: plot_cumulative_fixed(ax_zoom, kmf_nsf, color=COL_NSF)
finalize_axes(ax_zoom, title=f"Kaplan-Meier Cumulative fixed — VioMod (≤ {ZOOM_DAYS} days)", xlim=ZOOM_DAYS)

xt = np.arange(0, ZOOM_DAYS + 1, 7)   # 0, 7, 14, 21, 28
ax_zoom.set_xticks(xt)
ax_zoom.set_xticklabels([str(int(x)) for x in xt])

if kmfs:
    add_at_risk_counts(*kmfs, ax=ax_zoom, rows_to_show=["At risk", "Events"])

fig_zoom.tight_layout()
# fig_zoom.savefig("KM_VIOMOD_self_vs_nonself_30days.png", dpi=150, bbox_inches="tight")
fig_zoom.savefig("KM_VIOMOD_self_vs_nonself_30days.pdf", bbox_inches="tight")

print("Saved: KM_VIOMOD_self_vs_nonself_full.(png|pdf), KM_VIOMOD_self_vs_nonself_30days.(png|pdf)")
