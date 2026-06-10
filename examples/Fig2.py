"""
Baseflow-trend scatter by zone, GHM ensemble + DBH + δPS (low-res and Daymet hi-res)
vs observations. Period: 1989-1999.

Produces a single 3x4 figure:
  bf_trend_hres_scatter_zones_dmt_GSWP3_zone0228_DBH.png

Flag LOAD_BF_FROM_DISK:
  True  -> skip the baseflow separation pipeline entirely; load cached
           per-model daily baseflow .npy files from BF_DIR.
  False -> run the full pipeline: read streamflow for the GHMs + dPL
           models we need, do baseflow separation, save to BF_DIR.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

from post.read_GHMs_dPLs import (
    read_GHM_ISIMIP2a_daily, do_baseflow_separation, read_dPL_ISIMIP2a_daily,
    converting_daily_to_yearly, calculate_yearly_trends, read_dPL_Daymet_daily,
)


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# ---------------------------------------------------------------------------
BASE_DIR    = Path("D:/DR")
DATA_DIR    = BASE_DIR / "data"
M_DIR       = BASE_DIR / "M"
BF_DIR      = M_DIR / "bf"                       # cached daily baseflow arrays
FIG_OUT_DIR = BASE_DIR / "evaluation_figures"
DPL_DIR     = M_DIR / "daymet_1223_1023_PUB"
CPI_DIR     = M_DIR / "upscale" / "CPI"

ATTR_NAME_JSON  = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024_name.json"
ATTR_TRAINED    = DATA_DIR / "tr_1223basins" / "attr1223_1023_daymet_20240826.npy"
ATTR_2003       = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024.npy"
ZONES_SHP       = DATA_DIR / "Zones" / "Zones_0228.shp"
CPI_SHP         = CPI_DIR / "merged_shpfile_GHMs.shp"
CPI_FEATHER     = CPI_DIR / "grid_gage_2003_CPI_intersect.feather"

BF_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Behavior flag --------------------------------------------------------
LOAD_BF_FROM_DISK = True   # True = skip baseflow separation, load cached .npy

# --- Run settings ---------------------------------------------------------
TR_START          = "1989-01-01"
TR_END            = "1999-12-31"
FLOW_AVAIL_THRESH = 0.90
MIN_CPI           = 0.33
BF_METHOD         = "Furey"
DATASET           = "1223"

GHM_MODELS = ["Watergap2_2c", "WAYS", "CLM40", "DBH", "H08", "JULES_B1", "JULES_W1",
              "LPJmL", "MATSIRO", "MPI_HM", "ORCHIDEE", "PCR_GLOBWB", "VIC",
              "WATERGAP2_ISIMIP2a", "WEB_DHM_SG"]

# The figure needs only these 5 daily baseflow arrays.
REQUIRED_BF_KEYS = [
    f"mSS_{BF_METHOD}_daily",
    f"mSS_Daymet_{BF_METHOD}_daily",
    f"GHM_ens_{BF_METHOD}_daily",
    f"DBH_{BF_METHOD}_daily",
    f"obs_flow_{BF_METHOD}_daily",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rel_L2(y_sim, y_obs):
    num = np.sqrt(np.nansum((y_sim - y_obs) ** 2))
    den = np.sqrt(np.nansum(y_obs ** 2))
    return num / den


# Significance-stars helper for legend labels
SIG_LEVEL_3 = 0.001
SIG_LEVEL_2 = 0.01
SIG_LEVEL_1 = 0.05
NS_LABEL    = "ns"
SHOW_P_VALUE = False


def p_to_stars(p):
    if p < SIG_LEVEL_3: return "***"
    if p < SIG_LEVEL_2: return "**"
    if p < SIG_LEVEL_1: return "*"
    return NS_LABEL


def format_sig_label(model_name, sim, obs, force_prefix=True):
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() < 3:
        prefix = model_name if (force_prefix and model_name) else ""
        return f"{prefix}NA, L2: NA, r2: NA"

    _, _, r, p, _ = scipy.stats.linregress(sim[mask], obs[mask])
    sig    = p_to_stars(p)
    l2_val = rel_L2(sim[mask], obs[mask])
    r2_val = r ** 2
    prefix = model_name if force_prefix else ""

    if SHOW_P_VALUE:
        p_txt = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
        return f"{prefix}{sig}, p: {p_txt}, L2: {l2_val:.2f}, r2: {r2_val:.2f}"
    return f"{prefix}{sig}, L2: {l2_val:.2f}, r2: {r2_val:.2f}"


# ---------------------------------------------------------------------------
# 1. CPI intersection — always runs (cheap, needed for gdf_points + indexing)
# ---------------------------------------------------------------------------
with open(ATTR_NAME_JSON, "r") as f:
    a_name_np = json.load(f)

attr_trained_dt    = np.load(ATTR_TRAINED)
site_no_trained_dt = attr_trained_dt[:, a_name_np.index("site_no_int")]
attr2003           = np.load(ATTR_2003)
site_no_2003       = attr2003[:, a_name_np.index("site_no_int")]

cpi = pd.read_feather(CPI_FEATHER)
cpi = cpi.loc[cpi["CPI"] > MIN_CPI].reset_index(drop=True)
cpi["points_GHM"] = list(zip(cpi["lat_grid_GHM"], cpi["lon_grid_GHM"]))

intersecting = {k: [] for k in [
    "CPI_2003", "sites_2003_CPI_intersect", "ind_sites_2003_CPI_intersect",
    "lat_grid_2003", "lon_grid_2003",
]}

for p in set(cpi["points_GHM"]):
    sub = cpi.loc[cpi["points_GHM"] == p].reset_index(drop=True)
    row = sub.loc[sub["CPI"] == sub["CPI"].max()].reset_index(drop=True).iloc[0]

    intersecting["CPI_2003"].append(row["CPI"])
    intersecting["sites_2003_CPI_intersect"].append(row["site_no_int"])
    intersecting["ind_sites_2003_CPI_intersect"].append(
        int(np.where(site_no_2003 == row["site_no_int"])[0][0]))
    intersecting["lat_grid_2003"].append(p[0])
    intersecting["lon_grid_2003"].append(p[1])

site_ind_2003 = intersecting["ind_sites_2003_CPI_intersect"]
site_names    = intersecting["sites_2003_CPI_intersect"]


# ---------------------------------------------------------------------------
# 2. Load (or compute) baseflow arrays + obs_flow for the availability filter
# ---------------------------------------------------------------------------
def bf_path(key):
    return BF_DIR / f"{key}.npy"


sim_bf_dict = {}

if LOAD_BF_FROM_DISK:
    missing = [k for k in REQUIRED_BF_KEYS if not bf_path(k).exists()]
    if missing:
        raise FileNotFoundError(
            "LOAD_BF_FROM_DISK=True but these files are missing in "
            f"{BF_DIR}:\n  " + "\n  ".join(missing) +
            "\nSet LOAD_BF_FROM_DISK=False to compute them, or fix the path."
        )
    print(f"Loading cached baseflow arrays from {BF_DIR}")
    for key in REQUIRED_BF_KEYS:
        sim_bf_dict[key] = np.load(bf_path(key))

    # obs_flow time series is still needed by calculate_yearly_trends for the
    # FLOW_AVAIL_THRESH filter. Reading it from the dPL output is fast.
    _, _, obs_flow = read_dPL_ISIMIP2a_daily(
        model_name="mSS", site_ind_list=site_ind_2003,
        start_date=TR_START, end_date=TR_END,
        include_ssflow=False, read_obs_flow_flag=True, dir0=str(DPL_DIR))

else:
    print("Running full baseflow-separation pipeline...")

    # --- GHM individual models (needed for ensemble + DBH directly) -------
    for m in GHM_MODELS:
        key  = f"{m}_{BF_METHOD}_daily"
        path = bf_path(key)
        if path.exists():
            sim_bf_dict[key] = np.load(path)
            continue
        qtot = read_GHM_ISIMIP2a_daily(
            item_name="qtot", start_date=TR_START, end_date=TR_END,
            lat_list=intersecting["lat_grid_2003"],
            lon_list=intersecting["lon_grid_2003"],
            model_name_GHM=m, CLM_name="GSWP3")
        bf = do_baseflow_separation(streamflow=qtot, start_date=TR_START,
                                    end_date=TR_END, sites_name_cols_list=site_names,
                                    baseflow_sep_method=[BF_METHOD])
        sim_bf_dict[key] = bf[BF_METHOD].to_numpy()
        np.save(path, sim_bf_dict[key])
        print(f"  GHM {m} done")

    # --- GHM ensemble (mean across all GHM_MODELS) ------------------------
    ens_key  = f"GHM_ens_{BF_METHOD}_daily"
    ens_path = bf_path(ens_key)
    if ens_path.exists():
        sim_bf_dict[ens_key] = np.load(ens_path)
    else:
        stack = np.stack([sim_bf_dict[f"{m}_{BF_METHOD}_daily"] for m in GHM_MODELS])
        sim_bf_dict[ens_key] = np.mean(stack, axis=0)
        np.save(ens_path, sim_bf_dict[ens_key])
    print("  GHM ensemble done")

    # --- dPL mSS (ISIMIP2a GSWP3 forcing, low-res) ------------------------
    key = f"mSS_{BF_METHOD}_daily"
    if not bf_path(key).exists():
        flow_sim, _, obs_flow = read_dPL_ISIMIP2a_daily(
            model_name="mSS", site_ind_list=site_ind_2003,
            start_date=TR_START, end_date=TR_END,
            include_ssflow=False, read_obs_flow_flag=True, dir0=str(DPL_DIR))
        bf = do_baseflow_separation(streamflow=flow_sim, start_date=TR_START,
                                    end_date=TR_END, sites_name_cols_list=site_names,
                                    baseflow_sep_method=[BF_METHOD])
        sim_bf_dict[key] = bf[BF_METHOD].to_numpy()
        np.save(bf_path(key), sim_bf_dict[key])
    else:
        sim_bf_dict[key] = np.load(bf_path(key))
        _, _, obs_flow = read_dPL_ISIMIP2a_daily(
            model_name="mSS", site_ind_list=site_ind_2003,
            start_date=TR_START, end_date=TR_END,
            include_ssflow=False, read_obs_flow_flag=True, dir0=str(DPL_DIR))
    print("  dPL mSS done")

    # --- dPL mSS Daymet (high-res) ----------------------------------------
    key = f"mSS_Daymet_{BF_METHOD}_daily"
    if not bf_path(key).exists():
        flow_sim, _ = read_dPL_Daymet_daily(
            model_name="mSS", site_ind_list=site_ind_2003,
            start_date=TR_START, end_date=TR_END,
            include_ssflow=False, read_obs_flow_flag=False, dir0=str(DPL_DIR))
        bf = do_baseflow_separation(streamflow=flow_sim, start_date=TR_START,
                                    end_date=TR_END, sites_name_cols_list=site_names,
                                    baseflow_sep_method=[BF_METHOD])
        sim_bf_dict[key] = bf[BF_METHOD].to_numpy()
        np.save(bf_path(key), sim_bf_dict[key])
    else:
        sim_bf_dict[key] = np.load(bf_path(key))
    print("  dPL mSS Daymet done")

    # --- obs_flow baseflow -------------------------------------------------
    key = f"obs_flow_{BF_METHOD}_daily"
    if not bf_path(key).exists():
        bf = do_baseflow_separation(streamflow=obs_flow, start_date=TR_START,
                                    end_date=TR_END, sites_name_cols_list=site_names,
                                    baseflow_sep_method=[BF_METHOD])
        sim_bf_dict[key] = bf[BF_METHOD].to_numpy()
        np.save(bf_path(key), sim_bf_dict[key])
    else:
        sim_bf_dict[key] = np.load(bf_path(key))
    print("  obs_flow done")


# ---------------------------------------------------------------------------
# 3. Daily -> yearly, compute trends
# ---------------------------------------------------------------------------
sim_bf_dict_yearly = converting_daily_to_yearly(
    daily_data_dict=sim_bf_dict, start_date=TR_START, end_date=TR_END)

slope_dict, _ = calculate_yearly_trends(
    yearly_data_dict=sim_bf_dict_yearly,
    start_date=TR_START, end_date=TR_END,
    flow_obs_daily=obs_flow,
    flow_percentage_availability=FLOW_AVAIL_THRESH)


# ---------------------------------------------------------------------------
# 4. Zones & spatial points
# ---------------------------------------------------------------------------
gdf_points = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in zip(
        intersecting["lat_grid_2003"], intersecting["lon_grid_2003"])],
    crs="EPSG:4326",
)
shapefile = gpd.read_file(ZONES_SHP).to_crs(gdf_points.crs)


# ---------------------------------------------------------------------------
# 5. Plot — bf_trend_hres_scatter_zones_dmt_GSWP3_zone0228_DBH
# ---------------------------------------------------------------------------
COLORS  = ["black", "red", "orange", "blue"]
MARKERS = ["s", "o", "+", "*"]
ZONE7_LEGEND_FS  = 14
DEFAULT_LEGEND_FS = 17

slope_dPL     = np.array(slope_dict[f"mSS_{BF_METHOD}_yearly"])
slope_dPL_hr  = np.array(slope_dict[f"mSS_Daymet_{BF_METHOD}_yearly"])
slope_ghm_ens = np.array(slope_dict[f"GHM_ens_{BF_METHOD}_yearly"])
slope_dbh     = np.array(slope_dict[f"DBH_{BF_METHOD}_yearly"])
slope_obs     = np.array(slope_dict[f"obs_flow_{BF_METHOD}_yearly"])


def _plot_panel(ax_p, obs, sims, labels, force_prefix):
    """sims = [dPL_low, GHM_ens, DBH, dPL_high]; labels = matching name prefixes."""
    arrs = [obs] + sims
    mn, mx = np.nanmin(np.concatenate(arrs)), np.nanmax(np.concatenate(arrs))
    ax_p.plot([mn, mx], [mn, mx], lw=1.5, label=None)

    for i, (sim, name) in enumerate(zip(sims, labels)):
        lab = format_sig_label(name, sim, obs, force_prefix=force_prefix)
        ax_p.scatter(obs, sim, c=COLORS[i],
                     s=240 if i == 3 else 200,
                     marker=MARKERS[i], label=lab)
    ax_p.tick_params(axis="both", labelsize=21)


fig, axs = plt.subplots(3, 4, figsize=(24, 18))
ax = axs.flatten()

j = 0
zone_titles = []
for _, polygon_row in shapefile.iterrows():
    pgdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)
    idx = gpd.sjoin(gdf_points, pgdf, how="inner", predicate="intersects").index.tolist()
    print(len(idx))

    if len(idx) <= 10:
        continue

    zone_id = polygon_row["new_FID"]
    is_zone7 = (zone_id == 7)
    names = (["δPS-", "GHM(ens)-", "DBH-", "δPS(Daymet)-"]
             if is_zone7 else ["", "", "", ""])

    _plot_panel(
        ax[j],
        obs=slope_obs[idx],
        sims=[slope_dPL[idx], slope_ghm_ens[idx], slope_dbh[idx], slope_dPL_hr[idx]],
        labels=names,
        force_prefix=is_zone7,
    )
    ax[j].legend(loc="upper left",
                 fontsize=ZONE7_LEGEND_FS if is_zone7 else DEFAULT_LEGEND_FS,
                 frameon=True, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    zone_titles.append(f"Zone {zone_id}")
    j += 1

# Final panel: all zones combined
_plot_panel(
    ax[11],
    obs=slope_obs,
    sims=[slope_dPL, slope_ghm_ens, slope_dbh, slope_dPL_hr],
    labels=["", "", "", ""],
    force_prefix=False,
)
ax[11].legend(loc="upper left", fontsize=DEFAULT_LEGEND_FS, frameon=True,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.4)
zone_titles.append("All zones")

for i, a in enumerate(ax):
    if i < len(zone_titles):
        a.set_title(zone_titles[i], fontsize=22, pad=6)

fig.patch.set_facecolor("white")
fig.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.94,
                    wspace=0.20, hspace=0.15)
fig.text(0.5, 0.02, "Observed baseflow trend (mm/year/year)",
         ha="center", va="center", fontsize=29)
fig.text(0.02, 0.5, "Simulated baseflow trend (mm/year/year)",
         ha="center", va="center", rotation="vertical", fontsize=29)
fig.suptitle(
    f"Baseflow trend comparison of GHM and δ models in {TR_START[:4]}-{TR_END[:4]}",
    fontsize=30, y=0.985,
)

out_png = FIG_OUT_DIR / "bf_trend_hres_scatter_zones_dmt_GSWP3_zone0228_DBH.png"
plt.savefig(out_png, dpi=600)
plt.close("all")
print(f"Saved figure: {out_png}")
print("END")