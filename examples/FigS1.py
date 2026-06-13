"""
qtot_STemp.png — two-row evaluation of GHM/δ streamflow and stream
temperature performance.

Row 1 (spans both columns): Streamflow KGE — 15 GHMs (GSWP3-forced),
  3 δ models on GSWP3 (ISIMIP2a), 3 δ models on Daymet, plus ensembles.
Row 2 left: Stream temperature NSE — δ models on Daymet + GSWP3.
Row 2 right: Stream temperature RMSE — same models.

NOTE on naming. The TEMPERATURE reads use one function
(read_dPL_Daymet_daily_general) for both forcings, distinguished by the
simulation period (1980-2023 = Daymet, 1961-2011 = GSWP3). This means
the temp-dict key `mSS_daily` is Daymet, and `mSS_GSWP3_daily` is GSWP3.
The STREAMFLOW dict uses the opposite convention: `mSS_daily` is GSWP3
(ISIMIP2a) and `mSS_Daymet_daily` is Daymet. Both label dicts below
encode the per-context labeling.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from post.stat_plots import plotBoxFig, statError
from post.read_GHMs_dPLs import (
    read_GHM_ISIMIP2a_daily, read_dPL_ISIMIP2a_daily,
    read_dPL_Daymet_daily_general,
)


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# ---------------------------------------------------------------------------
BASE_DIR    = Path("D:/DR")
DATA_DIR    = BASE_DIR / "data"
M_DIR       = BASE_DIR / "M"
FIG_OUT_DIR = BASE_DIR / "evaluation_figures"
QTOT_CACHE  = M_DIR / "qtot" / "sim_qtot_dict.npz"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
QTOT_CACHE.parent.mkdir(parents=True, exist_ok=True)

DPL_DIR        = M_DIR / "daymet_1223_1023_PUB"
ATTR_2003_NPY  = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024.npy"
ATTR_2003_JSON = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024_name.json"
ZONES_SHP      = DATA_DIR / "Zones" / "Zones_0228.shp"
CPI_FEATHER    = M_DIR / "upscale" / "CPI" / "grid_gage_2003_CPI_intersect.feather"
MIN_CPI        = 0.33

# Streamflow evaluation window
FLOW_START, FLOW_END = "2000-01-01", "2009-12-31"
# Stream temperature evaluation window
TEMP_START, TEMP_END = "1981-01-01", "2009-12-31"

GHM_MODELS = ["Watergap2_2c", "WAYS", "CLM40", "DBH", "H08", "JULES_B1",
              "JULES_W1", "LPJmL", "MATSIRO", "MPI_HM", "ORCHIDEE",
              "PCR_GLOBWB", "VIC", "WATERGAP2_ISIMIP2a", "WEB_DHM_SG"]
DPL_MODELS = ["mSS", "HyS", "HVS"]      # integrated δ models only


# Styling --------------------------------------------------------------------
COLORS = {
    "CLM40": "pink",               "MATSIRO": "magenta",
    "H08": "orange",               "LPJmL": "khaki",
    "JULES-W1": "darkkhaki",       "WATERGAP2-2C": "lightyellow",
    "PCR-GLOBWB": "lavender",      "WAYS": "mediumspringgreen",
    "MPI-HM": "thistle",           "JULES-B1": "cadetblue",
    "ORCHIDEE": "darkcyan",        "DBH": "olive",
    "VIC": "bisque",               "WATERGAP2-ISIMIP2a": "peru",
    "WEB-DHM-SG": "plum",          "GHMs ens": "grey",
    # δ models (any forcing); same color used for solid fill and edge
    "δPS": "blue",  "δHcS": "green",  "δHS": "red",  "δ ens": "black",
    "δPS (Daymet)": "navy",        "δHcS (Daymet)": "darkolivegreen",
    "δHS (Daymet)": "darkred",     "δ ens (Daymet)": "gray",
}

# sim_qtot key → display label (and implicit box order)
FLOW_LABELS = {
    "Watergap2_2c_daily":       "WATERGAP2-2C",
    "WAYS_daily":               "WAYS",
    "CLM40_daily":              "CLM40",
    "DBH_daily":                "DBH",
    "H08_daily":                "H08",
    "JULES_B1_daily":           "JULES-B1",
    "JULES_W1_daily":           "JULES-W1",
    "LPJmL_daily":              "LPJmL",
    "MATSIRO_daily":            "MATSIRO",
    "MPI_HM_daily":             "MPI-HM",
    "ORCHIDEE_daily":           "ORCHIDEE",
    "PCR_GLOBWB_daily":         "PCR-GLOBWB",
    "VIC_daily":                "VIC",
    "WATERGAP2_ISIMIP2a_daily": "WATERGAP2-ISIMIP2a",
    "WEB_DHM_SG_daily":         "WEB-DHM-SG",
    "GHM_ens_daily":            "GHMs ens",
    "mSS_daily":                "δPS",          # GSWP3 (ISIMIP2a)
    "HyS_daily":                "δHcS",
    "HVS_daily":                "δHS",
    "dPL_ens_daily":            "δ ens",
    "mSS_Daymet_daily":         "δPS (Daymet)",
    "HyS_Daymet_daily":         "δHcS (Daymet)",
    "HVS_Daymet_daily":         "δHS (Daymet)",
    "dPL_ens_Daymet_daily":     "δ ens (Daymet)",
}

# sim_temp key → display label
TEMP_LABELS = {
    "mSS_daily":       "δPS (Daymet)",
    "HyS_daily":       "δHcS (Daymet)",
    "HVS_daily":       "δHS (Daymet)",
    "mSS_GSWP3_daily": "δPS",
    "HyS_GSWP3_daily": "δHcS",
    "HVS_GSWP3_daily": "δHS",
}


def flow_box_style(label):
    """Any δ model: white fill + thick colored edge. GHMs: solid fill."""
    if "δ" in label:
        return "whitesmoke", COLORS[label], 6.0
    return COLORS[label], "black", 0.6


def temp_box_style(label):
    """δ-Daymet only gets the white-fill thick-edge treatment."""
    if "δ" in label and "Daymet" in label:
        return "whitesmoke", COLORS[label], 4.0
    return COLORS[label], "black", 0.6


def legend_patches(faces, edges, lws, labels):
    return [
        FancyBboxPatch(
            (0, 0), 1, 1, facecolor=f, edgecolor=e, linewidth=w,
            boxstyle="square,pad=0.1", label=lab,
        )
        for f, e, w, lab in zip(faces, edges, lws, labels)
    ]


# ---------------------------------------------------------------------------
# 1. Basin attribute table
# ---------------------------------------------------------------------------
with open(ATTR_2003_JSON, "r") as f:
    a_name = json.load(f)

attr2003     = np.load(ATTR_2003_NPY)
site_no_2003 = attr2003[:, a_name.index("site_no_int")]


# ---------------------------------------------------------------------------
# 2. CPI intersection — gauge basins whose best GHM grid cell has CPI > 0.33
# ---------------------------------------------------------------------------
cpi_df = pd.read_feather(CPI_FEATHER)
cpi_df = cpi_df.loc[cpi_df["CPI"] > MIN_CPI].reset_index(drop=True)
cpi_df["points_GHM"] = list(zip(cpi_df["lat_grid_GHM"], cpi_df["lon_grid_GHM"]))

ind_sites, lat_grid, lon_grid = [], [], []
for p in set(cpi_df["points_GHM"]):
    sub = cpi_df.loc[cpi_df["points_GHM"] == p]
    row = sub.loc[sub["CPI"] == sub["CPI"].max()].iloc[0]
    ind = np.where(site_no_2003 == row["site_no_int"])[0]
    if not len(ind):
        continue
    ind_sites.append(int(ind[0]))
    lat_grid.append(p[0])
    lon_grid.append(p[1])
print(f"CPI-intersected basins: {len(ind_sites)}")


# ---------------------------------------------------------------------------
# 3. Streamflow data — cached to QTOT_CACHE
# ---------------------------------------------------------------------------
if QTOT_CACHE.exists():
    print(f"Loading streamflow cache: {QTOT_CACHE}")
    with np.load(QTOT_CACHE) as data:
        sim_qtot = {k: data[k] for k in data.files}
else:
    print("Building streamflow cache from scratch...")
    sim_qtot = {}

    # GHM models — GSWP3 forcing
    for m in GHM_MODELS:
        sim_qtot[m + "_daily"] = read_GHM_ISIMIP2a_daily(
            item_name="qtot",
            start_date=FLOW_START, end_date=FLOW_END,
            lat_list=lat_grid, lon_list=lon_grid,
            model_name_GHM=m, CLM_name="GSWP3",
        )
        print(m)
    sim_qtot["GHM_ens_daily"] = np.mean(
        np.stack([sim_qtot[m + "_daily"] for m in GHM_MODELS]), axis=0
    )
    print("GHM_ens")

    # δ models — GSWP3 (ISIMIP2a)
    obs_flow = None
    for m in DPL_MODELS:
        flow_sim, _, obs_flow = read_dPL_ISIMIP2a_daily(
            model_name=m, site_ind_list=ind_sites,
            start_date=FLOW_START, end_date=FLOW_END,
            include_ssflow=False, read_obs_flow_flag=True,
            dir0=str(DPL_DIR),
        )
        sim_qtot[m + "_daily"] = flow_sim
        print(m + " (GSWP3)")
    sim_qtot["dPL_ens_daily"] = np.mean(
        np.stack([sim_qtot[m + "_daily"] for m in DPL_MODELS]), axis=0
    )
    sim_qtot["obs_flow_daily"] = obs_flow
    print("dPL_ens (GSWP3)")

    # δ models — Daymet forcing
    for m in DPL_MODELS:
        sim_qtot[m + "_Daymet_daily"] = read_dPL_Daymet_daily_general(
            item="flow_sim", model_name=m,
            start_date_mask=FLOW_START, end_date_mask=FLOW_END,
            site_ind_list=ind_sites,
            start_date_sim="1980-01-01", end_date_sim="2023-01-01",
            dir0=str(DPL_DIR),
        )
        print(m + " (Daymet)")
    sim_qtot["dPL_ens_Daymet_daily"] = np.mean(
        np.stack([sim_qtot[m + "_Daymet_daily"] for m in DPL_MODELS]), axis=0
    )
    print("dPL_ens (Daymet)")

    np.savez_compressed(QTOT_CACHE, **sim_qtot)
    print(f"Saved: {QTOT_CACHE}")


# ---------------------------------------------------------------------------
# 4. Stream temperature — δ models on Daymet (1980-2023) and GSWP3 (1961-2011)
# ---------------------------------------------------------------------------
sim_temp = {}

for m in DPL_MODELS:
    sim_temp[m + "_daily"] = read_dPL_Daymet_daily_general(
        item="temp_sim", model_name=m,
        start_date_mask=TEMP_START, end_date_mask=TEMP_END,
        site_ind_list=np.arange(2003),
        start_date_sim="1980-01-01", end_date_sim="2023-01-01",
        dir0=str(DPL_DIR),
    )
sim_temp["temp_obs_daily"] = read_dPL_Daymet_daily_general(
    item="00010_Mean", model_name="mSS",
    start_date_mask=TEMP_START, end_date_mask=TEMP_END,
    site_ind_list=np.arange(2003),
    start_date_sim="1980-01-01", end_date_sim="2023-01-01",
    dir0=str(DPL_DIR),
)
for m in DPL_MODELS:
    sim_temp[m + "_GSWP3_daily"] = read_dPL_Daymet_daily_general(
        item="temp_sim", model_name=m,
        start_date_mask=TEMP_START, end_date_mask=TEMP_END,
        site_ind_list=np.arange(2003),
        start_date_sim="1961-01-01", end_date_sim="2011-01-01",
        dir0=str(DPL_DIR),
    )


# ---------------------------------------------------------------------------
# 5. CONUS index — basins inside any zone polygon (for temp panels only)
# ---------------------------------------------------------------------------
lat_all = attr2003[:, a_name.index("lat")]
lon_all = attr2003[:, a_name.index("lon")]
gdf_points = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in zip(lat_all, lon_all)],
    crs="EPSG:4326",
)
shapefile = gpd.read_file(ZONES_SHP).to_crs(gdf_points.crs)
conus_idx = sorted(gpd.sjoin(gdf_points, shapefile, how="inner",
                             predicate="intersects").index.unique().tolist())


# ---------------------------------------------------------------------------
# 6. Figure: 2x2 gridspec, top row spans both columns
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(24, 16))
plt.subplots_adjust(left=0.06, bottom=0.03, right=0.99, top=0.83,
                    wspace=0.13, hspace=0.24)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

# ---- a) Streamflow KGE (top row) ------------------------------------------
ax1 = fig.add_subplot(gs[0, :])
obs_daily = sim_qtot["obs_flow_daily"]
boxes, faces, edges, lws, labels = [], [], [], [], []
for key, label in FLOW_LABELS.items():
    if key not in sim_qtot:
        continue
    stats = statError(np.swapaxes(sim_qtot[key], 1, 0),
                      np.swapaxes(obs_daily, 1, 0))
    boxes.append(stats["KGE"])
    f, e, w = flow_box_style(label)
    faces.append(f); edges.append(e); lws.append(w); labels.append(label)

ax1 = plotBoxFig([boxes], label1=["a) Streamflow KGE"],
                 colorLst=faces, edgecolor_list=edges,
                 label1_font_size=32, sharey=False, figsize=(12, 5), axin=ax1,
                 add_horizontal_line=False, widths=0.6,
                 line_width_list=lws, ylim=[-0.6, 1.0])
ax1[0].tick_params(axis="y", labelsize=28)
ax1[0].legend(
    handles=legend_patches(faces, edges, lws, labels),
    loc="upper center", frameon=False, ncol=6,
    bbox_to_anchor=(0.49, 1.38), fontsize=24,
)

# ---- b) NSE and c) RMSE for stream temperature ----------------------------
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

temp_obs = sim_temp["temp_obs_daily"][:, conus_idx]
nse_box, rmse_box = [], []
faces, edges, lws, labels = [], [], [], []
for key, label in TEMP_LABELS.items():
    if key not in sim_temp:
        continue
    stats = statError(np.swapaxes(sim_temp[key][:, conus_idx], 1, 0),
                      np.swapaxes(temp_obs, 1, 0))
    nse_box.append(stats["NSE"])
    rmse_box.append(stats["RMSE"])
    f, e, w = temp_box_style(label)
    faces.append(f); edges.append(e); lws.append(w); labels.append(label)

ax2 = plotBoxFig([nse_box], label1=["b) Stream temperature NSE"],
                 colorLst=faces, edgecolor_list=edges,
                 label1_font_size=28, sharey=False, figsize=(12, 5), axin=ax2,
                 add_horizontal_line=False, widths=0.6,
                 line_width_list=lws, ylim=None)
ax2[0].tick_params(axis="y", labelsize=28)

ax3 = plotBoxFig([rmse_box], label1=["c) Stream temperature RMSE"],
                 colorLst=faces, edgecolor_list=edges,
                 label1_font_size=28, sharey=False, figsize=(12, 5), axin=ax3,
                 add_horizontal_line=False, widths=0.6,
                 line_width_list=lws, ylim=None)
ax3[0].tick_params(axis="y", labelsize=28)

ax2[0].legend(
    handles=legend_patches(faces, edges, lws, labels),
    loc="upper center", frameon=False, ncol=6,
    bbox_to_anchor=(1, 1.16), fontsize=28,
)

fig.patch.set_facecolor("white")
fig.suptitle(
    "Streamflow and stream temperature performance of GHMs and δ models",
    fontsize=34, y=0.985,
)

out_png = FIG_OUT_DIR / "qtot_STemp.png"
plt.savefig(out_png, dpi=600)
plt.close("all")
print(f"Saved: {out_png}")
print("END")