"""
ISIMIP2b recharge maps — one 3x2 figure per trend period.

For each period (2008-2050, 2050-2099, 2008-2099):
  a) Average δPS recharge in RCP6.0           (mm/year)
  b) Average GHM-ensemble recharge in RCP6.0  (mm/year)
  c) δPS recharge trend in RCP6.0             (mm/year/year)
  d) GHM-ensemble recharge trend in RCP6.0
  e) δPS recharge trend in RCP8.5
  f) GHM-ensemble recharge trend in RCP8.5

Loads pre-computed yearly recharge ensembles from the saved .npz, computes
per-period linear trends with calculate_yearly_trends, plots with
plot_multiple_shapefiles. Zone numbers (new_FID) are annotated on panel f.
"""

from pathlib import Path
import json
import os

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Point

from post.stat_plots import plot_multiple_shapefiles
from post.read_GHMs_dPLs import calculate_yearly_trends


# ---------------------------------------------------------------------------
# Configuration — edit BASE_DIR once; everything else is derived.
# ---------------------------------------------------------------------------
BASE_DIR    = Path("D:/DR")
DATA_DIR    = BASE_DIR / "data"
M_DIR       = BASE_DIR / "M"
FIG_OUT_DIR = BASE_DIR / "evaluation_figures"
FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARLY_NPZ     = M_DIR / "qr_2b" / "sim_recharge_dict_yearly_2007_2100.npz"
ATTR_2003_NPY  = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024.npy"
ATTR_2003_JSON = DATA_DIR / "ts_2003basins" / "attr2003_mswep_03122024_name.json"
ZONES_SHP      = DATA_DIR / "Zones" / "Zones_0228.shp"

# The saved .npz starts at 2007 → index 1 is 2008. We slice 2008..2099 (92 yrs).
NPZ_START_YEAR = 2007
SLICE_START    = 2008
SLICE_END      = 2099

PERIODS = [(2008, 2050), (2050, 2099), (2008, 2099)]

MODELS = [
    "dPL_ens_mSS_rcp60_yearly",   # δPS RCP6.0
    "GHM_ens_rcp60_yearly",       # GHM ensemble RCP6.0
    "dPL_ens_mSS_rcp85_yearly",   # δPS RCP8.5
    "GHM_ens_rcp85_yearly",       # GHM ensemble RCP8.5
]

PANEL_TITLES = [
    "a) Average recharge in δPS in RCP6.0",
    "b) Average recharge in GHM ensemble in RCP6.0",
    "c) Recharge trend in δPS in RCP6.0",
    "d) Recharge trend in GHMs ensemble in RCP6.0",
    "e) Recharge trend in δPS in RCP8.5",
    "f) Recharge trend in GHMs ensemble in RCP8.5",
]

# Hand-tuned label positions for the 12 zones (annotated on panel f)
ZONE_ANNOTATE_XY = [
    (-70, 38),    (-78, 30),    (-82, 47),    (-90, 27.5),
    (-94, 47),    (-102, 27),   (-106, 48),   (-107.8, 30),
    (-118.2, 48), (-113, 30.5), (-123, 50),   (-123, 34),
]

MEAN_CMAP   = plt.cm.plasma
MEAN_RANGE  = [50, 600]
TREND_CMAP  = plt.cm.seismic.reversed()
TREND_RANGE = [-2, 2]


# ---------------------------------------------------------------------------
# 1. Load yearly recharge ensembles, slice to 2008–2099
# ---------------------------------------------------------------------------
print(f"Loading: {YEARLY_NPZ}")
loaded = np.load(YEARLY_NPZ)

i0 = SLICE_START - NPZ_START_YEAR              # 1
i1 = SLICE_END   - NPZ_START_YEAR + 1          # 93
sim_yearly = {m: np.array(loaded[m][i0:i1, :]) for m in MODELS}
loaded.close()


# ---------------------------------------------------------------------------
# 2. Per-period sliced data + linear trends
# ---------------------------------------------------------------------------
period_data   = {}   # {"2008 - 2099": {model: (T, n_basins)}}
period_slopes = {}   # {"2008 - 2099": {model: (n_basins,)}}

for y1, y2 in PERIODS:
    pkey  = f"{y1} - {y2}"
    nyrs  = y2 - y1 + 1
    start = y1 - SLICE_START
    data  = {m: sim_yearly[m][start:start + nyrs, :] for m in MODELS}
    period_data[pkey] = data

    slopes, _ = calculate_yearly_trends(
        yearly_data_dict=data,
        start_date=str(y1),
        end_date=str(y2 + 1),
        flow_obs_daily=None,
        flow_percentage_availability=0.0,
        consider_obs_flow_percentage=False,
    )
    period_slopes[pkey] = slopes
    print(f"  trends computed for {pkey}")


# ---------------------------------------------------------------------------
# 3. Spatial setup — basin points + zone polygons
# ---------------------------------------------------------------------------
with open(ATTR_2003_JSON, "r") as f:
    a_name = json.load(f)

attr2003 = np.load(ATTR_2003_NPY)
lat_all  = attr2003[:, a_name.index("lat")]
lon_all  = attr2003[:, a_name.index("lon")]

gdf_points = gpd.GeoDataFrame(
    geometry=[Point(lon, lat) for lat, lon in zip(lat_all, lon_all)],
    crs="EPSG:4326",
)
shapefile = gpd.read_file(ZONES_SHP).to_crs(gdf_points.crs)

# Pre-compute zone → basin-index mapping once (it's the same for every panel)
zone_index = {}
for i, polygon_row in shapefile.iterrows():
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)
    zone_index[i] = gpd.sjoin(
        gdf_points, poly_gdf, how="inner", predicate="intersects",
    ).index.tolist()


def _split_by_zone(values):
    """Split a (n_basins,) array into per-zone {zone_idx: subarray} dicts."""
    data_dict, lat_dict, lon_dict = {}, {}, {}
    for i, idx in zone_index.items():
        data_dict[i] = values[idx]
        lat_dict[i]  = lat_all[idx]
        lon_dict[i]  = lon_all[idx]
    return data_dict, lat_dict, lon_dict


# ---------------------------------------------------------------------------
# 4. Plot one 3x2 figure per period
# ---------------------------------------------------------------------------
for pkey in period_data:
    fig, axs = plt.subplots(3, 2, figsize=(24, 24),
                            subplot_kw={"projection": ccrs.PlateCarree()})
    ax = axs.flatten()
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95,
                        wspace=0.01, hspace=0.01)

    # Panels a, b — period-mean recharge (mm/year)
    for j, model in enumerate(MODELS[:2]):
        mean_vals = np.nanmean(period_data[pkey][model], axis=0).squeeze()
        dd, ld, lod = _split_by_zone(mean_vals)
        ax[j] = plot_multiple_shapefiles(
            dd, shapefile, ld, lod,
            point_colors=MEAN_CMAP, colorbar_range=MEAN_RANGE,
            title=PANEL_TITLES[j], ax=ax[j],
        )

    # Panels c, d, e, f — period trends (mm/year/year)
    for j, model in enumerate(MODELS):
        slopes = np.array(period_slopes[pkey][model])
        dd, ld, lod = _split_by_zone(slopes)
        ax[j + 2] = plot_multiple_shapefiles(
            dd, shapefile, ld, lod,
            point_colors=TREND_CMAP, colorbar_range=TREND_RANGE,
            title=PANEL_TITLES[j + 2], ax=ax[j + 2], show_cbar=True,
        )

    # Zone-number annotations on panel f
    bbox = dict(facecolor="white", alpha=0.7, edgecolor="None")
    for i, polygon_row in shapefile.iterrows():
        x_lab, y_lab = ZONE_ANNOTATE_XY[i]
        ax[5].text(x_lab, y_lab, str(polygon_row["new_FID"]),
                   fontsize=30, ha="center", va="center",
                   color="black", bbox=bbox)

    fig.suptitle(
        f"Average recharge (mm/year) and recharge trends (mm/year/year) "
        f"prediction between {pkey}",
        fontsize=31, y=0.985,
    )

    y1, y2 = pkey.replace(" ", "").split("-")
    out_png = FIG_OUT_DIR / f"map1_shp_recharge_dPL_GHM_2b_ens_rcp85_rcp60_{y1}_{y2}.png"
    plt.savefig(out_png, dpi=600)
    plt.close("all")
    print(f"Saved: {out_png}")

print("END")