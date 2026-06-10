"""
Per-zone boxplots of yearly recharge trends, comparing δPS dPL ensembles
against individual GHM-on-climate-model ensembles, for two RCPs and two
periods. Produces 4 figures:

  qr_trends_future_2b_zones_GHM_dPL_ens_rcp60_2008_2050.png
  qr_trends_future_2b_zones_GHM_dPL_ens_rcp60_2008_2099.png
  qr_trends_future_2b_zones_GHM_dPL_ens_rcp85_2008_2050.png
  qr_trends_future_2b_zones_GHM_dPL_ens_rcp85_2008_2099.png

Per panel (one panel per climate zone):
  Boxes (left → right): δPS-mSS, δPS-HVS, δPS-HyS, δ-ens, GHMs-ens,
  then one box per individual GHM (clm45, cwatm, …).
  δ-models are styled with white fill + colored thick edges; GHMs are
  styled with their fill color + thin black edges.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from post.stat_plots import plotBoxFig
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

# Saved .npz spans 2007..2099; index 1 = 2008. We slice 2008..2099 (92 yrs).
NPZ_START_YEAR  = 2007
SLICE_START     = 2008
SLICE_END       = 2099

# Each figure is one (rcp, period) combination
RCPS    = ["rcp85", "rcp60"]
PERIODS = [(2008, 2050), (2008, 2099)]

GHM_MODELS = ["clm45", "cwatm", "matsiro", "h08",
              "lpjml", "jules-w1", "watergap2-2c", "pcr-globwb"]


# ---------------------------------------------------------------------------
# Style — colors per model and how to display each one in the boxplot
# ---------------------------------------------------------------------------
COLORS = {
    # GHMs (filled boxes, thin black edge)
    "clm45":        "pink",
    "cwatm":        "cyan",
    "matsiro":      "magenta",
    "h08":          "orange",
    "lpjml":        "khaki",
    "jules-w1":     "darkkhaki",
    "watergap2-2c": "lightyellow",
    "pcr-globwb":   "lavender",
    "GHMs ens":     "grey",
    # δ models (white fill, colored thick edge)
    "δPS":   "blue",
    "δHS":   "red",
    "δHcS":  "green",
    "δ ens": "black",
}

DPL_BOX_STYLE = dict(face="whitesmoke", lw=4.0)  # white-ish fill, thick edge
GHM_BOX_STYLE = dict(face=None,         lw=0.6)  # filled, thin black edge


def box_style(label):
    """Return (face_color, edge_color, line_width) for a box."""
    if "δ" in label:
        return DPL_BOX_STYLE["face"], COLORS[label], DPL_BOX_STYLE["lw"]
    return COLORS[label], "black", GHM_BOX_STYLE["lw"]


# ---------------------------------------------------------------------------
# 1. Load yearly recharge .npz once; build per-period slope dicts
# ---------------------------------------------------------------------------
print(f"Loading: {YEARLY_NPZ}")
loaded = np.load(YEARLY_NPZ)

i0 = SLICE_START - NPZ_START_YEAR              # 1  (year 2008)
i1 = SLICE_END   - NPZ_START_YEAR + 1          # 93 (exclusive)
sim_yearly_full = {k: np.array(loaded[k][i0:i1, :]) for k in loaded.files}
loaded.close()

period_slopes = {}
for y1, y2 in PERIODS:
    pkey  = f"{y1} - {y2}"
    start = y1 - SLICE_START
    nyrs  = y2 - y1 + 1
    sliced = {k: sim_yearly_full[k][start:start + nyrs, :]
              for k in sim_yearly_full}
    slopes, _ = calculate_yearly_trends(
        yearly_data_dict=sliced,
        start_date=str(y1), end_date=str(y2 + 1),
        flow_obs_daily=None, flow_percentage_availability=0.0,
        consider_obs_flow_percentage=False,
    )
    period_slopes[pkey] = slopes
    print(f"  trends computed for {pkey}")

sim_yearly_full = None    # not needed once trends are done


# ---------------------------------------------------------------------------
# 2. Spatial setup — basin points + zone polygons + zone→basin index map
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

zone_info = []  # list of (subplot_idx, zone_label, basin_indices)
for i, polygon_row in shapefile.iterrows():
    pgdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)
    idx  = gpd.sjoin(gdf_points, pgdf, how="inner",
                     predicate="intersects").index.tolist()
    zone_info.append((i, str(polygon_row["new_FID"]), idx))


# ---------------------------------------------------------------------------
# 3. Build one figure per (rcp, period)
# ---------------------------------------------------------------------------
def build_box_lists(slope_dict, rcp, basin_idx):
    """
    Return (data_list, labels, face_colors, edge_colors, line_widths)
    for the boxplot of one zone.
    """
    candidates = [
        # (key in slope_dict,                       legend label)
        (f"dPL_ens_mSS_{rcp}_yearly",               "δPS"),
        (f"dPL_ens_HVS_{rcp}_yearly",               "δHS"),
        (f"dPL_ens_HyS_{rcp}_yearly",               "δHcS"),
        (f"dPL_ens_{rcp}_yearly",                   "δ ens"),
        (f"GHM_ens_{rcp}_yearly",                   "GHMs ens"),
    ] + [
        (f"GHM_ens_{m}_{rcp}_yearly", m) for m in GHM_MODELS
    ]

    data, labels, faces, edges, lws = [], [], [], [], []
    for key, label in candidates:
        if key not in slope_dict:
            continue
        data.append(np.array(slope_dict[key])[basin_idx])
        labels.append(label)
        face, edge, lw = box_style(label)
        faces.append(face)
        edges.append(edge)
        lws.append(lw)
    return data, labels, faces, edges, lws


for rcp in RCPS:
    rcp_pretty = f"{rcp[3]}.{rcp[4]}"   # "rcp60" -> "6.0"
    for y1, y2 in PERIODS:
        pkey       = f"{y1} - {y2}"
        slope_dict = period_slopes[pkey]

        fig, axs = plt.subplots(2, 6, figsize=(24, 16))
        ax = axs.flatten()
        plt.subplots_adjust(left=0.04, bottom=0.03, right=0.99, top=0.90,
                            wspace=0.33, hspace=0.1)

        last_legend = None   # remember the per-zone labels/colors for the legend

        for subplot_i, zone_label, basin_idx in zone_info:
            if len(basin_idx) <= 4:
                continue

            data, labels, faces, edges, lws = build_box_lists(
                slope_dict, rcp, basin_idx
            )

            ax[subplot_i] = plotBoxFig(
                [data], label1=[f"Zone {zone_label}"],
                colorLst=faces, edgecolor_list=edges,
                label1_font_size=22, sharey=False, figsize=(12, 5),
                axin=ax[subplot_i], add_horizontal_line=True,
                widths=0.6, line_width_list=lws, ylim=None,
            )
            ax[subplot_i][0].tick_params(axis="y", labelsize=22)
            last_legend = (labels, faces, edges, lws)

        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Recharge trend comparison of ensembles of hydrologic models "
            f"in rcp {rcp_pretty} between {y1} - {y2} (mm/year/year)",
            fontsize=30, y=0.991,
        )

        # Build legend from the last populated zone (all zones use the same models)
        labels, faces, edges, lws = last_legend
        legend_patches = [
            FancyBboxPatch(
                (0, 0), 1, 1,
                facecolor=f, edgecolor=e, linewidth=w,
                boxstyle="square,pad=0.1", label=lab,
            )
            for lab, f, e, w in zip(labels, faces, edges, lws)
        ]
        plt.legend(handles=legend_patches, loc="upper center", frameon=False,
                   ncol=7, bbox_to_anchor=(-2.7, 2.27), fontsize=20)

        out_png = FIG_OUT_DIR / (
            f"qr_trends_future_2b_zones_GHM_dPL_ens_{rcp}_{y1}_{y2}.png"
        )
        plt.savefig(out_png, dpi=600)
        plt.close("all")
        print(f"Saved: {out_png}")

print("END")