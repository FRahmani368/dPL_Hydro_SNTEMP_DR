import numpy as np
import pandas as pd
import os
import glob
from matplotlib.lines import Line2D
# put these at the very top of the file
os.environ["MPLBACKEND"] = "Agg"   # non-GUI backend, safe for scripts/jobs

import matplotlib
matplotlib.use("Agg")              # (belt-and-suspenders; okay to keep)
import matplotlib.pyplot as plt
import json
import pickle
import shutil
import scipy.stats
import math
import scipy
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import string
import random
# from mpl_toolkits import basemap   # matplotlib==3.2.2 needed
import torch
# import cartopy.crs as ccrs
# import cartopy.feature as cfeaturegit push origin
import warnings
from sklearn.model_selection import KFold
from post.stat_plots import flatData, plotMap, plotBoxFig, statError, plot_multiple_shapefiles
import geopandas as gpd
from shapely.geometry import Point
import pymannkendall as mk
from post.read_GHMs_dPLs import (
    converting_daily_to_monthly, calculate_yearly_trends, read_dPL_recharge_fr_ISIMIP2b,
    converting_monthly_to_yearly, read_GHM_ISIMIP2b_2003basins_monthly, converting_daily_to_yearly_precip
)


### first element padding
# def adaptive_moving_average(arr, window):
#     # Pad the beginning with `window` values equal to arr[0]
#     padded_arr = np.pad(arr, (window-1, 0), mode='constant', constant_values=arr[0])

#     # Preallocate result array
#     result = np.empty_like(arr, dtype=float)

#     # Standard moving average using full window
#     for i in range(len(arr)):
#         result[i] = np.mean(padded_arr[i : i + window])

#     return result
def median_curve(x, y, nbins=80, min_bin=15):
    """
    Return a 'median regression' curve: (x_med_per_bin, y_med_per_bin).
    - x, y: 1D arrays (flattened points)
    - nbins: number of x-quantile bins
    - min_bin: require at least this many points in a bin
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        return np.array([]), np.array([])

    # Quantile bin edges (ensures bins across the occupied x-range)
    q = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, q)

    # Guard: remove duplicate edges (when x is clustered)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([]), np.array([])

    xmed, ymed = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # include left, exclude right (last bin handled by edges)
        sel = (x >= lo) & (x < hi) if hi > lo else np.zeros_like(x, dtype=bool)
        if sel.sum() >= min_bin:
            xmed.append(np.median(x[sel]))
            ymed.append(np.median(y[sel]))
    xmed = np.asarray(xmed); ymed = np.asarray(ymed)

    # Optionally do a light moving median smoothing (3-point window)
    if xmed.size >= 5:
        order = np.argsort(xmed)
        xmed = xmed[order]; ymed = ymed[order]
        # 3-pt running median on y
        ypad = np.pad(ymed, (1,1), mode='edge')
        ysm = np.stack([ypad[:-2], ypad[1:-1], ypad[2:]], axis=1)
        ymed = np.median(ysm, axis=1)

    return xmed, ymed

### mirro padding
def adaptive_moving_average(arr, window):
    if len(arr) < window:
        raise ValueError("Array length must be at least equal to the window size.")

    # Pad with the first (window - 1) actual values (not repeated first value)
    pad = arr[:window - 1][::-1]  # reverse for natural weighting from most recent
    padded_arr = np.concatenate([pad, arr])

    # Preallocate result array
    result = np.empty_like(arr, dtype=float)

    # Compute the moving average
    for i in range(len(arr)):
        result[i] = np.mean(padded_arr[i: i + window])

    return result


# GHM_model_name_list  =["pcrglobwb", "CLM45", "Matsiro", "Cwatm",
#             "Watergap2", "Watergap2_2c", "LPJmL", "H08"]
GHM_model_name_list = ["clm45", "cwatm", "matsiro", "h08", "lpjml", "jules-w1", "watergap2-2c", "pcr-globwb", ]
GHM_climate_forcing_list = ["gfdl-esm2m", "hadgem2-es", "ipsl-cm5a-lr", "miroc5"]

dPL_model_name_list = ["HVN", "mSS", "HyS", "mSN", "HyN", "HVS", ]
dPL_model_name_list_real = ["\u03B4H", "\u03B4PS", "\u03B4HcS", "\u03B4P", "\u03B4Hc", "\u03B4HS", ]
CLM_name_list = ["gfdl_esm2m", "hadgem2_es", "ipsl_cm5a_lr", "miroc5"]  # ,
CLM_scenario_list = ["rcp60", "rcp85"]
tr_time_start = "2007-01-01"  # "2007-01-01"    #
tr_time_end1 = "2099-12-31"  # "2050-12-31"  #

colors_dictionary_models = {"clm45": "pink",
                            "cwatm": "cyan",
                            "matsiro": "magenta",
                            "h08": "orange",
                            "lpjml": "khaki",
                            "jules-w1": "darkkhaki",
                            "watergap2-2c": "lightyellow",
                            "pcr-globwb": "lavender",
                            f"\u03B4H": "orange",
                            f"\u03B4HS": "red",
                            f"\u03B4PS": "blue",
                            f"\u03B4P": "lightskyblue",
                            f"\u03B4HcS": "green",
                            f"\u03B4Hc": "lawngreen",
                            f"\u03B4_ens": "black",
                            f"\u03B4 ens": "black",
                            'dPL_ens_mSS': "blue",
                            'dPL_ens_HVS': "red",
                            'dPL_ens_HyS': "green",
                            'dPL_ens': "black",  # needs to be fixes later
                            "dPL_ens_gfdl-esm2m": "teal",
                            "dPL_ens_hadgem2-es": "violet",
                            "dPL_ens_ipsl-cm5a-lr": "purple",
                            "dPL_ens_miroc5": "crimson",
                            "GHM_ens": "grey",
                            "GHMs ens": "grey",
                            'GHM_ens_gfdl-esm2m': "yellow",
                            'GHM_ens_hadgem2-es': "orchid",
                            'GHM_ens_ipsl-cm5a-lr': "lightseagreen",
                            'GHM_ens_miroc5': "maroon",

                            }
dataset = "1223"

# Windows server
main_dir = r"D:\\DR\\M"  ##daymet_1223_1023_PUB/
p_data = r"D:\\DR\\data"
p_data_out = r"D:\\DR\\data"
### basin path
shp_path0 = r"D:\\DR\\data\\Zones"
fig_out_dir = r"D:\\DR\\evaluation_figures"


## MAC
# main_dir = r"/Volumes/Extreme_SSD/Frshd/P/M"
# p_data = r"/Volumes/Extreme_SSD/inputs_2003"
# p_data_out = r"/Volumes/Extreme_SSD/Frshd/inputs"
# ### basin path
# shp_path0 = r"/Volumes/Extreme_SSD/Frshd/inputs"
# fig_out_dir = r"/Users/farshidrahmani/PhD/recharge_paper"


# qr_GHM_dir0 = os.path.join(p_data_out, "out_2003_np_qr_ISIMIP2b_GHMs")  # GHMs_out_qr_2005soc

attr_name_file_path = os.path.join(p_data, "ts_2003basins", "attr2003_mswep_03122024_name.json")
with open(attr_name_file_path, 'r') as json_file:
    # Load the JSON data from the file
    a_name_np = json.load(json_file)
d1 = "attr" + dataset + "_1023_daymet_20240826.npy"
attr_trained_dt = np.load(os.path.join(p_data, "tr_1223basins", d1))
site_no_trained_dt = attr_trained_dt[:, a_name_np.index("site_no_int")]
attr2003 = np.load(os.path.join(p_data, "ts_2003basins", "attr2003_mswep_03122024.npy"))
site_no_2003 = attr2003[:, a_name_np.index("site_no_int")]
## reading monthly recharge values for future 2006-2099 from Miroc5 isimip2b
# Dictionary to store results
sim_recharge_dict = dict()
sim_recharge_dict_yearly = dict()

## reread it:
# Load the saved `.npz` file
loaded_data = np.load(os.path.join(main_dir, "qr_2b", "sim_recharge_dict_monthly_2007_2100.npz"))
# Convert back to dictionary
temp_dict = {key: loaded_data[key] for key in loaded_data}
sim_recharge_dict.update(temp_dict)
temp_dict = None
print("dPL_ens")
#########################

####################

## reread it:
# Load the saved `.npz` file
loaded_data = np.load(os.path.join(main_dir, "qr_2b", "sim_recharge_dict_yearly_2007_2100.npz"))
# Convert back to dictionary
temp_dict = {key: loaded_data[key] for key in loaded_data}
sim_recharge_dict_yearly.update(temp_dict)
temp_dict = None
# print("converted from monthly to yearly")

# calculating trends
## we can do different time ranges
y1 = [2008, 2008, 2024, 2050, 2024]
y2 = [2050, 2099, 2099, 2099, 2050]
sim_recharge_dict_yearly_mod = dict()
for key in sim_recharge_dict_yearly.keys():
    sim_recharge_dict_yearly_mod[key] = np.array(sim_recharge_dict_yearly[key][1:93, :])
# years = np.arange(2008, 2100)
slope_dict_dict = dict()
intercept_dict_dict = dict()
sim_recharge_dict_yearly_mod_dict = dict()
for j in range(len(y1)):
    years = np.arange(y1[j], y2[j] + 1)
    data_dict = dict()
    for key in sim_recharge_dict_yearly_mod.keys():
        data_dict[key] = sim_recharge_dict_yearly_mod[key][y1[j] - 2008: y1[j] - 2008 + len(years), :]
    sim_recharge_dict_yearly_mod_dict[f"{str(y1[j])} - {str(y2[j])}"] = data_dict

    # slope, intercept = calculate_yearly_trends(yearly_data_dict=data_dict,
    #                                            start_date=str(y1[j]),
    #                                            end_date=str(y2[j] + 1),
    #                                            flow_obs_daily=None,
    #                                            flow_percentage_availability=0.0,
    #                                            consider_obs_flow_percentage=False)
    #
    # slope_dict_dict[f"{str(y1[j])} - {str(y2[j])}"] = slope
    # intercept_dict_dict[f"{str(y1[j])} - {str(y2[j])}"] = intercept
    print(f"{str(y1[j])} - {str(y2[j])} done")

# slope_dict, intercept_dict =calculate_yearly_trends(yearly_data_dict=sim_recharge_dict_yearly,
#                                                     start_date=tr_time_start,
#                                                     end_date=tr_time_end1,
#                                                     flow_obs_daily=None,
#                                                     flow_percentage_availability=0.0,
#                                                     consider_obs_flow_percentage=False)
print("trends calculated")



models_list = ["dPL_ens_mSS_rcp60_yearly", "GHM_ens_rcp60_yearly",
               "dPL_ens_mSS_rcp85_yearly", "GHM_ens_rcp85_yearly",
               "dPL_ens_HVS_rcp60_yearly", "dPL_ens_HVS_rcp85_yearly",
               "dPL_ens_HyS_rcp60_yearly", "dPL_ens_HyS_rcp85_yearly"]
sim_rech_dict = dict()
for key in sim_recharge_dict_yearly_mod_dict.keys():
    data_dict = dict()
    for m in models_list:
        data_dict[m] = sim_recharge_dict_yearly_mod_dict[key][m]
    sim_rech_dict[key] = data_dict

sim_recharge_dict_yearly_mod_dict = None
sim_recharge_dict = None
sim_recharge_dict_yearly_dict = None
sim_recharge_dict_yearly = None
loaded_data = None
data_dict = None
### boxplots for different zones
# temp_model_name_list = ["mSN_rcp60", "mSS_rcp60", "HVS_rcp60", "HyS_rcp60",
#                         "mSN_rcp85", "mSS_rcp85", "HVS_rcp85", "HyS_rcp85"]
# temp_model_name_list_real = ["PRMS (rcp60)", "PRMS-SNTEMP (rcp60)", "HBV-SNTEMP (rcp60)", "HBV cap-SNTEMP (rcp60)",
#                             "PRMS (rcp85)", "PRMS-SNTEMP (rcp85)", "HBV-SNTEMP (rcp85)", "HBV cap-SNTEMP (rcp85)"]
################
### Read precipitation data
# 2b
start_time_2b = "2006-01-01"
end_time_2b = "2099-12-31"

f_path_list_85 = glob.glob(os.path.join(p_data_out, "ISIMIP2b", "*rcp85*.npy"))
f_path_list_60 = glob.glob(os.path.join(p_data_out, "ISIMIP2b", "*rcp60*.npy"))
time_range_2b = pd.date_range(start=start_time_2b, end=end_time_2b, freq="D")
#
# json_file_path = os.path.join(p_data_out, "ISIMIP2b", f_path_list_60[0].split(".npy")[0] + "_name.json")
# with open(json_file_path, "r") as json_file:
#     f_name_np_2b = json.load(json_file)

precip_list = ["precip_rcp60_ens_yearly", "precip_rcp85_ens_yearly"]
# precip_dict = dict()
# for p_i, path_list in enumerate([f_path_list_60, f_path_list_85]):
#     data_dict = dict()
#     f_list = [np.load(path)[:, :, :] for path in path_list]
#     f_array = np.stack(f_list)
#     f_list = None
#     f_2b_ens = np.mean(f_array, axis=0)
#     f_array = None
#     data = dict()
#     key2 = precip_list[p_i]
#     data[key2] = f_2b_ens[:, :, f_name_np_2b.index('prcp(mm/day)')].squeeze().T
#     data_yearly = converting_daily_to_yearly_precip(data,
#                                                     start_date=start_time_2b,  # starting_time,
#                                                     end_date=end_time_2b, )  # ending_time)
#     for j in range(len(y1)):
#         years = np.arange(y1[j], y2[j] + 1)
#         key = f"{str(y1[j])} - {str(y2[j])}"
#         # starting_time = f"{y1[j]}-10-01"
#         # ending_time = f"{y2[j]}-09-30"
#         # mask = (time_range_2b >= starting_time) & (time_range_2b <=ending_time)
#
#         # data_dict[key2] = (data_yearly[key2][:, y1[j] - 2008: y1[j] - 2008 + len(years)])
#         precip_dict.setdefault(key, {})[key2] = (data_yearly[key2][y1[j] - 2008: y1[j] - 2008 + len(years), :])
#     # precip_dict[key] = data_dict
#         print(f"{key} is done")

precip_file_path = os.path.join(p_data_out, "isimip2b", "precip_dict.pkl")
# # ## save
# with open(precip_file_path, "wb") as f:
#     pickle.dump(precip_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
# # load
with open(precip_file_path, "rb") as f:
    precip_dict = pickle.load(f)



### load climate regions
file_name_rcp60 = f"2b_climate_labels_rcp60.json"
file_path_rcp60 = os.path.join(p_data, "climate_classifications", file_name_rcp60)
with open(file_path_rcp60, "r", encoding="utf-8") as f:
    loaded = json.load(f)
# labels_rn2 = np.array(loaded["labels_N_over_P"])
labels_pet_rcp60 = np.array(loaded["labels_Ep_over_P"])

file_name_rcp85 = f"2b_climate_labels_rcp85.json"
file_path_rcp85 = os.path.join(p_data, "climate_classifications", file_name_rcp85)
with open(file_path_rcp85, "r", encoding="utf-8") as f:
    loaded = json.load(f)
#
# labels_rn2 = np.array(loaded["labels_N_over_P"])
labels_pet_rcp85 = np.array(loaded["labels_Ep_over_P"])

# temp_model_name_list = list(sim_recharge_dict_yearly.keys())
# temp_model_name_list_real = ["PRMS-SNTEMP (rcp60)", "HBV cap-SNTEMP (rcp60)", "HBV-SNTEMP (rcp60)",
#                              "PRMS-SNTEMP (rcp85)", "HBV cap-SNTEMP (rcp85)", "HBV-SNTEMP (rcp85)"]
temp_ens_list = ["GHM_ens", "dPL_ens"]
divisions_labels = []
points = []
lat = attr2003[:, a_name_np.index("lat")]
lon = attr2003[:, a_name_np.index("lon")]
for i in range(len(lat)):
    # points.append((lon[i], lat[i]))
    points.append((lat[i], lon[i]))
gdf_points = gpd.GeoDataFrame(geometry=[Point(longitude, latitude) for latitude, longitude in points],
                              crs="EPSG:4326")

# # reading recharge observations
# rech_obs_moeck_dir = r"/scratch/fzr5082/PGML_STemp_results/inputs/rech_obs/"
# rech_obs_points = gpd.read_file(os.path.join(rech_obs_moeck_dir, "Moeck_2020_5207_glob_recharge_point.shp"))

### basin path

shapefile = gpd.read_file(os.path.join(shp_path0, "Zones_0228.shp"))
shapefile = shapefile.to_crs(gdf_points.crs)
########################################
#######################################
## making a 2 X2 figure with four maps for qr vs precipitation in 2b

# models_list = ["dPL_ens_mSS_rcp60_yearly", "GHM_ens_rcp60_yearly",
#                "dPL_ens_mSS_rcp85_yearly", "GHM_ens_rcp85_yearly", ]
title_list = [f"a) \u03B4PS in RCP6.0",
              "b) GHMs ensemble in RCP6.0",
              "c) \u03B4PS in RCP8.5",
              "d) GHMs ensemble in RCP8.5"]

b1 = 0
b2 = 2003
colorbar_range = [-2, 2]
trend_time_period_list = sim_rech_dict.keys()  # ['2008 - 2050', '2008 - 2099', '2024 - 2099', '2051 - 2099', '2024 - 2050']

#### precip to recharge graphs
models_list = ["dPL_ens_mSS_rcp60_yearly", "GHM_ens_rcp60_yearly",
               "dPL_ens_mSS_rcp85_yearly", "GHM_ens_rcp85_yearly", ]


# Map labels -> colors (you can tweak)
REGION_COLORS = {
    "wet-cold":  "#56B4E9",  # sky blue  (you asked for cold-wet to be blue/sky blue)
    "wet-warm":  "#009E73",  # green
    "dry-warm":  "#E69F00",  # orange
    "dry-cold":  "#CC79A7",  # magenta-ish
}

# Accept some synonyms / sloppy casing
CANON = {
    "cold-wet": "wet-cold",
    "warm-wet": "wet-warm",
    "warm-dry": "dry-warm",
    "cold-dry": "dry-cold",
    "wet-cold": "wet-cold",
    "wet-warm": "wet-warm",
    "dry-warm": "dry-warm",
    "dry-cold": "dry-cold",
}
def colorize(labels_array):
    # labels_array shape: (2003,)
    colors = []
    for lab in labels_array:
        key = CANON.get(str(lab).strip().lower(), None)
        colors.append(REGION_COLORS.get(key, "#000000"))  # default black if missing
    return np.array(colors)

# If you want a legend with color patches:
REGION_ORDER = ["wet-cold", "wet-warm", "dry-warm", "dry-cold"]
LEGEND_HANDLES = [
    Line2D([0],[0], marker='o', linestyle='',
           color=REGION_COLORS["wet-cold"],  label="Wet–Cold",  markersize=14),
    Line2D([0],[0], marker='o', linestyle='',
           color=REGION_COLORS["wet-warm"],  label="Wet–Warm",  markersize=14),
    Line2D([0],[0], marker='o', linestyle='',
           color=REGION_COLORS["dry-warm"],  label="Dry–Warm",  markersize=14),
    Line2D([0],[0], marker='o', linestyle='',
           color=REGION_COLORS["dry-cold"],  label="Dry–Cold",  markersize=14),
]
# =====================================



max_val = 3800   # maximum value for yearly precipitation
axis_font_size = 28

# labels: np.ndarray, shape (2003,), your climate-region labels
# rcp_scenario_list = ["rcp60", "rcp85"]
# for l_i, labels in enumerate([labels_pet_rcp60, labels_pet_rcp85]):
# point_colors_all = colorize(labels)  # precompute once
point_colors_rcp60 = colorize(labels_pet_rcp60)
point_colors_rcp85 = colorize(labels_pet_rcp85)
# #
# rcp_scenario = rcp_scenario_list[l_i]
for i, trend_time_period in enumerate(trend_time_period_list):
    fig, axs = plt.subplots(2, 2, figsize=(24, 24))
    ax = axs.flatten()
    plt.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.9, wspace=0.2, hspace=0.2)

    ## subplot 0
    data_dict = dict()
    lat_dict = dict()
    lon_dict = dict()

    rech_dPL_ens_60 = (sim_rech_dict[trend_time_period][models_list[0]])
    rech_GHM_ens_60 = (sim_rech_dict[trend_time_period][models_list[1]])

    rech_dPL_ens_85 = (sim_rech_dict[trend_time_period][models_list[2]])
    rech_GHM_ens_85 = (sim_rech_dict[trend_time_period][models_list[3]])

    precip_60 = (precip_dict[trend_time_period][precip_list[0]])
    precip_85 = (precip_dict[trend_time_period][precip_list[1]])
    yaxis_list = [rech_dPL_ens_60, rech_GHM_ens_60,
               rech_dPL_ens_85, rech_GHM_ens_85]
    xaxis_list = [precip_60, precip_60,
              precip_85, precip_85]
    point_colors_list = [point_colors_rcp60, point_colors_rcp60,
                         point_colors_rcp85, point_colors_rcp85]


    for j in range(4):
        point_colors_all = point_colors_list[j]
        x = xaxis_list[j]  # shape (T, L)
        y = yaxis_list[j]  # shape (T, L)
        T, L = x.shape

        # 1) Finite mask on the 2D grid
        m2d = np.isfinite(x) & np.isfinite(y)  # (T, L)

        # 2) Flatten x,y with the SAME order as the mask
        xf = x[m2d]  # (Nvalid,)
        yf = y[m2d]  # (Nvalid,)

        # 3) Build a color grid by repeating the per-point color across time
        #    point_colors_all: (L,)  ->  colors2d: (T, L)  ->  flatten with the same mask
        colors2d = np.tile(point_colors_all, (T, 1))  # repeat along time
        cf = colors2d[m2d]


        ax[j].scatter(x, y, s=12, alpha=0.8, c=cf)
        ax[j].set_title(title_list[j], fontsize=axis_font_size)

        ax[j].plot([0, max_val], [0, max_val], color="lightblue", lw=1)
        ax[j].set_xlim(0, max_val)
        ax[j].set_ylim(0, max_val)
        ax[j].tick_params(axis='y', labelsize=axis_font_size)
        ax[j].tick_params(axis='x', labelsize=axis_font_size)
        ax[j].set_ylabel("Groundwater recharge (mm/year)", fontsize=axis_font_size)
        ax[j].set_xlabel("Precipitation (mm/year)", fontsize=axis_font_size)
        ax[j].grid(True, ls=":")

        # ---------- median curves per climate region (by color) ----------
        # Use the same mapping you used for colors:
        REGION_COLORS = {
            "wet-cold": "#56B4E9",  # sky blue
            "wet-warm": "#009E73",
            "dry-warm": "#E69F00",
            "dry-cold": "#CC79A7",
        }
        REGION_ORDER = ["wet-cold", "wet-warm", "dry-warm", "dry-cold"]  # plotting order

        for region in REGION_ORDER:
            col = REGION_COLORS[region]
            sel = (cf == col)
            if np.any(sel):
                xs, ys = median_curve(xf[sel], yf[sel], nbins=40, min_bin=15)
                if xs.size:
                    # bold line through the medians (this is your "median regression" curve)
                    ax[j].plot(xs, ys, color=col, lw=4, alpha=0.95, zorder=4)
                    # optional: a thin white outline
                    ax[j].plot(xs, ys, color='white', lw=7, alpha=0.5, zorder=3)

        # add legend once (top-left panel), or put on each if you prefer
        if j == 0:
            ax[j].legend(handles=LEGEND_HANDLES, fontsize=axis_font_size-4,
                         frameon=True, loc="upper left")

    fig.suptitle(
        f"Yearly recharge vs. precipitation across models, {trend_time_period}",
        fontsize=34, y=2.0
    )
    # plt.title(f"Recharge and precipitation relationships in different models in {trend_time_period}",
    #           fontsize=34, y=2.3, x=-0.1)
    fig.text(0.08, 0.96,
             f"Recharge and precipitation relationships in different models in {trend_time_period}",
             fontsize=40, ha='left', va='top')

    fig_name = f"rech_precip_{trend_time_period}.png"
    plt.savefig(os.path.join(fig_out_dir, fig_name), dpi=300)
    plt.close("all")
    print(f"plot {trend_time_period}")


print("end")



