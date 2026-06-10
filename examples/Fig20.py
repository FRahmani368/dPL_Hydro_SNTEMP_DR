import os
import pandas as pd
import numpy as np
import json
import glob
import baseflow
import xarray as xr
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import shutil
import numpy as np
import math
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import string
import random
# from mpl_toolkits import basemap   # matplotlib==3.2.2 needed
import os
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from sklearn.model_selection import KFold
from post.stat_plots import flatData, plotMap, plotBoxFig, statError
import geopandas as gpd
from shapely.geometry import Point
from post.read_GHMs_dPLs import (
                read_GHM_ISIMIP2a_daily, do_baseflow_separation, read_dPL_ISIMIP2a_daily,
                converting_daily_to_monthly, converting_daily_to_yearly, calculate_yearly_trends,
                read_dPL_Daymet_daily
                    )
from sklearn.metrics import r2_score

def calculate_r2(obs, sim):
    """Computes the squared Pearson correlation coefficient (r^2)."""
    correlation_matrix = np.corrcoef(obs, sim)
    r2 = correlation_matrix[0, 1] ** 2
    return r2

def calculate_R2(obs, sim):
    """Computes the coefficient of determination (R^2)."""
    return r2_score(obs, sim)

def Rel_L2(y_sim, y_obs):
    nominator = (np.nansum((y_sim - y_obs)**2))**0.5
    denominator = (np.nansum((y_obs)**2))**0.5
    return nominator / denominator


dataset = "1223"
tr_time_start = "1989-01-01"
tr_time_end1 = "1999-12-31"
tr_time_end2 = "2000-01-01"
flow_percentage_availability = 0.90 # a threshold for calculating baseflow trend
include_ssflow=True
minimum_CPI_threshold = 0.33
# p_data = r"/scratch/fzr5082/PGML_STemp_results/inputs"
# dir0=r"/scratch/fzr5082/PGML_STemp_results/models/daymet_1223_1023_PUB/"
# fig_out_dir = r"/scratch/fzr5082/PGML_STemp_results/models"
p_data = r"D:\\DR\\data"
dir0 = r"D:\\DR\\M\\daymet_1223_1023_PUB"
fig_out_dir = r"D:\\DR"
shp_path0 = r"D:\\DR\\data\\Zones"
attr_name_file_path = os.path.join(p_data, "ts_2003basins", "attr2003_mswep_03122024_name.json")
with open(attr_name_file_path, 'r') as json_file:
    # Load the JSON data from the file
    a_name_np = json.load(json_file)
### finding the indices of the basins that were f_out in small and big dataset
# d1 = "attr" + dataset + "_mswep_03122024.npy"
# attr1031 = np.load(os.path.join(p_data, d1))
d1 = "attr" + dataset + "_1023_daymet_20240826.npy"
attr_trained_dt = np.load(os.path.join(p_data, "tr_1223basins", d1))
site_no_trained_dt = attr_trained_dt[:, a_name_np.index("site_no_int")]

attr2003 = np.load(os.path.join(p_data, "ts_2003basins", "attr2003_mswep_03122024.npy"))
site_no_2003 = attr2003[:, a_name_np.index("site_no_int")]
# ind_1223 = np.where(np.isin(site_no_2003, site_no_trained_dt))[0]

## CPI method
# shp_upstr_GHM = gpd.read_file(os.path.join("/scratch/fzr5082/PGML_STemp_results/models/upscale/merged_shpfile_GHMs.shp"))
# flow_gage_grid_2003_CPI = pd.read_feather("/scratch/fzr5082/PGML_STemp_results/models/upscale/grid_gage_2003_CPI_intersect.feather")
shp_upstr_GHM = gpd.read_file(os.path.join(r"D:\\DR\\M\\upscale\\CPI\\merged_shpfile_GHMs.shp"))
flow_gage_grid_2003_CPI = pd.read_feather(r"D:\\DR\\M\\upscale\\CPI\\grid_gage_2003_CPI_intersect.feather")
flow_gage_grid_2003_CPI = flow_gage_grid_2003_CPI.loc[
    flow_gage_grid_2003_CPI["CPI"] > minimum_CPI_threshold].reset_index(drop=True)
points = []
for i in range(len(flow_gage_grid_2003_CPI)):
    lat = flow_gage_grid_2003_CPI.loc[i, "lat_grid_GHM"]
    lon = flow_gage_grid_2003_CPI.loc[i, "lon_grid_GHM"]
    points.append((lat, lon))
flow_gage_grid_2003_CPI["points_GHM"] = points
all_points_GHM = list(set(points))
intersecting_dict = {
    "CPI_2003": [],
    "sites_2003_CPI_intersect": [],
    "ind_sites_2003_CPI_intersect": [],
    "GHM_points_CPI_intersect_2003": [],
    "lat_grid_2003": [],
    "lon_grid_2003": [],
    "upstr_area_GHM_2003": [],
    "GAGESII_area_2003": [],
    "CPI_1223": [],
    "sites_1223_CPI_intersect": [],
    "ind_sites_1223_CPI_intersect": [],
    "GHM_points_CPI_intersect_1223": [],
    "lat_grid_1223": [],
    "lon_grid_1223": [],
    "upstr_area_GHM_1223": [],
    "GAGESII_area_1223": [],
}

for p in all_points_GHM:
    CPI_list = flow_gage_grid_2003_CPI.loc[flow_gage_grid_2003_CPI["points_GHM"] == p, "CPI"].reset_index(drop=True)
    row_flow_gage_grid_2003_CPI = flow_gage_grid_2003_CPI.loc[(flow_gage_grid_2003_CPI["points_GHM"] == p) &
                                                              (flow_gage_grid_2003_CPI[
                                                                   "CPI"] == CPI_list.max())].reset_index(drop=True)

    CPI_value = row_flow_gage_grid_2003_CPI["CPI"][0]
    intersecting_dict["CPI_2003"].append(CPI_value)

    site_no_int = row_flow_gage_grid_2003_CPI["site_no_int"][0]
    intersecting_dict["sites_2003_CPI_intersect"].append(site_no_int)

    ind_site_2003 = np.where(site_no_2003 == site_no_int)[0][0]
    intersecting_dict["ind_sites_2003_CPI_intersect"].append(ind_site_2003)

    intersecting_dict["GHM_points_CPI_intersect_2003"].append(p)
    intersecting_dict["lat_grid_2003"].append(p[0])
    intersecting_dict["lon_grid_2003"].append(p[1])

    upstr_area_GHM = row_flow_gage_grid_2003_CPI["upstr_area_GHM"][0]
    intersecting_dict["upstr_area_GHM_2003"].append(upstr_area_GHM)

    GAGESII_area = row_flow_gage_grid_2003_CPI["GAGESII_area"][0]
    intersecting_dict["GAGESII_area_2003"].append(GAGESII_area)

    if site_no_int in site_no_trained_dt:
        intersecting_dict["CPI_1223"].append(CPI_value)

        intersecting_dict["sites_1223_CPI_intersect"].append(site_no_int)

        ind_site_trained_dt = np.where(site_no_trained_dt == site_no_int)[0][0]
        intersecting_dict["ind_sites_1223_CPI_intersect"].append(ind_site_trained_dt)

        intersecting_dict["GHM_points_CPI_intersect_1223"].append(p)
        intersecting_dict["lat_grid_1223"].append(p[0])
        intersecting_dict["lon_grid_1223"].append(p[1])

        intersecting_dict["upstr_area_GHM_1223"].append(upstr_area_GHM)

        intersecting_dict["GAGESII_area_1223"].append(GAGESII_area)

sim_bf_dict = dict()
GHM_model_name_list = ["Watergap2_2c", "WAYS", "CLM40", "DBH", "H08", "JULES_B1", "JULES_W1",
                       "LPJmL", "MATSIRO", "MPI_HM", "ORCHIDEE", "PCR_GLOBWB", "VIC",  # "SWBM",
                       "WATERGAP2_ISIMIP2a",
                       "WEB_DHM_SG"]
dPL_model_name_list = ["mSS", "mSN", "HyS", "HyN", "HVS", "HVN"]  #
baseflow_sep_method_list = ["Furey"]
##### Reading streamflow GHM

for model_name_GHM in GHM_model_name_list:
    for baseflow_sep_method in baseflow_sep_method_list:
        # check if it has been estimated before
        file_name = model_name_GHM + "_" + baseflow_sep_method + "_daily.npy"

        if os.path.exists(os.path.join(fig_out_dir, "bf", file_name)):
            data_temp = np.load(os.path.join(fig_out_dir, "bf", file_name))
            sim_bf_dict[file_name.split(".npy")[0]] = data_temp
        else:
            qtot_daily = read_GHM_ISIMIP2a_daily(item_name="qtot",
                                                 start_date=tr_time_start,
                                                 end_date=tr_time_end1,
                                                 lat_list=intersecting_dict["lat_grid_2003"],
                                                 lon_list=intersecting_dict["lon_grid_2003"],
                                                 model_name_GHM=model_name_GHM,
                                                 CLM_name="GSWP3",
                                                 )

            baseflow_dict = do_baseflow_separation(streamflow=qtot_daily,
                                                   start_date=tr_time_start,
                                                   end_date=tr_time_end1,
                                                   sites_name_cols_list=intersecting_dict["sites_2003_CPI_intersect"],
                                                   baseflow_sep_method=[baseflow_sep_method])
            key = model_name_GHM + "_" + baseflow_sep_method + "_daily"
            sim_bf_dict[key] = (baseflow_dict[baseflow_sep_method]).to_numpy()
            np.save(os.path.join(fig_out_dir, "M", "bf", file_name), sim_bf_dict[key])
    print(model_name_GHM)
# ### calculating the GHM ensemble for baseflow
for baseflow_sep_method in baseflow_sep_method_list:
    ensemble_data = list()
    file_name = "GHM_ens_" + baseflow_sep_method + "_daily.npy"
    if os.path.exists(os.path.join(fig_out_dir,"M", "bf", file_name)):
        data_temp = np.load(os.path.join(fig_out_dir, "bf", file_name))
        sim_bf_dict[file_name.split(".npy")[0]] = data_temp
    else:
        for model_name_GHM in GHM_model_name_list:
            ensemble_data.append((sim_bf_dict[model_name_GHM + "_" + baseflow_sep_method + "_daily"]))
        ensemble_data = np.stack(ensemble_data)
        sim_bf_dict[file_name] = np.mean(ensemble_data, axis=0)
        np.save(os.path.join(fig_out_dir, "bf", file_name), sim_bf_dict[file_name])
print("GHM_ens")
ensemble_data = None
#########################
### reading Streamflow dPL
for model_name in dPL_model_name_list:
    for baseflow_sep_method in baseflow_sep_method_list:
        # check if it has been estimated before
        file_name = model_name + "_" + baseflow_sep_method + "_daily.npy"
        if os.path.exists(os.path.join(fig_out_dir,"M", "bf", file_name)):
            data_temp = np.load(os.path.join(fig_out_dir,"M", "bf", file_name))
            sim_bf_dict[file_name.split(".npy")[0]] = data_temp
        else:
            flow_sim, gwflow_sim, obs_flow = read_dPL_ISIMIP2a_daily(model_name=model_name,
                                                                     site_ind_list=intersecting_dict[
                                                                         "ind_sites_2003_CPI_intersect"],
                                                                     start_date=tr_time_start,
                                                                     end_date=tr_time_end1,
                                                                     include_ssflow=False,
                                                                     read_obs_flow_flag=True,
                                                                     dir0=dir0)
            baseflow_dict = do_baseflow_separation(streamflow=flow_sim,
                                                   start_date=tr_time_start,
                                                   end_date=tr_time_end1,
                                                   sites_name_cols_list=intersecting_dict["sites_2003_CPI_intersect"],
                                                   baseflow_sep_method=[baseflow_sep_method])
            sim_bf_dict[file_name.split(".npy")[0]] = (baseflow_dict[baseflow_sep_method]).to_numpy()
            np.save(os.path.join(fig_out_dir, "M", "bf", file_name), (baseflow_dict[baseflow_sep_method]).to_numpy())

    print(model_name)
    ####################
# ### calculating the dPL ensemble for baseflow  (only on integrated models)
for baseflow_sep_method in baseflow_sep_method_list:
    ensemble_data = list()
    file_name = "dPL_ens_" + baseflow_sep_method + "_daily.npy"
    if os.path.exists(os.path.join(fig_out_dir, "M", "bf", file_name)):
        data_temp = np.load(os.path.join(fig_out_dir, "M", "bf", file_name))
        sim_bf_dict[file_name.split(".npy")[0]] = data_temp
    else:
        for model_name in ["mSS", "HyS", "HVS"]:
            ensemble_data.append((sim_bf_dict[model_name + "_" + baseflow_sep_method + "_daily"]))
        ensemble_data = np.stack(ensemble_data)
        sim_bf_dict[file_name] = np.mean(ensemble_data, axis=0)
        np.save(os.path.join(fig_out_dir, "bf", file_name), sim_bf_dict[file_name])
print("dPL_ens")
# adding obs_flow baseflow to the dict
## read the obs_flow from one of the dPL models
_, _, obs_flow = read_dPL_ISIMIP2a_daily(model_name=dPL_model_name_list[0],
                                         site_ind_list=intersecting_dict["ind_sites_2003_CPI_intersect"],
                                         start_date=tr_time_start,
                                         end_date=tr_time_end1,
                                         include_ssflow=False,
                                         read_obs_flow_flag=True,
                                         dir0=dir0)
model_name = "obs_flow"
for baseflow_sep_method in baseflow_sep_method_list:
    file_name = model_name + "_" + baseflow_sep_method + "_daily.npy"
    if os.path.exists(os.path.join(fig_out_dir,"M", "bf", file_name)):
        data_temp = np.load(os.path.join(fig_out_dir,"M", "bf", file_name))
        sim_bf_dict[file_name.split(".npy")[0]] = data_temp
    else:
        baseflow_dict = do_baseflow_separation(streamflow=obs_flow,
                                               start_date=tr_time_start,
                                               end_date=tr_time_end1,
                                               sites_name_cols_list=intersecting_dict["sites_2003_CPI_intersect"],
                                               baseflow_sep_method=[baseflow_sep_method])
        sim_bf_dict[model_name + "_" + baseflow_sep_method + "_daily"] = (baseflow_dict[baseflow_sep_method]).to_numpy()
        np.save(os.path.join(fig_out_dir, "M","bf", file_name), (baseflow_dict[baseflow_sep_method]).to_numpy())
print("obs_flow")
############################
### read dPL Daymet daily
for model_name in dPL_model_name_list:
    for baseflow_sep_method in baseflow_sep_method_list:
        # check if it has been estimated before
        file_name = model_name + "_Daymet_" + baseflow_sep_method + "_daily.npy"
        if os.path.exists(os.path.join(fig_out_dir, "M","bf", file_name)):
            data_temp = np.load(os.path.join(fig_out_dir, "M", "bf", file_name))
            sim_bf_dict[file_name.split(".npy")[0]] = data_temp
        else:
            flow_sim, gwflow = read_dPL_Daymet_daily(model_name=model_name,
                                                     site_ind_list=intersecting_dict["ind_sites_2003_CPI_intersect"],
                                                     start_date=tr_time_start,
                                                     end_date=tr_time_end1,
                                                     include_ssflow=False,
                                                     read_obs_flow_flag=False,
                                                     dir0=dir0)
            baseflow_dict = do_baseflow_separation(streamflow=flow_sim,
                                                   start_date=tr_time_start,
                                                   end_date=tr_time_end1,
                                                   sites_name_cols_list=intersecting_dict["sites_2003_CPI_intersect"],
                                                   baseflow_sep_method=[baseflow_sep_method])
            sim_bf_dict[file_name.split(".npy")[0]] = (baseflow_dict[baseflow_sep_method]).to_numpy()
            np.save(os.path.join(fig_out_dir, "M", "bf", file_name), (baseflow_dict[baseflow_sep_method]).to_numpy())

# converting daily to monthly
sim_bf_dict_monthly = converting_daily_to_monthly(daily_data_dict=sim_bf_dict,
                                                  start_date=tr_time_start,
                                                  end_date=tr_time_end1)
# converting daily to yearly
sim_bf_dict_yearly = converting_daily_to_yearly(daily_data_dict=sim_bf_dict,
                                                start_date=tr_time_start,
                                                end_date=tr_time_end1)
# calculating trends
slope_dict, intercept_dict = calculate_yearly_trends(yearly_data_dict=sim_bf_dict_yearly,
                                                     start_date=tr_time_start,
                                                     end_date=tr_time_end1,
                                                     flow_obs_daily=obs_flow,
                                                     flow_percentage_availability=flow_percentage_availability)

###################################
##################################
### boxplots for different zones
temp_model_name_list = ["mSN", "mSS", "HVS", "HyS"]
temp_model_name_list_real = ["PRMS", "PRMS-SNTEMP", "HBV-SNTEMP", "HBV cap-SNTEMP"]
temp_ens_list = ["GHM_ens", "dPL_ens", "obs_flow"]
divisions_labels = []
points = []
lat = intersecting_dict["lat_grid_2003"]
lon = intersecting_dict["lon_grid_2003"]
for i in range(len(lat)):
    # points.append((lon[i], lat[i]))
    points.append((lat[i], lon[i]))
gdf_points = gpd.GeoDataFrame(geometry=[Point(longitude, latitude) for latitude, longitude in points],
                              crs="EPSG:4326")

# # reading recharge observations
# rech_obs_moeck_dir = r"/scratch/fzr5082/PGML_STemp_results/inputs/rech_obs/"
# rech_obs_points = gpd.read_file(os.path.join(rech_obs_moeck_dir, "Moeck_2020_5207_glob_recharge_point.shp"))

### basin path
# shp_path0 = r"/scratch/fzr5082/PGML_STemp_results/inputs/Zones"
shapefile = gpd.read_file(os.path.join(shp_path0, "Zones_0228.shp"))
shapefile = shapefile.to_crs(gdf_points.crs)

######################### subploting all zones in one
colors = ["grey", "cyan", "magenta", "orange",
          "khaki", "darkkhaki", "lightyellow", "lavender", "lightgray", "y", "hotpink", "ghostwhite",
          "yellow", "azure", "pink", "lightskyblue", "royalblue", "midnightblue",
          "lawngreen", "darkviolet",
          "red", "blue", "black", "green", ]

######################
# baseflow trends
fig, axs = plt.subplots(2, 6, figsize=(24, 16))
ax = axs.flatten()
plt.subplots_adjust(left=0.04,
                    bottom=0.03,
                    right=0.99,
                    top=0.90,
                    wspace=0.29,  # 0.4
                    hspace=0.1)
for i, polygon_row in shapefile.iterrows():
    # Create a GeoDataFrame for the individual polygon
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)

    # Spatial join or intersection
    clipped_points_ind = gpd.sjoin(gdf_points, polygon_gdf, how='inner', predicate='intersects').index.tolist()
    ####  baseflow trends values
    databox = []
    print(len(clipped_points_ind))
    if len(clipped_points_ind) > 4:
        for key in slope_dict.keys():
            if (key.split("_Furey")[0] in temp_model_name_list) or (key.split("_Furey")[0] in GHM_model_name_list) or (
                    key.split("_Furey")[0] in temp_ens_list):
                databox.append(np.array(slope_dict[key])[clipped_points_ind])
                label1 = ["Zone " + str(i)]
        ax[i] = plotBoxFig([databox], label1=label1, colorLst=colors,
                           label1_font_size=22, sharey=False, figsize=(12, 5), axin=ax[i],
                           add_horizontal_line=False, widths=0.6)
        ax[i][0].tick_params(axis='y', labelsize=22)
fig.patch.set_facecolor('white')
title = "Baseflow trend comparison of GSWP3 in GHM and dPL(trained on Daymet) models in " + tr_time_start[
    :4] + "_" + tr_time_end1[:4] + " (mm/year)"
fig.suptitle(title, fontsize=31, y=0.985)
## make the legends labels
legends_labels = list()
for key in slope_dict.keys():
    model_name = key.split("_Furey_yearly")[0]
    if (model_name in GHM_model_name_list) or (model_name in temp_ens_list):
        legends_labels.append(model_name)
    elif model_name in temp_model_name_list:
        ind = temp_model_name_list.index(model_name)
        legends_labels.append(temp_model_name_list_real[ind])
    elif model_name == "obs_flow":
        legends_labels.append("Observation")

# Create a discrete legend with custom rectangle sizes

legend_patches = [
    mpatches.Patch(color=colors[i], label=legends_labels[i],
                   linewidth=1.75, edgecolor="b")  # Optional: remove edge lines
    for i in range(len(legends_labels))
]

plt.legend(
    handles=legend_patches,
    loc='upper center',
    frameon=False,
    ncol=8,
    bbox_to_anchor=(-2.7, 2.27),
    fontsize=16
)
plt.savefig(os.path.join(fig_out_dir, "evaluation_figures", "bf_trends_zones_dmt_GSWP3" + ".png"), dpi=300)
plt.close("all")
print("END")
######################


##############################
## baseflow daily values -->KGE
fig, axs = plt.subplots(2, 6, figsize=(24, 16))
ax = axs.flatten()
plt.subplots_adjust(left=0.04,
                    bottom=0.03,
                    right=0.99,
                    top=0.90,
                    wspace=0.29,  # 0.4
                    hspace=0.1)
for i, polygon_row in shapefile.iterrows():
    # Create a GeoDataFrame for the individual polygon
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)

    # Spatial join or intersection
    clipped_points_ind = gpd.sjoin(gdf_points, polygon_gdf, how='inner', predicate='intersects').index.tolist()
    ####  baseflow values
    print(len(clipped_points_ind))
    if len(clipped_points_ind) > 4:
        sim_bf = list()
        obs_bf = list()
        for key in sim_bf_dict.keys():
            if (key.split("_Furey")[0] in temp_model_name_list) or (key.split("_Furey")[0] in GHM_model_name_list) or (
                    key.split("_Furey")[0] in temp_ens_list):
                if "obs_flow" not in key:
                    sim_bf.append(sim_bf_dict[key][:, clipped_points_ind])
                    obs_bf.append(sim_bf_dict["obs_flow_Furey_daily"][:, clipped_points_ind])

        statDictLst = [
            statError(np.swapaxes(x, 1, 0), np.swapaxes(y, 1, 0))
            for (x, y) in zip(sim_bf, obs_bf)
        ]

        label1 = ["Zone " + str(i)]

        databox = []
        for val in range(len(statDictLst)):
            databox.append(statDictLst[val]["KGE"])

        ax[i] = plotBoxFig([databox], label1=label1, colorLst=colors,
                           label1_font_size=22, sharey=False, figsize=(12, 5), axin=ax[i],
                           add_horizontal_line=False, widths=0.6, ylim=[-0.8, 1.0])
        ax[i][0].tick_params(axis='y', labelsize=22)
fig.patch.set_facecolor('white')
title = "Baseflow comparison of GSWP3 in GHM and dPL(trained on Daymet) models in " + tr_time_start[
    :4] + "_" + tr_time_end1[:4] + " (mm/year)"
fig.suptitle(title, fontsize=31, y=0.985)
## make the legends labels
legends_labels = list()
for key in sim_bf_dict.keys():
    model_name = key.split("_Furey_daily")[0]
    if (model_name in GHM_model_name_list) or (model_name in temp_ens_list):
        legends_labels.append(model_name)
    elif model_name in temp_model_name_list:
        ind = temp_model_name_list.index(model_name)
        legends_labels.append(temp_model_name_list_real[ind])

# Create a discrete legend with custom rectangle sizes

legend_patches = [
    mpatches.Patch(color=colors[i], label=legends_labels[i],
                   linewidth=1.75, edgecolor="b")  # Optional: remove edge lines
    for i in range(len(legends_labels))
]

plt.legend(
    handles=legend_patches,
    loc='upper center',
    frameon=False,
    ncol=8,
    bbox_to_anchor=(-2.7, 2.27),
    fontsize=16
)
plt.savefig(os.path.join(fig_out_dir, "evaluation_figures", "bf_sim_KGE_zones_dmt_GSWP3" + ".png"), dpi=300)
plt.close("all")
print("END")

###################
### scatter plot GHM ens and dPL
fig = plt.figure(1, figsize=(13, 13))
colors = ['black', 'blue', 'red', 'orange', 'pink', 'green', 'yellow']
markers = ['s', "D", "o", "*", "+", "D", "x"]

# item1_list = ["mSN", "HVN", "HyN"] #, "SwN"]    # "HVN",
# item2_list = ["mSS",   "HVS", "HyS"] #, "SwS"]    # "HVS",

plt.plot(np.array([-20, 80]), np.array([-20, 80]), lw=1.5)
# Adjust the margins
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.09)
obs = np.array(slope_dict['obs_flow_Furey_yearly'])
sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])
sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])
sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])
sim_clm = np.array(slope_dict['CLM40_Furey_yearly'])

mask = (obs == obs) & (sim_dPL == sim_dPL)
_, _, r_dpl, _, _ = scipy.stats.linregress(sim_dPL[mask], obs[mask])
plt.scatter(slope_dict['obs_flow_Furey_yearly'], slope_dict['mSS_Furey_yearly'], c=colors[0], s=200, marker=markers[0])

mask = (obs == obs) & (sim_ghm == sim_ghm)
_, _, r_ghm, _, _ = scipy.stats.linregress(sim_ghm[mask], obs[mask])
plt.scatter(slope_dict['obs_flow_Furey_yearly'], slope_dict['GHM_ens_Furey_yearly'], c=colors[1], s=200,
            marker=markers[2])

mask = (obs == obs) & (sim_clm == sim_clm)
_, _, r_clm, _, _ = scipy.stats.linregress(sim_clm[mask], obs[mask])
plt.scatter(slope_dict['obs_flow_Furey_yearly'], slope_dict['CLM40_Furey_yearly'], c=colors[3], s=200,
            marker=markers[2])

mask = (obs == obs) & (sim_dPL_high_res == sim_dPL_high_res)
_, _, r_dpl_high_res, _, _ = scipy.stats.linregress(sim_dPL_high_res[mask], obs[mask])
plt.scatter(slope_dict['obs_flow_Furey_yearly'], slope_dict['mSS_Daymet_Furey_yearly'], c=colors[2], s=250,
            marker=markers[4])

#     leg = "r= " + str("{:.3f}".format(r))
leg = ["-", "dPL-R2:" + str((r_dpl ** 2).round(3)), "GHM (ens)-R2:" + str((r_ghm ** 2).round(3)),
       "CLM40-R2:" + str((r_clm ** 2).round(3)), "dPL(high res)-R2:" + str((r_dpl_high_res ** 2).round(3))]  # "HBV",
plt.legend(leg, loc='lower right', title='Models', fontsize=25)
# plt.xlim(0.81, 0.94)
# plt.ylim(0.81, 0.94)
plt.grid()
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.xlabel("observed baseflow trends", fontsize=35)
plt.ylabel("simulated baseflow trends", fontsize=35)
plt.title("Observed and simulated baseflow \n trends on dPL and GHM models", fontsize=40)
plt.savefig(os.path.join(fig_out_dir, "evaluation_figures", "clm40_hres_bf_trend_scatter.png"), dpi=300)
plt.close("all")
print("END")

#######################################
## scatter plots for zones

# fig, axs = plt.subplots(3, 4, figsize=(24, 18))
# ax = axs.flatten()
# j = 0
# zone_list_label = list()
# for i, polygon_row in shapefile.iterrows():
#     # Create a GeoDataFrame for the individual polygon
#     polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)

#     # Spatial join or intersection
#     clipped_points_ind = gpd.sjoin(gdf_points, polygon_gdf, how='inner', predicate='intersects').index.tolist()
#     #### Baseflow values
#     print(len(clipped_points_ind))
#     if len(clipped_points_ind) > 10:
#         sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])[clipped_points_ind]
#         sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])[clipped_points_ind]
#         sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])[clipped_points_ind]
#         sim_clm = np.array(slope_dict['CLM40_Furey_yearly'])[clipped_points_ind]
#         obs = np.array(slope_dict['obs_flow_Furey_yearly'])[clipped_points_ind]
#         min_val = np.nanmin(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))
#         max_val = np.nanmax(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))
#         ax[j].plot(np.array([min_val, max_val]), np.array([min_val, max_val]), lw=1.5, label="1:1 Line")
#         bias = np.nansum(np.abs(sim_dPL - obs))
#         mask = (obs==obs) & (sim_dPL==sim_dPL)
#         _, _, r, _, _ = scipy.stats.linregress(sim_dPL[mask], obs[mask])
#         ax[j].scatter(obs, sim_dPL,
#                       c=colors[0], s=200, marker=markers[0], label="dPL, R2: " + str((r**2).round(3)))
#         bias = np.nansum(np.abs(sim_ghm - obs))
#         mask = (obs==obs) & (sim_ghm==sim_ghm)
#         _, _, r, _, _ = scipy.stats.linregress(sim_ghm[mask], obs[mask])
#         ax[j].scatter(obs, sim_ghm,
#                       c=colors[1], s=200, marker=markers[2], label="GHM (ens), R2: " + str((r**2).round(3)))
#         bias = np.nansum(np.abs(sim_clm - obs))
#         mask = (obs==obs) & (sim_clm==sim_clm)
#         _, _, r, _, _ = scipy.stats.linregress(sim_clm[mask], obs[mask])
#         ax[j].scatter(obs, sim_clm,
#                       c=colors[3], s=200, marker=markers[3], label="CLM40, R2: " + str((r**2).round(3)))
#         bias = np.nansum(np.abs(sim_dPL_high_res - obs))
#         mask = (obs==obs) & (sim_dPL_high_res==sim_dPL_high_res)
#         _, _, r, _, _ = scipy.stats.linregress(sim_dPL_high_res[mask], obs[mask])
#         ax[j].scatter(obs, sim_dPL_high_res,
#                       c=colors[2], s=240, marker=markers[4], label="dPL(high res), R2: " + str((r**2).round(3)))

#         ax[j].legend(loc="upper left", fontsize=14)  # Adjust legend location and font size
#         # Increase font size of x and y tick labels
#         ax[j].tick_params(axis='both', labelsize=21)
#         j = j + 1
#         zone_list_label.append("zone " + str(i))
#     ## for the subplot in number 12, I put all combined. zince zone 5 does not have enough points, one slot gets free.
# sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])
# sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])
# sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])
# sim_clm = np.array(slope_dict['CLM40_Furey_yearly'])
# obs = np.array(slope_dict['obs_flow_Furey_yearly'])
# j = 11
# min_val = np.nanmin(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm)))
# max_val = np.nanmax(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm)))
# ax[j].plot(np.array([min_val, max_val]), np.array([min_val, max_val]), lw=1.5, label="1:1 Line")
# bias = np.nansum(np.abs(sim_dPL - obs))
# print(bias)
# mask = (obs==obs) & (sim_dPL==sim_dPL)
# _, _, r, _, _ = scipy.stats.linregress(sim_dPL[mask], obs[mask])
# ax[j].scatter(obs, sim_dPL,
#                 c=colors[0], s=200, marker=markers[0], label="dPL, R2: " + str((r**2).round(3)))
# bias = np.nansum(np.abs(sim_ghm - obs))
# print(bias)
# mask = (obs==obs) & (sim_ghm==sim_ghm)
# _, _, r, _, _ = scipy.stats.linregress(sim_ghm[mask], obs[mask])
# ax[j].scatter(obs, sim_ghm,
#                 c=colors[1], s=200, marker=markers[2], label="GHM (ens), R2: " + str((r**2).round(3)))
# bias = np.nansum(np.abs(sim_clm - obs))
# print(bias)
# mask = (obs==obs) & (sim_clm==sim_clm)
# _, _, r, _, _ = scipy.stats.linregress(sim_clm[mask], obs[mask])
# ax[j].scatter(obs, sim_clm,
#                 c=colors[3], s=200, marker=markers[3], label="CLM40, R2: " + str((r**2).round(3)))
# bias = np.nansum(np.abs(sim_dPL_high_res - obs))
# print(bias)
# mask = (obs==obs) & (sim_dPL_high_res==sim_dPL_high_res)
# _, _, r, _, _ = scipy.stats.linregress(sim_dPL_high_res[mask], obs[mask])
# ax[j].scatter(obs, sim_dPL_high_res,
#                 c=colors[2], s=200, marker=markers[4], label="dPL (high res), R2: " + str((r**2).round(3)))

# ax[j].legend(loc="upper left", fontsize=14)  # Adjust legend location and font size
# # Increase font size of x and y tick labels
# ax[j].tick_params(axis='both', labelsize=21)
# zone_list_label.append("all zones")


# # Add shared labels
# # fig.supxlabel("Observed baseflow trend (mm/year/year)", fontsize=24)
# # fig.supylabel("Simulated baseflow trend (mm/year/year)", fontsize=24)
# fig.text(0.5, 0.02, "Observed baseflow trend (mm/year/year)", ha='center', va='center', fontsize=29)  # Shared x-axis label
# fig.text(0.02, 0.5, "Simulated baseflow trend (mm/year/year)", ha='center', va='center', rotation='vertical', fontsize=29)  # Shared y-axis label
# # # Add x-axis labels to all subplots
# for i, a in enumerate(ax):
#     # a.set_xlabel("Observed baseflow trend in " + zone_list_label[i], fontsize=20)
#     a.set_title(zone_list_label[i], fontsize=22, pad=6)

# # # Add y-axis labels only to the first subplot of each row
# # rows = 3
# # cols = 4
# # for r in range(rows):
# #     first_col_index = r * cols
# #     ax[first_col_index].set_ylabel("Simulated baseflow trend", fontsize=22)

# fig.patch.set_facecolor('white')
# fig.subplots_adjust(
#         left=0.06,
#         bottom=0.06,
#         right=0.99,
#         top=0.94,
#         wspace=0.20,    # 0.4
#         hspace=0.15
# )

# title = "Baseflow trend comparison of GSWP3 in GHM and dPL(trained on Daymet) models in " + tr_time_start[:4] + "_" + tr_time_end1[:4] #+ " (mm/year/mm)"
# fig.suptitle(title, fontsize=30, y=0.985)

# plt.savefig(os.path.join(fig_out_dir, "evaluation_figures", "bf_trend_hres_scatter_zones_dmt_GSWP3"+ ".png"), dpi=300)
# plt.close("all")
# print("END")


#################################
########## scatter plots for new zones###
### basin path
# shp_path0 = r"/scratch/fzr5082/PGML_STemp_results/inputs/Zones"

shapefile = gpd.read_file(os.path.join(shp_path0, "Zones_0228.shp"))
shapefile = shapefile.to_crs(gdf_points.crs)

colors = ['black', 'red', 'orange', 'blue', 'pink', 'green', 'yellow']
markers = ['s', "o", "+", "*", "D", "x", "D", ]

fig, axs = plt.subplots(3, 4, figsize=(24, 18))
ax = axs.flatten()
j = 0
zone_list_label = list()
for i, polygon_row in shapefile.iterrows():
    # Create a GeoDataFrame for the individual polygon
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)

    # Spatial join or intersection
    clipped_points_ind = gpd.sjoin(gdf_points, polygon_gdf, how='inner', predicate='intersects').index.tolist()
    #### Baseflow values
    print(len(clipped_points_ind))
    if len(clipped_points_ind) > 10:
        sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])[clipped_points_ind]
        sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])[clipped_points_ind]
        sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])[clipped_points_ind]
        sim_clm = np.array(slope_dict['DBH_Furey_yearly'])[clipped_points_ind]
        obs = np.array(slope_dict['obs_flow_Furey_yearly'])[clipped_points_ind]
        min_val = np.nanmin(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))
        max_val = np.nanmax(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))
        ax[j].plot(np.array([min_val, max_val]), np.array([min_val, max_val]), lw=1.5, label=None)
        bias = np.nansum(np.abs(sim_dPL - obs))
        mask = (obs == obs) & (sim_dPL == sim_dPL)
        _, _, r, _, _ = scipy.stats.linregress(sim_dPL[mask], obs[mask])
        # ax[j].scatter(obs, sim_dPL,
        #               c=colors[0], s=200, marker=markers[0], label="\u03B4PS, R2: " + str((r**2).round(3)))
        # R2_dPL_dmt_low = calculate_R2(obs[mask] , sim_dPL[mask])
        if polygon_row["new_FID"] == 7:
            m_name = "\u03B4PS-"
        else:
            m_name = ""
        label_dPL_low_res = f"{m_name}L2: {(Rel_L2(sim_dPL[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
        ax[j].scatter(obs, sim_dPL,
                      c=colors[0], s=200, marker=markers[0], label=label_dPL_low_res)
        bias = np.nansum(np.abs(sim_ghm - obs))
        mask = (obs == obs) & (sim_ghm == sim_ghm)
        _, _, r, _, _ = scipy.stats.linregress(sim_ghm[mask], obs[mask])
        # ax[j].scatter(obs, sim_ghm,
        #               c=colors[1], s=200, marker=markers[2], label="GHM (ens), R2: " + str((r**2).round(3)))
        # R2_GHM_ens = calculate_R2(obs[mask] , sim_ghm[mask])
        if polygon_row["new_FID"] == 7:
            m_name = "GHM(ens)-"
        else:
            m_name = ""
        label_GHM_ens = f"{m_name}L2: {(Rel_L2(sim_ghm[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
        ax[j].scatter(obs, sim_ghm,
                      c=colors[1], s=200, marker=markers[1], label=label_GHM_ens)
        bias = np.nansum(np.abs(sim_clm - obs))
        mask = (obs == obs) & (sim_clm == sim_clm)
        _, _, r, _, _ = scipy.stats.linregress(sim_clm[mask], obs[mask])
        # ax[j].scatter(obs, sim_clm,
        #               c=colors[3], s=200, marker=markers[3], label="CLM40, R2: " + str((r**2).round(3)))
        # R2_clm = calculate_R2(obs[mask] , sim_clm[mask])
        if polygon_row["new_FID"] == 7:
            m_name = "DBH-"
        else:
            m_name = ""
        label_GHM_ens = f"{m_name}L2: {(Rel_L2(sim_clm[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
        ax[j].scatter(obs, sim_clm,
                      c=colors[2], s=200, marker=markers[2], label=label_GHM_ens)
        bias = np.nansum(np.abs(sim_dPL_high_res - obs))
        mask = (obs == obs) & (sim_dPL_high_res == sim_dPL_high_res)
        _, _, r, _, _ = scipy.stats.linregress(sim_dPL_high_res[mask], obs[mask])
        # ax[j].scatter(obs, sim_dPL_high_res,
        #               c=colors[2], s=240, marker=markers[4], label="\u03B4PS(high res), R2: " + str((r**2).round(3)))
        R2_dPL_dmt_high = calculate_R2(obs[mask], sim_dPL_high_res[mask])
        if polygon_row["new_FID"] == 7:
            m_name = "\u03B4PS(Daymet)-"
        else:
            m_name = ""
        label_dPL_high_res = f"{m_name}L2: {(Rel_L2(sim_dPL_high_res[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
        ax[j].scatter(obs, sim_dPL_high_res,
                      c=colors[3], s=240, marker=markers[3], label=label_dPL_high_res)

        ax[j].legend(loc="upper left", fontsize=17)  # Adjust legend location and font size
        # Increase font size of x and y tick labels
        ax[j].tick_params(axis='both', labelsize=21)
        j = j + 1
        zone_list_label.append("Zone " + str(polygon_row["new_FID"]))
    ## for the subplot in number 12, I put all combined. zince zone 5 does not have enough points, one slot gets free.
sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])
sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])
sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])
sim_clm = np.array(slope_dict['DBH_Furey_yearly'])
obs = np.array(slope_dict['obs_flow_Furey_yearly'])
j = 11
min_val = np.nanmin(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm)))
max_val = np.nanmax(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm)))
ax[j].plot(np.array([min_val, max_val]), np.array([min_val, max_val]), lw=1.5, label=None)
bias = np.nansum(np.abs(sim_dPL - obs))
print(bias)
mask = (obs == obs) & (sim_dPL == sim_dPL)
_, _, r, _, _ = scipy.stats.linregress(sim_dPL[mask], obs[mask])
# ax[j].scatter(obs, sim_dPL,
#                 c=colors[0], s=200, marker=markers[0], label="\u03B4PS, R2: " + str((r**2).round(3)))
# R2_dPL_dmt_low = calculate_R2(obs[mask] , sim_dPL[mask])
label_dPL_low_res = f"L2: {(Rel_L2(sim_dPL[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
ax[j].scatter(obs, sim_dPL,
              c=colors[0], s=200, marker=markers[0], label=label_dPL_low_res)
bias = np.nansum(np.abs(sim_ghm - obs))
print(bias)
mask = (obs == obs) & (sim_ghm == sim_ghm)
_, _, r, _, _ = scipy.stats.linregress(sim_ghm[mask], obs[mask])
# ax[j].scatter(obs, sim_ghm,
#                 c=colors[1], s=200, marker=markers[2], label="GHM (ens), R2: " + str((r**2).round(3)))
R2_GHM_ens = calculate_R2(obs[mask], sim_ghm[mask])
label_GHM_ens = f"L2: {(Rel_L2(sim_ghm[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
ax[j].scatter(obs, sim_ghm,
              c=colors[1], s=200, marker=markers[1], label=label_GHM_ens)
bias = np.nansum(np.abs(sim_clm - obs))
print(bias)
mask = (obs == obs) & (sim_clm == sim_clm)
_, _, r, _, _ = scipy.stats.linregress(sim_clm[mask], obs[mask])
# ax[j].scatter(obs, sim_clm,
#                 c=colors[3], s=200, marker=markers[3], label="CLM40, R2: " + str((r**2).round(3)))
# R2_clm = calculate_R2(obs[mask] , sim_clm[mask])
label_GHM_ens = f"L2: {(Rel_L2(sim_clm[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
ax[j].scatter(obs, sim_clm,
              c=colors[2], s=200, marker=markers[2], label=label_GHM_ens)
bias = np.nansum(np.abs(sim_dPL_high_res - obs))
print(bias)
mask = (obs == obs) & (sim_dPL_high_res == sim_dPL_high_res)
_, _, r, _, _ = scipy.stats.linregress(sim_dPL_high_res[mask], obs[mask])
# ax[j].scatter(obs, sim_dPL_high_res,
#                 c=colors[2], s=200, marker=markers[4], label="\u03B4PS(high res), R2: " + str((r**2).round(3)))
# R2_dPL_dmt_high = calculate_R2(obs[mask] , sim_dPL_high_res[mask])
label_dPL_high_res = f"L2: {(Rel_L2(sim_dPL_high_res[mask], obs[mask])):.2f}, r2: {(r ** 2):.2f}"
ax[j].scatter(obs, sim_dPL_high_res,
              c=colors[3], s=240, marker=markers[3], label=label_dPL_high_res)
ax[j].legend(loc="upper left", fontsize=17)  # Adjust legend location and font size
# Increase font size of x and y tick labels
ax[j].tick_params(axis='both', labelsize=21)
zone_list_label.append("All zones")

# Add shared labels
# fig.supxlabel("Observed baseflow trend (mm/year/year)", fontsize=24)
# fig.supylabel("Simulated baseflow trend (mm/year/year)", fontsize=24)
fig.text(0.5, 0.02, "Observed baseflow trend (mm/year/year)", ha='center', va='center',
         fontsize=29)  # Shared x-axis label
fig.text(0.02, 0.5, "Simulated baseflow trend (mm/year/year)", ha='center', va='center', rotation='vertical',
         fontsize=29)  # Shared y-axis label
# # Add x-axis labels to all subplots
for i, a in enumerate(ax):
    # a.set_xlabel("Observed baseflow trend in " + zone_list_label[i], fontsize=20)
    a.set_title(zone_list_label[i], fontsize=22, pad=6)

# # Add y-axis labels only to the first subplot of each row
# rows = 3
# cols = 4
# for r in range(rows):
#     first_col_index = r * cols
#     ax[first_col_index].set_ylabel("Simulated baseflow trend", fontsize=22)

fig.patch.set_facecolor('white')
fig.subplots_adjust(
    left=0.06,
    bottom=0.06,
    right=0.99,
    top=0.94,
    wspace=0.20,  # 0.4
    hspace=0.15
)

title = "Baseflow trend comparison of GHM and \u03B4 models in " + tr_time_start[:4] + "-" + tr_time_end1[
    :4]  # + " (mm/year/mm)"
fig.suptitle(title, fontsize=30, y=0.985)

plt.savefig(
    os.path.join(fig_out_dir, "evaluation_figures", "bf_trend_hres_scatter_zones_dmt_GSWP3_zone0228_DBH" + ".png"),
    dpi=600)
plt.close("all")
print("END")

############################
####Adding p-value for baseflow
#################################
########## scatter plots for new zones###
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import geopandas as gpd

# =================================================
# USER-DEFINED SIGNIFICANCE THRESHOLDS
# Change these however you want
# Example:
#   p < 0.001  -> ***
#   p < 0.01   -> **
#   p < 0.05   -> *
#   otherwise  -> ns
# =================================================
SIG_LEVEL_3 = 0.001  # *** very very strong significance
SIG_LEVEL_2 = 0.01  # ** very strong significance
SIG_LEVEL_1 = 0.05  # * significant
NS_LABEL = "ns"  # label for non-significant

# If True, legend will also show the p-value next to stars
SHOW_P_VALUE = False

# Legend sizes
DEFAULT_LEGEND_SIZE = 17
ZONE7_LEGEND_SIZE = 14
ZONE12_LEGEND_SIZE = 17


# =================================================
# Convert p-value to significance stars
# =================================================
def p_to_stars(p):
    if p < SIG_LEVEL_3:
        return "***"
    elif p < SIG_LEVEL_2:
        return "**"
    elif p < SIG_LEVEL_1:
        return "*"
    else:
        return NS_LABEL


# =================================================
# Build legend label
# force_prefix=False means do not write model name
# =================================================
def format_sig_label(model_name, sim, obs, force_prefix=True):
    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() < 3:
        if force_prefix and model_name != "":
            return f"{model_name}NA, L2: NA, r2: NA"
        else:
            return f"NA, L2: NA, r2: NA"

    slope, intercept, r, p_value, std_err = scipy.stats.linregress(sim[mask], obs[mask])

    sig_star = p_to_stars(p_value)
    l2_val = Rel_L2(sim[mask], obs[mask])
    r2_val = r ** 2

    prefix = model_name if force_prefix else ""

    if SHOW_P_VALUE:
        if p_value < 0.001:
            p_txt = f"{p_value:.1e}"
        else:
            p_txt = f"{p_value:.3f}"
        return f"{prefix}{sig_star}, p: {p_txt}, L2: {l2_val:.2f}, r2: {r2_val:.2f}"
    else:
        return f"{prefix}{sig_star}, L2: {l2_val:.2f}, r2: {r2_val:.2f}"


# =================================================
# Read shapefile
# =================================================
shapefile = gpd.read_file(os.path.join(shp_path0, "Zones_0228.shp"))
shapefile = shapefile.to_crs(gdf_points.crs)

colors = ['black', 'red', 'orange', 'blue', 'pink', 'green', 'yellow']
markers = ['s', 'o', '+', '*', 'D', 'x', 'D']

fig, axs = plt.subplots(3, 4, figsize=(24, 18))
ax = axs.flatten()

j = 0
zone_list_label = []

# =================================================
# Loop through zones
# =================================================
for i, polygon_row in shapefile.iterrows():
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)

    clipped_points_ind = gpd.sjoin(
        gdf_points, polygon_gdf, how='inner', predicate='intersects'
    ).index.tolist()

    print(len(clipped_points_ind))

    if len(clipped_points_ind) > 10:
        sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])[clipped_points_ind]
        sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])[clipped_points_ind]
        sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])[clipped_points_ind]
        sim_clm = np.array(slope_dict['DBH_Furey_yearly'])[clipped_points_ind]
        obs = np.array(slope_dict['obs_flow_Furey_yearly'])[clipped_points_ind]

        min_val = np.nanmin(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))
        max_val = np.nanmax(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))

        ax[j].plot(
            np.array([min_val, max_val]),
            np.array([min_val, max_val]),
            lw=1.5,
            label=None
        )

        zone_id = polygon_row["new_FID"]

        # For Zone 7: keep model names
        # For all other individual zones: no model names
        if zone_id == 7:
            dpl_name = "\u03B4PS-"
            ghm_name = "GHM(ens)-"
            clm_name = "DBH-"
            dpl_hr_name = "\u03B4PS(Daymet)-"
            use_prefix = True
        else:
            dpl_name = ""
            ghm_name = ""
            clm_name = ""
            dpl_hr_name = ""
            use_prefix = False

        label_dPL_low_res = format_sig_label(dpl_name, sim_dPL, obs, force_prefix=use_prefix)
        label_GHM_ens = format_sig_label(ghm_name, sim_ghm, obs, force_prefix=use_prefix)
        label_clm = format_sig_label(clm_name, sim_clm, obs, force_prefix=use_prefix)
        label_dPL_high_res = format_sig_label(dpl_hr_name, sim_dPL_high_res, obs, force_prefix=use_prefix)

        ax[j].scatter(
            obs, sim_dPL,
            c=colors[0], s=200, marker=markers[0],
            label=label_dPL_low_res
        )

        ax[j].scatter(
            obs, sim_ghm,
            c=colors[1], s=200, marker=markers[1],
            label=label_GHM_ens
        )

        ax[j].scatter(
            obs, sim_clm,
            c=colors[2], s=200, marker=markers[2],
            label=label_clm
        )

        ax[j].scatter(
            obs, sim_dPL_high_res,
            c=colors[3], s=240, marker=markers[3],
            label=label_dPL_high_res
        )

        # Smaller legend only for Zone 7
        if zone_id == 7:
            ax[j].legend(
                loc="upper left",
                fontsize=ZONE7_LEGEND_SIZE,
                frameon=True,
                borderpad=0.3,
                labelspacing=0.3,
                handletextpad=0.4
            )
        else:
            ax[j].legend(
                loc="upper left",
                fontsize=DEFAULT_LEGEND_SIZE,
                frameon=True,
                borderpad=0.3,
                labelspacing=0.3,
                handletextpad=0.4
            )

        ax[j].tick_params(axis='both', labelsize=21)

        zone_list_label.append("Zone " + str(zone_id))
        j += 1

# =================================================
# Last subplot: all zones combined
# No model names here to save space
# =================================================
sim_dPL = np.array(slope_dict['mSS_Furey_yearly'])
sim_dPL_high_res = np.array(slope_dict['mSS_Daymet_Furey_yearly'])
sim_ghm = np.array(slope_dict['GHM_ens_Furey_yearly'])
sim_clm = np.array(slope_dict['DBH_Furey_yearly'])
obs = np.array(slope_dict['obs_flow_Furey_yearly'])

j = 11

min_val = np.nanmin(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))
max_val = np.nanmax(np.concatenate((obs, sim_dPL, sim_ghm, sim_clm, sim_dPL_high_res)))

ax[j].plot(
    np.array([min_val, max_val]),
    np.array([min_val, max_val]),
    lw=1.5,
    label=None
)

# No model names in Zone 12 / All zones
label_dPL_low_res = format_sig_label("", sim_dPL, obs, force_prefix=False)
label_GHM_ens = format_sig_label("", sim_ghm, obs, force_prefix=False)
label_clm = format_sig_label("", sim_clm, obs, force_prefix=False)
label_dPL_high_res = format_sig_label("", sim_dPL_high_res, obs, force_prefix=False)

ax[j].scatter(
    obs, sim_dPL,
    c=colors[0], s=200, marker=markers[0],
    label=label_dPL_low_res
)

ax[j].scatter(
    obs, sim_ghm,
    c=colors[1], s=200, marker=markers[1],
    label=label_GHM_ens
)

ax[j].scatter(
    obs, sim_clm,
    c=colors[2], s=200, marker=markers[2],
    label=label_clm
)

ax[j].scatter(
    obs, sim_dPL_high_res,
    c=colors[3], s=240, marker=markers[3],
    label=label_dPL_high_res
)

ax[j].legend(
    loc="upper left",
    fontsize=ZONE12_LEGEND_SIZE,
    frameon=True,
    borderpad=0.3,
    labelspacing=0.3,
    handletextpad=0.4
)

ax[j].tick_params(axis='both', labelsize=21)
zone_list_label.append("All zones")

# =================================================
# Shared labels and titles
# =================================================
fig.text(
    0.5, 0.02,
    "Observed baseflow trend (mm/year/year)",
    ha='center', va='center', fontsize=29
)

fig.text(
    0.02, 0.5,
    "Simulated baseflow trend (mm/year/year)",
    ha='center', va='center', rotation='vertical', fontsize=29
)

for i, a in enumerate(ax):
    if i < len(zone_list_label):
        a.set_title(zone_list_label[i], fontsize=22, pad=6)

fig.patch.set_facecolor('white')

fig.subplots_adjust(
    left=0.06,
    bottom=0.06,
    right=0.99,
    top=0.94,
    wspace=0.20,
    hspace=0.15
)

title = "Baseflow trend comparison of GHM and \u03B4 models in " + tr_time_start[:4] + "-" + tr_time_end1[:4]
fig.suptitle(title, fontsize=30, y=0.985)

plt.savefig(
    os.path.join(
        fig_out_dir,
        "evaluation_figures",
        "bf_trend_hres_scatter_zones_dmt_GSWP3_zone0228_DBH.png"
    ),
    dpi=600
)

plt.close("all")
print("END")