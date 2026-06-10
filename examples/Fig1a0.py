from os.path import exists

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import scipy
import json
import scipy.stats
import shutil
import numpy as np
import scipy.stats
import math
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
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
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
from matplotlib.patches import FancyBboxPatch
from post.stat_plots import flatData, plotMap, plotBoxFig, statError
from post.read_GHMs_dPLs import (
    read_GHM_ISIMIP2a_daily, read_dPL_ISIMIP2a_daily, read_GHM_ISIMIP2a_2003basins,
    converting_daily_to_monthly, converting_daily_to_yearly, 
    read_GHM_ISIMIP2a_monthly, converting_monthly_to_yearly, read_dPL_recharge_Daymet_daily
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

def find_nearest_point(point, point_gdf):
    nearest_geom = point_gdf.geometry.apply(lambda x: point.distance(x)).idxmin()
    return point_gdf.loc[nearest_geom]

# Function to find the nearest point and return its index
def find_nearest_point_index(point, point_gdf):
    nearest_geom_index = point_gdf.geometry.apply(lambda x: point.distance(x)).idxmin()
    return nearest_geom_index 


p0 = r"D:\\DR\\M"
fig_out_dir = r"D:\\DR"
#######################################
### WATERGAP2_2C
## Load the original NetCDF file to get the latitude, longitude, and time grids
p0_GHMs = r"D:\\DR\\data\\ISIMIP2a"
rech_GHM = xr.open_dataset(os.path.join(p0_GHMs, "GHMs", "Watergap2_2c", "GSWP3",
                                           "watergap2-2c_gswp3_nobc_hist_nosoc_co2_qr_global_monthly_1901_2010.nc4"))["qr"]
bounds_USA = [-127, 20.9, -66, 50.4]
rech_GHM_USA = rech_GHM.sel(lat=slice(bounds_USA[3], bounds_USA[1]),
                                            lon=slice(bounds_USA[0], bounds_USA[2]))
start_date = "1901-01-01"
end_date = "2011-01-01"
date_range = pd.date_range(start=start_date,end=end_date, freq='ME')
# ind = date_range >= "1972-01-01"
ind = (date_range > "1980-12-31") & (date_range < "2011-01-01")
rech_GHM_USA2 = rech_GHM_USA[ind, :, :] * 86400 * 30   
rech_GHM_USA_yearly = rech_GHM_USA2.sum(dim="time") / np.floor(rech_GHM_USA2.shape[0] / 12)   # monthly to yearly
####################################
# ## WAYS
# p0_GHMs = r"/scratch/fzr5082/PGML_STemp_results/data/ISIMIP2a"
# time_ranges = ["1971_1980", "1981_1990", "1991_2000", "2001_2010"]
# GHM_list = []
# for t_range in time_ranges:
#     rech_GHM = xr.open_dataset(os.path.join(p0_GHMs, "GHMs", "WAYS", "GSWP3",
#                                            "ways_gswp3_nobc_hist_nosoc_noco2_qr_global_daily_"+ t_range + ".nc4"))["qr"]

#     # clipping with bounds of USA
#     bounds_USA = [-127, 20.9, -66, 50.4]
#     rech_GHM_USA = rech_GHM.sel(lat=slice(bounds_USA[3], bounds_USA[1]),
#                                             lon=slice(bounds_USA[0], bounds_USA[2]))
#     GHM_list.append(rech_GHM_USA)
# rech_GHM_USA = xr.concat(GHM_list, dim='time')

# start_date = "1971-01-01"
# end_date = "2010-12-31"
# date_range = pd.date_range(start=start_date,end=end_date, freq='D')
# ind = date_range >= "1971-01-01"
# rech_GHM_USA2 = rech_GHM_USA[ind, :, :]
# # rech_GHM_USA_yearly = rech_GHM_USA.sum(dim="time") / np.floor(rech_GHM_USA2.shape[0] / 12)   # monthly to yearly
# rech_GHM_USA_yearly = rech_GHM_USA.sum(dim="time") / np.floor(rech_GHM_USA2.shape[0] / 365)     # daily to yearly
##################################
# taking 208 huc12 basins (highest resolution)
p1 = r"D:\\DR\\data\\recharge_208basins"
recharge_huc12_obs_208 = pd.read_csv(os.path.join(p1, "huc12_recharge_208.csv"), header=0)
obs_208 = np.array(recharge_huc12_obs_208["Groundwater recharge [mm/y]_mean"])
site_no_208 = recharge_huc12_obs_208["site_no"].tolist()
p2 = r"D:\\DR\\data\\ts_data_4231"
attr4231 = np.load(os.path.join(p2, "attr_HUC12_4231_grid_clip_20250224.npy"))
json_file_path = os.path.join(p2, "attr_HUC12_4231_grid_clip_20250224_name.json")
with open(json_file_path, "r") as json_file:
    attr_name_np_4231 = json.load(json_file)
site_no_4231 = attr4231[:, attr_name_np_4231.index("site_no_int")].tolist()
recharge_4231 = read_dPL_recharge_Daymet_daily(model_name="mSS", 
                                                start_date_mask="1980-12-31",
                                                end_date_mask="2010-12-31",
                                                site_ind_list=np.arange(4231), 
                                                start_date_sim="1980-01-01", 
                                                end_date_sim="2023-01-01", 
                                                dir0=r"D:\\DR\\M\\daymet_1223_1023_PUB_huc12_dmt_4231\\")
ind_208 = list()
for i, s in enumerate(site_no_208):
    if s in site_no_4231:
        ind_208.append(np.where(np.array(site_no_4231)==s)[0][0])
attr208 = attr4231[ind_208, :]
recharge_208 = read_dPL_recharge_Daymet_daily(model_name="mSS", 
                                                start_date_mask="1980-12-31",
                                                end_date_mask="2010-12-31",
                                                site_ind_list=np.array(ind_208), 
                                                start_date_sim="1980-01-01", 
                                                end_date_sim="2023-01-01", 
                                                dir0=r"D:\\DR\\M\\daymet_1223_1023_PUB_huc12_dmt_4231\\")

#################################
p0_dPL = r"D:\\DR\\M\\upscale"
rech_dPL_big_basins = xr.open_dataset(os.path.join(p0_dPL, "mSS_recharge_grid.nc"))["mSS_recharge"]
rech_dPL_huc12_dmt = xr.open_dataset(os.path.join(p0_dPL, "mSS_recharge_grid_huc12_dmt_4231_1980_2023.nc"))["mSS_recharge_huc12_dmt"]
rech_dPL_huc12_GSWP3 = xr.open_dataset(os.path.join(p0_dPL, "mSS_recharge_grid_huc12_GSWP3_4238_1962_2011.nc"))["mSS_recharge_huc12_GSWP3"]
# rech_dPL = xr.open_dataset(os.path.join(p0_dPL, "mSS_recharge_grid_GSWP3_huc12_4328_1961_2011.nc"))["mSS_recharge_huc12"]
start_date = '1980-12-31' #"1962-01-01"
end_date = '2022-12-31'    #"2010-12-31"
date_range = pd.date_range(start=start_date,end=end_date, freq='D')
ind = (date_range > "1972-01-01") & (date_range < "2011-01-01")
rech_dPL_big_basins_1980_2011 = rech_dPL_big_basins[ind, :, :]
rech_dPL_big_basins_mean_yearly = rech_dPL_big_basins_1980_2011.sum(dim="time") / np.floor(rech_dPL_big_basins_1980_2011.shape[0]/ 365)


rech_dPL_huc12_dmt_1980_2011 = rech_dPL_huc12_dmt[ind, :, :]
rech_dPL_huc12_dmt_mean_yearly = rech_dPL_huc12_dmt_1980_2011.sum(dim="time") / np.floor(rech_dPL_huc12_dmt_1980_2011.shape[0]/ 365)


start_date = '1962-01-01' #"1962-01-01"
end_date = '2010-12-31'    #"2010-12-31"
date_range = pd.date_range(start=start_date,end=end_date, freq='D')
# ind = date_range > "1972-01-01"
ind = (date_range > "1980-12-31")
rech_dPL_huc12_GSWP3_1962_2011 = rech_dPL_huc12_GSWP3[ind, :, :]
rech_dPL_huc12_GSWP3_mean_yearly = rech_dPL_huc12_GSWP3_1962_2011.sum(dim="time") / np.floor(rech_dPL_huc12_GSWP3_1962_2011.shape[0]/ 365)


rech_obs_mean = xr.open_dataset(os.path.join(p0_dPL, "obs_mean_recharge_grid.nc"))["obs_recharge"]

correlation_big_basins = xr.corr(rech_obs_mean.stack(z=('lat', 'lon')), rech_dPL_big_basins_mean_yearly.stack(z=('lat', 'lon')), dim='z')
correlation_huc12_dmt = xr.corr(rech_obs_mean.stack(z=('lat', 'lon')), rech_dPL_huc12_dmt_mean_yearly.stack(z=('lat', 'lon')), dim='z')
correlation_huc12_GSWP3 = xr.corr(rech_obs_mean.stack(z=('lat', 'lon')), rech_dPL_huc12_GSWP3_mean_yearly.stack(z=('lat', 'lon')), dim='z')
correlation_GHM = xr.corr(rech_obs_mean.stack(z=('lat', 'lon')), rech_GHM_USA_yearly.stack(z=('lat', 'lon')), dim='z')


#################################################
######### ploting
fig = plt.figure(1, figsize = (13,13))
colors = ['black', 'blue', 'red', 'orange', 'pink', 'green', 'yellow']
markers = ['s', "D", "o", "*", "+", "D", "x"]



plt.plot(np.array([0,800]), np.array([0,800]), lw = 1.5, label="_nolegend_")
# Adjust the margins
plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.09)

mask = ~np.isnan(rech_obs_mean.values.flatten()) 
item1 = rech_obs_mean.values.flatten()[mask]
item2 = rech_dPL_big_basins_mean_yearly.values.flatten()[mask]
item3 = rech_dPL_huc12_dmt_mean_yearly.values.flatten()[mask]
item4 = rech_GHM_USA_yearly.values.flatten()[mask]
item5 = rech_dPL_huc12_GSWP3_mean_yearly.values.flatten()[mask]
mask_rain_dPL = ~((item1 > 200) & (item2 < 50))
mask_rain_GHM = ~((item1 > 200) & (item3 < 50))


mask2 = ~np.isnan(obs_208) 
item6 = obs_208[mask2]
rech_208_high_res = recharge_208[:, mask2]
item7 = np.nansum(rech_208_high_res, axis=0)/np.floor((recharge_208.shape[0]) / 365)
# plt.scatter(item1[mask_rain_dPL], item2[mask_rain_dPL], 
#                 c = colors[0], s = 200, marker = markers[0])
# plt.scatter(item1[mask_rain_GHM], item3[mask_rain_GHM], 
#                 c = colors[1], s = 200, marker = markers[1])
plt.scatter(item1, item4, 
                c = colors[2], s = 200, marker = markers[2])
R2_watergap = calculate_R2(item1 , item4)

plt.scatter(item1, item5, 
                c = colors[0], s = 200, marker = markers[0])
R2_dPL_GSWP3 = calculate_R2(item1 , item5)

plt.scatter(item1, item3, 
                c = colors[1], s = 200, marker = markers[1])
R2_dPL_dmt_low = calculate_R2(item1 , item3)

plt.scatter(item6, item7, 
                c = colors[3], s = 200, marker = markers[3])
R2_dPL_dmt_high = calculate_R2(item6 , item7)

#     leg = "r= " + str("{:.3f}".format(r))
leg = [f"Watergap2_2c(GSWP3, 0.5\u00b0)-L2: {(Rel_L2(item4, item1)):.2f}, R2: {(R2_watergap):.2f}", 
        f"\u03B4PS(GSWP3, HUC12 to 0.5\u00b0)-L2: {(Rel_L2(item5, item1)):.2f}, R2: {(R2_dPL_GSWP3):.2f}",
        f"\u03B4PS(Daymet, HUC12 to 0.5\u00b0)-L2: {(Rel_L2(item3, item1)):.2f}, R2: {(R2_dPL_dmt_low):.2f}", 
        f"\u03B4PS(Daymet, HUC12)-L2: {(Rel_L2(item7, item6)):.2f}, R2: {(R2_dPL_dmt_high):.2f}"]    # "HBV",
plt.legend(leg, loc='upper right', title = 'Models', fontsize=25, title_fontsize=25)
plt.xlim(-10.0, 800)
plt.ylim(-10.0, 800)
plt.grid()
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.xlabel("Recharge observations (mm/year)", fontsize=35)
plt.ylabel("Recharge simulations (mm/year)", fontsize=35)
plt.title("Observed and simulated recharge \n in \u03B4 and GHM modelings", fontsize=35)
os.makedirs(os.path.join(p0, "evaluation_figures"), exist_ok=True)
plt.savefig(os.path.join(p0, "evaluation_figures",  "rech_dPL_huc12_dmt_GSWP3_hist" +  ".png"), dpi=600)
print("END")

############################
##############################
##############################
### Boxplot historical for recharge evaluations on large 2003 basins and small 208 basins:
## recharge 2003 basins from differentiable models trained on Daymet
# read attrs lat and lon
p_data = r"D:\\DR\\data"
attr_name_file_path = os.path.join(p_data, "ts_2003basins", "attr2003_mswep_03122024_name.json")
with open(attr_name_file_path, 'r') as json_file:
    # Load the JSON data from the file
    a_name_np = json.load(json_file)
  
attr_trained_dt = np.load(os.path.join(p_data, "tr_1223basins", "attr1223_1023_daymet_20240826.npy"))
site_no_trained_dt = attr_trained_dt[:, a_name_np.index("site_no_int")]

attr2003 = np.load(os.path.join(p_data, "ts_2003basins", "attr2003_mswep_03122024.npy"))
site_no_2003 = attr2003[:, a_name_np.index("site_no_int")]
ind_trained_dt = list()
for i, s in enumerate(site_no_2003):
    if s in site_no_trained_dt:
        ind_trained_dt.append(i)

points2003 = []
lat2003 = attr2003[:, a_name_np.index("lat")].tolist()   #attr_trained_dt
lon2003 = attr2003[:, a_name_np.index("lon")].tolist()   #attr_trained_dt




sim_recharge_dict = dict()
# ## recharge for 4231 HUC12
# recharge_sim_monthly = converting_daily_to_monthly(daily_data_dict={"mSS_HUC12_dmt_4231" + "_daily": recharge_4231}, 
#                                                         start_date="1980-12-31", 
#                                                         end_date="2010-12-31")
# sim_recharge_dict["mSS_HUC12_dmt_4231" + "_monthly"] = recharge_sim_monthly["mSS_HUC12_dmt_4231" + "_monthly"][0:,:]
# recharge for 208 HUC12 basins
recharge_sim_monthly = converting_daily_to_monthly(daily_data_dict={"mSS_HUC12_dmt" + "_daily": recharge_208}, 
                                                        start_date="1980-12-31", 
                                                        end_date="2010-12-31")
sim_recharge_dict["mSS_HUC12_dmt" + "_monthly"] = recharge_sim_monthly["mSS_HUC12_dmt" + "_monthly"][0:,:]
# sim_recharge_dict["mSS_HUC12_dmt"] = sim_recharge_dict["mSS_HUC12_dmt" + "_monthly"][0:,:]
## recharge for 2003 large HUC8 basins
dPL_model_name_list = ["mSS", "HyS", "HVS"]#   , "mSN", "HyN", "HVN"
for model_name in dPL_model_name_list:
    recharge_sim_daily = read_dPL_recharge_Daymet_daily(model_name=model_name, 
                                                    start_date_mask="1980-12-31",
                                                    end_date_mask="2010-12-31",
                                                    site_ind_list=np.arange(2003), 
                                                    start_date_sim="1961-01-01", 
                                                    end_date_sim="2011-01-01", 
                                                    dir0=r"D:\\DR\\M\\daymet_1223_1023_PUB")
    # converting daily to monthly
    recharge_sim_monthly = converting_daily_to_monthly(daily_data_dict={model_name + "_daily": recharge_sim_daily}, 
                                                        start_date="1980-12-31", 
                                                        end_date="2010-12-31")
    sim_recharge_dict[model_name + "_monthly"] = recharge_sim_monthly[model_name + "_monthly"][0:,:]
    print(model_name)
############ reading GHMs outputs
qr_data = r"D:\\DR\\data\\out_qr_isimip2a_gages_2003"
GHMs_list  = ["WAYS_nosoc", "Watergap2_2c_nosoc"]#, "Watergap2_2c_pressoc", "Watergap2_2c_varsoc"]
tr_time_start = "1971-01-01"
tr_time_end1 = "2010-12-31"
for model_name_GHM in GHMs_list:
    # for monthly data read
    if "Watergap2_2c" in model_name_GHM:
        recharge_sim_monthly = read_GHM_ISIMIP2a_2003basins(item_name="qr",
                                            model_name_GHM=model_name_GHM, 
                                            CLM_name="GSWP3",
                                            start_date_sim="1901-01-01", 
                                            end_date_sim="2010-12-31",
                                            start_date_mask=tr_time_start,
                                            end_date_mask=tr_time_end1, 
                                            site_no_2003_list=np.arange(2003), 
                                            site_no_subset_list=np.arange(2003), 
                                            freq="MS",
                                            p0_GHM=r"D:\\DR\\data\\out_qr_isimip2a_gages_2003\\"
                                            )
                                            
                                
        sim_recharge_dict[model_name_GHM + "_monthly"] = recharge_sim_monthly
        print(model_name_GHM)
    # for daily data read
    elif "WAYS" in model_name_GHM:
        recharge_sim_daily = read_GHM_ISIMIP2a_2003basins(item_name="qr",
                                                model_name_GHM=model_name_GHM, 
                                                CLM_name="GSWP3",
                                                start_date_sim="1971-01-01", 
                                                end_date_sim="2010-12-31",
                                                start_date_mask=tr_time_start,
                                                end_date_mask=tr_time_end1, 
                                                site_no_2003_list=np.arange(2003), 
                                                site_no_subset_list=np.arange(2003), 
                                                freq="D",
                                                p0_GHM=r"D:\\DR\\data\\out_qr_isimip2a_gages_2003\\"
                                                )   
        # converting daily to monthly
        recharge_sim_monthly = converting_daily_to_monthly(daily_data_dict={model_name_GHM + "_daily": recharge_sim_daily}, 
                                                    start_date=tr_time_start, 
                                                    end_date=tr_time_end1)
        sim_recharge_dict[model_name_GHM + "_monthly"] = recharge_sim_monthly[model_name_GHM + "_monthly"]
        print(model_name_GHM)


### reading zones shapefile
### basin path
shp_path0 = r"D:\\DR\\data\\Zones"
shapefile = gpd.read_file(os.path.join(shp_path0, "Zones_0228.shp"))
shapefile = shapefile.to_crs("EPSG:4326")

### based on my own division
## making a point gpd file first

for i in range(len(lat2003)):
    #points.append((lon[i], lat[i]))
    points2003.append((lat2003[i], lon2003[i]))
gdf_points_2003 = gpd.GeoDataFrame(geometry=[Point(longitude, latitude) for latitude, longitude in points2003],
                              crs="EPSG:4326")
#points for 208 huc12 basins
points208 = []
lat208 = attr208[:, attr_name_np_4231.index("lat")].tolist()   #attr_trained_dt
lon208 = attr208[:, attr_name_np_4231.index("lon")].tolist()   #attr_trained_dt
for i in range(len(lat208)):
    #points.append((lon[i], lat[i]))
    points208.append((lat208[i], lon208[i]))
gdf_points_208 = gpd.GeoDataFrame(geometry=[Point(longitude, latitude) for latitude, longitude in points208],
                              crs="EPSG:4326")


#points for 4231 huc12 basins
points4231 = []
lat4231 = attr4231[:, attr_name_np_4231.index("lat")].tolist()   #attr_trained_dt
lon4231 = attr4231[:, attr_name_np_4231.index("lon")].tolist()   #attr_trained_dt
for i in range(len(lat4231)):
    #points.append((lon[i], lat[i]))
    points4231.append((lat4231[i], lon4231[i]))
gdf_points_4231 = gpd.GeoDataFrame(geometry=[Point(longitude, latitude) for latitude, longitude in points4231],
                              crs="EPSG:4326")

# reading recharge observations
rech_obs_moeck_dir = r"D:\\DR\\data\\rech_obs"
rech_obs_points = gpd.read_file(os.path.join(rech_obs_moeck_dir, "Moeck_2020_5207_glob_recharge_point.shp"))

# Convert geometry to WKT (Well-Known Text) to make it orderable
rech_obs_points["geometry_wkt"] = rech_obs_points.geometry.apply(lambda geom: geom.wkt)

# Group by the WKT representation of geometry and compute mean
rech_obs_points_avg = rech_obs_points.groupby("geometry_wkt").mean().reset_index()

# Convert WKT back to geometry
rech_obs_points_avg["geometry"] = rech_obs_points_avg["geometry_wkt"].apply(lambda wkt: gpd.GeoSeries.from_wkt([wkt])[0])

# Drop the WKT column (optional)
rech_obs_points_avg = rech_obs_points_avg.drop(columns=["geometry_wkt"])

# Convert back to GeoDataFrame
rech_obs_points_avg = gpd.GeoDataFrame(rech_obs_points_avg, geometry="geometry", crs=rech_obs_points.crs)
############# qr boxplots historical
colors_dictionary_models ={"clm45": "pink", 
                           "cwatm": "cyan", 
                           "matsiro": "magenta", 
                           "h08": "orange", 
                           "lpjml": "khaki", 
                           "jules-w1": "darkkhaki", 
                           "watergap2-2c": "lightyellow", 
                           "watergap2-2c_nosoc": "lightyellow",
                            "watergap2-2c_pressoc": "gold",
                            "watergap2-2c_varsoc": "yellow",
                           "pcr-globwb": "lavender", 
                           "WAYS_nosoc": "mediumspringgreen",
                           f"\u03B4H": "orange", 
                           f"\u03B4HS": "red",
                           f"\u03B4PS": "blue", 
                           f"\u03B4P": "lightskyblue",
                           f"\u03B4HcS": "green", 
                           f"\u03B4Hc": "lawngreen",
                           "\u03B4PS-HUC12": "dodgerblue",
                           "\u03B4PS-HUC12_4231": "lightskyblue",
                           "pseudo observations": "black",
                           f"\u03B4_ens": "black", 
                           f"\u03B4 ens": "black",    
                           'dPL_ens_mSS':"blue",
                           'dPL_ens_HVS':"red",
                           'dPL_ens_HyS':"green",
                           'dPL_ens':"black",      # needs to be fixes later
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
legend_correction_dict = {'mSS_HUC12_dmt_monthly': "\u03B4PS-HUC12", 
                          'mSS_HUC12_dmt_4231_monthly': "\u03B4PS-HUC12_4231", 
                          'mSS_monthly': "\u03B4PS", 
                          'HyS_monthly': "\u03B4HcS", 
                          'HVS_monthly': "\u03B4HS", 
                          'WAYS_nosoc_monthly': "WAYS_nosoc",
                          'Watergap2_2c_nosoc_monthly': "watergap2-2c_nosoc",
                          'Watergap2_2c_pressoc_monthly': "watergap2-2c_pressoc", 
                          'Watergap2_2c_varsoc_monthly': "watergap2-2c_varsoc",
                          }
# fig, axs = plt.subplots(2, 6, figsize=(24, 16))
# ax = axs.flatten()
# plt.subplots_adjust( left=0.04,
#                     bottom=0.03, 
#                     right=0.99, 
#                     top=0.90, 
#                 wspace=0.33,    # 0.4
#                 hspace=0.1) 
# legends_labels = list()
# for i, polygon_row in shapefile.iterrows():
#     # Create a GeoDataFrame for the individual polygon
#     polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)
#     # recharge observations
#     # clipped_rech_obs = gpd.sjoin(rech_obs_points, polygon_gdf, how='inner', predicate='intersects')
#     clipped_rech_obs = gpd.sjoin(rech_obs_points, polygon_gdf, how='inner', predicate='intersects')
#     clipped_rech_obs = clipped_rech_obs.drop(columns=["index_right"])
#     clipped_rech_obs_3857 = clipped_rech_obs.to_crs("EPSG:3857")
    
#     # Spatial join or intersection
#     clipped_points_ind_2003 = gpd.sjoin(gdf_points_2003, polygon_gdf, how='inner', predicate='intersects').index.tolist()
#     clipped_points_ind_208 = gpd.sjoin(gdf_points_208, polygon_gdf, how='inner', predicate='intersects').index.tolist()


#     clipped_points_2003 = gpd.sjoin(gdf_points_2003, polygon_gdf, how='inner', predicate='intersects')
#     ## buffer
#     clipped_gdf_points_2003_3857 = clipped_points_2003.to_crs("EPSG:3857")
#     # Step 1: Create a 100-km buffer around each point in the first shapefile
#     clipped_rech_obs_3857["buffer"] = clipped_rech_obs_3857.geometry.buffer(100000)
#     clipped_rech_obs_3857_buffered = gpd.GeoDataFrame(clipped_rech_obs_3857.drop(columns="geometry"), geometry=clipped_rech_obs_3857["buffer"], crs=clipped_rech_obs_3857.crs)
#     # Step 2: Spatial join to find indices of points in `points_2` that fall within any buffer
#     points_within_100km_2003 = gpd.sjoin(clipped_gdf_points_2003_3857, clipped_rech_obs_3857_buffered, how="inner", predicate="within")
#     # Step 3: Extract the indices of `points_2` that are within 100 km
#     nearest_indices_2003 = points_within_100km_2003.index.unique().tolist()
#     print(f"{i} , rech obs number {len(clipped_rech_obs)},   indices_2003: {len(points_within_100km_2003)}")


#     # # Apply to each selected point
#     # clipped_points_2003 = gpd.sjoin(gdf_points_2003, polygon_gdf, how='inner', predicate='intersects')
#     # nearest_points_list_2003 = clipped_rech_obs.apply(lambda row: find_nearest_point(row.geometry, clipped_points_2003), axis=1)

#     # # Convert to GeoDataFrame
#     # nearest_points_gdf = gpd.GeoDataFrame(nearest_points_list_2003, geometry='geometry', crs=shapefile.crs)
#     # nearest_indices_2003 = nearest_points_gdf.apply(
#     #                             lambda row: find_nearest_point_index(row.geometry, gdf_points_2003), axis=1
#     #                                             )
#     ####  baseflow trends values
#     databox = []
#     ## make the legends labels
    
#     colorLst = list()
#     edgecolor_list = list()
#     line_width_list = list()
#     #print(len(clipped_points_ind_2003))
#     for key in sim_recharge_dict.keys():
#         ## taking an average of simulations in time
#         if np.array(sim_recharge_dict[key]).shape[1] == 2003:
#             # mask_ind = np.array(clipped_points_ind_2003)
#             if len(clipped_rech_obs) > 4:
#                 mask_ind = np.array(nearest_indices_2003)
#             else:
#                 mask_ind = np.array(clipped_points_ind_2003)
#         elif np.array(sim_recharge_dict[key]).shape[1] == 208:
#             mask_ind = np.array(clipped_points_ind_208)
#         #print(f"mask_ind: {len(clipped_points_ind_2003)}, {len(clipped_points_ind_208)}")
#         if len(mask_ind) > 4:
#             data = np.nansum(np.array(sim_recharge_dict[key])[:, mask_ind], axis=0) / ((np.array(sim_recharge_dict[key]).shape[0]/12))
#             #print(f"{i}: simulation - {len(data)}")
#             databox.append(data)

        

#             label1 = ["Zone " + str(polygon_row["new_FID"])]
#             leg = legend_correction_dict[key]
#             # legends_labels.append(leg)
            
#             if ("\u03B4" in leg):
#                 colorLst.append("whitesmoke")
#                 edgecolor_list.append(colors_dictionary_models[leg])
#                 line_width_list.append(4.0)
#             else:
#                 colorLst.append(colors_dictionary_models[leg])
#                 edgecolor_list.append("black")
#                 line_width_list.append(0.6)
#         if i == 0:
#             legends_labels.append(leg)
#     ## checking if there is any recharge obs in the zone
#     clipped_rech_obs = gpd.sjoin(rech_obs_points, polygon_gdf, how='inner', predicate='intersects')
#     # Count unique spatial points in the clipped recharge observations
#     unique_points_count = clipped_rech_obs["geometry"].nunique()
#     if unique_points_count >= 5: # len(clipped_rech_obs) > 5:
#         data = np.array(clipped_rech_obs["recharge(m"])
#         #print(f"{i}: observation - {len(data)}")
#         # databox.append(np.asarray(data, dtype=object))
#         databox.append(data)
#         leg = "pseudo observations"
#         colorLst.append(colors_dictionary_models[leg])
#         edgecolor_list.append("black")
#         line_width_list.append(0.6)
#     if i == 0:
#         legends_labels.append(leg)
#     ax[i] = plotBoxFig([databox], label1=label1, colorLst=colorLst, edgecolor_list=edgecolor_list,
#                     label1_font_size=22, sharey = False, figsize=(12,5), axin=ax[i],
#                     add_horizontal_line=False, widths= 0.6, line_width_list=line_width_list, ylim=None)    # ylim_list_0799[i]
#     ax[i][0].tick_params(axis='y', labelsize=22)
# fig.patch.set_facecolor('white')
# title = f"Average recharge comparison of GSWP3 in GHM and \u03B4 models (trained on Daymet) models in 1962-2010 (mm/year)"
# fig.suptitle(title, fontsize=28, y=0.991)
# # making legends, edge_color, and boxplots colors for legends patches
# colorLst = list()
# edgecolor_list = list()
# line_width_list = list()
# for leg in legends_labels:
#     if ("\u03B4" in leg):
#         colorLst.append("whitesmoke")
#         edgecolor_list.append(colors_dictionary_models[leg])
#         line_width_list.append(4.0)
#     else:
#         colorLst.append(colors_dictionary_models[leg])
#         edgecolor_list.append("black")
#         line_width_list.append(0.6)
# legend_patches = [
#                 FancyBboxPatch(
#                     (0, 0), 1, 1,  
#                     facecolor=colorLst[i], 
#                     edgecolor=edgecolor_list[i], 
#                     linewidth=line_width_list[i],  
#                     boxstyle="square,pad=0.1",  # Makes it look like a boxplot box
#                     label=legends_labels[i]
#                 )  
#                 for i in range(len(legends_labels))
#             ]
# # legend_patches = [
# #     mpatches.Patch(color=colorLst[i], label=legends_labels[i], 
# #                 linewidth=1.75, edgecolor="b")  # Optional: remove edge lines
# #     for i in range(len(legends_labels))
# # ]

# plt.legend(
#     handles=legend_patches,
#     loc='upper center',
#     frameon=False,
#     ncol=4,
#     bbox_to_anchor=(-2.7, 2.28),
#     fontsize=25
# )
# fig_name = f"qr_2a_boxplot_zones.png"
# plt.savefig(os.path.join(fig_out_dir, "evaluation_figures", fig_name), dpi=300)
# plt.close("all")
# print("END")


#########################
########################
######################
fig, axs = plt.subplots(2, 6, figsize=(24, 16))
ax = axs.flatten()
plt.subplots_adjust( left=0.04,
                    bottom=0.03, 
                    right=0.99, 
                    top=0.90, 
                wspace=0.33,    # 0.4
                hspace=0.1) 
legends_labels = list()
for i, polygon_row in shapefile.iterrows():
    # Create a GeoDataFrame for the individual polygon
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_row.geometry], crs=shapefile.crs)
    # recharge observations
    # clipped_rech_obs = gpd.sjoin(rech_obs_points, polygon_gdf, how='inner', predicate='intersects')
    clipped_rech_obs = gpd.sjoin(rech_obs_points, polygon_gdf, how='inner', predicate='intersects')
    clipped_rech_obs = clipped_rech_obs.drop(columns=["index_right"])
    clipped_rech_obs_3857 = clipped_rech_obs.to_crs("EPSG:3857")
    unique_points_count = clipped_rech_obs["geometry"].nunique()
    
    # Spatial join or intersection
    clipped_points_2003 = gpd.sjoin(gdf_points_2003, polygon_gdf, how='inner', predicate='intersects')
    clipped_points_ind_2003 = gpd.sjoin(gdf_points_2003, polygon_gdf, how='inner', predicate='intersects').index.tolist()
    clipped_points_ind_208 = gpd.sjoin(gdf_points_208, polygon_gdf, how='inner', predicate='intersects').index.tolist()

    # clipped_points_4231 = gpd.sjoin(gdf_points_4231, polygon_gdf, how='inner', predicate='intersects')


    
    ## buffer
    # clipped_points_4231_3857 = clipped_points_4231.to_crs("EPSG:3857")
    # clipped_points_4231_3857 = clipped_points_4231_3857.drop(columns=["index_right"])
    clipped_gdf_points_2003_3857 = clipped_points_2003.to_crs("EPSG:3857")
    clipped_gdf_points_2003_3857 = clipped_gdf_points_2003_3857.drop(columns=["index_right"])
    # Step 1: Create a 100-km buffer around each point in the first shapefile
    clipped_rech_obs_3857["buffer"] = clipped_rech_obs_3857.geometry.buffer(100000)
    clipped_rech_obs_3857_buffered = gpd.GeoDataFrame(clipped_rech_obs_3857.drop(columns="geometry"), geometry=clipped_rech_obs_3857["buffer"], crs=clipped_rech_obs_3857.crs)
    # Step 2: Spatial join to find indices of points in `points_2` that fall within any buffer
    points_within_km_2003 = gpd.sjoin(clipped_gdf_points_2003_3857, clipped_rech_obs_3857_buffered, how="inner", predicate="within")
    # Step 3: Extract the indices of `points_2` that are within 100 km
    nearest_indices_2003 = points_within_km_2003.index.unique().tolist()
    print(f"{i} , rech obs number {len(clipped_rech_obs)},   indices_2003: {len(nearest_indices_2003)}")

    # points_within_km_4231 = gpd.sjoin(clipped_points_4231_3857, clipped_rech_obs_3857_buffered, how="inner", predicate="within")
    # # Step 3: Extract the indices of `points_2` that are within 100 km
    # nearest_indices_4231 = points_within_km_4231.index.unique().tolist()

    # # Apply to each selected point
    # clipped_points_2003 = gpd.sjoin(gdf_points_2003, polygon_gdf, how='inner', predicate='intersects')
    # nearest_points_list_2003 = clipped_rech_obs.apply(lambda row: find_nearest_point(row.geometry, clipped_points_2003), axis=1)

    # # Convert to GeoDataFrame
    # nearest_points_gdf = gpd.GeoDataFrame(nearest_points_list_2003, geometry='geometry', crs=shapefile.crs)
    # nearest_indices_2003 = nearest_points_gdf.apply(
    #                             lambda row: find_nearest_point_index(row.geometry, gdf_points_2003), axis=1
    #                                             )
    ####  baseflow trends values
    databox = []
    ## make the legends labels
    
    colorLst = list()
    edgecolor_list = list()
    line_width_list = list()
    #print(len(clipped_points_ind_2003))
    for key in sim_recharge_dict.keys():
        ## taking an average of simulations in time
        flag = "keep"
        if np.array(sim_recharge_dict[key]).shape[1] == 2003:
            # mask_ind = np.array(clipped_points_ind_2003)
            if unique_points_count > 10:            # len(clipped_rech_obs)   nearest_indices_2003    unique_points_count
                mask_ind = np.array(nearest_indices_2003)
            else:
                mask_ind = np.array(clipped_points_ind_2003)
            flag = "keep"
        elif np.array(sim_recharge_dict[key]).shape[1] == 208:
            mask_ind = np.array(clipped_points_ind_208)
            if len(mask_ind) < 10:
                flag = "pass"
        # elif np.array(sim_recharge_dict[key]).shape[1] == 4231:
        #     mask_ind = np.array(nearest_indices_4231)
        #     if len(mask_ind) < 10:
        #         flag = "pass"
        print(f"len mask_ind: {len(mask_ind)}")
        if flag == "keep":
            data = np.nansum(np.array(sim_recharge_dict[key])[:, mask_ind], axis=0) / ((np.array(sim_recharge_dict[key]).shape[0]/12))
            #print(f"{i}: simulation - {len(data)}")
            databox.append(data)

        

            # label1 = ["Zone " + str(polygon_row["new_FID"])]
            leg = legend_correction_dict[key]
            # legends_labels.append(leg)
            
            if ("\u03B4" in leg):
                colorLst.append("whitesmoke")
                edgecolor_list.append(colors_dictionary_models[leg])
                line_width_list.append(4.0)
            else:
                colorLst.append(colors_dictionary_models[leg])
                edgecolor_list.append("black")
                line_width_list.append(0.6)
        if i == 0:
            legends_labels.append(leg)
    ## checking if there is any recharge obs in the zone
    # clipped_rech_obs = gpd.sjoin(rech_obs_points, polygon_gdf, how='inner', predicate='intersects')
    # Count unique spatial points in the clipped recharge observations
    unique_points_count = clipped_rech_obs["geometry"].nunique()
    if unique_points_count > 10: # len(clipped_rech_obs) > 5:
        data = np.array(clipped_rech_obs["recharge(m"])
        #print(f"{i}: observation - {len(data)}")
        # databox.append(np.asarray(data, dtype=object))
        databox.append(data)
        leg = "pseudo observations"
        colorLst.append(colors_dictionary_models[leg])
        edgecolor_list.append("black")
        line_width_list.append(0.6)
    if i == 0:   # because in i=0 we have all the boxes
        legends_labels.append(leg)
    label1 = ["Zone " + str(polygon_row["new_FID"])]
    ax[i] = plotBoxFig([databox], label1=label1, colorLst=colorLst, edgecolor_list=edgecolor_list,
                    label1_font_size=22, sharey = False, figsize=(12,5), axin=ax[i],
                    add_horizontal_line=False, widths= 0.6, line_width_list=line_width_list, ylim=None)    # ylim_list_0799[i]
    ax[i][0].tick_params(axis='y', labelsize=22)
fig.patch.set_facecolor('white')
title = f"Average recharge comparison of GSWP3 in GHM and \u03B4 models (trained on Daymet) models in 1962-2010 (mm/year)"
fig.suptitle(title, fontsize=28, y=0.991)
# making legends, edge_color, and boxplots colors for legends patches
colorLst = list()
edgecolor_list = list()
line_width_list = list()
for leg in legends_labels:
    if ("\u03B4" in leg):
        colorLst.append("whitesmoke")
        edgecolor_list.append(colors_dictionary_models[leg])
        line_width_list.append(4.0)
    else:
        colorLst.append(colors_dictionary_models[leg])
        edgecolor_list.append("black")
        line_width_list.append(0.6)
legend_patches = [
                FancyBboxPatch(
                    (0, 0), 1, 1,  
                    facecolor=colorLst[i], 
                    edgecolor=edgecolor_list[i], 
                    linewidth=line_width_list[i],  
                    boxstyle="square,pad=0.1",  # Makes it look like a boxplot box
                    label=legends_labels[i]
                )  
                for i in range(len(legends_labels))
            ]
# legend_patches = [
#     mpatches.Patch(color=colorLst[i], label=legends_labels[i], 
#                 linewidth=1.75, edgecolor="b")  # Optional: remove edge lines
#     for i in range(len(legends_labels))
# ]

plt.legend(
    handles=legend_patches,
    loc='upper center',
    frameon=False,
    ncol=4,
    bbox_to_anchor=(-2.7, 2.28),
    fontsize=25
)
fig_name = f"qr_2a_boxplot_zones.png"
plt.savefig(os.path.join(fig_out_dir, "evaluation_figures", fig_name), dpi=300)
plt.close("all")
print("END")
