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
# import bs
import pymannkendall as mk

def read_GHM_ISIMIP2a_daily(item_name, start_date, end_date, lat_list, lon_list, model_name_GHM, CLM_name="GSWP3",
                        p0_GHM=r"D:\P\inputs\isimip2a\GHMs"):
    dir0_GHM = os.path.join(p0_GHM, model_name_GHM, CLM_name)
    files_path_list = glob.glob(os.path.join(dir0_GHM, "*" + item_name + "*" + "daily" + "*.nc*"))
    files_path_list.sort()
    bounds_USA = [-127, 20.9, -66, 50.4]
    datasets = []
    for file_path in files_path_list:
        ds = xr.open_dataset(file_path, engine="netcdf4")[item_name]  # Open each file
        #### clipping with bounds of USA
        ds_temp = ds.sel(lat=slice(bounds_USA[3], bounds_USA[1]),
                                                lon=slice(bounds_USA[0], bounds_USA[2]))
        datasets.append(ds_temp)
        ds = None
        ds_temp = None
    combined_ds = xr.concat(datasets, dim="time")
    datasets = None
    full_time_range = pd.date_range(
                                    start=str(combined_ds["time"].min().values),
                                    end=str(combined_ds["time"].max().values),
                                    freq="D"  # Daily frequency
                                )

    ds_time_range = list(set(combined_ds["time"].dropna(dim="time").values))
    ds_time_range.sort()
    pandas_time_range = pd.to_datetime([str(time) for time in ds_time_range])
    data_np = np.full((len(pandas_time_range), len(lat_list)), np.nan)
    for j, s in enumerate(lat_list):
        lat_GHM = lat_list[j]
        lon_GHM = lon_list[j]
        data_np[:, j] = combined_ds.sel(lat=lat_GHM, lon=lon_GHM).values * 86400   ## toconvert it to mm/day
    combined_ds = None 
    data_df = pd.DataFrame(data_np, index=pandas_time_range, columns=lat_list)
    # Reindex the DataFrame to match the full time range
    data_df = data_df.reindex(full_time_range)
    # Interpolate missing dates with nearest values
    data_df = data_df.interpolate(method="linear", axis=0)
    interpolated_data_np = data_df.to_numpy()
    mask_time = (full_time_range >= start_date) & (full_time_range <= end_date)

    return interpolated_data_np[mask_time, :]

def do_baseflow_separation(streamflow, start_date, end_date, sites_name_cols_list, baseflow_sep_method=["all"]):
    cols = ["time"] + sites_name_cols_list
    df = pd.DataFrame(columns = cols)
    date_range = pd.date_range(start=start_date,end=end_date, freq='D')
    df["time"] = date_range
    df[sites_name_cols_list] = streamflow
    df = df.set_index("time")
    if baseflow_sep_method == ["all"]:
        baseflow_dict, df_bfi, df_kge = baseflow.separation(df, return_kge=True, return_bfi=True) 
    else:
        # baseflow_dict, df_bfi, df_kge = baseflow.separation(df, return_kge=True, return_bfi=True, method=baseflow_sep_method) #, "Local" , method=["UKIH"]
        baseflow_dict = baseflow.separation(df, return_kge=False, return_bfi=False,
                                                    method=baseflow_sep_method)

    return baseflow_dict

def read_GHM_ISIMIP2a_monthly(item_name, start_date, end_date, lat_list, lon_list, model_name_GHM, CLM_name="GSWP3",
                        p0_GHM=r"D:\P\inputs\isimip2a\GHMs"):
    dir0_GHM = os.path.join(p0_GHM, model_name_GHM, CLM_name)
    files_path_list = glob.glob(os.path.join(dir0_GHM, "*" + item_name + "*" + "monthly" + "*.nc*"))
    files_path_list.sort()
    bounds_USA = [-127, 20.9, -66, 50.4]
    datasets = []
    for file_path in files_path_list:
        ds = xr.open_dataset(file_path, engine="netcdf4")[item_name]  # Open each file
        #### clipping with bounds of USA
        ds_temp = ds.sel(lat=slice(bounds_USA[3], bounds_USA[1]),
                                                lon=slice(bounds_USA[0], bounds_USA[2]))
        datasets.append(ds_temp)
        ds = None
        ds_temp = None
    combined_ds = xr.concat(datasets, dim="time")
    datasets = None
    full_time_range = pd.date_range(
                                    start=str(combined_ds["time"].min().values),
                                    end=str(combined_ds["time"].max().values),
                                    freq="MS"  # Daily frequency
                                )

    ds_time_range = list(set(combined_ds["time"].dropna(dim="time").values))
    ds_time_range.sort()
    pandas_time_range = pd.to_datetime([str(time) for time in ds_time_range])
    data_np = np.full((len(pandas_time_range), len(lat_list)), np.nan)
    for j, s in enumerate(lat_list):
        lat_GHM = lat_list[j]
        lon_GHM = lon_list[j]
        data_np[:, j] = combined_ds.sel(lat=lat_GHM, lon=lon_GHM).values * 86400 * 30  ## toconvert it to mm/month
    combined_ds = None 
    data_df = pd.DataFrame(data_np, index=pandas_time_range, columns=lat_list)
    # Reindex the DataFrame to match the full time range
    data_df = data_df.reindex(full_time_range)
    # Interpolate missing dates with nearest values
    data_df = data_df.interpolate(method="linear", axis=0)
    interpolated_data_np = data_df.to_numpy()
    mask_time = (full_time_range >= start_date) & (full_time_range <= end_date)

    return interpolated_data_np[mask_time, :]


def read_dPL_ISIMIP2a_daily(model_name, site_ind_list, start_date, end_date, include_ssflow=False,
                            read_obs_flow_flag=True,
                            dir0=r"D:\P\M\daymet_1223_1023_PUB"):
    flow_sim = np.full((17897, int(len(site_ind_list)), 10), np.nan)   #1961-2011
    gwflow_sim = np.full((17897, int(len(site_ind_list)), 10), np.nan)
    
    folds = glob.glob(os.path.join(os.path.join(dir0 , "*" + model_name + "*")))
    for i in range(10):
        name0 = "k" + str(i + 1) + "_"
        dir0_flow = [element for element in folds if name0 in element]
        dir1_flow = glob.glob(os.path.join(dir0_flow[0], "*"))
        dir11_flow = glob.glob(os.path.join(dir1_flow[0], "*")) 
        m4 = os.path.join(dir11_flow[0], "ts1961_2011")

        data = np.load(os.path.join(m4, "flow_sim.npy"))
        flow_sim[:,:, i] = data[:,site_ind_list, 0]
        if include_ssflow == True:
            data = np.load(os.path.join(m4, "gwflow.npy")) + np.load(os.path.join(m4, "ssflow.npy"))
        else:
            data = np.load(os.path.join(m4, "gwflow.npy"))
        gwflow_sim[:,:, i] = data[:,site_ind_list, 0]
    

    date_range_total = pd.date_range(start="1962-01-01", end="2010-12-31", freq="D")
    mask_time = (date_range_total >= start_date) & (date_range_total <= end_date)  # from october 01 to September 30

    flow_sim = np.nanmean(flow_sim[mask_time, :, :], axis=2)
    gwflow_sim = np.nanmean(gwflow_sim[mask_time, :, :], axis=2)
    if read_obs_flow_flag == True:
        obs_flow = np.load(os.path.join(m4, "00060_Mean.npy"))
        obs_flow = obs_flow[mask_time, :]
        obs_flow = obs_flow[:, site_ind_list]
        return flow_sim, gwflow_sim, obs_flow
    else:
        return flow_sim, gwflow_sim
    
def read_dPL_Daymet_daily(model_name, site_ind_list, start_date, end_date, include_ssflow=False,
                            read_obs_flow_flag=True,
                            dir0=r"D:\P\M\daymet_1223_1023_PUB"):
    flow_sim = np.full((15341, int(len(site_ind_list)), 10), np.nan)   #19801231-20230101, last day excluded
    gwflow_sim = np.full((15341, int(len(site_ind_list)), 10), np.nan)
    
    folds = glob.glob(os.path.join(os.path.join(dir0 , "*" + model_name + "*")))
    for i in range(10):
        name0 = "k" + str(i + 1) + "_"
        dir0_flow = [element for element in folds if name0 in element]
        dir1_flow = glob.glob(os.path.join(dir0_flow[0], "*"))
        dir11_flow = glob.glob(os.path.join(dir1_flow[0], "*")) 
        m4 = os.path.join(dir11_flow[0], "ts1980_2023")

        data = np.load(os.path.join(m4, "flow_sim.npy"))
        flow_sim[:,:, i] = data[:,site_ind_list, 0]
        if include_ssflow == True:
            data = np.load(os.path.join(m4, "gwflow.npy")) + np.load(os.path.join(m4, "ssflow.npy"))
        else:
            data = np.load(os.path.join(m4, "gwflow.npy"))
        gwflow_sim[:,:, i] = data[:,site_ind_list, 0]
    

    date_range_total = pd.date_range(start="1980-12-31", end="2022-12-31", freq="D")
    mask_time = (date_range_total >= start_date) & (date_range_total <= end_date)  # from october 01 to September 30

    flow_sim = np.nanmean(flow_sim[mask_time, :, :], axis=2)
    gwflow_sim = np.nanmean(gwflow_sim[mask_time, :, :], axis=2)
    if read_obs_flow_flag == True:
        obs_flow = np.load(os.path.join(m4, "00060_Mean.npy"))
        obs_flow = obs_flow[mask_time, :]
        obs_flow = obs_flow[:, site_ind_list]
        return flow_sim, gwflow_sim, obs_flow
    else:
        return flow_sim, gwflow_sim




def converting_daily_to_monthly(daily_data_dict, start_date, end_date):
    date_range = pd.date_range(start=start_date,end=end_date, freq='D')
    monthly_data_dict = dict()
    for key in daily_data_dict.keys():
        monthly_data = []
        for site_idx in range(daily_data_dict[key].shape[1]):
            # For each site, create a DataFrame with the date range and the 10 model outputs
            site_data = pd.DataFrame(daily_data_dict[key][:, site_idx], index=date_range)
            # Resample by month, applying standard deviation over the 10 models
            monthly_data.append(np.expand_dims(site_data.resample("ME").apply(np.nansum).to_numpy(), axis=1))
            # Store the monthly standard deviation for this site
            # monthly_std_dev.append(monthly_std.values)
        monthly_data_all = np.concatenate(monthly_data, axis=1)
        model_name = key.split("_daily")[0]
        monthly_data_dict[model_name + "_monthly"] = monthly_data_all.squeeze()
    return monthly_data_dict

def converting_monthly_to_yearly(monthly_data_dict, start_date, end_date):
    date_range = pd.date_range(start=start_date,end=end_date, freq='MS')
    yearly_data_dict = dict()
    for key in monthly_data_dict.keys():
        site_idx = np.arange(monthly_data_dict[key].shape[1]).tolist()
        cols = ["time"] + site_idx
        df = pd.DataFrame(columns = cols)
        df["time"] = date_range
        df[site_idx] = monthly_data_dict[key]
        df = df.set_index("time")
        model_name = key.split("_monthly")[0]
        yearly_data_dict[model_name + "_yearly"] = (df[site_idx].resample('A-SEP').sum()).to_numpy()
    return yearly_data_dict


def converting_daily_to_yearly(daily_data_dict, start_date, end_date):
    date_range = pd.date_range(start=start_date,end=end_date, freq='D')
    yearly_data_dict = dict()
    for key in daily_data_dict.keys():
        site_idx = np.arange(daily_data_dict[key].shape[1]).tolist()
        cols = ["time"] + site_idx
        df = pd.DataFrame(columns = cols)
        df["time"] = date_range
        df[site_idx] = daily_data_dict[key]
        df = df.set_index("time")
        model_name = key.split("_daily")[0]
        yearly_data_dict[model_name + "_yearly"] = (df[site_idx].resample('YE-SEP').sum()).to_numpy()
    return yearly_data_dict

def converting_daily_to_yearly_precip(daily_data_dict, start_date, end_date):
    date_range = pd.date_range(start=start_date,end=end_date, freq='D')
    yearly_data_dict = dict()
    for key in daily_data_dict.keys():
        site_idx = np.arange(daily_data_dict[key].shape[1]).tolist()
        cols = ["time"] + site_idx
        df = pd.DataFrame(columns = cols)
        df["time"] = date_range
        df[site_idx] = daily_data_dict[key]
        df = df.set_index("time")
        model_name = key
        yearly_data_dict[model_name] = (df[site_idx].resample('YE-SEP').sum()).to_numpy()
    return yearly_data_dict

def calculate_yearly_trends(yearly_data_dict, start_date, end_date, flow_obs_daily, flow_percentage_availability,
                            consider_obs_flow_percentage=True):
    years = np.arange(int(start_date[:4]), int(end_date[:4])) 
    slope_dict=dict()
    intercept_dict = dict()
    for key in yearly_data_dict.keys():
        intercept_temp = []
        slope_temp = []
        for i in range(yearly_data_dict[key].shape[1]):
            ## if you want to consider obs_flow_percentage
            if consider_obs_flow_percentage == True:
                ## checking if there are %95 of flow observations available
                obs = flow_obs_daily[:, i]
                if (len(obs[obs==obs]) > flow_percentage_availability * len(obs)):
                    slope, intercept = np.polyfit(years, yearly_data_dict[key][1:-1, i], 1)   # [key][1:-1, i]     [:, i]
                    intercept_temp.append(intercept)
                    slope_temp.append(slope)
                else:
                    intercept_temp.append(np.nan)
                    slope_temp.append(np.nan)
            else:
                slope, intercept = np.polyfit(years, yearly_data_dict[key][:, i], 1)  #[1:-1, i]    [:, i]
                intercept_temp.append(intercept)
                slope_temp.append(slope)
        intercept_dict[key] = intercept_temp
        slope_dict[key] = slope_temp
    return slope_dict, intercept_dict


def read_GHM_ISIMIP2a_2003basins(item_name, 
                                model_name_GHM, 
                                CLM_name,
                                start_date_sim, 
                                end_date_sim,
                                start_date_mask,
                                end_date_mask, 
                                site_no_2003_list, 
                                site_no_subset_list, 
                                freq,
                                p0_GHM=r"D:\P\inputs\out_qr_isimip2a_gages_2003"):

    # Search for file
    file_name = f"{model_name_GHM}*{item_name}*{CLM_name}*2003*.npy*"
    file_paths = glob.glob(os.path.join(p0_GHM, file_name))

    # Handle missing or multiple files
    if len(file_paths) == 0:
        print(f"WARNING: No file found for {file_name}. Skipping...")
        return None
    elif len(file_paths) > 1:
        print(f"WARNING: Multiple files found for {file_name}. Using the first one.")

    file_path = file_paths[0]  # Use the first match

    # Load file
    try:
        qr0 = np.load(file_path)
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}. Error: {e}")
        return None

    # Initialize NaN-filled array
    qr = np.full((qr0.shape[0], len(site_no_subset_list)), np.nan)

    # Map values to correct indices
    for i, s in enumerate(site_no_subset_list):
        indices = np.where(site_no_2003_list == s)[0]
        if len(indices) == 0:
            print(f"WARNING: Site {s} not found in site_no_2003_list. Skipping...")
            continue
        qr[:, i] = qr0[:, indices[0], 0]

    # Create time range mask
    full_time_range = pd.date_range(start=start_date_sim, end=end_date_sim, freq=freq)
    mask_time = (full_time_range >= start_date_mask) & (full_time_range <= end_date_mask)

    return qr[mask_time, :]



def read_GHM_ISIMIP2b_2003basins_monthly(item_name, 
                                        start_date, 
                                        end_date, 
                                        site_no_2003_list, 
                                        site_no_subset_list, 
                                        model_name_GHM, 
                                        CLM_name,
                                        rcp,
                                        p0_GHM=r"D:\P\inputs\GHMs_out_qr_2005soc"):

    # Search for file
    file_name = f"{model_name_GHM}*{item_name}*{CLM_name}*{rcp}*2003*.npy*"
    file_paths = glob.glob(os.path.join(p0_GHM, file_name))

    # Handle missing or multiple files
    if len(file_paths) == 0:
        print(f"WARNING: No file found for {file_name}. Skipping...")
        return None
    elif len(file_paths) > 1:
        print(f"WARNING: Multiple files found for {file_name}. Using the first one.")

    file_path = file_paths[0]  # Use the first match

    # Load file
    try:
        qr0 = np.load(file_path)
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}. Error: {e}")
        return None

    # Initialize NaN-filled array
    qr = np.full((qr0.shape[0], len(site_no_subset_list)), np.nan)

    # Map values to correct indices
    for i, s in enumerate(site_no_subset_list):
        indices = np.where(site_no_2003_list == s)[0]
        if len(indices) == 0:
            print(f"WARNING: Site {s} not found in site_no_2003_list. Skipping...")
            continue
        qr[:, i] = qr0[:, indices[0], 0]

    # Create time range mask
    full_time_range = pd.date_range(start="2006-01-01", end="2099-12-31", freq="MS")
    mask_time = (full_time_range >= start_date) & (full_time_range <= end_date)

    return qr[mask_time, :]

def read_dPL_recharge_fr_ISIMIP2b(model_name, start_date, end_date, site_ind_list,
                            dir0=r"D:\P\M\daymet_1223_1023_PUB"):
    recharge = np.full((33968, int(len(site_ind_list)), 10), np.nan)   #2007-2099
    folds = glob.glob(os.path.join(os.path.join(dir0 , "*" + model_name + "*")))

    for i in range(10):
        # 
        name0 = "k" + str(i + 1) + "_"
        dir0_flow = [element for element in folds if name0 in element]
        dir1_flow = glob.glob(os.path.join(dir0_flow[0], "*"))
        dir11_flow = glob.glob(os.path.join(dir1_flow[0], "*")) 
        m4 = os.path.join(dir11_flow[0], "ts2006_2100")     ## GSWP3

        ## if the item is recharge, then we need to read it differently based on the model applied

        if "HBV_capillary" in m4:
    #         evapfactor1 = np.load(os.path.join(m4, "evapfactor.npy"))
            excs1 = np.load(os.path.join(m4, "excs.npy"))
            perc1 = np.load(os.path.join(m4, "percolation.npy"))
            capillary = np.load(os.path.join(m4, "capillary.npy"))
            recharge1 = np.load(os.path.join(m4, "recharge.npy"))

            data = recharge1 - capillary + excs1
#             recharge = excs1
        elif "HBV" in m4:
    #         evapfactor1 = np.load(os.path.join(m4, "evapfactor.npy"))
            excs1 = np.load(os.path.join(m4, "excs.npy"))
            perc1 = np.load(os.path.join(m4, "percolation.npy"))
            recharge1 = np.load(os.path.join(m4, "recharge.npy"))

            data = recharge1 + excs1
#             recharge = excs1
        elif "marrmot_PRMS" in m4:
    #         flux_gad1 = np.load(os.path.join(m4, "flux_gad.npy"))
    #         flux_inf1 = np.load(os.path.join(m4, "flux_inf.npy"))
    #         flux_ea1 = np.load(os.path.join(m4, "flux_ea.npy"))
    #         flux_pc1 = np.load(os.path.join(m4, "flux_pc.npy"))
            flux_qres1 = np.load(os.path.join(m4, "flux_qres.npy"))
            flux_sep1 = np.load(os.path.join(m4, "flux_sep.npy"))
        #     sink1 = np.load(os.path.join(p01, "sink.npy"))
            data =  flux_sep1 + flux_qres1
        elif "SACSMA_with_snow" in m4:
            flux_twexls1 = np.load(os.path.join(m4, "flux_twexls.npy"))
            flux_pcfws1 = np.load(os.path.join(m4, "flux_pcfws.npy"))
            flux_Rls1 = np.load(os.path.join(m4, "flux_Rls.npy"))
            flux_pcfw1 = np.load(os.path.join(m4, "flux_pcfw.npy"))
            flux_Rlp1 = np.load(os.path.join(m4, "flux_Rlp.npy"))
            flux_Elztw = np.load(os.path.join(m4, "flux_Elztw.npy"))
            flux_Euzfw = np.load(os.path.join(m4, "flux_Euzfw.npy"))
            AET_hydro = np.load(os.path.join(m4, "AET_hydro.npy"))
            flux_Twexu = np.load(os.path.join(m4, "flux_Twexu.npy"))
            flux_pc1 = np.load(os.path.join(m4, "flux_pc.npy"))
            flux_Ru = np.load(os.path.join(m4, "flux_Ru.npy"))
    #         flux_pctw1 = np.load(os.path.join(m4, "flux_pctw.npy"))
            flux_twexlp1 = np.load(os.path.join(m4, "flux_twexlp.npy"))
    #         flux_pcfwp1 = flux_pcfw1 - flux_pcfws1
            flux_twexl1 = flux_twexlp1 + flux_twexls1
            ssflow = np.load(os.path.join(m4, "ssflow.npy"))
            # data = flux_Twexu - flux_Euzfw - flux_Ru
            data = flux_twexls1 + flux_pcfws1
        else:
            print("no such model for recharge")
            exit()

        recharge[:,:, i] = data[:,site_ind_list, 0]
    date_range_total = pd.date_range(start="2007-01-01", end="2099-12-31", freq="D")
    mask_time = (date_range_total >= start_date) & (date_range_total <= end_date)  # from october 01 to September 30
    recharge = np.nanmean(recharge[mask_time, :, :], axis=2)
    
    return recharge

def read_dPL_recharge_Daymet_daily(model_name, start_date_mask, end_date_mask, site_ind_list,
                                    start_date_sim, end_date_sim,
                            dir0=r"D:\P\M\daymet_1223_1023_PUB"):
    ## the last date of sim has not been considered
        # reducing the first 365 days because of warm-up
    actual_start_date_sim = pd.to_datetime(start_date_sim) + pd.Timedelta(days=365)
    actual_end_date_sim = pd.to_datetime(end_date_sim) - pd.Timedelta(days=1)
    date_range_total = pd.date_range(start=actual_start_date_sim, end=actual_end_date_sim, freq="D")
    recharge = np.full((int(len(date_range_total)), int(len(site_ind_list)), 10), np.nan)   #1980-12-31-2022-12-31
    # recharge = np.full((17897, int(len(site_ind_list)), 10), np.nan)   #1980-12-31-2022-12-31
    folds = glob.glob(os.path.join(os.path.join(dir0 , "*" + model_name + "*")))
    ## finding the folder name: 
    folder_name = f"ts{start_date_sim[:4]}_{end_date_sim[:4]}"

    for i in range(10):
        # 
        name0 = "k" + str(i + 1) + "_"
        dir0_flow = [element for element in folds if name0 in element]
        dir1_flow = glob.glob(os.path.join(dir0_flow[0], "*"))
        dir11_flow = glob.glob(os.path.join(dir1_flow[0], "*")) 
        m4 = os.path.join(dir11_flow[0], folder_name)     ## GSWP3: "ts1961_2011"

        ## if the item is recharge, then we need to read it differently based on the model applied

        if "HBV_capillary" in m4:
    #         evapfactor1 = np.load(os.path.join(m4, "evapfactor.npy"))
            excs1 = np.load(os.path.join(m4, "excs.npy"))
            perc1 = np.load(os.path.join(m4, "percolation.npy"))
            capillary = np.load(os.path.join(m4, "capillary.npy"))
            recharge1 = np.load(os.path.join(m4, "recharge.npy"))

            data = recharge1 - capillary + excs1
#             recharge = excs1
        elif "HBV" in m4:
    #         evapfactor1 = np.load(os.path.join(m4, "evapfactor.npy"))
            excs1 = np.load(os.path.join(m4, "excs.npy"))
            perc1 = np.load(os.path.join(m4, "percolation.npy"))
            recharge1 = np.load(os.path.join(m4, "recharge.npy"))

            data = recharge1 + excs1
#             recharge = excs1
        elif "marrmot_PRMS" in m4:
    #         flux_gad1 = np.load(os.path.join(m4, "flux_gad.npy"))
    #         flux_inf1 = np.load(os.path.join(m4, "flux_inf.npy"))
    #         flux_ea1 = np.load(os.path.join(m4, "flux_ea.npy"))
    #         flux_pc1 = np.load(os.path.join(m4, "flux_pc.npy"))
            flux_qres1 = np.load(os.path.join(m4, "flux_qres.npy"))
            flux_sep1 = np.load(os.path.join(m4, "flux_sep.npy"))
        #     sink1 = np.load(os.path.join(p01, "sink.npy"))
            data =  flux_sep1 + flux_qres1
        elif "SACSMA_with_snow" in m4:
            flux_twexls1 = np.load(os.path.join(m4, "flux_twexls.npy"))
            flux_pcfws1 = np.load(os.path.join(m4, "flux_pcfws.npy"))
            flux_Rls1 = np.load(os.path.join(m4, "flux_Rls.npy"))
            flux_pcfw1 = np.load(os.path.join(m4, "flux_pcfw.npy"))
            flux_Rlp1 = np.load(os.path.join(m4, "flux_Rlp.npy"))
            flux_Elztw = np.load(os.path.join(m4, "flux_Elztw.npy"))
            flux_Euzfw = np.load(os.path.join(m4, "flux_Euzfw.npy"))
            AET_hydro = np.load(os.path.join(m4, "AET_hydro.npy"))
            flux_Twexu = np.load(os.path.join(m4, "flux_Twexu.npy"))
            flux_pc1 = np.load(os.path.join(m4, "flux_pc.npy"))
            flux_Ru = np.load(os.path.join(m4, "flux_Ru.npy"))
    #         flux_pctw1 = np.load(os.path.join(m4, "flux_pctw.npy"))
            flux_twexlp1 = np.load(os.path.join(m4, "flux_twexlp.npy"))
    #         flux_pcfwp1 = flux_pcfw1 - flux_pcfws1
            flux_twexl1 = flux_twexlp1 + flux_twexls1
            ssflow = np.load(os.path.join(m4, "ssflow.npy"))
            # data = flux_Twexu - flux_Euzfw - flux_Ru
            data = flux_twexls1 + flux_pcfws1
        else:
            print("no such model for recharge")
            exit()

        recharge[:,:, i] = data[:,site_ind_list, 0]
    # date_range_total = pd.date_range(start="1962-01-01", end="2010-12-31", freq="D")
    mask_time = (date_range_total >= start_date_mask) & (date_range_total <= end_date_mask)  # from october 01 to September 30
    recharge = np.nanmean(recharge[mask_time, :, :], axis=2)
    
    return recharge




def read_dPL_Daymet_daily_general(item, model_name, start_date_mask, end_date_mask, site_ind_list,
                                  start_date_sim, end_date_sim,
                                  include_ssflow=False,
                                  dir0=r"D:\P\M\daymet_1223_1023_PUB"):
    ## the last date of sim has not been considered
        # reducing the first 365 days because of warm-up
    actual_start_date_sim = pd.to_datetime(start_date_sim) + pd.Timedelta(days=365)
    actual_end_date_sim = pd.to_datetime(end_date_sim) - pd.Timedelta(days=1)
    date_range_total = pd.date_range(start=actual_start_date_sim, end=actual_end_date_sim, freq="D")
    sim = np.full((int(len(date_range_total)), int(len(site_ind_list)), 10), np.nan)   #1980-12-31-2022-12-31
    # recharge = np.full((17897, int(len(site_ind_list)), 10), np.nan)   #1980-12-31-2022-12-31
    folds = glob.glob(os.path.join(os.path.join(dir0 , "*" + model_name + "*")))
    ## finding the folder name: 
    folder_name = f"ts{start_date_sim[:4]}_{end_date_sim[:4]}"

    item_name = item + ".npy"

    for i in range(10):
        # 
        name0 = "k" + str(i + 1) + "_"
        dir0_flow = [element for element in folds if name0 in element]
        dir1_flow = glob.glob(os.path.join(dir0_flow[0], "*"))
        dir11_flow = glob.glob(os.path.join(dir1_flow[0], "*")) 
        m4 = os.path.join(dir11_flow[0], folder_name)     ## GSWP3: "ts1961_2011"

        if item == "gwflow":
            if include_ssflow == True:
                data = np.load(os.path.join(m4, "gwflow.npy")) + np.load(os.path.join(m4, "ssflow.npy"))
            else:
                data = np.load(os.path.join(m4, "gwflow.npy"))
        else:
            data = np.load(os.path.join(m4, item_name))
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        sim[:,:, i] = data[:,site_ind_list, 0]
    

    mask_time = (date_range_total >= start_date_mask) & (date_range_total <= end_date_mask)
    sim = np.nanmean(sim[mask_time, :, :], axis=2)
    return sim



def Mann_Kendal_Theil_Sen_Test(input_dict, alpha=0.05):
    ### Mann_Kendall test
    # trend: tells the trend (increasing, decreasing or no trend)
    # h: True (if trend is present) or False (if the trend is absence)
    # p: p-value of the significance test
    # z: normalized test statistics
    # Tau: Kendall Tau
    # s: Mann-Kendal's score
    # var_s: Variance S
    # slope: Theil-Sen estimator/slope
    # intercept: intercept of Kendall-Theil Robust Line, for seasonal test, full period cycle consider as unit time step
    mann_kendall = {}

    # Iterate over the keys in the input dictionary
    for key, data in input_dict.items():
        num_columns = data.shape[1]
        
        # Pre-allocate lists for results
        trend, h, p, z, tau, s, var_s, slope, intercept = ([] for _ in range(9))

        # Perform the tests for each column in the dataset
        for col in range(num_columns):
            results = mk.original_test(data[:, col], alpha=alpha)
            trend.append(results.trend)
            h.append(results.h)
            p.append(results.p)
            z.append(results.z)
            tau.append(results.Tau)
            s.append(results.s)
            var_s.append(results.var_s)
            slope.append(results.slope)
            intercept.append(results.intercept)

        # Store results in a dictionary
        mann_kendall[key] = {
            "trend": trend,
            "h": h,
            "p": p,
            "z": z,
            "Tau": tau,
            "s": s,
            "var_s": var_s,
            "slope": slope,
            "intercept": intercept
        }

    return mann_kendall

        