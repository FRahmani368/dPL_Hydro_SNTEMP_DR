from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import torch
from core.load_data.time import tRange2Array
from datetime import datetime, timedelta

class Data_Reader(ABC):
    @abstractmethod
    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        raise NotImplementedError


class DataFrame_dataset(Data_Reader):
    def __init__(self, tRange):
        self.time = tRange2Array(tRange)

    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        inputfile = os.path.join(os.path.realpath(args["forcing_path"]))
        inputfile_attr = os.path.join(os.path.realpath(args["attr_path"]))
        if inputfile.endswith(".csv"):
            dfMain = pd.read_csv(inputfile)
            dfMain_attr = pd.read_csv(inputfile_attr)
        elif inputfile.endswith(".feather"):
            dfMain = pd.read_feather(inputfile)
            dfMain_attr = pd.read_feather(inputfile_attr)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain["site_no"].unique()
        tLst = tRange2Array(args["tRange"])
        tLstobs = tRange2Array(args["tRange"])
        # nt = len(tLst)
        ntobs = len(tLstobs)
        nNodes = len(sites)

        varLst_forcing = []
        varLst_attr = []
        for var in varLst:
            if var in dfMain.columns:
                varLst_forcing.append(var)
            elif var in dfMain_attr.columns:
                varLst_attr.append(var)
            else:
                print(var, "the var is not in forcing file nor in attr file")
        xt = dfMain.loc[:, varLst_forcing].values
        g = dfMain.reset_index(drop=True).groupby("site_no")
        xtg = [xt[i.values, :] for k, i in g.groups.items()]
        x = np.array(xtg)

        ## for attr
        if len(varLst_attr) > 0:
            x_attr_t = dfMain_attr.loc[:, varLst_attr].values
            x_attr_t = np.expand_dims(x_attr_t, axis=2)
            xattr = np.repeat(x_attr_t, x.shape[1], axis=2)
            xattr = np.transpose(xattr, (0, 2, 1))
            x = np.concatenate((x, xattr), axis=2)

        data = x
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        # if os.path.isdir(out):
        #     pass
        # else:
        #     os.makedirs(out)
        # np.save(os.path.join(out, 'x.npy'), data)
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return np.swapaxes(data, 1, 0)

    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        inputfile = os.path.join(os.path.realpath(args["forcing_path"]))
        if inputfile.endswith(".csv"):
            dfMain = pd.read_csv(inputfile)
            inputfile2 = os.path.join(
                os.path.realpath(args["attr_path"])
            )  #   attr
            dfC = pd.read_csv(inputfile2)
        elif inputfile.endswith(".feather"):
            dfMain = pd.read_feather(inputfile)
            inputfile2 = os.path.join(
                os.path.realpath(args["attr_path"])
            )  #   attr
            dfC = pd.read_feather(inputfile2)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain["site_no"].unique()
        nNodes = len(sites)
        c = np.empty([nNodes, len(varLst)])

        for k, kk in enumerate(sites):
            data = dfC.loc[dfC["site_no"] == kk, varLst].to_numpy().squeeze()
            c[k, :] = data

        data = c
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return data

class numpy_dataset(Data_Reader):
    def __init__(self, tRange):
        self.time = tRange2Array(tRange)

        # These are default forcings and attributes that are read from the dataset
        self.all_forcings_name = ['Lwd', 'PET_hargreaves(mm/day)', 'prcp(mm/day)',
                                'Pres', 'RelHum', 'SpecHum', 'srad(W/m2)',
                                'tmean(C)', 'tmax(C)', 'tmin(C)', 'Wind', 'ccov',
                                'vp(Pa)', "00060_Mean", "00010_Mean",'dayl(s)']  #
        self.attrLst_name = ['aridity', 'p_mean', 'ETPOT_Hargr', 'NDVI', 'FW', 'SLOPE_PCT', 'SoilGrids1km_sand',
                             'SoilGrids1km_clay', 'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel',
                             'HWSD_sand', 'HWSD_silt', 'ELEV_MEAN_M_BASIN', 'meanTa', 'permafrost',
                             'permeability','seasonality_P', 'seasonality_PET', 'snow_fraction',
                             'snowfall_fraction','T_clay','T_gravel','T_sand', 'T_silt','Porosity',
                             "DRAIN_SQKM", "lat", "site_no_int", "stream_length_square", "lon"]

    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        inputfile = os.path.join(os.path.realpath(args["forcing_path"]))
        inputfile_attr = os.path.join(os.path.realpath(args["attr_path"]))
        if inputfile.endswith(".npy"):
            forcing_main = np.load(inputfile)
            attr_main = np.load(inputfile_attr)
        elif inputfile.endswith(".pt"):
            forcing_main = torch.load(inputfile)
            attr_main = torch.load(inputfile_attr)
        else:
            print("data type is not supported")
            exit()

        varLst_index_forcing = []
        varLst_index_attr = []
        for var in varLst:
            if var in self.all_forcings_name:
                varLst_index_forcing.append(self.all_forcings_name.index(var))
            elif var in self.attrLst_name:
                varLst_index_attr.append(self.attrLst_name.index(var))
            else:
                print(var, "the var is not in forcing file nor in attr file")
                exit()

        x = forcing_main[:, :, varLst_index_forcing]
        ## for attr
        if len(varLst_index_attr) > 0:
            x_attr_t = attr_main[:, varLst_index_attr]
            x_attr_t = np.expand_dims(x_attr_t, axis=2)
            xattr = np.repeat(x_attr_t, x.shape[1], axis=2)
            xattr = np.transpose(xattr, (0, 2, 1))
            x = np.concatenate((x, xattr), axis=2)

        data = x
        tLst = tRange2Array(args["tRange"])
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        # if os.path.isdir(out):
        #     pass
        # else:
        #     os.makedirs(out)
        # np.save(os.path.join(out, 'x.npy'), data)
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return np.swapaxes(data, 1, 0)

    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        inputfile = os.path.join(os.path.realpath(args["attr_path"]))
        if inputfile.endswith(".npy"):
            dfC = np.load(inputfile)
        elif inputfile.endswith(".pt"):
            dfC = torch.load(inputfile)
        else:
            print("data type is not supported")
            exit()

        varLst_index_attr = []
        for var in varLst:
            if var in self.attrLst_name:
                varLst_index_attr.append(self.attrLst_name.index(var))
            else:
                print(var, "the var is not in forcing file nor in attr file")
                exit()
        c = dfC[:, varLst_index_attr]

        data = c
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return data

def loadData(args, trange):

    out_dict = dict()
    # Todo: I should this section to a wrapper class
    inputfile_forcing = os.path.join(os.path.realpath(args["forcing_path"]))
    if inputfile_forcing.endswith(".feather") or inputfile_forcing.endswith(".csv"):
        read_data = DataFrame_dataset(tRange=trange)
    elif inputfile_forcing.endswith(".npy") or inputfile_forcing.endswith(".pt"):
        read_data = numpy_dataset(tRange=trange)
    # getting inputs for NN model:
    out_dict["x_NN"] = read_data.getDataTs(args, varLst=args["varT_NN"])
    out_dict["c_NN"] = read_data.getDataConst(args, varLst=args["varC_NN"])
    obs_raw = read_data.getDataTs(args, varLst=args["target"])
    ## converting the
    if "00060_Mean" in args["target"]:
        out_dict["obs"] = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
                                                                         out_dict["c_NN"],
                                                                         obs_raw)
    if args["hydro_model_name"] != "None":
        out_dict["x_hydro_model"] = read_data.getDataTs(args, varLst=args["varT_hydro_model"])
        out_dict["c_hydro_model"] = read_data.getDataConst(args, varLst=args["varC_hydro_model"])
    if args["temp_model_name"] != "None":
        out_dict["x_temp_model"] = read_data.getDataTs(args, varLst=args["varT_temp_model"])
        out_dict["c_temp_model"] = read_data.getDataConst(args, varLst=args["varC_temp_model"])
        # preparing the airT matrix for calculating source water temp
        # airT_memory = max(0, max([args["res_time_lenF_srflow"],
        #                       args["res_time_lenF_ssflow"],
        #                       args["res_time_lenF_bas_shallow"],
        #                       args["res_time_lenF_gwflow"]]) - args["rho"])
        airT_memory = max([args["res_time_lenF_srflow"],
                              args["res_time_lenF_ssflow"],
                              args["res_time_lenF_bas_shallow"],
                              args["res_time_lenF_gwflow"]])
        init_time = datetime.strptime(str(trange[0]), "%Y%m%d")
        new_init_time = init_time - timedelta(days=airT_memory) + timedelta(days=args["warm_up"])
        new_init_time_str = new_init_time.strftime("%Y%m%d")
        # updating read_data class
        read_data.time = tRange2Array([int(new_init_time_str), trange[1]])
        out_dict["airT_mem_temp_model"] = read_data.getDataTs(args, varLst=['tmean(C)'])
        # out_dict["airT_memory"] = airT_memory
    return out_dict


def converting_flow_from_ft3_per_sec_to_mm_per_day(args, c_NN_sample, obs_sample):
    varTar_NN = args["target"]
    if "00060_Mean" in varTar_NN:
        obs_flow_v = obs_sample[:, :, varTar_NN.index("00060_Mean")]
        varC_NN = args["varC_NN"]
        if "DRAIN_SQKM" in varC_NN:
            area_name = "DRAIN_SQKM"
        elif "area_gages2" in varC_NN:
            area_name = "area_gages2"
        # area = (c_NN_sample[:, varC_NN.index(area_name)]).unsqueeze(0).repeat(obs_flow_v.shape[0], 1)  # torch version
        area = np.expand_dims(c_NN_sample[:, varC_NN.index(area_name)], axis=0).repeat(obs_flow_v.shape[0], 0)  # np ver
        obs_sample[:, :, varTar_NN.index("00060_Mean")] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample