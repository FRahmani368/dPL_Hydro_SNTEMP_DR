# read camels dataset
import os
import pandas as pd
import numpy as np
import datetime as dt
from core import hydroDL
from core.hydroDL import utils, pathCamels
from pandas.api.types import is_numeric_dtype, is_string_dtype
import time
import json
from . import Dataframe

# module variable
# tRange = [19800101, 20150101]
# tRangeobs = [19800101, 20150101]    #[19801001, 20161001] #  streamflow observations
# tLst = utils.time.tRange2Array(tRange)
# tLstobs = utils.time.tRange2Array(tRangeobs)
# nt = len(tLst)
# ntobs = len(tLstobs)

###############  for a paper with two upstream node with temp data  ###############
# forcingLst = ['Q3Tw', 'Q7Tw', 'Q3Q', 'Q5Q', 'Q7Q',
#      'Q9Q', 'Btamean', 'V', 'Pressure', 'SkyCov', 'SR.sum', 'Precip',
#       ]
###################################################################################

###############   for stream_temp Module  ####################
##forcingLst = ['basin_ccov', 'basin_humid', 'basin_rain',
##     'basin_tave_air', 'basin_gwflow', 'basin_potet', 'basin_sroff',
##   'basin_ssflow', 'basin_swrad', 'basin_tave_gw',
## 'basin_tave_ss', 'network_width','outlet_width', 'outlet_outflow', 'gw_tau', 'ss_tau']  #, 'gw_tau', 'ss_tau' 'obs_discharge'

##attrLstSel = ['hru_elev', 'hru_slope', 'network_elev',
##     'outlet_elev', 'network_length', 'network_slope', 'outlet_slope',
##   'basin_area']
################################################################

##############   Water Temperature for CONUS scale  ##########
# forcingLst = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'tmax(C)',
#        'tmin(C)', 'vp(Pa)', '00060_Mean']   #, 'pred_discharge' , '00060_Mean' ,'combine_discharge', 'combine_discharge' 'swe(mm)' ,'outlet_outflow',, 'pred_discharge', , '00060_Mean'
# attrLstSel = ['DRAIN_SQKM',
#
#        'STREAMS_KM_SQ_KM',
#        'STOR_NID_2009', 'FORESTNLCD06', 'PLANTNLCD06',
#        'SLOPE_PCT', 'RAW_DIS_NEAREST_MAJ_DAM',
#
# 'PERDUN', 'RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALL_MAJ_DAMS',
# 'T_MIN_BASIN', 'T_MINSTD_BASIN', 'RH_BASIN',  'RAW_AVG_DIS_ALLDAMS', 'PPTAVG_BASIN',
# 'HIRES_LENTIC_PCT','T_AVG_BASIN', 'T_MAX_BASIN','T_MAXSTD_BASIN',  'NDAMS_2009', 'ELEV_MEAN_M_BASIN'
#               ]

########################################################################
############# Streamflow prediction for CONUS scale  ##########################
# attrLstSel = ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'DRAIN_SQKM',
#       'HYDRO_DISTURB_INDX', 'STREAMS_KM_SQ_KM', 'BFI_AVE', 'NDAMS_2009',
#       'STOR_NID_2009', 'RAW_DIS_NEAREST_DAM', 'FRAGUN_BASIN', 'DEVNLCD06',
#       'FORESTNLCD06', 'PLANTNLCD06', 'PERMAVE', 'RFACT',
#       'PPTAVG_BASIN', 'BARRENNLCD06', 'DECIDNLCD06', 'EVERGRNLCD06',
#       'MIXEDFORNLCD06', 'SHRUBNLCD06', 'GRASSNLCD06', 'WOODYWETNLCD06',
#       'EMERGWETNLCD06', 'GEOL_REEDBUSH_DOM_PCT',
#        'STRAHLER_MAX', 'MAINSTEM_SINUOUSITY', 'REACHCODE', 'ARTIFPATH_PCT',
#       'ARTIFPATH_MAINSTEM_PCT', 'HIRES_LENTIC_PCT', 'PERDUN', 'PERHOR',
#       'TOPWET', 'CONTACT', 'CANALS_PCT', 'RAW_AVG_DIS_ALLCANALS',
#        'NPDES_MAJ_DENS', 'RAW_AVG_DIS_ALL_MAJ_NPDES',
#       'RAW_AVG_DIS_ALL_MAJ_DAMS', 'FRESHW_WITHDRAWAL', 'PCT_IRRIG_AG',
#       'POWER_NUM_PTS', 'POWER_SUM_MW', 'ROADS_KM_SQ_KM', 'PADCAT1_PCT_BASIN',
#       'PADCAT2_PCT_BASIN']   # 'GEOL_REEDBUSH_SITE', , 'AWCAVE'
##############################################################################
def readGageInfo(dirDB):
    gageFile = os.path.join(
        dirDB,
        "basin_timeseries_v1p2_metForcing_obsFlow",
        "basin_dataset_public_v1p2",
        "basin_metadata",
        "gauge_information.txt",
    )

    data = pd.read_csv(gageFile, sep="\t", header=None, skiprows=1)
    # header gives some troubles. Skip and hardcode
    fieldLst = ["huc", "id", "name", "lat", "lon", "area"]
    out = dict()
    for s in fieldLst:
        if s == "name":
            out[s] = data[fieldLst.index(s)].values.tolist()
        else:
            out[s] = data[fieldLst.index(s)].values
    return out


def readUsgsGage(args, usgsId, forcing_data, *, readQc=False):
    ##ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    ##huc = gageDict['huc'][ind]
    ##usgsFile = os.path.join(dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
    ##                  'basin_dataset_public_v1p2', 'usgs_streamflow',
    ##                    str(huc).zfill(2),
    ##                      '%08d_streamflow_qc.txt' % (usgsId))
    ##dataTemp = pd.read_csv(usgsFile, sep=r'\s+', header=None)
    ##obs = dataTemp[4].values
    obs = forcing_data.loc[
        forcing_data["site_no"] == usgsId, args["optData"]["target"]
    ].to_numpy()
    ##obs[obs < 0] = np.nan
    if readQc is True:
        qcDict = {"A": 1, "A:e": 2, "M": 3}
        qc = np.array([qcDict[x] for x in dataTemp[5]])
    if len(obs) != ntobs:
        ##  out = np.full([ntobs], np.nan)
        ##dfDate = dataTemp[[1, 2, 3]]
        ##dfDate.columns = ['year', 'month', 'day']
        ##date = pd.to_datetime(dfDate).values.astype('datetime64[D]')
        if "datetime" in forcing_data.columns:
            date = forcing_data.loc[forcing_data["site_no"] == usgsId, "datetime"]
        elif "date" in forcing_data.columns:
            date = forcing_data.loc[forcing_data["site_no"] == usgsId, "date"]
        [C, ind1, ind2] = np.intersect1d(date, tLstobs, return_indices=True)
        out[ind2] = obs
        if readQc is True:
            outQc = np.full([ntobs], np.nan)
            outQc[ind2] = qc
    else:
        out = obs
        if readQc is True:
            outQc = qc

    if readQc is True:
        return out, outQc
    else:
        return out


def readUsgs(args, usgsIdLst, dfMain):
    t0 = time.time()
    y = np.empty([len(usgsIdLst), ntobs])
    for k in range(len(usgsIdLst)):
        dataObs = readUsgsGage(args, usgsIdLst[k], dfMain)
        y[k, :] = dataObs.squeeze()
    print("read Streamflow data", time.time() - t0)
    return y


def readForcingGage(usgsId, forcing_data, varLst, *, dataset="nldas"):
    # dataset = daymet or maurer or nldas
    ##forcingLst = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    ##ind = np.argwhere(gageDict['id'] == usgsId)[0][0]
    ##huc = gageDict['huc'][ind]

    ##dataFolder = os.path.join(
    ##  dirDB, 'basin_timeseries_v1p2_metForcing_obsFlow',
    ##'basin_dataset_public_v1p2', 'basin_mean_forcing')
    if dataset == "daymet":
        tempS = "cida"
    else:
        tempS = dataset
    ## dataFile = os.path.join(dataFolder, dataset,
    ##                       str(huc).zfill(2),
    ##                     '%08d_lump_%s_forcing_leap.txt' % (usgsId, tempS))
    ##dataTemp = pd.read_csv(dataFile, sep=r'\s+', header=None, skiprows=4)
    dataTemp = forcing_data.loc[forcing_data["site_no"] == usgsId]
    nf = len(varLst)
    out = np.empty([nt, nf])
    for k in range(nf):
        # assume all files are of same columns. May check later.
        ##ind = forcingLst.index(varLst[k])
        ##out[:, k] = dataTemp[ind + 4].values
        out[:, k] = dataTemp[varLst[k]].values
    return out


def readForcing(usgsIdLst, dfMain, varLst):
    t0 = time.time()

    x = np.empty([len(usgsIdLst), nt, len(varLst)])

    for k in range(len(usgsIdLst)):
        data = readForcingGage(usgsIdLst[k], dfMain, varLst)
        x[k, :, :] = data
    print("read forcing data", time.time() - t0)
    return x


def readAttrAll(*, saveDict=False):
    dataFolder = os.path.join(dirDB, "camels_attributes_v2.0", "camels_attributes_v2.0")
    fDict = dict()  # factorize dict
    varDict = dict()
    varLst = list()
    outLst = list()
    keyLst = ["topo", "clim", "hydro", "vege", "soil", "geol"]

    for key in keyLst:
        dataFile = os.path.join(dataFolder, "camels_" + key + ".txt")
        dataTemp = pd.read_csv(dataFile, sep=";")
        varLstTemp = list(dataTemp.columns[1:])
        varDict[key] = varLstTemp
        varLst.extend(varLstTemp)
        k = 0
        nGage = len(gageDict["id"])
        outTemp = np.full([nGage, len(varLstTemp)], np.nan)
        for field in varLstTemp:
            if is_string_dtype(dataTemp[field]):
                value, ref = pd.factorize(dataTemp[field], sort=True)
                outTemp[:, k] = value
                fDict[field] = ref.tolist()
            elif is_numeric_dtype(dataTemp[field]):
                outTemp[:, k] = dataTemp[field].values
            k = k + 1
        outLst.append(outTemp)
    out = np.concatenate(outLst, 1)
    if saveDict is True:
        fileName = os.path.join(dataFolder, "dictFactorize.json")
        with open(fileName, "w") as fp:
            json.dump(fDict, fp, indent=4)
        fileName = os.path.join(dataFolder, "dictAttribute.json")
        with open(fileName, "w") as fp:
            json.dump(varDict, fp, indent=4)
    return out, varLst


def readAttr(usgsIdLst, varLst):
    attrAll, varLstAll = readAttrAll()
    indVar = list()
    for var in varLst:
        indVar.append(varLstAll.index(var))
    idLstAll = gageDict["id"]
    C, indGrid, ind2 = np.intersect1d(idLstAll, usgsIdLst, return_indices=True)
    temp = attrAll[indGrid, :]
    out = temp[:, indVar]
    return out


def calStat(x):
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatgamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def read_attr_data(args, idLst=None):
    """
    to read attribute file with respect to ID list (site_no)
    """
    if args.dataType == "csv":
        inputfiles = os.path.join(
            os.path.sep,
            os.getcwd(),
            args.dir_data_root,
            args.dataset_name,
            "CAMELSattr." + args.dataType,
        )  # forcing
        attr = pd.read_csv(inputfiles)
    elif args.dataType == "feather":
        inputfiles = os.path.join(
            os.path.sep,
            os.getcwd(),
            args.dir_data_root,
            args.dataset_name,
            "CAMELSattr." + args.dataType,
        )  # forcing
        attr = pd.read_csv(inputfiles)
    else:
        print("Error: data file is not supported")
        exit()
    if idLst is None:
        out = attr
    else:
        out = attr.loc[attr["site_no"].isin(idLst)]
    return out


def calStatbasinnorm(
    y, x, args
):  # for daily streamflow normalized by basin area and precipitation
    """

    :param y: parameter that wantes to be normalized
    :param x: x is forcing+attr numpy matrix
    :param args: config file
    :return: y
    """
    y[y == (-999)] = np.nan
    y[y < 0] = 0
    force_attr_list = args["varT_NN"] + args["varC_NN"]
    # attr_data = read_attr_data(args, idLst=idLst)
    if "DRAIN_SQKM" in force_attr_list:
        area_name = "DRAIN_SQKM"
    elif "area_gages2" in force_attr_list:
        area_name = "area_gages2"
    basinarea = x[:, :, force_attr_list.index(area_name)]  #  'DRAIN_SQKM'
    if "PPTAVG_BASIN" in force_attr_list:
        p_mean_name = "PPTAVG_BASIN"
    elif "p_mean" in force_attr_list:
        p_mean_name = "p_mean"
    meanprep = x[:, :, force_attr_list.index(p_mean_name)]  #   'PPTAVG_BASIN'
    # basinarea = attr_data['DRAIN_SQKM']
    ## meanprep = readAttr(gageDict['id'], ['PPTAVG_BASIN'])
    # meanprep = attr_data['PPTAVG_BASIN'] #  anual average precipitation
    # meanprep = readAttr(gageDict['id'], ['q_mean'])
    # temparea = np.tile(basinarea, ( x.shape[1], 1)).transpose()
    # tempprep = np.tile(meanprep, ( x.shape[1],1)).transpose()
    temparea = np.expand_dims(basinarea, axis=2)
    tempprep = np.expand_dims(meanprep, axis=2)
    flowua = (y * 0.0283168 * 3600 * 24) / (
        (temparea * (10**6)) * (tempprep * 10 ** (-2)) / 365
    )  # unit (m^3/day)/(m^3/day)
    a = flowua.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics plus 0.1 for 0 values
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]


def calStatAll(args, x, y):
    statDict = dict()
    # ##idLst = gageDict['id']
    # if args.dataType == 'csv':
    #     inputfiles = os.path.join(os.path.sep, os.getcwd(), args.dir_data_root, args.dataset_name,
    #                               'CAMELSdata.' + args.dataType)  # forcing
    #     dfMain = pd.read_csv(inputfiles)
    #     idLst = dfMain['site_no'].unique()
    # elif args.dataType == 'feather':
    #     inputfiles = os.path.join(os.path.sep, os.getcwd(), args.dir_data_root, args.dataset_name,
    #                               'CAMELSdata.' + args.dataType)  # forcing
    #     dfMain = pd.read_csv(inputfiles)
    #     idLst = dfMain['site_no'].unique()
    # else:
    #     print("Error: data file is not supported")
    #     exit()
    #
    # # usgs streamflow
    # y = readUsgs(args, idLst, dfMain)
    # # statDict['usgsFlow'] = calStatgamma(y)
    # ##statDict['00060_Mean'] = calStatbasinnorm(y)
    for i, target_name in enumerate(args["target"]):
        if target_name == "00060_Mean":
            statDict[args["target"][i]] = calStatbasinnorm(y[:, :, i: i+1], x, args)
        # elif args['optData']['target'] == ['combine_discharge']:
        #     statDict['00060_Mean'] = calStatbasinnorm(y, idLst)
        else:
            statDict[args["target"][i]] = calStat(y[:, :, i: i+1])
    # USGS streamflow
    # statDict['00060_Mean'] = calStatbasinnorm(y, x, args)
    # forcing
    forcingLst = args["varT_NN"] + args["varC_NN"]
    # x = readForcing(idLst, dfMain, varLst=forcingLst)
    for k in range(len(forcingLst)):
        var = forcingLst[k]
        if var == "prcp(mm/day)":
            statDict[var] = calStatgamma(x[:, :, k])
        elif (var == "00060_Mean") or (var == "combine_discharge"):
            statDict[var] = calStatbasinnorm(x[:, :, k : k + 1], x, args)
        else:
            statDict[var] = calStat(x[:, :, k])
    # const attribute
    ##attrData, attrLst = readAttrAll()
    # attrLst = attrLstSel
    # attrData = np.empty([len(idLst), len(attrLst)])
    # for i, ii in enumerate(attrLst):
    #     attrData[:,i] = attr_data[ii].values
    # for k in range(len(attrLst)):
    #     var = attrLst[k]
    #     statDict[var] = calStat(attrData[:, k])
    statFile = os.path.join(args["out_dir"], "Statistics_basinnorm.json")
    with open(statFile, "w") as fp:
        json.dump(statDict, fp, indent=4)


def transNorm(x, varLst, *, toNorm):
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)
    x_temp = x.copy()
    for k in range(len(varLst)):
        var = varLst[k]

        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if (
                    var == "prcp(mm/day)"
                    or var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    x_temp[:, :, k] = np.log10(np.sqrt(x_temp[:, :, k]) + 0.1)

                out[:, :, k] = (x_temp[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if (
                    var == "prcp(mm/day)"
                    or var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    x_temp[:, k] = np.log10(np.sqrt(x_temp[:, k]) + 0.1)
                out[:, k] = (x_temp[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x_temp[:, :, k] * stat[3] + stat[2]
                if (
                    var == "prcp(mm/day)"
                    or var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2

            elif len(x.shape) == 2:
                out[:, k] = x_temp[:, k] * stat[3] + stat[2]
                if (
                    var == "prcp(mm/day)"
                    or var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2

    return out


def basinNorm(x, gageid, toNorm):
    # for regional training, gageid should be numpyarray
    # if type(gageid) is str:
    # if gageid == 'All':
    #   gageid = gageDict['id']
    nd = len(x.shape)
    meanprep = attr_data["PPTAVG_BASIN"]  # 'PPTAVG_BASIN'
    basinarea = attr_data["DRAIN_SQKM"]  #   'DRAIN_SQKM'

    # basinarea = readAttr(gageid, ['DRAIN_SQKM'])
    #  meanprep = readAttr(gageid, ['PPTAVG_BASIN'])
    # # meanprep = readAttr(gageid, ['q_mean'])  #this line was ponded from the beginning
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basinarea, (x.shape[1], 1)).transpose()
    tempprep = np.tile(meanprep, (x.shape[1], 1)).transpose()
    if toNorm is True:
        flow = (x * 0.0283168 * 3600 * 24) / (
            (temparea * (10**6)) * (tempprep * 10 ** (-2)) / 365
        )  # (m^3/day)/(m^3/day)
    else:

        flow = (
            x
            * ((temparea * (10**6)) * (tempprep * 10 ** (-2)) / 365)
            / (0.0283168 * 3600 * 24)
        )
    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def createSubsetAll(opt, **kw):
    if opt == "all":
        idLst = gageDict["id"]
        subsetFile = os.path.join(dirDB, "Subset", "all.csv")
        np.savetxt(subsetFile, idLst, delimiter=",", fmt="%d")


# Define and initialize module variables
if os.path.isdir(pathCamels["DB"]):
    dirDB = pathCamels["DB"]
    gageDict = readGageInfo(dirDB)
    statFile = os.path.join(dirDB, "Statistics_basinnorm.json")
    if not os.path.isfile(statFile):
        calStatAll()
    with open(statFile, "r") as fp:
        statDict = json.load(fp)
else:
    dirDB = None
    gageDict = None
    statDict = None


def initcamels(args, x, y):
    # reinitialize module variable
    global dirDB, gageDict, statDict, forcing_data, attr_data, TempTarget
    stats_directory = args["out_dir"]
    statFile = os.path.join(stats_directory, "Statistics_basinnorm.json")

    if not os.path.isfile(statFile):
        calStatAll(args, x, y)
    with open(statFile, "r") as fp:
        statDict = json.load(fp)


class DataframeCamels(Dataframe):
    def __init__(self, *, subset="All", tRange):
        self.subset = subset
        self.time = utils.time.tRange2Array(tRange)

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def getDataObs(self, args, *, doNorm=False, rmNan=False, basinnorm=False):

        inputfile = os.path.join(os.path.realpath(args["forcing_path"]))
        if inputfile.endswith(".csv"):
            dfMain = pd.read_csv(inputfiles)
        elif inputfile.endswith(".feather"):
            dfMain = pd.read_feather(inputfile)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain["site_no"].unique()
        tLst = utils.time.tRange2Array(args["tRange"])
        tLstobs = utils.time.tRange2Array(args["tRange"])
        # nt = len(tLst)
        ntobs = len(tLstobs)
        nNodes = len(sites)
        y = np.empty([nNodes, ntobs])

        # for k, kk in enumerate(sites):
        #     data = (
        #         dfMain.loc[dfMain["site_no"] == kk, args["optData"]["target"]]
        #         .to_numpy()
        #         .squeeze()
        #     )
        #     y[k, :] = data

        ## Farshid : trying a faster method compared to the upper six lines
        # ## (to check: np.array_equal(y, y2, equal_nan=True)
        yt = dfMain.loc[:, args["target"]].values
        g = dfMain.reset_index(drop=True).groupby("site_no")
        ytg = [yt[i.values, :] for k, i in g.groups.items()]
        y = np.array(ytg)

        data = y.squeeze()

        # data = readUsgs(self.usgsId)
        #   if basinnorm is True:
        #    for k in range(len(varLst)):
        #     var = varLst[k]
        #    stat = statDict[var]

        #       data = basinNorm(data, gageid=self.usgsId, toNorm=True)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLstobs, return_indices=True)
        data = data[:, ind2, :]
        # if doNorm is True:
        #     data = transNorm(data, TempTarget, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        # data[np.where(np.isnan(data))] = -99
        return data

    def getDataTs(self, args, *, varLst, doNorm=True, rmNan=True):
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
        tLst = utils.time.tRange2Array(args["tRange"])
        tLstobs = utils.time.tRange2Array(args["tRange"])
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
                print("the target is not in forcing file nor in attr file")
        # x = np.empty([nNodes, ntobs, len(varLst)])
        # for k, kk in enumerate(sites):
        #     data = dfMain.loc[dfMain["site_no"] == kk, varLst].to_numpy()
        #     x[k, :, :] = data

        ## Farshid : trying a faster method compared to the upper four lines ##
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
        return data

    def getDataConst(self, args, *, varLst, doNorm=True, rmNan=True):
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
