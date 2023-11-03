import os
import numpy as np
import pandas as pd
from core.load_data.time import tRange2Array
class DataFrame_dataset:
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
def loadData(args, trange):

    out_dict = dict()
    df = DataFrame_dataset(tRange=trange)
    # getting inputs for NN model:
    out_dict["x_NN"] = df.getDataTs(args, varLst=args["varT_NN"])
    out_dict["c_NN"] = df.getDataConst(args, varLst=args["varC_NN"])
    out_dict["obs"] = df.getDataTs(args, varLst=args["target"])
    if args["hydro_model_name"] != "None":
        out_dict["x_hydro_model"] = df.getDataTs(args, varLst=args["varT_hydro_model"])
        out_dict["c_hydro_model"] = df.getDataConst(args, varLst=args["varC_hydro_model"])
    if args["temp_model_name"] != "None":
        out_dict["x_temp_model"] = df.getDataTs(args, varLst=args["varT_temp_model"])
        out_dict["c_temp_model"] = df.getDataConst(args, varLst=args["varC_temp_model"])
    return out_dict