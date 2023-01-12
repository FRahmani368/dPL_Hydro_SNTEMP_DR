import os

# import core.hydroDL
from collections import OrderedDict
import numpy as np
import json
from core import hydroDL
from core.hydroDL import utils
import datetime as dt
import pandas as pd
import random
import torch

global TempTarget


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain
    )
    return mDict


def readMasterFile(out):
    mFile = os.path.join(out, "master.json")
    with open(mFile, "r") as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    print("read master file " + mFile)
    return mDict


def writeMasterFile(mDict):
    if mDict is str:
        out = mDict
    else:
        out = mDict["out"]
    if not os.path.isdir(out):
        os.makedirs(out)
    mFile = os.path.join(out, "master.json")
    with open(mFile, "w") as fp:
        json.dump(mDict, fp, indent=4)
    print("write master file " + mFile)
    return out


def loadModel(out, epoch=None):
    if epoch is None:
        mDict = readMasterFile(out)
        epoch = mDict["train"]["nEpoch"]
    model = hydroDL.model.train.loadModel(out, epoch)
    return model


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    mDict = readMasterFile(out)
    if mDict["data"]["name"] == "hydroDL.data.camels.DataframeCamels":
        target = ["Streamflow"]
    else:
        target = mDict["data"]["target"]
    if type(target) is not list:
        target = [target]
    nt = len(target)
    lossName = mDict["loss"]["name"]
    if epoch is None:
        epoch = mDict["train"]["nEpoch"]

    fileNameLst = list()
    for k in range(nt):
        testName = "_".join([subset, str(tRange[0]), str(tRange[1]), "ep" + str(epoch)])
        fileName = "_".join([testName, target[k]])
        fileNameLst.append(fileName)
        if lossName == "hydroDL.model.crit.SigmaLoss":
            fileName = "_".join([testName, target[k], "SigmaX"])
            fileNameLst.append(fileName)
    if doMC is not False:
        mcFileNameLst = list()
        for fileName in fileNameLst:
            fileName = "_".join([testName, target[k], "SigmaMC" + str(doMC)])
            mcFileNameLst.append(fileName)
        fileNameLst = fileNameLst + mcFileNameLst

    # sum up to file path list
    filePathLst = list()
    for fileName in fileNameLst:
        if suffix is not None:
            fileName = fileName + "_" + suffix
        filePath = os.path.join(out, fileName + ".csv")
        filePathLst.append(filePath)
    return filePathLst


# def readPred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
#     mDict = readMasterFile(out)
#     dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
#     for k in range(len(filePathLst)):
#         filePath = filePathLst[k]
#         dataPred[:, :, k] = pd.read_csv(
#             filePath, dtype=np.float, header=None).values
#     isSigmaX = False
#     if mDict['loss']['name'] == 'hydroDL.model.crit.SigmaLoss':
#         isSigmaX = True
#         pred = dataPred[:, :, ::2]
#         sigmaX = dataPred[:, :, 1::2]
#     else:
#         pred = dataPred


def mvobs(data, mvday, rmNan=True):
    obslen = data.shape[1] - mvday + 1  # The length of training daily data
    ngage = data.shape[0]
    mvdata = np.full((ngage, obslen, 1), np.nan)
    for ii in range(obslen):
        tempdata = data[:, ii : ii + mvday, :]
        tempmean = np.nanmean(tempdata, axis=1)
        mvdata[:, ii, 0] = tempmean[:, 0]
    if rmNan is True:
        mvdata[np.where(np.isnan(mvdata))] = 0
    return mvdata


def calFDC(data):
    # data = Ngrid * Nday
    Ngrid, Nday = data.shape
    FDC100 = np.full([Ngrid, 100], np.nan)
    for ii in range(Ngrid):
        tempdata0 = data[ii, :]
        tempdata = tempdata0[~np.isnan(tempdata0)]
        # deal with no data case for some gages
        if len(tempdata) == 0:
            tempdata = np.full(Nday, 0)
        # sort from large to small
        temp_sort = np.sort(tempdata)[::-1]
        # select 100 quantile points
        Nlen = len(tempdata)
        ind = (np.arange(100) / 100 * Nlen).astype(int)
        FDCflow = temp_sort[ind]
        if len(FDCflow) != 100:
            raise Exception("unknown assimilation variable")
        else:
            FDC100[ii, :] = FDCflow

    return FDC100


def loadData(args, readX=True, readY=True):

    df = hydroDL.data.camels.DataframeCamels(
        subset=args["optData"]["subset"], tRange=args["optData"]["tRange"]
    )
    x = df.getDataTs(
        args,
        varLst=args["optData"]["varT"],
        doNorm=args["optData"]["doNorm"][0],
        rmNan=args["optData"]["rmNan"][0],
    )
    y = df.getDataObs(
        args, doNorm=args["optData"]["doNorm"][1], rmNan=args["optData"]["rmNan"][1]
    )  # doNorm=optData['doNorm'][1]     optData['rmNan'][1]
    c = df.getDataConst(
        args,
        varLst=args["optData"]["varC"],
        doNorm=args["optData"]["doNorm"][0],
        rmNan=args["optData"]["rmNan"][0],
    )
    c_PRMS = df.getDataConst(
        args,
        varLst=args["optData"]["varC_PRMS"],
        doNorm=args["optData"]["doNorm"][0],
        rmNan=args["optData"]["rmNan"][0],
    )
    x_PRMS = df.getDataTs(
        args,
        varLst=args["optData"]["varT_PRMS"],
        doNorm=args["optData"]["doNorm"][0],
        rmNan=args["optData"]["rmNan"][0],
    )

    return df, x, y, c, c_PRMS, x_PRMS


def train(mDict):
    if mDict is str:
        mDict = readMasterFile(mDict)
    out = mDict["out"]
    optData = mDict["data"]
    optModel = mDict["model"]
    optLoss = mDict["loss"]
    optTrain = mDict["train"]

    # fix the random seed
    if optTrain["seed"] is None:
        # generate random seed
        randomseed = int(np.random.uniform(low=0, high=1e6))
        optTrain["seed"] = randomseed
        print("random seed updated!")
    else:
        randomseed = optTrain["seed"]

    random.seed(randomseed)
    torch.manual_seed(randomseed)
    np.random.seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # data
    df, x, y, c = loadData(optData)
    # c = None # tempora test
    if c is None:
        if type(x) is tuple:
            nx = x[0].shape[-1]
        else:
            nx = x.shape[-1]
    else:
        if type(x) is tuple:
            nx = x[0].shape[-1] + c.shape[-1]
        else:
            nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]

    # loss
    if eval(optLoss["name"]) is hydroDL.model.crit.SigmaLoss:
        lossFun = hydroDL.model.crit.SigmaLoss(prior=optLoss["prior"])
        optModel["ny"] = ny * 2
    elif eval(optLoss["name"]) is hydroDL.model.crit.RmseLoss:
        lossFun = hydroDL.model.crit.RmseLoss()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.NSELoss:
        lossFun = hydroDL.model.crit.NSELoss()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.NSELosstest:
        lossFun = hydroDL.model.crit.NSELosstest()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.MSELoss:
        lossFun = hydroDL.model.crit.MSELoss()
        optModel["ny"] = ny

    # model
    if optModel["nx"] != nx:
        print("updated nx by input data")
        optModel["nx"] = nx
    if eval(optModel["name"]) is hydroDL.model.rnn.CudnnLstmModel:
        if type(x) is tuple:
            x = np.concatenate([x[0], x[1]], axis=2)
            if c is None:
                nx = x.shape[-1]
            else:
                nx = x.shape[-1] + c.shape[-1]
            optModel["nx"] = nx
            print("Concatenate input and obs, update nx by obs")
        model = hydroDL.model.rnn.CudnnLstmModel(
            nx=optModel["nx"], ny=optModel["ny"], hiddenSize=optModel["hiddenSize"]
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.CpuLstmModel:
        model = hydroDL.model.rnn.CpuLstmModel(
            nx=optModel["nx"], ny=optModel["ny"], hiddenSize=optModel["hiddenSize"]
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.LstmCloseModel:
        model = hydroDL.model.rnn.LstmCloseModel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            hiddenSize=optModel["hiddenSize"],
            fillObs=True,
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.AnnModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=optModel["nx"], ny=optModel["ny"], hiddenSize=optModel["hiddenSize"]
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.AnnCloseModel:
        model = hydroDL.model.rnn.AnnCloseModel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            hiddenSize=optModel["hiddenSize"],
            fillObs=True,
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLSTMmodel:
        daobsOption = optData["daObs"]
        if type(daobsOption) is list:
            if len(daobsOption) - 3 >= 7:
                # using 1dcnn only when number of obs larger than 7
                optModel["nobs"] = len(daobsOption)
                convpara = optModel["convNKS"]
                model = hydroDL.model.rnn.CNN1dLSTMmodel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    nobs=optModel["nobs"] - 3,
                    hiddenSize=optModel["hiddenSize"],
                    nkernel=convpara[0],
                    kernelSize=convpara[1],
                    stride=convpara[2],
                    poolOpt=optModel["poolOpt"],
                )
                print("CNN1d Kernel is used!")
            else:
                if type(x) is tuple:
                    x = np.concatenate([x[0], x[1]], axis=2)
                    nx = x.shape[-1] + c.shape[-1]
                    optModel["nx"] = nx
                    print("Concatenate input and obs, update nx by obs")
                model = hydroDL.model.rnn.CudnnLstmModel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    hiddenSize=optModel["hiddenSize"],
                )
                optModel["name"] = "hydroDL.model.rnn.CudnnLstmModel"
                print("Too few obserservations, not using cnn kernel")
        else:
            raise Exception("CNN kernel used but daobs option is not obs list")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLSTMInmodel:
        daobsOption = optData["daObs"]
        if type(daobsOption) is list:
            if len(daobsOption) - 3 >= 7:
                # using 1dcnn only when number of obs larger than 7
                optModel["nobs"] = len(daobsOption)
                convpara = optModel["convNKS"]
                model = hydroDL.model.rnn.CNN1dLSTMInmodel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    nobs=optModel["nobs"] - 3,
                    hiddenSize=optModel["hiddenSize"],
                    nkernel=convpara[0],
                    kernelSize=convpara[1],
                    stride=convpara[2],
                    poolOpt=optModel["poolOpt"],
                )
                print("CNN1d Kernel is used!")
            else:
                if type(x) is tuple:
                    x = np.concatenate([x[0], x[1]], axis=2)
                    nx = x.shape[-1] + c.shape[-1]
                    optModel["nx"] = nx
                    print("Concatenate input and obs, update nx by obs")
                model = hydroDL.model.rnn.CudnnLstmModel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    hiddenSize=optModel["hiddenSize"],
                )
                optModel["name"] = "hydroDL.model.rnn.CudnnLstmModel"
                print("Too few obserservations, not using cnn kernel")
        else:
            raise Exception("CNN kernel used but daobs option is not obs list")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLCmodel:
        LCrange = optData["lckernel"]
        tLCLst = utils.time.tRange2Array(LCrange)
        optModel["nobs"] = x[1].shape[-1]
        convpara = optModel["convNKS"]
        model = hydroDL.model.rnn.CNN1dLCmodel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            nobs=optModel["nobs"],
            hiddenSize=optModel["hiddenSize"],
            nkernel=convpara[0],
            kernelSize=convpara[1],
            stride=convpara[2],
            poolOpt=optModel["poolOpt"],
        )
        print("CNN1d Local calibartion Kernel is used!")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLCInmodel:
        LCrange = optData["lckernel"]
        tLCLst = utils.time.tRange2Array(LCrange)
        optModel["nobs"] = x[1].shape[-1]
        convpara = optModel["convNKS"]
        model = hydroDL.model.rnn.CNN1dLCInmodel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            nobs=optModel["nobs"],
            hiddenSize=optModel["hiddenSize"],
            nkernel=convpara[0],
            kernelSize=convpara[1],
            stride=convpara[2],
            poolOpt=optModel["poolOpt"],
        )
        print("CNN1d Local calibartion Kernel is used!")

    # train
    if optTrain["saveEpoch"] > optTrain["nEpoch"]:
        optTrain["saveEpoch"] = optTrain["nEpoch"]

    # train model
    writeMasterFile(mDict)
    model = hydroDL.model.train.trainModel(
        model,
        x,
        y,
        c,
        lossFun,
        nEpoch=optTrain["nEpoch"],
        miniBatch=optTrain["miniBatch"],
        saveEpoch=optTrain["saveEpoch"],
        saveFolder=out,
    )


def test(
    out,
    TempTarget,
    forcing_path,
    attr_path,
    *,
    tRange,
    subset,
    doMC=False,
    suffix=None,
    batchSize=None,
    epoch=None,
    reTest=False,
    basinnorm=False
):
    mDict = readMasterFile(out)

    optData = mDict["data"]
    optData["subset"] = subset
    optData["tRange"] = tRange
    if "damean" not in optData.keys():
        optData["damean"] = False
    if "dameanopt" not in optData.keys():
        optData["dameanopt"] = 0
    if "davar" not in optData.keys():
        optData["davar"] = "streamflow"
    elif type(optData["davar"]) is list:
        optData["davar"] = "".join(optData["davar"])

    # generate file names and run model
    filePathLst = namePred(out, tRange, subset, epoch=epoch, doMC=doMC, suffix=suffix)
    print("output files:", filePathLst)
    for filePath in filePathLst:
        if not os.path.isfile(filePath):
            reTest = True
    if reTest is True:
        print("Runing new results")
        df, x, obs, c = loadData(optData, TempTarget, forcing_path, attr_path, out)
        # c = None # temporary test
        model = loadModel(out, epoch=epoch)
        hydroDL.model.train.testModel(
            model, x, c, batchSize=batchSize, filePathLst=filePathLst, doMC=doMC
        )
    else:
        print("Loaded previous results")
        df, x, obs, c = loadData(
            optData, TempTarget, forcing_path, attr_path, readX=False
        )

    # load previous result - readPred
    mDict = readMasterFile(out)
    dataPred = np.zeros([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(filePath, dtype=np.float, header=None).values
    isSigmaX = False
    if mDict["loss"]["name"] == "hydroDL.model.crit.SigmaLoss" or doMC is not False:
        isSigmaX = True
        pred = dataPred[:, :, ::2]
        sigmaX = dataPred[:, :, 1::2]
    else:
        # for i in range(len(filePathLst)):
        #    ktemp = dataPred[:,:,i]
        # pred=ktemp
        pred = dataPred

    if optData["doNorm"][1] is True:
        if eval(optData["name"]) is hydroDL.data.dbCsv.DataframeCsv:
            target = optData["target"]
            if type(optData["target"]) is not list:
                target = [target]
            nTar = len(target)
            for k in range(nTar):
                pred[:, :, k] = hydroDL.data.dbCsv.transNorm(
                    pred[:, :, k],
                    rootDB=optData["rootDB"],
                    fieldName=target[k],
                    fromRaw=False,
                )
                obs[:, :, k] = hydroDL.data.dbCsv.transNorm(
                    obs[:, :, k],
                    rootDB=optData["rootDB"],
                    fieldName=target[k],
                    fromRaw=False,
                )
                if isSigmaX is True:
                    sigmaX[:, :, k] = hydroDL.data.dbCsv.transNormSigma(
                        sigmaX[:, :, k],
                        rootDB=optData["rootDB"],
                        fieldName=target[k],
                        fromRaw=False,
                    )
        elif eval(optData["name"]) is hydroDL.data.camels.DataframeCamels:
            pred = hydroDL.data.camels.transNorm(pred, TempTarget, toNorm=False)
            obs = hydroDL.data.camels.transNorm(obs, TempTarget, toNorm=False)
        if basinnorm is True:
            if type(subset) is list:
                gageid = np.array(subset)
            elif type(subset) is str:
                gageid = subset
            pred = hydroDL.data.camels.basinNorm(pred, gageid=gageid, toNorm=False)
            obs = hydroDL.data.camels.basinNorm(obs, gageid=gageid, toNorm=False)
    if isSigmaX is True:
        return df, pred, obs, sigmaX
    else:
        return df, pred, obs, x
