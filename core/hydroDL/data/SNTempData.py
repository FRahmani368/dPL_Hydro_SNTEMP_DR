import os
import pandas as pd
import numpy as np
import datetime as dt
from hydroDL import utils, pathCamels
from pandas.api.types import is_numeric_dtype, is_string_dtype
import time
import json
from . import Dataframe


# module variable
tRange = [19801001, 20161001]
tRangeobs = [19801001, 20161001]  # Stream Temperature  observations
tLst = utils.time.tRange2Array(tRange)
tLstobs = utils.time.tRange2Array(tRangeobs)
nt = len(tLst)
ntobs = len(tLstobs)
forcingLst = ['seg_ccov',
             'seg_humid',
             'seg_outflow',
             'seg_rain',
             'seg_shade',
             'seg_tave_air',
             'seg_upstream_inflow',
             'seg_width',
             'seginc_gwflow',
             'seginc_potet',
             'seginc_sroff',
             'seginc_ssflow',
             'seginc_swrad']

attrLstSel = ['hru_elev',
           'hru_slope',
          ]


class DataframeCamels(Dataframe):
    def __init__(self, *, subset='All', tRange):
        self.subset = subset
        if subset == 'All':  # change to read subset later
#            self.usgsId = gageDict['id']
 #           crd = np.zeros([len(self.usgsId), 2])
  #          crd[:, 0] = gageDict['lat']
   #         crd[:, 1] = gageDict['lon']
    #        self.crd = crd
     #   elif type(subset) is list:
      #      self.usgsId = np.array(subset)
       #     crd = np.zeros([len(self.usgsId), 2])
        #    C, ind1, ind2 = np.intersect1d(self.usgsId, gageDict['id'], return_indices=True)
         #   crd[:, 0] = gageDict['lat'][ind2]
          #  crd[:, 1] = gageDict['lon'][ind2]
           # self.crd = crd
        else:
            #raise Exception('The format of subset is not correct!')
        self.time = utils.time.tRange2Array(tRange)

    def getGeo(self):
        return self.crd

    def getT(self):
        return self.time

    def getDataObs(self, *, doNorm=True, rmNan=True, basinnorm = True):
        data = readUsgs(self.usgsId)
        if basinnorm is True:
            data = basinNorm(data, gageid=self.usgsId, toNorm=True)
        data = np.expand_dims(data, axis=2)
        C, ind1, ind2 = np.intersect1d(self.time, tLstobs, return_indices=True)
        data = data[:, ind2, :]
        if doNorm is True:
            data = transNorm(data, 'usgsFlow', toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
            # data[np.where(np.isnan(data))] = -99
        return data

    def getDataTs(self, *, varLst=forcingLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        # read ts forcing
        data = readForcing(self.usgsId, varLst) # data:[gage*day*variable]
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data

    def getDataConst(self, *, varLst=attrLstSel, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        data = readAttr(self.usgsId, varLst)
        if doNorm is True:
            data = transNorm(data, varLst, toNorm=True)
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data