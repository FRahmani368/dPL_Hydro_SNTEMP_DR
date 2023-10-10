import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
# from .dropout import DropMask, createMask
# from . import cnn
import csv
import numpy as np
import random


class HBVMul(torch.nn.Module):
    """HBV Model Pytorch version"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMul, self).__init__()
        self.parameters_bound = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0,2.9], [0,6.5]]

    def forward(self, x, parameters, mu, muwts, rtwts, bufftime=0, outstate=False, routOpt=False, comprout=False):
        # HBV(P, ETpot, T, parameters)
        #
        # Runs the HBV-light hydrological model (Seibert, 2005). NaN values have to be
        # removed from the inputs.
        #
        # Input:
        #     P = array with daily values of precipitation (mm/d)
        #     ETpot = array with daily values of potential evapotranspiration (mm/d)
        #     T = array with daily values of air temperature (deg C)
        #     parameters = array with parameter values having the following structure and scales:
        #         BETA[1,6]; CET; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         MAXBAS; PERC[0,10]; UZL[0,100]; PCORR; TT[-2.5,2.5]; CFMAX[0.5,10]; SFCF; CFR[0,0.1]; CWH[0,0.2]
        #
        #
        # Output, all in mm:
        #     Qsim = daily values of simulated streamflow
        #     SM = soil storage
        #     SUZ = upper zone storage
        #     SLZ = lower zone storage
        #     SNOWPACK = snow depth
        #     ETact = actual evaporation

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBVMul()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, mu, muwts, rtwts,
                                                                      bufftime=0, outstate=True, routOpt=False, comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        Pm= P.unsqueeze(2).repeat(1,1,mu)
        T = x[bufftime:, :, 1]
        Tm = T.unsqueeze(2).repeat(1,1,mu)
        ETpot = x[bufftime:, :, 2]
        ETpm = ETpot.unsqueeze(2).repeat(1,1,mu)


        ## scale the parameters
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0,2.9], [0,6.5]]

        parBETA = parascaLst[0][0] + parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        # parCET = parameters[:,1]
        parFC = parascaLst[1][0] + parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        # parMAXBAS = parameters[:,7]
        parPERC = parascaLst[6][0] + parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        # parPCORR = parameters[:,10]
        parTT = parascaLst[8][0] + parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        # parSFCF = parameters[:,13]
        parCFR = parascaLst[10][0] + parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        Nstep, Ngrid = P.size()
        # Apply correction factor to precipitation
        # P = parPCORR.repeat(Nstep, 1) * P

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).cuda()
        # # Debug for the state variables
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT).type(torch.float32))
            # SNOW = SNOW * parSFCF

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (Tm[t, :, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            # ## log for displaying
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

            # # for debug state variables
            # SMlog[t,:] = SM.detach().cpu().numpy()

        # get the primary average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routOpt is True: # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * mu)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = parascaLst[-2][0] + rtwts[:,0]*(parascaLst[-2][1]-parascaLst[-2][0])
            tempb = parascaLst[-1][0] + rtwts[:,1]*(parascaLst[-1][1]-parascaLst[-1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the primary average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs