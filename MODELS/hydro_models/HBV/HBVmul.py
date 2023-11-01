import math
import torch
from MODELS.PET_models.potet import get_potet
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
        self.parameters_bound = [[0 , 1.0], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2]]
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]
        self.activation_sigmoid = torch.nn.Sigmoid()

    def UH_gamma(self, a, b, lenF=10):
        # UH. a [time (same all time steps), batch, var]
        m = a.shape
        lenF = min(a.shape[0], lenF)
        w = torch.zeros([lenF, m[1], m[2]])
        aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
        theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
        t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
        t = t.cuda(aa.device)
        denom = (aa.lgamma().exp()) * (theta ** aa)
        mid = t ** (aa - 1)
        right = torch.exp(-t / theta)
        w = 1 / denom * mid * right
        w = w / w.sum(0)  # scale to 1 for each UH

        return w

    def UH_conv(self, x, UH, viewmode=1):
        # UH is a vector indicating the unit hydrograph
        # the convolved dimension will be the last dimension
        # UH convolution is
        # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
        # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
        # hence we flip the UH
        # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
        # view
        # x: [batch, var, time]
        # UH:[batch, var, uhLen]
        # batch needs to be accommodated by channels and we make use of groups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html

        mm = x.shape;
        nb = mm[0]
        m = UH.shape[-1]
        padd = m - 1
        if viewmode == 1:
            xx = x.view([1, nb, mm[-1]])
            w = UH.view([nb, 1, m])
            groups = nb

        y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
        if padd != 0:
            y = y[:, :, 0:-padd]
        return y.view(mm)

    def source_flow_calculation(self, args, flow_out, c_hydro_model):
        varC_hydro_model = args["varC_hydro_model"]
        if "DRAIN_SQKM" in varC_hydro_model:
            area_name = "DRAIN_SQKM"
        elif "area_gages2" in varC_hydro_model:
            area_name = "area_gages2"
        else:
            print("area of basins are not available among attributes dataset")
        area = c_hydro_model[:, varC_hydro_model.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
            flow_out["flow_sim"].shape[
                0], 1, 1)
        # flow calculation. converting mm/day to m3/sec
        srflow = (1000 / 86400) * area * flow_out["srflow"].repeat(1, 1, args["nmul"])  # Q_t - gw - ss
        ssflow = (1000 / 86400) * area * flow_out["ssflow"].repeat(1, 1, args["nmul"])  # ras
        gwflow = (1000 / 86400) * area * flow_out["gwflow"].repeat(1, 1, args["nmul"])
        srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
        ssflow = torch.clamp(ssflow, min=0.0)
        gwflow = torch.clamp(gwflow, min=0.0)
        return srflow, ssflow, gwflow
    def param_bounds_2D(self, params, num, bounds, ndays, nmul):

        out_temp = (
                params[:, num * nmul: (num + 1) * nmul]
                * (bounds[1] - bounds[0])
                + bounds[0]
        )
        out = out_temp.unsqueeze(0).repeat(ndays, 1, 1).reshape(
            ndays, params.shape[0], nmul
        )
        return out
    def forward(self, x_hydro_model, c_hydro_model, params, args, PET_param, muwts=None, warm_up=0, init=False, routing=False, comprout=False):
        nmul = args["nmul"]
        # HBV(P, ETpot, T, parameters)
        #
        # Runs the HBV-light hydrological model (Seibert, 2005). NaN values have to be
        # removed from the inputs.

        PRECS = 1e-5

        # Initialization
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                initmodel = HBVMul().to(args["device"])
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, c_hydro_model, params, args, PET_param,
                                                                      muwts=None, warm_up=0, init=True, routing=False,
                                                                      comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x_hydro_model.shape[1]
            SNOWPACK = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args["device"])
            MELTWATER = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args["device"])
            SM = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args["device"])
            SUZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args["device"])
            SLZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args["device"])
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()
        vars = args["varT_hydro_model"]
        vars_c = args["varC_hydro_model"]
        P = x_hydro_model[warm_up:, :, vars.index("prcp(mm/day)")]
        Pm= P.unsqueeze(2).repeat(1, 1, nmul)
        Tmaxf = x_hydro_model[warm_up:, :, vars.index("tmax(C)")].unsqueeze(2).repeat(1, 1, nmul)
        Tminf = x_hydro_model[warm_up:, :, vars.index("tmin(C)")].unsqueeze(2).repeat(1, 1, nmul)
        mean_air_temp = Tmaxf #(Tmaxf + Tminf) / 2

        if args["potet_module"] == "potet_hamon":
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.004, 0.008], ndays=No_days, nmul=args["nmul"])
            PET = get_potet(
                args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=PET_coef
            )     # mm/day
        elif args["potet_module"] == "potet_hargreaves":

            day_of_year = x_hydro_model[warm_up:, :, vars.index("dayofyear")].unsqueeze(-1).repeat(1, 1, nmul)
            lat = c_hydro_model[:, vars_c.index("lat")].unsqueeze(0).unsqueeze(-1).repeat(day_of_year.shape[0], 1, nmul)
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
            #                                   nmul=args["nmul"])

            PET = get_potet(
                args=args, tmin=Tminf, tmax=Tmaxf,
                tmean=mean_air_temp, lat=lat,
                day_of_year=day_of_year
            )
            # AET = PET_coef * PET     # here PET_coef converts PET to Actual ET here
        elif args["potet_module"] == "dataset":
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
            #                                 nmul=args["nmul"])
            # here PET_coef converts PET to Actual ET
            PET = x_hydro_model[warm_up:, :, vars.index(args["potet_dataset_name"])].unsqueeze(-1).repeat(1, 1, nmul)
            # AET = PET_coef * PET

        ## scale the parameters
        No_days = x_hydro_model.shape[0] - warm_up
        parBETA = self.param_bounds_2D(params, 0, bounds=self.parameters_bound[0], ndays=No_days, nmul=nmul)
        parFC = self.param_bounds_2D(params, 1, bounds=self.parameters_bound[1], ndays=No_days, nmul=nmul)
        parK0 = self.param_bounds_2D(params, 2, bounds=self.parameters_bound[2], ndays=No_days, nmul=nmul)
        parK1 = self.param_bounds_2D(params, 3, bounds=self.parameters_bound[3], ndays=No_days, nmul=nmul)
        parK2 = self.param_bounds_2D(params, 4, bounds=self.parameters_bound[4], ndays=No_days, nmul=nmul)
        parLP = self.param_bounds_2D(params, 5, bounds=self.parameters_bound[5], ndays=No_days, nmul=nmul)
        parPERC = self.param_bounds_2D(params, 6, bounds=self.parameters_bound[6], ndays=No_days, nmul=nmul)
        parUZL = self.param_bounds_2D(params, 7, bounds=self.parameters_bound[7], ndays=No_days, nmul=nmul)
        parTT = self.param_bounds_2D(params, 8, bounds=self.parameters_bound[8], ndays=No_days, nmul=nmul)
        parCFMAX = self.param_bounds_2D(params, 9, bounds=self.parameters_bound[9], ndays=No_days, nmul=nmul)
        parCFR = self.param_bounds_2D(params, 10, bounds=self.parameters_bound[10], ndays=No_days, nmul=nmul)
        parCWH = self.param_bounds_2D(params, 11, bounds=self.parameters_bound[11], ndays=No_days, nmul=nmul)
        if routing == True:
            conv_params = params[:, len(self.parameters_bound):]
            # conv_params = self.activation_sigmoid(conv_params)
            tempa = self.param_bounds_2D(conv_params, 0, bounds=self.conv_routing_hydro_model_bound[0], ndays=No_days,
                                         nmul=1)
            tempb = self.param_bounds_2D(conv_params, 1, bounds=self.conv_routing_hydro_model_bound[1], ndays=No_days,
                                         nmul=1)

        Nstep, Ngrid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(Nstep, 1) * P

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(args["device"])
        Q0_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args["device"])
        Q1_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args["device"])
        Q2_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args["device"])
        # # Debug for the state variables
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())
        AET = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(args["device"])

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (Tmaxf[t, :, :] >= parTT[t, :, :]).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tmaxf[t, :, :] < parTT[t, :, :]).type(torch.float32))
            # SNOW = SNOW * parSFCF

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX[t, :, :] * (Tmaxf[t, :, :] - parTT[t, :, :])
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR[t, :, :] * parCFMAX[t, :, :] * (parTT[t, :, :] - Tmaxf[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH[t, :, :] * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC[t, :, :]) ** parBETA[t, :, :]
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
            excess = SM - parFC[t, :, :]
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP[t, :, :] * parFC[t, :, :])
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PET[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            AET[t,:,:] = ETact
            SM = torch.clamp(SM - ETact, min=PRECS)  # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC[t, :, :])
            SUZ = SUZ - PERC
            Q0 = parK0[t, :, :] * torch.clamp(SUZ - parUZL[t, :, :], min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1[t, :, :] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2[t, :, :] * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2
            # # for debug state variables
            # SMlog[t,:] = SM.detach().cpu().numpy()

        # get the primary average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routing is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * nmul)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            UH = self.UH_gamma(tempa, tempb, lenF=15)  # lenF: folter
            # UH = self.UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = self.UH_conv(rf, UH).permute([2, 0, 1])
            # do routing individually for Q0, Q1, and Q2
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            Q0_rout = self.UH_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])   # dim:gage*var*time
            Q1_rout = self.UH_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            Q2_rout = self.UH_conv(rf_Q2, UH).permute([2, 0, 1])


            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, nmul)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the primary average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if init is True:     # means we are in warm up
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return dict(flow_sim=torch.clamp(Q0_rout + Q1_rout + Q2_rout, min=0.0),
                        srflow=torch.clamp(Q0_rout, min=0.0),
                        ssflow=torch.clamp(Q1_rout, min=0.0),
                        gwflow=torch.clamp(Q2_rout, min=0.0),
                        AET_hydro=AET.mean(-1, keepdim=True),
                        PET_hydro=PET.mean(-1, keepdim=True))

