import numpy as np
import os
import pandas as pd
import torch
from MODELS.potet import get_potet

class prms_marrmot(torch.nn.Module):
    def __init__(self):
        super(prms_marrmot, self).__init__()

    def smoothThreshold_temperature_logistic(self, T, Tt, r = 0.01):
        # By transforming the equation above to Sf = f(P,T,Tt,r)
        # Sf = P * 1/ (1+exp((T-Tt)/r))
        # T       : current temperature
        # Tt      : threshold temperature below which snowfall occurs
        # r       : [optional] smoothing parameter rho, default = 0.01
        # calculate multiplier
        out = 1 / (1 + torch.exp((T - Tt) / r))

        return out


    def snowfall_1(self, In, T, p1, varargin = 0.01):
        out = In * ( 1 - self.smoothThreshold_temperature_logistic(T, p1, r=varargin))
        return out

    def rainfall_1(self, In, T, p1, varargin = 0.01):
        # inputs:
        # p1   - temperature threshold above which rainfall occurs [oC]
        # T    - current temperature [oC]
        # In   - incoming precipitation flux [mm/d]
        # varargin(1) - smoothing variable r (default 0.01)
        out = In * (1 - self.smoothThreshold_temperature_logistic(T, p1, r=varargin))
        return out

    def split_1(self, p1, In):
        # inputs:
        # p1   - fraction of flux to be diverted [-]
        # In   - incoming flux [mm/d]
        out = p1 * In
        return out

    def smoothThreshold_storage_logistic(self, args, S, Smax, r=0.01, e=5.0):
        Smax = torch.where(Smax < 0.0,
                           torch.zeros(Smax.shape, dtype=torch.float32, device=args["device"]),
                           Smax)

        out = torch.where(r * Smax == 0.0,
                          1 / (1 + torch.exp((S - Smax + r * e * Smax) / r)),
                          1 / (1 + torch.exp((S - Smax + r * e * Smax) / (r * Smax))))
        return out

    def interception_1(self, args, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)

        out = In * (1 - self.smoothThreshold_storage_logistic(args, S, Smax, varargin_r, varargin_e))
        return out

    def melt_1(self, p1, p2, T, S, dt):
        # Constraints:  f <= S/dt
        # inputs:
        # p1   - degree-day factor [mm/oC/d]
        # p2   - temperature threshold for snowmelt [oC]
        # T    - current temperature [oC]
        # S    - current storage [mm]
        # dt   - time step size [d]
        out = torch.min(p1 * (T - p2), S / dt)
        out = torch.clamp(out, min=0.0)
        return out

    def saturation_1(self, args, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)
        out = In * (1 - self.smoothThreshold_storage_logistic(args, S, Smax, varargin_r, varargin_e))
        return out

    def saturation_8(self, p1, p2, S, Smax, In):
        # description: Description:  Saturation excess flow from a store with different degrees
        # of saturation (min-max linear variant)
        # inputs:
        # p1   - minimum fraction contributing area [-]
        # p2   - maximum fraction contributing area [-]
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # In   - incoming flux [mm/d]
        out = (p1 + (p2 - p1) * S / Smax) * In
        return out

    def effective_1(self, In1, In2):
        # description: General effective flow (returns flux [mm/d]) Constraints:  In1 > In2
        # inputs:
        # In1  - first flux [mm/d]
        # In2  - second flux [mm/d]
        out = torch.clamp(In1 - In2, min=0.0)
        return out

    def recharge_7(selfself, p1, fin):
        # Description:  Constant recharge limited by incoming flux
        # p1   - maximum recharge rate [mm/d]
        # fin  - incoming flux [mm/d]
        out = torch.min(p1, fin)
        return out

    def recharge_2(self, p1, S, Smax, flux):
        # Description:  Recharge as non-linear scaling of incoming flux
        # Constraints:  S >= 0
        # inputs:
        # p1   - recharge scaling non-linearity [-]
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # flux - incoming flux [mm/d]
        S = torch.clamp(S, min=0.0)
        out = flux * ((S / Smax) ** p1)
        return out

    def interflow_4(self, p1, p2, S):
        # Description:  Combined linear and scaled quadratic interflow
        # Constraints: f <= S
        #              S >= 0     - prevents numerical issues with complex numbers
        # inputs:
        # p1   - time coefficient [d-1]
        # p2   - scaling factor [mm-1 d-1]
        # S    - current storage [mm]
        S = torch.clamp(S, min=0.0)
        out = torch.min(S, p1 * S + p2 * (S ** 2))
        return out

    def baseflow_1(self, p1, S):
        # Description:  Outflow from a linear reservoir
        # inputs:
        # p1   - time scale parameter [d-1]
        # S    - current storage [mm]
        out = p1 * S
        return out

    def evap_1(self, S, Ep, dt):
        # Description:  Evaporation at the potential rate
        # Constraints:  f <= S/dt
        # inputs:
        # S    - current storage [mm]
        # Ep   - potential evaporation rate [mm/d]
        # dt   - time step size
        out = torch.min(S / dt, Ep)
        return out

    def evap_7(self, S, Smax, Ep, dt):
        # Description:  Evaporation scaled by relative storage
        # Constraints:  f <= S/dt
        # input:
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # Ep   - potential evapotranspiration rate [mm/d]
        # dt   - time step size [d]
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def evap_15(self, args, Ep, S1, S1max, S2, S2min, dt):
        # Description:  Scaled evaporation if another store is below a threshold
        #  Constraints:  f <= S1/dt
        # inputs:
        # Ep    - potential evapotranspiration rate [mm/d]
        # S1    - current storage in S1 [mm]
        # S1max - maximum storage in S1 [mm]
        # S2    - current storage in S2 [mm]
        # S2min - minimum storage in S2 [mm]
        # dt    - time step size [d]

        # this needs to be checked because in MATLAB version there is a min function that does not make sense to me
        out = (S1 / S1max * Ep) * self.smoothThreshold_storage_logistic(args, S2, S2min, S1 / dt)
        return out

    def multi_comp_semi_static_params(
        self, params, param_no, args, interval=30, method="average"
    ):
        # seperate the piece for each interval
        nmul = args["nmul"]
        param = params[:, :, param_no * nmul : (param_no + 1) * nmul]
        no_basins, no_days = param.shape[0], param.shape[1]
        interval_no = math.floor(no_days / interval)
        remainder = no_days % interval
        param_name_list = list()
        if method == "average":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = torch.mean(
                        param[:, 0:remainder, :], 1, keepdim=True
                    ).repeat((1, remainder, 1))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = torch.mean(
                    param[
                        :,
                        ((i * interval) + remainder) : (
                            ((i + 1) * interval) + remainder
                        ),
                        :,
                    ],
                    1,
                    keepdim=True,
                ).repeat((1, interval, 1))
                param_name_list.append(param_name)
        elif method == "single_val":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = (param[:, 0:1, :]).repeat((1, remainder, 1))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = (
                    param[
                        :,
                        (((i) * interval) + remainder) : (((i) * interval) + remainder)
                        + 1,
                        :,
                    ]
                ).repeat((1, interval, 1))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list_prms"]:
            out_temp = (
                params[:, -1, num * nmul : (num + 1) * nmul]
                * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                + args["marrmot_paramCalLst"][num][0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list_prms"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param_prms"][
                    args["semi_static_params_list_prms"].index(num)
                ],
                method=args["method_for_semi_static_param_prms"][
                    args["semi_static_params_list_prms"].index(num)
                ],
            )
            out = (
                out_temp * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                + args["marrmot_paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                params[:, :, num * nmul : (num + 1) * nmul]
                * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                + args["marrmot_paramCalLst"][num][0]
            )
        return out

    def ODE_approx_IE(args, t, S1_old, S2_old, S3_old, S4_old, S5_old, S6_old, S7_old,
                      delta_S1, delta_S2, delta_S3, delta_S4, delta_S5, delta_S6, delta_S7):
        return S1_old


    def forward(self, x, c_PRMS, params, args, warm_up=0, init=False):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args["optData"]["varT_PRMS"]
        vars_c_PRMS = args["optData"]["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, 0:warm_up, :]
                warm_up_model = prms_marrmot()
                S1, S2, S3, S4, S5, S6, S7 = warm_up_model(xinit, c_PRMS, paramsinit, args, warm_up=0, init=True)
        else:
            S1 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            S2 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            S3 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            S4 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            S5 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            S6 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            S7 = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2

        ## parameters for prms_marrmot. there are 18 parameters in it
        tt = self.multi_comp_parameter_bounds(params, 0, args)
        ddf = self.multi_comp_parameter_bounds(params, 1, args)
        alpha = self.multi_comp_parameter_bounds(params, 2, args)
        beta = self.multi_comp_parameter_bounds(params, 3, args)
        stor = self.multi_comp_parameter_bounds(params, 4, args)
        retip = self.multi_comp_parameter_bounds(params, 5, args)
        fscn = self.multi_comp_parameter_bounds(params, 6, args)
        scx = self.multi_comp_parameter_bounds(params, 7, args)
        scn = fscn * scx
        flz = self.multi_comp_parameter_bounds(params, 8, args)
        stot = self.multi_comp_parameter_bounds(params, 9, args)
        remx = (1 - flz) * stot
        smax = flz * stot
        cgw = self.multi_comp_parameter_bounds(params, 10, args)
        resmax = self.multi_comp_parameter_bounds(params, 11, args)
        k1 = self.multi_comp_parameter_bounds(params, 12, args)
        k2 = self.multi_comp_parameter_bounds(params, 13, args)
        k3 = self.multi_comp_parameter_bounds(params, 14, args)
        k4 = self.multi_comp_parameter_bounds(params, 15, args)
        k5 = self.multi_comp_parameter_bounds(params, 16, args)
        k6 = self.multi_comp_parameter_bounds(params, 17, args)
        #################
        # inputs
        Precip = (
            x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        mean_air_temp = (Tmaxf + Tminf) / 2
        dayl = (
            x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Ngrid, Ndays = Precip.shape[0], Precip.shape[1]
        hamon_coef = torch.ones(dayl.shape, dtype=torch.float32, device=args["device"]) * 0.006  # this can be param
        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        )

        # initialize the Q_sim
        Q_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])

        for t in range(Ndays):
            delta_t = 1 # timestep (day)
            P = Precip[:, t, :]
            Ep = PET[:, t, :]
            T = mean_air_temp[:, t, :]

            # fluxes
            flux_ps = self.snowfall_1(P, T, tt[:, t, :])
            flux_pr = self.rainfall_1(P, T, tt[:, t, :])
            flux_pim = self.split_1(1 - beta[:, t, :], flux_pr)
            flux_psm = self.split_1(beta[:, t, :], flux_pr)
            flux_pby = self.split_1(1 - alpha[:, t, :], flux_psm)
            flux_pin = self.split_1(alpha[:, t, :], flux_psm)
            flux_ptf = self.interception_1(args, flux_pin, S2, stor[:, t, :])
            flux_m = self.melt_1(ddf[:, t, :], tt[:, t, :], T, S1, delta_t)
            flux_mim = self.split_1(1 - beta[:, t, :], flux_m)
            flux_msm = self.split_1(beta[:, t, :], flux_m)
            flux_sas = self.saturation_1(args, flux_pim + flux_mim, S3, retip[:, t, :])
            flux_sro = self.saturation_8(scn[:, t, :], scx[:, t, :], S4, remx[:, t, :], flux_msm + flux_ptf + flux_pby)
            flux_inf = self.effective_1(flux_msm + flux_ptf + flux_pby, flux_sro)
            flux_pc = self.saturation_1(args, flux_inf, S4, remx[:, t, :])
            flux_excs = self.saturation_1(args, flux_pc, S5, smax[:, t, :])
            flux_sep = self.recharge_7(cgw[:, t, :], flux_excs)
            flux_qres = self.effective_1(flux_excs, flux_sep)
            flux_gad = self.recharge_2(k2[:, t, :], S6, resmax[:, t, :], k1[:, t, :])
            flux_ras = self.interflow_4(k3[:, t, :], k4[:, t, :], S6)
            flux_bas = self.baseflow_1(k5[:, t, :], S7)
            flux_snk = self.baseflow_1(k6[:, t, :], S7)    # represents transbasin gw or undergage streamflow
            flux_ein = self.evap_1(S2, beta[:, t, :] * Ep, delta_t)
            flux_eim = self.evap_1(S3, (1 - beta[:, t, :]) * Ep, delta_t)
            flux_ea = self.evap_7(S4, remx[:, t, :], Ep - flux_ein - flux_eim, delta_t)
            flux_et = self.evap_15(args, Ep - flux_ein - flux_eim - flux_ea, S5, smax[:, t, :], S4, Ep - flux_ein - flux_eim, delta_t)

            # stores ODEs
            dS1 = flux_ps - flux_m
            dS2 = flux_pin - flux_ein - flux_ptf
            dS3 = flux_pim + flux_mim - flux_eim - flux_sas
            dS4 = flux_inf - flux_ea - flux_pc
            dS5 = flux_pc - flux_et - flux_excs
            dS6 = flux_qres - flux_gad - flux_ras
            dS7 = flux_sep + flux_gad - flux_bas - flux_snk






















