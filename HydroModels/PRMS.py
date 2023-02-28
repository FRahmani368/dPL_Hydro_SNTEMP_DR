import torch
import torch.nn as nn


class PRMS(nn.Module):
    def __init__(self, theta, delta_t, climate_data, args):
        super().__init__()

        # self.register_parameter("theta", theta)
        # self.theta = torch.nn.Parameter(theta)
        self.theta = theta
        self.delta_t = delta_t
        self.climate_data = climate_data
        self.args = args

    def forward(self, t, y):
        args = self.args
        delta_t = self.delta_t
        ##parameters
        ## parameters for prms_marrmot. there are 18 parameters in it
        # tt = torch.tensor([1.0]).repeat([y.shape[0]]).to(y)
        # ddf = torch.tensor([10.0]).repeat([y.shape[0]]).to(y)
        # alpha = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # beta = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # stor = torch.tensor([2.5]).repeat([y.shape[0]]).to(y)
        # retip = torch.tensor([25.0]).repeat([y.shape[0]]).to(y)
        # fscn = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # scx = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # scn = fscn * scx
        # flz = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # stot = torch.tensor([1000.0]).repeat([y.shape[0]]).to(y)
        # remx = (1 - flz) * stot
        # smax = flz * stot
        # cgw = torch.tensor([10.0]).repeat([y.shape[0]]).to(y)
        # resmax = torch.tensor([150.0]).repeat([y.shape[0]]).to(y)
        # k1 = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # k2 = torch.tensor([2.5]).repeat([y.shape[0]]).to(y)
        # k3 = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # k4 = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # k5 = torch.tensor([0.5]).repeat([y.shape[0]]).to(y)
        # k6 = torch.tensor([0.0]).repeat([y.shape[0]]).to(y)

        tt = self.theta[:, 0] * (args["marrmot_paramCalLst"][0][1] - args["marrmot_paramCalLst"][0][0]) \
             + args["marrmot_paramCalLst"][0][0]
        ddf = self.theta[:, 1] * (args["marrmot_paramCalLst"][1][1] - args["marrmot_paramCalLst"][1][0]) \
              + args["marrmot_paramCalLst"][1][0]
        alpha = self.theta[:, 2] * (args["marrmot_paramCalLst"][2][1] - args["marrmot_paramCalLst"][2][0]) \
                + args["marrmot_paramCalLst"][2][0]
        beta = self.theta[:, 3] * (args["marrmot_paramCalLst"][3][1] - args["marrmot_paramCalLst"][3][0]) \
               + args["marrmot_paramCalLst"][3][0]
        stor = self.theta[:, 4] * (args["marrmot_paramCalLst"][4][1] - args["marrmot_paramCalLst"][4][0]) \
               + args["marrmot_paramCalLst"][4][0]
        retip = self.theta[:, 5] * (args["marrmot_paramCalLst"][5][1] - args["marrmot_paramCalLst"][5][0]) \
                + args["marrmot_paramCalLst"][5][0]
        fscn = self.theta[:, 6] * (args["marrmot_paramCalLst"][6][1] - args["marrmot_paramCalLst"][6][0]) \
               + args["marrmot_paramCalLst"][6][0]
        scx = self.theta[:, 7] * (args["marrmot_paramCalLst"][7][1] - args["marrmot_paramCalLst"][7][0]) \
              + args["marrmot_paramCalLst"][7][0]
        scn = fscn * scx
        flz = self.theta[:, 8] * (args["marrmot_paramCalLst"][8][1] - args["marrmot_paramCalLst"][8][0]) \
              + args["marrmot_paramCalLst"][8][0]
        stot = self.theta[:, 9] * (args["marrmot_paramCalLst"][9][1] - args["marrmot_paramCalLst"][9][0]) \
               + args["marrmot_paramCalLst"][9][0]
        remx = (1 - flz) * stot
        smax = flz * stot
        cgw = self.theta[:, 10] * (args["marrmot_paramCalLst"][10][1] - args["marrmot_paramCalLst"][10][0]) \
              + args["marrmot_paramCalLst"][10][0]
        resmax = self.theta[:, 11] * (args["marrmot_paramCalLst"][11][1] - args["marrmot_paramCalLst"][11][0]) \
                 + args["marrmot_paramCalLst"][11][0]
        k1 = self.theta[:, 12] * (args["marrmot_paramCalLst"][12][1] - args["marrmot_paramCalLst"][12][0]) \
             + args["marrmot_paramCalLst"][12][0]
        k2 = self.theta[:, 13] * (args["marrmot_paramCalLst"][13][1] - args["marrmot_paramCalLst"][13][0]) \
             + args["marrmot_paramCalLst"][13][0]
        k3 = self.theta[:, 14] * (args["marrmot_paramCalLst"][14][1] - args["marrmot_paramCalLst"][14][0]) \
             + args["marrmot_paramCalLst"][14][0]
        k4 = self.theta[:, 15] * (args["marrmot_paramCalLst"][14][1] - args["marrmot_paramCalLst"][14][0]) \
             + args["marrmot_paramCalLst"][14][0]
        k5 = self.theta[:, 16] * (args["marrmot_paramCalLst"][14][1] - args["marrmot_paramCalLst"][14][0]) \
             + args["marrmot_paramCalLst"][14][0]
        k6 =  0.0 * self.theta[:, 17] * (args["marrmot_paramCalLst"][14][1] - args["marrmot_paramCalLst"][14][0]) \
             + args["marrmot_paramCalLst"][14][0]

        ##% stores
        S1 = y[:, 0].clone();
        S2 = y[:, 1].clone();
        S3 = y[:, 2].clone();
        S4 = y[:, 3].clone();
        S5 = y[:, 4].clone();
        S6 = y[:, 5].clone();
        S7 = y[:, 6].clone();
        dS = torch.zeros(y.shape).to(y)
        fluxes = torch.zeros((y.shape[0], 25)).to(y)

        climate_in = self.climate_data[int(t), :, :];  ##% climate at this step
        P = climate_in[:, 0];
        T = climate_in[:, 1];
        Ep = climate_in[:, 2];

        ##% fluxes functions
        flux_ps = self.snowfall_1(P, T, tt)
        flux_pr = self.rainfall_1(P, T, tt)
        flux_pim = self.split_1(1 - beta, flux_pr)
        flux_psm = self.split_1(beta, flux_pr)
        flux_pby = self.split_1(1 - alpha, flux_psm)
        flux_pin = self.split_1(alpha, flux_psm)
        flux_ptf = self.interception_1(flux_pin, S2, stor)
        flux_m = self.melt_1(ddf, tt, T, S1, delta_t)
        flux_mim = self.split_1(1 - beta, flux_m)
        flux_msm = self.split_1(beta, flux_m)
        flux_sas = self.saturation_1(flux_pim + flux_mim, S3, retip)
        flux_sro = self.saturation_8(scn, scx, S4, remx, flux_msm + flux_ptf + flux_pby)
        flux_inf = self.effective_1(flux_msm + flux_ptf + flux_pby, flux_sro)
        flux_pc = self.saturation_1(flux_inf, S4, remx)
        flux_excs = self.saturation_1(flux_pc, S5, smax)
        flux_sep = self.recharge_7(cgw, flux_excs)
        flux_qres = self.effective_1(flux_excs, flux_sep)
        flux_gad = self.recharge_2(k2, S6, resmax, k1)
        flux_ras = self.interflow_4(k3, k4, S6)
        flux_bas = self.baseflow_1(k5, S7)
        flux_snk = self.baseflow_1(k6, S7)  # represents transbasin gw or undergage streamflow
        flux_ein = self.evap_1(S2, beta * Ep, delta_t)
        flux_eim = self.evap_1(S3, (1 - beta) * Ep, delta_t)
        flux_ea = self.evap_7(S4, remx, Ep - flux_ein - flux_eim, delta_t)
        flux_et = self.evap_15(Ep - flux_ein - flux_eim - flux_ea, S5, smax, S4,
                               Ep - flux_ein - flux_eim, delta_t)

        # stores ODEs
        dS[:, 0] = flux_ps - flux_m
        dS[:, 1] = flux_pin - flux_ein - flux_ptf
        dS[:, 2] = flux_pim + flux_mim - flux_eim - flux_sas
        dS[:, 3] = flux_inf - flux_ea - flux_pc
        dS[:, 4] = flux_pc - flux_et - flux_excs
        dS[:, 5] = flux_qres - flux_gad - flux_ras
        dS[:, 6] = flux_sep + flux_gad - flux_bas - flux_snk

        fluxes = torch.cat((flux_ps.unsqueeze(1), flux_pr.unsqueeze(1),
                            flux_pim.unsqueeze(1), flux_psm.unsqueeze(1),
                            flux_pby.unsqueeze(1), flux_pin.unsqueeze(1),
                            flux_ptf.unsqueeze(1), flux_m.unsqueeze(1),
                            flux_mim.unsqueeze(1), flux_msm.unsqueeze(1),
                            flux_sas.unsqueeze(1), flux_sro.unsqueeze(1),
                            flux_inf.unsqueeze(1), flux_pc.unsqueeze(1),
                            flux_excs.unsqueeze(1), flux_qres.unsqueeze(1),
                            flux_sep.unsqueeze(1), flux_gad.unsqueeze(1),
                            flux_ras.unsqueeze(1), flux_bas.unsqueeze(1),
                            flux_snk.unsqueeze(1), flux_ein.unsqueeze(1),
                            flux_eim.unsqueeze(1), flux_ea.unsqueeze(1),
                            flux_et.unsqueeze(1)), dim=1)

        return dS, fluxes

    def smoothThreshold_temperature_logistic(self, T, Tt, r=0.01):
        # By transforming the equation above to Sf = f(P,T,Tt,r)
        # Sf = P * 1/ (1+exp((T-Tt)/r))
        # T       : current temperature
        # Tt      : threshold temperature below which snowfall occurs
        # r       : [optional] smoothing parameter rho, default = 0.01
        # calculate multiplier
        # temp_out = torch.clamp((T - Tt) / r, min=-10.0, max=10.0)
        # out = 1 / (1 + torch.exp(temp_out))
        # out = 1 / (1 + torch.exp((T - Tt) / r))
        # out = torch.zeros(T.shape).to(T) + 0.1
        m = torch.nn.Sigmoid()
        out = m(-(T - Tt) / r)
        return out

    def snowfall_1(self, In, T, p1, varargin=0.01):
        out = In * (self.smoothThreshold_temperature_logistic(T, p1, r=varargin))
        return out

    def rainfall_1(self, In, T, p1, varargin=0.01):
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

    def smoothThreshold_storage_logistic(self, S, Smax, r=0.01, e=5.0):
        Smax = torch.clamp(Smax, min=0.0)

        # out = torch.where(r * Smax == 0.0,
        #                   1 / (1 + torch.exp((S - Smax + r * e * Smax) / r)),
        #                   1 / (1 + torch.exp((S - Smax + r * e * Smax) / (r * Smax))))

        # out = torch.where(r * Smax == 0.0,
        #                   torch.sigmoid(-(S - Smax + r * e * Smax - torch.max(S - Smax + r * e * Smax)) / (r- torch.max(S - Smax + r * e * Smax))),
        #                   torch.sigmoid(-(S - Smax + r * e * Smax - torch.max(S - Smax + r * e * Smax)) / (r * Smax - torch.max(S - Smax + r * e * Smax))))
        # temp_out1 = torch.clamp((S - Smax + r * e * Smax) / r, min=-10.0, max=10.0)
        # temp_out2 = torch.clamp((S - Smax + r * e * Smax) / (r * Smax), min=-10.0, max=10.0)
        m = torch.nn.Sigmoid()
        out = torch.where(r * Smax == 0.0,
                          m(-(S - Smax + r * e * Smax) / r),
                          m(-(S - Smax + r * e * Smax) / (r * Smax)))
        # out = 1 / (1 + torch.exp((S - Smax + r * e * Smax) / r))
        # out = (torch.zeros(S.shape).to(S)) + 0.05
        return out

    def interception_1(self, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)

        out = In * (1 - self.smoothThreshold_storage_logistic(S, Smax, varargin_r, varargin_e))
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

    def saturation_1(self, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)
        out = In * (1 - self.smoothThreshold_storage_logistic(S, Smax, varargin_r, varargin_e))
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

    def recharge_7(self, p1, fin):
        # Description:  Constant recharge limited by incoming flux
        # p1   - maximum recharge rate [mm/d]
        # fin  - incoming flux [mm/d]
        out = torch.minimum(p1, fin)
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
        Ep = torch.clamp(Ep, min=0.0)
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def evap_15(self, Ep, S1, S1max, S2, S2min, dt):
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
        Ep = torch.clamp(Ep, min=0.0)
        out = (S1 / S1max * Ep) * self.smoothThreshold_storage_logistic(S2, S2min, S1 / dt)
        out = torch.clamp(out, min=0.0)
        return out

    def multi_comp_semi_static_params(
            self, params, param_no, args, interval=30, method="average"
    ):
        # seperate the piece for each interval
        nmul = args["nmul"]
        param = params[:, :, param_no * nmul: (param_no + 1) * nmul]
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
                    ((i * interval) + remainder): (
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
                    (((i) * interval) + remainder): (((i) * interval) + remainder)
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
                    params[:, -1, num * nmul: (num + 1) * nmul]
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
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                    + args["marrmot_paramCalLst"][num][0]
            )
        return out
