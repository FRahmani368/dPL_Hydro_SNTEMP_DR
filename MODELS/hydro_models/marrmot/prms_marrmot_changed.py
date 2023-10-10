import pandas as pd
import torch
from MODELS.PET_models.potet import get_potet
# from functorch import vmap, jacrev, jacfwd, vjp
import torch.nn.functional as F

class prms_marrmot_changed(torch.nn.Module):
    def __init__(self, args, settings={'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}):
        super(prms_marrmot, self).__init__()
        self.args = args
        self.settings = settings

    def smoothThreshold_temperature_logistic(self, T, Tt, r=0.01):
        # By transforming the equation above to Sf = f(P,T,Tt,r)
        # Sf = P * 1/ (1+exp((T-Tt)/r))
        # T       : current temperature
        # Tt      : threshold temperature below which snowfall occurs
        # r       : [optional] smoothing parameter rho, default = 0.01
        # calculate multiplier
        out = 1 / (1 + torch.exp((T - Tt) / r))

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

    def smoothThreshold_storage_logistic(self, S, Smax, r=2, e=1):
        Smax = torch.where(Smax < 0.0,
                           torch.zeros(Smax.shape, dtype=torch.float32, device=self.args["device"]),
                           Smax)

        out = torch.where(r * Smax == 0.0,
                          1 / (1 + torch.exp((S - Smax + r * e * Smax) / r)),
                          1 / (1 + torch.exp((S - Smax + r * e * Smax) / (r * Smax))))
        return out

    def interception_1(self, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)

        out = In * (1 - self.smoothThreshold_storage_logistic(S, Smax))   # , varargin_r, varargin_e
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
        out = In * (1 - self.smoothThreshold_storage_logistic(S, Smax))   #, varargin_r, varargin_e
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

    def ODE_approx_IE(self, args, t, S1_old, S2_old, S3_old, S4_old, S5_old, S6_old, S7_old,
                      delta_S1, delta_S2, delta_S3, delta_S4, delta_S5, delta_S6, delta_S7):
        return S1_old




    def Run_for_one_day(self,S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6):
        # storages:
        # There are 7 reservoirs in PRMS
        if S_tensor.dim() == 3:
            S1 = S_tensor[:, 0, :]
            S2 = S_tensor[:, 1, :]
            S3 = S_tensor[:, 2, :]
            S4 = S_tensor[:, 3, :]
            S5 = S_tensor[:, 4, :]
            S6 = S_tensor[:, 5, :]
            S7 = S_tensor[:, 6, :]
        elif S_tensor.dim() == 2:  # mostly for calculating Jacobian!
            S1 = S_tensor[:, 0]
            S2 = S_tensor[:, 1]
            S3 = S_tensor[:, 2]
            S4 = S_tensor[:, 3]
            S5 = S_tensor[:, 4]
            S6 = S_tensor[:, 5]
            S7 = S_tensor[:, 6]

        delta_t = 1  # timestep (day)
        # P = Precip[:, t, :]
        # Ep = PET[:, t, :]
        # T = mean_air_temp[:, t, :]

        # fluxes
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
        dS1 = flux_ps - flux_m
        dS2 = flux_pin - flux_ein - flux_ptf
        dS3 = flux_pim + flux_mim - flux_eim - flux_sas
        dS4 = flux_inf - flux_ea - flux_pc
        dS5 = flux_pc - flux_et - flux_excs
        dS6 = flux_qres - flux_gad - flux_ras
        dS7 = flux_sep + flux_gad - flux_bas - flux_snk

        # dS_tensor dimension: [ batch, 7, nmul]
        dS_tensor = torch.cat((dS1.unsqueeze(1),
                               dS2.unsqueeze(1),
                               dS3.unsqueeze(1),
                               dS4.unsqueeze(1),
                               dS5.unsqueeze(1),
                               dS6.unsqueeze(1),
                               dS7.unsqueeze(1)), dim=1)

        fluxes_tensor = torch.cat((flux_ps.unsqueeze(1), flux_pr.unsqueeze(1),
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
        return dS_tensor, fluxes_tensor

    # def error_func(self,y,t, ):
    def error_func(self, S_tensor_new, S_tensor_old, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6, dt=1):
        delta_S,_ = self.Run_for_one_day(S_tensor_old, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6);  ##bs*ny
        # err = (y - self.y0)/self.dt - delta_S
        err = (S_tensor_new - S_tensor_old) / dt - delta_S
        return err  ##bs*ny

    # def ODEsolver_NR(self, y0, dt, t):
    def ODEsolver_NR(self, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6):
        ALPHA = 1e-4;  # criteria for decrease
        MIN_LAMBDA = 0.1;  # min lambda
        MAX_LAMBDA = 0.5;  # max lambda
        # y = y0;  # initial guess
        y = torch.clone(S_tensor)  # initial guess
        # self.y0 = y0
        # self.dt = dt
        F = self.error_func(y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6)  # evaluate initial guess  bs*ny
        # bs, ny = y.shape
        # jac = torch.autograd.functional.jacobian(self.error_func, (y, t))  ##bs*ny*ny
        jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor,
                                                                   P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
        jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
        jac_new = jac_new.permute(2, 0, 1)

        if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
            exitflag = -1;  # matrix may be singular
        else:
            exitflag = 1;  # normal exit

        resnorm = torch.linalg.norm(F, float('inf'), dim=[1])  # calculate norm of the residuals
        resnorm0 = 100 * resnorm;
        # dy = torch.zeros(y.shape).to(y0);  # dummy values
        dy = torch.zeros(y.shape).to(S_tensor);  # dummy values
        ##%% solver
        Niter = 0;  # start counter
        # lambda_ = torch.tensor(1.0).to(y0)  # backtracking
        lambda_ = torch.tensor(1.0).to(S_tensor)  # backtracking
        while ((torch.max(resnorm) > self.settings["TolFun"] or lambda_ < 1) and exitflag >= 0 and Niter <=
               self.settings["MaxIter"]):
            if lambda_ == 1:
                ### Newton-Raphson solver
                Niter = Niter + 1;  ## increment counter
                ### update Jacobian, only if necessary
                if torch.max(resnorm / resnorm0) > 0.2:
                    jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
                    jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
                    jac_new = jac_new.permute(2, 0, 1)  ## bs*ny*ny

                    if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
                        exitflag = -1;  ## % matrix may be singular
                        break

                if torch.min(1 / torch.linalg.cond(jac_new, p=1)) <= 2.2204e-16:
                    dy = torch.bmm(torch.linalg.pinv(jac_new), -F.unsqueeze(-1)).squeeze(
                        -1);  ## bs*ny*ny  , bs*ny*1 = bs*ny
                else:
                    dy = -torch.linalg.lstsq(jac_new, F).solution;
                g = torch.bmm(F.unsqueeze(1), jac_new);  # %star; % gradient of resnorm  bs*1*ny, bs*ny*ny = bs*1*ny
                slope = torch.bmm(g, dy.unsqueeze(-1)).squeeze();  # %_star; % slope of gradient  bs*1*ny,bs*ny*1
                fold_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% objective function
                yold = torch.clone(y);  ##% initial value
                lambda_min = self.settings["TolX"] / torch.max(abs(dy) / torch.maximum(abs(yold), torch.tensor(1.0)));
            if lambda_ < lambda_min:
                exitflag = 2;  ##% x is too close to XOLD
                break
            elif torch.isnan(dy).any() or torch.isinf(dy).any():
                exitflag = -1;  ##% matrix may be singular
                break
            y = yold + dy * lambda_;  ## % next guess
            # F = self.error_func(y, t);  ## % evaluate this guess
            F = self.error_func(y, S_tensor, P, Ep, T,
                                tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                cgw, resmax, k1, k2, k3, k4, k5, k6)
            b = F[F!=F]
            if len(b) > 0:
                print("end")
            f_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% new objective function
            ###%% check for convergence
            lambda1 = lambda_;  ###% save previous lambda
            if torch.any(f_obj > fold_obj + ALPHA * lambda_ * slope):
                if lambda_ == 1:
                    a = torch.maximum(f_obj - fold_obj - slope, torch.tensor(0.0000001))
                    lambda_ = torch.min(-slope / 2.0 / (a));  ##% calculate lambda
                else:

                    A = 1 / (lambda1 - lambda2);  ##Scalar
                    B = torch.stack([torch.stack([1.0 / lambda1 ** 2.0, -1.0 / lambda2 ** 2.0]),
                                     torch.stack([-lambda2 / lambda1 ** 2.0, lambda1 / lambda2 ** 2.0])]);  ##2*2
                    C = torch.stack([f_obj - fold_obj - lambda1 * slope, f2_obj - fold_obj - lambda2 * slope]);  ##2*1
                    a = (A * B @ C)[0, :];
                    b = (A * B @ C)[1, :];
                    a = torch.maximum(a, torch.tensor(0.0000001))
                    b = torch.maximum(b, torch.tensor(0.0000001))
                    if torch.all(a == 0):
                        lambda_tmp = -slope / 2 / b;
                    else:
                        discriminant = b ** 2 - 3 * a * slope;
                        if torch.any(discriminant < 0):
                            lambda_tmp = MAX_LAMBDA * lambda1;
                        elif torch.any(b <= 0):
                            lambda_tmp = (-b + torch.sqrt(discriminant)) / 3 / a;
                        else:
                            lambda_tmp = -slope / (b + torch.sqrt(discriminant));

                    lambda_ = torch.min(
                        torch.minimum(lambda_tmp, torch.tensor(MAX_LAMBDA * lambda1)));  # % minimum step length

            elif torch.isnan(f_obj).any() or torch.isinf(f_obj).any():
                ## % limit undefined evaluation or overflow
                lambda_ = MAX_LAMBDA * lambda1;
            else:
                lambda_ = torch.tensor(1.0).to(S_tensor);  ### % fraction of Newton step

            if lambda_ < 1:
                lambda2 = lambda1;
                f2_obj = torch.clone(f_obj);  ##% save 2nd most previous value
                lambda_ = torch.maximum(lambda_, torch.tensor(MIN_LAMBDA * lambda1));  ###% minimum step length
                continue
            #lambda2 = lambda_
            resnorm0 = resnorm;  ##% old resnorm
            resnorm = torch.linalg.norm(F, float('inf'), dim=[1]);  ###% calculate new resnorm
        print("day ", "Iteration ", Niter, "Flag ", exitflag)
        return y, F, exitflag

    def ODEsolver_NR_modified(self, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6):
        ALPHA = 1e-4;  # criteria for decrease
        MIN_LAMBDA = 0.1;  # min lambda
        MAX_LAMBDA = 0.5;  # max lambda
        # y = y0;  # initial guess
        y = torch.clone(S_tensor)  # initial guess
        # self.y0 = y0
        # self.dt = dt
        F = self.error_func(y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6)  # evaluate initial guess  bs*ny
        # bs, ny = y.shape
        # jac = torch.autograd.functional.jacobian(self.error_func, (y, t))  ##bs*ny*ny
        jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor,
                                                                   P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
        jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
        jac_new = jac_new.permute(2, 0, 1)

        if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
            exitflag = -1;  # matrix may be singular
        else:
            exitflag = 1;  # normal exit

        resnorm = torch.linalg.norm(F, float('inf'), dim=[1])  # calculate norm of the residuals
        resnorm0 = 100 * resnorm;
        # dy = torch.zeros(y.shape).to(y0);  # dummy values
        dy = torch.zeros(y.shape).to(S_tensor);  # dummy values
        ##%% solver
        Niter = 0;  # start counter
        # lambda_ = torch.tensor(1.0).to(y0)  # backtracking
        # lambda_ = torch.tensor(1.0).to(S_tensor)  # backtracking
        lambda_ = torch.ones(S_tensor.shape).to(S_tensor)
        while ((torch.max(resnorm) > self.settings["TolFun"] or lambda_ < 1) and exitflag >= 0 and Niter <=
               self.settings["MaxIter"]):
            if lambda_ == 1:
                ### Newton-Raphson solver
                Niter = Niter + 1;  ## increment counter
                ### update Jacobian, only if necessary
                if torch.max(resnorm / resnorm0) > 0.2:
                    jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
                    jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
                    jac_new = jac_new.permute(2, 0, 1)  ## bs*ny*ny

                    if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
                        exitflag = -1;  ## % matrix may be singular
                        break

                if torch.min(1 / torch.linalg.cond(jac_new, p=1)) <= 2.2204e-16:
                    dy = torch.bmm(torch.linalg.pinv(jac_new), -F.unsqueeze(-1)).squeeze(
                        -1);  ## bs*ny*ny  , bs*ny*1 = bs*ny
                else:
                    dy = -torch.linalg.lstsq(jac_new, F).solution;
                g = torch.bmm(F.unsqueeze(1), jac_new);  # %star; % gradient of resnorm  bs*1*ny, bs*ny*ny = bs*1*ny
                slope = torch.bmm(g, dy.unsqueeze(-1)).squeeze();  # %_star; % slope of gradient  bs*1*ny,bs*ny*1
                fold_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% objective function
                yold = torch.clone(y);  ##% initial value
                lambda_min = self.settings["TolX"] / torch.max(abs(dy) / torch.maximum(abs(yold), torch.tensor(1.0)));
            if lambda_ < lambda_min:
                exitflag = 2;  ##% x is too close to XOLD
                break
            elif torch.isnan(dy).any() or torch.isinf(dy).any():
                exitflag = -1;  ##% matrix may be singular
                break
            y = yold + dy * lambda_;  ## % next guess
            # F = self.error_func(y, t);  ## % evaluate this guess
            F = self.error_func(y, S_tensor, P, Ep, T,
                                tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                cgw, resmax, k1, k2, k3, k4, k5, k6)
            a = F[F!=F]
            if len(a) > 0:
                print("end")
            f_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% new objective function
            ###%% check for convergence
            lambda1 = lambda_;  ###% save previous lambda

            if torch.any(f_obj > fold_obj + ALPHA * lambda_ * slope):
                if lambda_ == 1:
                    lambda_ = torch.min(-slope / 2.0 / (f_obj - fold_obj - slope));  ##% calculate lambda
                else:

                    A = 1 / (lambda1 - lambda2);  ##Scalar
                    B = torch.stack([torch.stack([1.0 / lambda1 ** 2.0, -1.0 / lambda2 ** 2.0]),
                                     torch.stack([-lambda2 / lambda1 ** 2.0, lambda1 / lambda2 ** 2.0])]);  ##2*2
                    C = torch.stack([f_obj - fold_obj - lambda1 * slope, f2_obj - fold_obj - lambda2 * slope]);  ##2*1
                    a = (A * B @ C)[0, :];
                    b = (A * B @ C)[1, :];

                    if torch.all(a == 0):
                        lambda_tmp = -slope / 2 / b;
                    else:
                        discriminant = b ** 2 - 3 * a * slope;
                        if torch.any(discriminant < 0):
                            lambda_tmp = MAX_LAMBDA * lambda1;
                        elif torch.any(b <= 0):
                            lambda_tmp = (-b + torch.sqrt(discriminant)) / 3 / a;
                        else:
                            lambda_tmp = -slope / (b + torch.sqrt(discriminant));

                    lambda_ = torch.min(
                        torch.minimum(lambda_tmp, torch.tensor(MAX_LAMBDA * lambda1)));  # % minimum step length

            elif torch.isnan(f_obj).any() or torch.isinf(f_obj).any():
                ## % limit undefined evaluation or overflow
                lambda_ = MAX_LAMBDA * lambda1;
            else:
                lambda_ = torch.tensor(1.0).to(S_tensor);  ### % fraction of Newton step

            if lambda_ < 1:
                lambda2 = lambda1;
                f2_obj = torch.clone(f_obj);  ##% save 2nd most previous value
                lambda_ = torch.maximum(lambda_, torch.tensor(MIN_LAMBDA * lambda1));  ###% minimum step length
                continue

            resnorm0 = resnorm;  ##% old resnorm
            resnorm = torch.linalg.norm(F, float('inf'), dim=[1]);  ###% calculate new resnorm
        # print("day ", t.detach().cpu().numpy(), "Iteration ", Niter, "Flag ", exitflag)
        return y, F, exitflag


    def f3D(self, x, c_PRMS, params, args, warm_up=0, init=False):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args["optData"]["varT_PRMS"]
        vars_c_PRMS = args["optData"]["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, 0:warm_up, :]
                warm_up_model = prms_marrmot(args=args)
                S_tensor = warm_up_model(xinit, c_PRMS, paramsinit, args, warm_up=0, init=True)
        else:
            # All storages in prms. There are 7.
            S_tensor = torch.zeros(
                [x.shape[0], 7, nmul], dtype=torch.float32, device=args["device"]
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
            P = Precip[:, t, :]
            Ep = PET[:, t, :]
            T = mean_air_temp[:, t, :]
            delta_S, _ = self.Run_for_one_day(S_tensor, P, Ep, T,
                                            tt[:, t, :], ddf[:, t, :], alpha[:, t, :], beta[:, t, :], stor[:, t, :],
                                              retip[:, t, :], fscn[:, t, :], scx[:, t, :], scn[:, t, :],
                                              flz[:, t, :], stot[:, t, :], remx[:, t, :], smax[:, t, :],
                                            cgw[:, t, :], resmax[:, t, :], k1[:, t, :], k2[:, t, :],
                                              k3[:, t, :], k4[:, t, :], k5[:, t, :], k6[:, t, :])

            S, error, exit_flag = self.ODEsolver_NR(S_tensor, P, Ep, T,
                                            tt[:, t, :], ddf[:, t, :], alpha[:, t, :], beta[:, t, :], stor[:, t, :],
                                              retip[:, t, :], fscn[:, t, :], scx[:, t, :], scn[:, t, :],
                                              flz[:, t, :], stot[:, t, :], remx[:, t, :], smax[:, t, :],
                                            cgw[:, t, :], resmax[:, t, :], k1[:, t, :], k2[:, t, :],
                                              k3[:, t, :], k4[:, t, :], k5[:, t, :], k6[:, t, :])
        return S_tensor, dS_tensor, fluxes_tensor
    def forward(self, x, c_PRMS, params, args, warm_up=0, init=False):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        bs = self.args["hyperparameters"]["batch_size"]
        vars = args["optData"]["varT_PRMS"]
        vars_c_PRMS = args["optData"]["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, 0:warm_up, :]
                warm_up_model = prms_marrmot(args=args)
                S_tensor, _, _ = warm_up_model(xinit, c_PRMS, paramsinit, args, warm_up=0, init=True)
        else:
            # All storages in prms. There are 7.
            # S_tensor = torch.zeros(
            #     [x.shape[0] * nmul, 7], dtype=torch.float32, device=args["device"]
            # ) + 2
            S_tensor = torch.tensor([15,7,3,8,22,10,10]).unsqueeze(0).to(x)

        tt = torch.tensor([1.0]).repeat([1, 365]).to(x)
        ddf = torch.tensor([10.0]).repeat([1, 365]).to(x)
        alpha = torch.tensor([0.5]).repeat([1, 365]).to(x)
        beta = torch.tensor([0.5]).repeat([1, 365]).to(x)
        stor = torch.tensor([2.5]).repeat([1, 365]).to(x)
        retip = torch.tensor([25.0]).repeat([1, 365]).to(x)
        fscn = torch.tensor([0.5]).repeat([1, 365]).to(x)
        scx = torch.tensor([0.5]).repeat([1, 365]).to(x)
        scn = fscn * scx
        flz = torch.tensor([0.5]).repeat([1, 365]).to(x)
        stot = torch.tensor([1000.0]).repeat([1, 365]).to(x)
        remx = (1 - flz) * stot
        smax = flz * stot
        cgw = torch.tensor([10.0]).repeat([1, 365]).to(x)
        resmax = torch.tensor([150.0]).repeat([1, 365]).to(x)
        k1 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k2 = torch.tensor([2.5]).repeat([1, 365]).to(x)
        k3 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k4 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k5 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k6 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        # ## parameters for prms_marrmot. there are 18 parameters in it
        # tt = self.multi_comp_parameter_bounds(params, 0, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # ddf = self.multi_comp_parameter_bounds(params, 1, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # alpha = self.multi_comp_parameter_bounds(params, 2, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # beta = self.multi_comp_parameter_bounds(params, 3, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # stor = self.multi_comp_parameter_bounds(params, 4, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # retip = self.multi_comp_parameter_bounds(params, 5, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # fscn = self.multi_comp_parameter_bounds(params, 6, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # scx = self.multi_comp_parameter_bounds(params, 7, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # scn = fscn * scx
        # flz = self.multi_comp_parameter_bounds(params, 8, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # stot = self.multi_comp_parameter_bounds(params, 9, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # remx = (1 - flz) * stot
        # smax = flz * stot
        # cgw = self.multi_comp_parameter_bounds(params, 10, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # resmax = self.multi_comp_parameter_bounds(params, 11, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k1 = self.multi_comp_parameter_bounds(params, 12, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k2 = self.multi_comp_parameter_bounds(params, 13, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k3 = self.multi_comp_parameter_bounds(params, 14, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k4 = self.multi_comp_parameter_bounds(params, 15, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k5 = self.multi_comp_parameter_bounds(params, 16, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k6 = self.multi_comp_parameter_bounds(params, 17, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        #################
        # inputs
        climate_path = r"G:\\Farshid\\GitHub\\MARRMoT\\MARRMoT\\prms_climate_data.csv"
        climate = torch.tensor((pd.read_csv(climate_path, header=None)).to_numpy()).to(x)
        Precip = climate[0:365,0:1].permute(1,0).unsqueeze(-1).repeat(1, 1, nmul).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        mean_air_temp = climate[0:365,2:3].permute(1,0).unsqueeze(-1).repeat(1, 1, nmul).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        PET = climate[0:365,1:2].permute(1,0).unsqueeze(-1).repeat(1, 1, nmul).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])


        # Precip = (
        #     x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        # ).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # mean_air_temp = ((Tmaxf + Tminf) / 2).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # dayl = (
        #     x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        # ).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        Ngrid, Ndays = Precip.shape[0], Precip.shape[1]
        # hamon_coef = torch.ones(dayl.shape, dtype=torch.float32, device=args["device"]) * 0.006  # this can be param
        # PET = get_potet(
        #     args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        # )

        # initialize the Q_sim
        Q_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])

        for t in range(Ndays):
            P = Precip[:, t]
            Ep = PET[:, t]
            T = mean_air_temp[:, t]
            delta_S, fluxes_tensor = self.Run_for_one_day(S_tensor, P, Ep, T,
                                              tt[:, t], ddf[:, t], alpha[:, t], beta[:, t],
                                              stor[:, t],
                                              retip[:, t], fscn[:, t], scx[:, t], scn[:, t],
                                              flz[:, t], stot[:, t], remx[:, t], smax[:, t],
                                              cgw[:, t], resmax[:, t], k1[:, t], k2[:, t],
                                              k3[:, t], k4[:, t], k5[:, t], k6[:, t])
            a = fluxes_tensor[fluxes_tensor!=fluxes_tensor]
            if len(a) > 0:
                print("end")
            S_tensor, error, exit_flag = self.ODEsolver_NR(S_tensor, P, Ep, T,
                                                    tt[:, t], ddf[:, t], alpha[:, t], beta[:, t],
                                                    stor[:, t],
                                                    retip[:, t], fscn[:, t], scx[:, t], scn[:, t],
                                                    flz[:, t], stot[:, t], remx[:, t], smax[:, t],
                                                    cgw[:, t], resmax[:, t], k1[:, t], k2[:, t],
                                                    k3[:, t], k4[:, t], k5[:, t], k6[:, t])






        return S_tensor, dS_tensor, fluxes_tensor
            # in marrmot code, there are three ways of solving it:
            # 1) Newton-Raphson
            # 2) fsolve
            # 3) isqnonlin
