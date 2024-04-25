import numpy as np
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
import math
from core.utils.small_codes import make_tensor
from core.load_data.time import tRange2Array, intersect
from MODELS.PET_models.potet import get_potet



def str_to_datetime(t):
    if isinstance(t, datetime.datetime):
        return t
    elif isinstance(t, str):
        year, month, day = t.split("-")
        t_datetime_format = datetime.datetime(int(year), int(month), int(day))
        return t_datetime_format


class STREAM_TEMP_EQ(nn.Module):
    def __init__(self):
        super(STREAM_TEMP_EQ, self).__init__()

        # self.a_srflow = nn.Parameter(torch.randn(args['no_basins'],
        #                                          args["res_time_params"]["lenF_srflow"],
        #                                          1))
        # self.b_srflow = nn.Parameter(torch.randn(args['no_basins'],
        #                                          args["res_time_params"]["lenF_srflow"],
        #                                          1))
        # self.bias_srflow = nn.Parameter(torch.randn(args['no_basins']))
        # self.a_ssflow = nn.Parameter(torch.randn(args['no_basins'],
        #                                          args["res_time_params"]["lenF_ssflow"],
        #                                          1))
        # self.b_ssflow = nn.Parameter(torch.randn(args['no_basins'],
        #                                          args["res_time_params"]["lenF_ssflow"],
        #                                          1))
        # self.bias_ssflow = nn.Parameter(torch.zeros(args['no_basins']))
        # self.a_gwflow = nn.Parameter(torch.randn(args['no_basins'],
        #                                          args["res_time_params"]["lenF_gwflow"],
        #                                          1))
        # self.b_gwflow = nn.Parameter(torch.randn(args['no_basins'],
        #                                          args["res_time_params"]["lenF_gwflow"],
        #                                          1))
        # self.bias_gwflow = nn.Parameter(torch.zeros(args['no_basins']))

        # self.shade_fraction = nn.ParameterList(
        #     [
        #         nn.Parameter(
        #             torch.tensor(config["initial_values"]['shade_fraction'])
        #         )
        #         for i in range(config["no_basins"])
        #     ]
        # )
        # self.shade = nn.Parameter(
        #             torch.randn(
        #                 (config["no_basins"], 1)
        #             )
        #         )
        # self.res_time_srflow = nn.Parameter(torch.zeros((99, 1)) + 1)
        # self.res_time_ssflow = nn.Parameter(torch.zeros((99, 1)) + 1)
        # self.res_time_gwflow = nn.Parameter(torch.zeros((99, 1)) + 1)
        # self.shade = nn.Parameter(torch.load("/home/fzr5082/PGML_STemp_results/data/shade.pt"))

    def atm_pressure(self, elev):
        ## from Jake's document
        # mmHg2mb = make_tensor(0.75061683)  # Unit conversion
        # mmHg2inHg = make_tensor(25.3970886)  # Unit conversion
        # P_sea = make_tensor(29.92126)  # Standard pressure ar sea level
        # A_g = make_tensor(9.80665)  # Acceleration due to gravity
        # M_a = make_tensor(0.0289644)  # Molar mass of air
        # R = make_tensor(8.31447)  # universal gas constant
        # T_sea = make_tensor(288.16)  # the standard temperature at sea level
        # P = (1 / mmHg2mb) * (mmHg2inHg) * (P_sea) * torch.exp(-A_g * M_a * elev / (R * T_sea))

        ## the code from stream_temp.f90
        P = 1013 - (0.1055 * elev)

        ## Note: both jakes and stream_temp are close to each other (error less than 0.013 for 99 basins)
        return P

    def atm_longwave_radiation_heat(self, T_a, e_a, shade_total, cloud_fraction, args):
        """
        :param T_a: air temperature in degree Celsius
        :param e_a: vapor pressure
        :return: Atmospheric longwave radiation
        """
        emissivity_air = 0.61 + 0.05 * torch.pow(e_a, 0.5)
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(
            make_tensor(10), (-8.0)
        )  # (J/s*m^2 * K^4)
        longwave_reflect_frac = make_tensor(
            args["STemp_default_params"]["longwave_reflect_fraction"]
        )
        # shade_fraction = make_t0ensor(args['STemp_default_params']['shade_fraction'])
        # cloud_fraction = make_tensor(args['STemp_default_params']['cloud_fractiontion'])

        # H_a = (
        #         (1 - longwave_reflect_frac) * (1 - shade_total) * (1 + 0.17 * torch.pow(cloud_fraction, 2))
        #         * emissivity_air * St_Boltzman_ct * torch.pow((T_a + 273.16), 4)
        # )
        H_a = (
                (3.354939e-8 + 2.74995e-9 * e_a ** 0.5)
                * (1 - shade_total)
                * (1 + 0.17 * cloud_fraction ** 2)
                * (T_a + 273.16) ** 4
        )
        return H_a

    def stream_friction_heat(self, top_width, slope, Q):
        H_f = (
                9805 * Q * slope / top_width
        )  # Q is the seg_inflow (total flow entering a segment)
        return H_f

    def shortwave_solar_radiation_heat(self, albedo, H_sw, shade_total):
        """
        :param albedo: albedo or fraction reflected by stream , dimensionless
        :param H_sw: the clear sky solar radiation in watt per sq meter (seginc_swrad)
        :return: daily average clear sky, shortwave solar radiation for each segment
        """
        # solar_shade_fraction = make_tensor(args['STemp_default_params']['shade_fraction'])
        H_s = (1 - albedo) * (1 - shade_total) * H_sw
        return H_s

    def riparian_veg_longwave_radiation_heat(
            self, T_a, iGrid, shade_fraction_riparian, args
    ):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(
            make_tensor(10), (-8.0)
        )  # (J/s*m^2 * K^4)
        emissivity_veg = make_tensor(args["STemp_default_params"]["emissivity_veg"])
        H_v = (
                emissivity_veg
                * St_Boltzman_ct
                * shade_fraction_riparian
                * torch.pow((T_a + 273.16), 4)
        )
        # H_v = emissivity_veg * St_Boltzman_ct * shade2[iGrid, :] * torch.pow((T_a + 273.16), 4)
        return H_v

    def ABCD_equations(
            self,
            T_a,
            swrad,
            e_a,
            E,
            elev,
            slope,
            top_width,
            up_inflow,
            T_g,
            iGrid,
            shade_fraction_riparian,
            albedo,
            shade_total,
            args,
            cloud_fraction,
    ):
        """

        :param T_a: average daily air temperature
        :param swrad: solar radiation
        :param e_a: vapor pressure
        :param E: Free-water surface-evaporation rate (assumed to be PET, potet in PRMS)
        :param elev: average basin elevation
        :param slope: average stream slope (seg_slope)
        :param top_width: average top width of the stream
        :param up_inflow: is the discharge (variable seg_inflow) which is from upstream
        :return:
        """
        e_s = 6.108 * torch.exp((17.26939 * T_a) / (237.3 + T_a))
        # e_s = 6.108 * torch.exp((17.26939 * T_0) / (237.3 + T_0))
        P = self.atm_pressure(
            elev
        )  # calculating atmosphere pressure based on elevation
        # chacking vapor pressure with saturation vapor pressure
        denom = e_s - e_a
        mask_denom = denom.ge(0)
        # converting negative values to zero
        denom1 = denom * mask_denom.int().float()
        # adding 0.01 to zero values as it is denominator
        mask_denom2 = denom1.eq(0)
        denom2 = denom1 + 0.01 * mask_denom2.int().float()

        B_c = 0.00061 * P / denom2
        B_c1 = 0.00061 * P / (e_s - e_a)
        K_g = make_tensor(1.65)
        delta_Z = make_tensor(args["STemp_default_params"]["delta_Z"])
        # we don't need H_a, because we hae swrad directly from inputs
        H_a = self.atm_longwave_radiation_heat(
            T_a, e_a, shade_total, cloud_fraction, args=args
        )
        ###############
        H_f = self.stream_friction_heat(top_width=top_width, slope=slope, Q=up_inflow)
        H_s = self.shortwave_solar_radiation_heat(
            albedo=albedo, H_sw=swrad, shade_total=shade_total
        )  # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(
            T_a, iGrid, shade_fraction_riparian, args=args
        )

        A = 5.4 * torch.pow(make_tensor(np.full((T_a.shape), 10)), (-8))
        B = torch.pow(make_tensor(10), 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (
                K_g / delta_Z
        )
        C = torch.pow(make_tensor(10), 6) * E * B_c * 2.36
        # Todo: need to check 10**6. it is in fortran code but it is not in the document
        D = (
                H_f
                + H_a
                + H_s
                + H_v
                + 2495 * torch.pow(make_tensor(10), 6) * E * ((B_c * T_a) - 1)
                + (T_g * K_g / delta_Z)
        )
        # D = H_a + swrad + H_v + 2495 * E * ((B_c * T_a) - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e, iter=50):
        def F(T_e):
            return (
                    A * torch.pow((T_e + 273.16), 4) - C * torch.pow(T_e, 2) + B * T_e - D
            )

        def Fprime(T_e):
            return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B

        ## solving the equation with Newton's method
        for i in range(iter):
            next_geuss = T_e - (F(T_e) / Fprime(T_e))
            T_e = next_geuss

        return T_e

    def finding_K1_K2(self, A, B, C, D, T_e, NEARZERO, T_0):
        """
        :param A: Constant coming from equilibrium temp equation
        :param B: Constant coming from equilibrium temp equation
        :param C: Constant coming from equilibrium temp equation
        :param T_e: equilibrium temperature
        :param H_i: initial net heat flux at temperature T_o, of the upstream inflow
        :param T_o: initial water temperature
        :return: K1 (first order thermal exchange coefficient), K2 (second order coefficient)
        """
        H_i = A * torch.pow((T_0 + 273.16), 4) - C * torch.pow(T_0, 2) + B * T_0 - D
        K1 = 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B
        denom = torch.pow((T_e - T_0), 2)
        mask_denom = denom.le(NEARZERO)
        denom1 = denom + mask_denom.int().float()
        # K2 = (H_i - (K1 * (T_e - T_0))) / denom1
        K2 = (-H_i + (K1 * (T_0 - T_e))) / denom1

        return K1, K2

    def srflow_ssflow_gwflow_portions(
            self,
            discharge,
            srflow_factor=make_tensor(0.40),
            ssflow_factor=make_tensor(0.3),
            gwlow_factor=make_tensor(0.3),
    ):
        srflow = srflow_factor * discharge
        ssflow = ssflow_factor * discharge
        gwflow = gwlow_factor * discharge
        return srflow, ssflow, gwflow

    def ave_temp_res_time(
            self, ave_air_temp, x, res_time, iGrid, iT, args, x_total_raw
    ):
        rho = x.shape[1]  # args['hyperparameters']['rho']
        tArray_Total = tRange2Array(args["optData"]["tRange"])
        tArray_train = tRange2Array(args["optData"]["t_train"])
        _, ind1, _ = intersect(tArray_Total, tArray_train)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["optData"]["varT"] + args["optData"]["varC"]
        temp_res = res_time
        with torch.no_grad():
            temp_res1 = temp_res.int()
        A = res_time.repeat(1, rho)
        B = torch.reshape(A, (res_time.shape[0], rho, res_time.shape[1]))
        ave_air = torch.zeros(
            (
                args["hyperparameters"]["batch_size"],
                args["hyperparameters"]["rho"],
                res_time.shape[1],
            ),
            device=args["device"],
        )
        for i in range(res_time.shape[1]):
            for s, station in enumerate(iGrid):
                array = np.zeros((x.shape[1], temp_res1[s, i].item()), dtype=np.int32)
                for j in range(temp_res1[s, i].item()):
                    array[:, j] = np.arange(
                        (ind1_tensor[0] + iT_tensor[s] - j).item(),
                        (ind1_tensor[0] + iT_tensor[s] - j + x.shape[1]).item(),
                    )
                tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
                max_add = torch.sum(tmax_temp, dim=1)
                tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
                min_add = torch.sum(tmin_temp, dim=1)
                ave_air[s, :, i] = (max_add + min_add) / 2  # (2 * res_time[station, i])
        ave_air_temp = ave_air / B
        # return ave_air
        return ave_air_temp

    def x_sample_air_temp(self, iGrid, iT, lenF, args, x_total_raw, time_range):
        """
        :param iGrid:
        :param iT:
        :param lenF: maximum number of days that it is needed to be considered in average
        :return:
        """
        rho = args["hyperparameters"]["rho"]
        tArray_Total = tRange2Array(args["optData"]["tRange"])
        tArray_sample = tRange2Array(time_range)
        _, ind1, _ = intersect(tArray_Total, tArray_sample)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["optData"]["varT"] + args["optData"]["varC"]
        ave_air = torch.zeros(
            (
                args["hyperparameters"]["batch_size"],
                args["hyperparameters"]["rho"],
                lenF,
            ),
            device=args["device"],
        )
        for s, station in enumerate(iGrid):
            array = np.zeros((rho, lenF), dtype=np.int32)
            for j in range(lenF):
                array[:, j] = np.arange(
                    (ind1_tensor[0] + iT_tensor[s] - j).item(),
                    (ind1_tensor[0] + iT_tensor[s] - j + rho).item(),
                )
            # array = np.flip(array, 1).copy()
            tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp
        return ave_air

    def x_sample_air_temp2(self, iGrid, iT, lenF, args, ave_air_total):
        rho = args["hyperparameters"]["rho"]
        # ave_air = torch.zeros((len(iGrid), args["hyperparameters"]["rho"],
        #                        lenF),
        #                       device=args["device"])
        a = min(ave_air_total.shape[1], args["hyperparameters"]["rho"])
        ave_air = torch.zeros((len(iGrid), a, lenF), device=args["device"])
        # array = np.array([np.arange(x, y) for x, y in zip(iT, iT + rho)])
        ave_air_temp = ave_air_total[iGrid, :, 0:lenF]
        for i in range(len(iGrid)):
            ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + a), :]
            # ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + ave_air_total.shape[1]), :]

        # ave_air_temp = ave_air_total[:, iT : iT + rho, 0:lenF]
        # ave_air = ave_air_temp[iGrid, :, :]
        return ave_air

    def ave_temp_general(self, args, x_total_raw_tensor, time_range):
        vars = args["varT"] + args["varC"]
        lenF_max = np.maximum(
            args["res_time_lenF_srflow"],
            np.maximum(
                args["res_time_lenF_ssflow"],
                args["res_time_lenF_gwflow"],
            ),
        )
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        c, ind1, ind2 = np.intersect1d(tArray_sample, tArray_Total, return_indices=True)
        ave_air = torch.zeros(
            (x_total_raw_tensor.shape[0], len(tArray_sample), lenF_max), device="cpu"
        )

        array = np.zeros((len(ind2), lenF_max), dtype=np.int32)
        for j in range(lenF_max):
            array[:, j] = np.arange(
                (ind2[0] - j).item(), (ind2[0] - j + len(ind2)).item()
            )

        for s in range(x_total_raw_tensor.shape[0]):
            tmax_temp = x_total_raw_tensor[s, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw_tensor[s, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp

        return ave_air

    def res_time_gamma(self, a, b, lenF):
        # UH. a [time (same all time steps), batch, var]
        # a = torch.abs(a)
        if a.dim() == 2:
            m = a.shape
            a1 = a.repeat(1, lenF)
            b1 = b.repeat(1, lenF)
            alpha = F.relu(a1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.1
            beta = F.relu(b1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 3:
            m = a.shape
            a1 = a.repeat(1, 1, lenF)
            b1 = b.repeat(1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.1
            beta = F.relu(b1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 4:
            m = a.shape
            a1 = a.repeat(1, 1, 1, lenF)
            b1 = b.repeat(1, 1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.1
            beta = F.relu(b1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.5
            x = (
                torch.linspace(0.001, 20, lenF)
                .view(lenF, 1, 1, 1)
                .repeat(1, m[0], m[1], m[2])
            )
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        return www

    def res_time_conv(self, x_sample, UH, bias, viewmode=1):
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
        # batch needs to be accommodated by channels and we make use of gr
        # ++++---------------------------------+
        #
        # oups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html
        if UH.shape[1] == 1:
            x = x_sample[:, 0:1, :]
            mm = x.shape
            nb = mm[0]
            m = UH.shape[-1]
            padd = m - 1
            if viewmode == 1:
                xx = x.view([1, nb, mm[-1]])
                w = UH.view([nb, 1, m])
                groups = nb

                # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
                # y = y[:, :, 0:-padd]

            x_sample1 = x_sample.permute(1, 0, 2)
            a = torch.arange(x_sample.shape[1])
            y = F.conv1d(
                x_sample1[a],
                torch.flip(w, [2]),
                groups=groups,
                padding=0,
                stride=1,
                bias=bias,
            )
            y = y.permute(1, 0, 2)
        elif UH.shape[1] > 1:
            w = torch.flip(UH, [2])
            y = x_sample * w
            y = y.sum(2)
            if bias is not None:
                y = y + bias
            y = y.unsqueeze(3)

        return y

    def lateral_flow_temperature(
            self, srflow, ssflow, gwflow, ave_air_temp, args, lat_temp_adj, NEARZERO=1e-6
    ):
        """
        :param srflow: surface runoff
        :param ssflow: subsurface runoff
        :param gwflow: qroundwaterflow
        :param res_time_srflow: residense time for surface runoff
        :param res_time_ssflow: residence time for subsurface runoff
        :param res_time_gwflow: residence time for groundwater flow
        :return: temperature of lateral flow
        """
        # with torch.no_grad():
        if args["res_time_params"]["type"] == "SNTEMP":
            mask_ave_air_temp = ave_air_temp.ge(0)
            ave_air_temp = ave_air_temp * mask_ave_air_temp.int().float()

            srflow_temp = ave_air_temp[:, :, :, 0]  # .clone().detach()
            ssflow_temp = ave_air_temp[:, :, :, 1]  # .clone().detach()
            gwflow_temp = ave_air_temp[:, :, :, 2]  # .clone().detach()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        elif args["res_time_params"]["type"] == "van Vliet":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0] - 1.5
            mask_srflow_temp = srflow_temp.ge(0.0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 1]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(5.0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )
        #
        # elif args["res_time_params"]["type"] is "Meisner":
        elif args["res_time_params"]["type"] == "Meisner":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0]
            mask_srflow_temp = srflow_temp.ge(0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 1]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        denom = gwflow + ssflow + srflow
        mask_denom = denom.eq(0.0)
        denom = denom + mask_denom.int().float()

        if args["lat_temp_adj"] == True:
            gwflow_temp = gwflow_temp + lat_temp_adj

        T_l = (
                      (gwflow * gwflow_temp) + (srflow * srflow_temp) + (ssflow * ssflow_temp)
              ) / denom

        mask_less_zero = T_l.le(NEARZERO)
        T_l[mask_less_zero] = 0.0
        return T_l, srflow_temp, ssflow_temp, gwflow_temp, lat_flow_temp

    def solving_SNTEMP_ODE_second_order(
            self,
            K1,
            K2,
            T_e,
            ave_width,
            q_l,
            L,
            args,
            T_0=make_tensor(0),
            Q_0=make_tensor(0.01),
            NEARZERO=1e-10,
    ):
        # # Note: as we assume that Q_0 is 0.01, we are always gaining flow with positive lateral flow or
        # # with zero lateral flow
        density = args["params"]["water_density"]
        c_w = args["params"]["C_w"]
        mask_q_l = q_l.eq(0)
        q_l_pos = q_l + mask_q_l.int().float()
        b = q_l + ((K1 * ave_width) / (density * c_w))
        mask_Q_0 = Q_0.eq(0)
        Q_0_pos = Q_0 + mask_Q_0.int().float()
        R_0 = torch.exp(-(b * L) / Q_0_pos)
        mask_K1 = K1.eq(0)
        K1_masked = K1 + mask_K1.int().float()
        denom = 1 + ((K2 / K1_masked) * (T_e - T_0) * (1 - R_0))
        mask_denom = denom.eq(0)
        denom_masked = denom + mask_denom.int().float()
        Tw = T_e - ((T_e - T_0) * R_0 / denom_masked)
        return Tw

    def parameter_bounds(self, params, num, args):
        if params.dim() == 3:
            if num in args["static_params_list"]:
                out_temp = (
                        params[:, -1, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
                out = (
                    out_temp.view(-1, 1)
                    .repeat(1, params.shape[1])
                    .view(params.shape[0], params.shape[1])
                )
            elif num in args["semi_static_params_list"]:
                out_temp = self.semi_static_params(
                    params,
                    num,
                    interval=args["interval_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                    method=args["method_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                )
                out = (
                        out_temp
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
            else:  # dynamic
                out = (
                        params[:, :, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
        elif params.dim() == 2:
            out_temp = (
                    params[:, num]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = (
                out_temp.view(-1, 1)
                .repeat(1, args["hyperparameters"]["rho"])
                .view(
                    args["hyperparameters"]["batch_size"],
                    args["hyperparameters"]["rho"],
                )
            )
        return out

    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
                method=args["method_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
            )
            out = (
                    out_temp * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
        return out

    def shade_modification(self, w1_shade, w2_shade, w3_shade, args):

        w1 = torch.ones((365, 1, 40), device=args["device"]) / 40

        w1_shade = w1_shade.unsqueeze(-1).repeat((1, 1, 365)).permute(0, 2, 1)
        B = F.conv1d(w1_shade, w1, padding=40, groups=365)
        w1_shade_mov = B[:, 0, 20:385]

        w2_shade = w2_shade.unsqueeze(-1).repeat((1, 1, 365)).permute(0, 2, 1)
        B = F.conv1d(w2_shade, w1, padding=40, groups=365)
        w2_shade_mov = B[:, 0, 20:385]

        w3_shade = w3_shade.unsqueeze(-1).repeat((1, 1, 365)).permute(0, 2, 1)
        B = F.conv1d(w3_shade, w1, padding=40, groups=365)
        no_shade_mov = B[:, 0, 20:385]  # [:, 0, 5:370]

        shade_fraction_riparian = w1_shade_mov
        # shade_fraction_riparian = w1_shade_mov / (w1_shade_mov + w2_shade_mov + no_shade_mov)  # + no_shade_mov
        shade_fraction_riparian = torch.clamp(
            shade_fraction_riparian, min=0.01, max=1.0
        )

        # shade_fraction_topo = w2_shade_mov / (w1_shade_mov + w2_shade_mov + no_shade_mov)
        shade_fraction_topo = (
                (1 - shade_fraction_riparian) * w2_shade_mov / (w2_shade_mov + no_shade_mov)
        )  # + no_shade_mov
        shade_fraction_topo = torch.clamp(shade_fraction_topo, min=0.01, max=1.0)
        shade_total = shade_fraction_riparian + shade_fraction_topo
        shade_total = torch.clamp(shade_total, min=0.01, max=1.0)

        return shade_fraction_riparian, shade_fraction_topo, shade_total

    def frac_modification(
            self, srflow_portion, ssflow_portion, gwflow_portion, Q_T, args
    ):
        nmul = args["nmul"]
        A = list()
        Q = gwflow_portion * Q_T
        for i in range(nmul):
            Q_gw = Q[:, :, i]
            gw_filter_size = args["frac_smoothening"]["gw_filter_size"]
            wgw = (
                    torch.ones(
                        (gwflow_portion.shape[1], 1, gw_filter_size), device=args["device"]
                    )
                    / gw_filter_size
            )

            Q_gw_por = (
                Q_gw.unsqueeze(-1)
                .repeat((1, 1, gwflow_portion.shape[1]))
                .permute(0, 2, 1)
            )
            B = F.conv1d(
                Q_gw_por, wgw, padding=gw_filter_size, groups=gwflow_portion.shape[1]
            )
            Q_gw_por_mov = B[
                           :,
                           0,
                           math.floor(gw_filter_size / 2): math.floor(gw_filter_size / 2)
                                                           + gwflow_portion.shape[1],
                           ]
            # Q_gw_por_mov = torch.clamp(Q_gw_por_mov, min=0.0, max=Q_T)
            Q_gw_por_mov = torch.max(
                torch.min(Q_gw_por_mov, Q_T[:, :, 0]), make_tensor(0.0)
            )
            gwflow_portion_new = Q_gw_por_mov / (
                    Q_T[:, :, 0] + 0.001
            )  # 0.001 is for not having nan values
            gwflow_portion_new = torch.clamp(gwflow_portion_new, min=0.01, max=1.0)
            A.append(gwflow_portion_new.unsqueeze(-1))
        gwflow_portion_new = torch.cat(A, dim=2)
        remain_frac = 1 - gwflow_portion_new

        if args["res_time_params"]["type"] != "Meisner":
            srflow_portion_new = (
                    srflow_portion * remain_frac / (srflow_portion + ssflow_portion + 0.001)
            )
            ssflow_portion_new = (
                    ssflow_portion * remain_frac / (srflow_portion + ssflow_portion + 0.001)
            )
        else:
            srflow_portion_new = remain_frac
            ssflow_portion_new = ssflow_portion * 0.0 + 0.01
        srflow_percentage = torch.clamp(srflow_portion_new, min=0.01, max=1.0)
        ssflow_percentage = torch.clamp(ssflow_portion_new, min=0.01, max=1.0)
        gwflow_percentage = torch.clamp(gwflow_portion_new, min=0.01, max=1.0)

        return srflow_percentage, ssflow_percentage, gwflow_percentage

    def semi_static_params(self, params, param_no, interval=30, method="average"):
        # seperate the piece for each interval
        param = params[:, :, param_no]
        no_basins, no_days = param.shape
        interval_no = math.floor(no_days / interval)
        remainder = no_days % interval
        param_name_list = list()
        if method == "average":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = torch.mean(param[:, 0:remainder], 1, keepdim=True).repeat(
                        (1, remainder)
                    )
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = torch.mean(
                    param[
                    :,
                    ((i * interval) + remainder): (
                            ((i + 1) * interval) + remainder
                    ),
                    ],
                    1,
                    keepdim=True,
                ).repeat((1, interval))
                param_name_list.append(param_name)
        elif method == "single_val":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = (param[:, 0:1]).repeat((1, remainder))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = (
                    param[
                    :,
                    (((i) * interval) + remainder): (((i) * interval) + remainder)
                                                    + 1,
                    ]
                ).repeat((1, interval))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

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

    def forward(
            self, x, params, iGrid, iT, args, air_sample_sr, air_sample_ss, air_sample_gw
    ):
        # restricting the params
        NEARZERO = args["NEARZERO"]

        # for all a and b
        a_srflow = self.multi_comp_parameter_bounds(params, 0, args)
        b_srflow = self.multi_comp_parameter_bounds(params, 1, args)
        a_ssflow = self.multi_comp_parameter_bounds(params, 2, args)
        b_ssflow = self.multi_comp_parameter_bounds(params, 3, args)
        a_gwflow = self.multi_comp_parameter_bounds(params, 4, args)
        b_gwflow = self.multi_comp_parameter_bounds(params, 5, args)
        w1_shade = self.multi_comp_parameter_bounds(params, 6, args)
        srflow_portion = self.multi_comp_parameter_bounds(params, 7, args)
        ssflow_portion = self.multi_comp_parameter_bounds(params, 8, args)
        gwflow_portion = self.multi_comp_parameter_bounds(params, 9, args)
        w2_shade = self.multi_comp_parameter_bounds(params, 10, args)
        width_coef_nom = self.multi_comp_parameter_bounds(params, 11, args)
        width_coef_denom = self.multi_comp_parameter_bounds(params, 12, args)
        hamon_coef = self.multi_comp_parameter_bounds(params, 13, args)
        w3_shade = self.multi_comp_parameter_bounds(params, 14, args)
        lat_temp_adj = self.multi_comp_parameter_bounds(params, 15, args)

        nmul = args["nmul"]
        vars = args["optData"]["varT"] + args["optData"]["varC"]
        with torch.no_grad():
            obsQ = (
                    x[:, :, vars.index("00060_Mean")].unsqueeze(-1).repeat(1, 1, nmul)
                    * 0.028316
            )  # converting cfs to cms
            precip = (
                x[:, :, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
            )
            up_inflow = make_tensor(torch.zeros(obsQ.size()))
            mean_air_temp = (
                ((x[:, :, vars.index("tmax(C)")] + x[:, :, vars.index("tmin(C)")]) / 2)
                .unsqueeze(-1)
                .repeat(1, 1, nmul)
            )
            dayl = x[:, :, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
            vp = 0.01 * x[:, :, vars.index("vp(Pa)")].unsqueeze(-1).repeat(
                1, 1, nmul
            )  # converting to mbar
            swrad = (
                (
                        x[:, :, vars.index("srad(W/m2)")]
                        * x[:, :, vars.index("dayl(s)")]
                        / 86400
                )
                .unsqueeze(-1)
                .repeat(1, 1, nmul)
            )
            elev = (
                x[:, :, vars.index("ELEV_MEAN_M_BASIN")]
                .unsqueeze(-1)
                .repeat(1, 1, nmul)
            )
            slope = 0.01 * x[:, :, vars.index("SLOPE_PCT")].unsqueeze(-1).repeat(
                1, 1, nmul
            )  # adding the percentage, it is a watershed slope not a stream slope
            # stream_density = x[:, :, vars.index("STREAMS_KM_SQ_KM")]
            # stream_length = 1000 * (stream_density * x[:, :, vars.index("DRAIN_SQKM")]).unsqueeze(-1).repeat(1,1,nmul)
            # stream_length = x[:, :, vars.index("stream_length_artificial")]
            # stream_length = x[:, :, vars.index("NHDlength_tot(m)")].unsqueeze(-1).repeat(1,1,nmul)
            stream_length = (
                x[:, :, vars.index("stream_length_artificial")]
                .unsqueeze(-1)
                .repeat(1, 1, nmul)
            )
            # basin_area = x[:, :, vars.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1,1,nmul)
        cloud_fraction = x[:, :, vars.index("ccov")].unsqueeze(-1).repeat(1, 1, nmul)
        albedo = args["STemp_default_params"]["albedo"]
        # top_width = make_tensor(np.full((x.shape[0], x.shape[1]), 10), has_grad=False)

        # PET = make_tensor(np.full((x.shape[0], x.shape[1]), 0.010 / 86400), has_grad=False)
        # hamon PET equation. We can add other methods too, such as Penman monteith
        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        )

        # d = torch.pow(q * n * (q + 1) / (p * torch.pow(slope, 0.5)), (3 / (5 + 3 * q)))

        # d = torch.pow(width_coef_nom * obsQ / (width_coef_denom * torch.pow(basin_area, width_A_coef)), width_exp)
        # top_width = 5 + p * torch.pow(d, q)
        # top_width = width_coef_nom * obsQ + width_coef_denom
        # top_width = width_coef_nom * (obsQ ** width_coef_denom) + 0.5
        top_width = (
                torch.abs(width_coef_nom)
                * torch.pow(torch.abs(obsQ + 0.0001), torch.abs(width_coef_denom))
                + 0.5
        )
        # top_width = make_tensor(torch.ones(width_coef_nom.shape) * 10.0, device=args["device"])
        # if p.dim() == 3:
        #     top_width = p * torch.pow(basin_area, q)
        # elif p.dim() == 2:
        #     top_width = p * torch.pow(basin_area, q)

        # masking surface runoff fraction with precipitation.
        # if there is not any precipitaton, it cannot be more than 0.01
        if args["make_sr_masked_by_precip"]:
            mask_precip = precip.ge(NEARZERO)
            srflow_portion = srflow_portion * mask_precip.int().float()
            srflow_portion = torch.clamp(srflow_portion, min=0.00, max=1.0)
        # gw fractions smoothening
        if args["frac_smoothening_mode"] == True:
            (
                srflow_percentage,
                ssflow_percentage,
                gwflow_percentage,
            ) = self.frac_modification(
                srflow_portion, ssflow_portion, gwflow_portion, obsQ, args
            )
        else:
            if args["res_time_params"]["type"] != "Meisner":
                gwflow_percentage = gwflow_portion
                srflow_percentage = ((1 - gwflow_portion) * srflow_portion) / (
                        srflow_portion + ssflow_portion
                )
                ssflow_percentage = ((1 - gwflow_portion) * ssflow_portion) / (
                        srflow_portion + ssflow_portion
                )
                # srflow_percentage = srflow_portion / (srflow_portion + ssflow_portion + gwflow_portion)
                # ssflow_percentage = ssflow_portion / (srflow_portion + ssflow_portion + gwflow_portion)
                # gwflow_percentage = gwflow_portion / (srflow_portion + ssflow_portion + gwflow_portion)
            else:
                gwflow_percentage = gwflow_portion
                srflow_percentage = 1 - gwflow_portion
                ssflow_percentage = 0.0001 * ssflow_portion
                # srflow_percentage = srflow_portion / (srflow_portion + gwflow_portion)
                # ssflow_percentage = 0.0001 * ssflow_portion
                # gwflow_percentage = gwflow_portion / (srflow_portion + gwflow_portion)

        # total shade (solar shade) is accumulative shade of vegetation and topography
        if args["shade_smoothening"] == True:
            (
                shade_fraction_riparian,
                shade_fraction_topo,
                shade_total,
            ) = self.shade_modification(w1_shade, w2_shade, w3_shade, args)
        else:
            shade_fraction_riparian = w1_shade
            shade_fraction_topo = (
                    (1 - shade_fraction_riparian) * w2_shade / (w2_shade + w3_shade)
            )
            shade_total = shade_fraction_riparian + shade_fraction_topo
            # shade_fraction_riparian = w1_shade / (w1_shade + w2_shade + w3_shade)
            # shade_fraction_topo = w2_shade / (w1_shade + w2_shade + w3_shade)
            # shade_total = shade_fraction_riparian + shade_fraction_topo

        srflow, ssflow, gwflow = self.srflow_ssflow_gwflow_portions(
            discharge=obsQ,
            srflow_factor=srflow_percentage,
            ssflow_factor=ssflow_percentage,
            gwlow_factor=gwflow_percentage,
        )

        w_srflow = self.res_time_gamma(
            a=a_srflow.unsqueeze(-1),
            b=b_srflow.unsqueeze(-1),
            lenF=args["res_time_params"]["lenF_srflow"],
        )

        # air_sample_sr = self.x_sample_air_temp(iGrid, iT, lenF=args['res_time_params']['lenF_srflow'],
        #                                        args=args, x_total_raw=x_total_raw,
        #                                        time_range=time_range)
        # air_sample_sr = self.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_srflow'],
        #                                         args=args, ave_air_total=ave_air_total)
        air_sample_sr = air_sample_sr.unsqueeze(-1).repeat(1, 1, 1, nmul)
        w_srflow = w_srflow.permute(1, 2, 0, 3)
        ave_air_sr = self.res_time_conv(
            air_sample_sr, w_srflow, bias=None
        )  # bias=None, sr_conv_bias

        # subsurface flow
        w_ssflow = self.res_time_gamma(
            a=a_ssflow.unsqueeze(-1),
            b=b_ssflow.unsqueeze(-1),
            lenF=args["res_time_params"]["lenF_ssflow"],
        )

        # air_sample_ss = self.x_sample_air_temp(iGrid, iT, lenF=args['res_time_params']['lenF_ssflow'],
        #                                        args=args, x_total_raw=x_total_raw,
        #                                        time_range=time_range)
        # air_sample_ss = self.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_ssflow'],
        #                                         args=args, ave_air_total=ave_air_total)
        air_sample_ss = air_sample_ss.unsqueeze(-1).repeat(1, 1, 1, nmul)
        w_ssflow = w_ssflow.permute(1, 2, 0, 3)
        ave_air_ss = self.res_time_conv(
            air_sample_ss, w_ssflow, bias=None
        )  # ss_conv_bias

        # groundwater flow
        w_gwflow = self.res_time_gamma(
            a=a_gwflow.unsqueeze(-1),
            b=b_gwflow.unsqueeze(-1),
            lenF=args["res_time_params"]["lenF_gwflow"],
        )

        # air_sample_gw = self.x_sample_air_temp(iGrid, iT, lenF=args['res_time_params']['lenF_gwflow'],
        #                                        args=args, x_total_raw=x_total_raw,
        #                                        time_range=time_range)
        # air_sample_gw = self.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_gwflow'],
        #                                         args=args, ave_air_total=ave_air_total)
        air_sample_gw = air_sample_gw.unsqueeze(-1).repeat(1, 1, 1, nmul)
        w_gwflow = w_gwflow.permute(1, 2, 0, 3)
        ave_air_gw = self.res_time_conv(
            air_sample_gw, w_gwflow, bias=None
        )  # gw_conv_bias

        ave_air_temp = torch.cat((ave_air_sr, ave_air_ss, ave_air_gw), dim=3)

        (
            T_0,
            srflow_temp,
            ssflow_temp,
            gwflow_temp,
            ave_air_temp_new,
        ) = self.lateral_flow_temperature(
            srflow=srflow,
            ssflow=ssflow,
            gwflow=gwflow,
            ave_air_temp=ave_air_temp,
            args=args,
            lat_temp_adj=lat_temp_adj,
        )

        # 'Correction factor to adjust the bias of the temperature of the lateral inflow'
        # Fortran code:
        # elseif(fs.le.NEARZERO)    (fs: surface runoff)
        # then
        # ! if this is true, then there is no flow from upstream, but there is lateral inflow
        # t_o = seg_tave_lat(i) + lat_temp_adj(i, Nowmonth)
        # if there is upstream flow, it should be weighted average temperature of all flows

        # if args['lat_temp_adj'] == "True":
        #     T_0 = T_l + lat_temp_adj
        # else:
        #     T_0 = T_l
        A, B, C, D = self.ABCD_equations(
            T_a=T_0,
            swrad=swrad,
            e_a=vp,
            elev=elev,
            slope=slope,
            top_width=top_width,
            up_inflow=0.0,
            E=PET,  # up_inflow
            T_g=gwflow_temp,
            iGrid=iGrid,
            shade_fraction_riparian=shade_fraction_riparian,
            albedo=albedo,
            shade_total=shade_total,
            args=args,
            cloud_fraction=cloud_fraction,
        )
        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D, T_e=T_0)
        K1, K2 = self.finding_K1_K2(
            A=A, B=B, C=C, D=D, T_e=T_e, NEARZERO=NEARZERO, T_0=T_0
        )

        Q_0 = make_tensor(np.full((obsQ.shape), 0.000001))
        # Q_0 = make_tensor(np.full((obsQ.shape[0], obsQ.shape[1]), 0))

        # T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, T_l, T_e, ave_width=top_width,
        #                                            q_l=obsQ, L=stream_length, args=args,
        #                                            T_0=T_0, Q_0=Q_0)

        # writing the original fortran code here
        # they assumed if Q_upstream==0 and q_lat > 0, they assume Q_upstream=q_lat, and q_lat=0
        # it prevents from dividing to zero
        T_w = self.solving_SNTEMP_ODE_second_order(
            K1,
            K2,
            T_e,
            ave_width=top_width,
            q_l=Q_0,
            L=stream_length,
            args=args,
            T_0=T_0,
            Q_0=obsQ,
        )

        if args["lat_temp_adj"] == True:
            return (
                torch.mean(T_w, -1).squeeze(),
                torch.mean(ave_air_temp_new, 2).squeeze(),
                torch.mean(gwflow_percentage, -1).squeeze(),
                torch.mean(ssflow_percentage, -1).squeeze(),
                torch.mean(w_gwflow, -1).squeeze(),
                torch.mean(w_ssflow, -1).squeeze(),
                torch.mean(PET, -1).squeeze(),
                torch.mean(shade_fraction_riparian, -1).squeeze(),
                torch.mean(shade_fraction_topo, -1).squeeze(),
                torch.mean(top_width, -1).squeeze(),
                torch.mean(cloud_fraction, -1).squeeze(),
                torch.mean(hamon_coef, -1).squeeze(),
                torch.mean(lat_temp_adj, -1).squeeze(),
            )
        else:
            return (
                torch.mean(T_w, -1).squeeze(),
                torch.mean(ave_air_temp_new, 2).squeeze(),
                torch.mean(gwflow_percentage, -1).squeeze(),
                torch.mean(ssflow_percentage, -1).squeeze(),
                torch.mean(w_gwflow, -1).squeeze(),
                torch.mean(w_ssflow, -1).squeeze(),
                torch.mean(PET, -1).squeeze(),
                torch.mean(shade_fraction_riparian, -1).squeeze(),
                torch.mean(shade_fraction_topo, -1).squeeze(),
                torch.mean(top_width, -1).squeeze(),
                torch.mean(cloud_fraction, -1).squeeze(),
                torch.mean(hamon_coef, -1).squeeze(),
                torch.mean(
                    torch.zeros(hamon_coef.shape, device=args["device"]), -1
                ).squeeze(),
            )


# this class needs a hydrology model as a backbone to provide flow of different sources
class SNTEMP_EQ(nn.Module):
    def __init__(self):
        super(SNTEMP_EQ, self).__init__()

    def atm_pressure(self, elev):
        ## from Jake's document
        # mmHg2mb = make_tensor(0.75061683)  # Unit conversion
        # mmHg2inHg = make_tensor(25.3970886)  # Unit conversion
        # P_sea = make_tensor(29.92126)  # Standard pressure ar sea level
        # A_g = make_tensor(9.80665)  # Acceleration due to gravity
        # M_a = make_tensor(0.0289644)  # Molar mass of air
        # R = make_tensor(8.31447)  # universal gas constant
        # T_sea = make_tensor(288.16)  # the standard temperature at sea level
        # P = (1 / mmHg2mb) * (mmHg2inHg) * (P_sea) * torch.exp(-A_g * M_a * elev / (R * T_sea))

        ## the code from stream_temp.f90
        P = 1013 - (0.1055 * elev)

        ## Note: both jakes and stream_temp are close to each other (error less than 0.013 for 99 basins)
        return P

    def atm_longwave_radiation_heat(self, T_a, e_a, shade_total, cloud_fraction, args):
        """
        :param T_a: air temperature in degree Celsius
        :param e_a: vapor pressure
        :return: Atmospheric longwave radiation
        """
        H_a = (
                (3.354939e-8 + 2.74995e-9 * e_a ** 0.5)
                * (1 - shade_total)
                * (1 + 0.17 * cloud_fraction ** 2)
                * (T_a + 273.16) ** 4
        )
        return H_a

    def stream_friction_heat(self, top_width, slope, Q):
        H_f = (
                9805 * Q * slope / top_width
        )  # Q is the seg_inflow (total flow entering a segment)
        return H_f

    def shortwave_solar_radiation_heat(self, albedo, H_sw, shade_total):
        """
        :param albedo: albedo or fraction reflected by stream , dimensionless
        :param H_sw: the clear sky solar radiation in watt per sq meter (seginc_swrad)
        :return: daily average clear sky, shortwave solar radiation for each segment
        """
        # solar_shade_fraction = make_tensor(args['STemp_default_params']['shade_fraction'])
        H_s = (1 - albedo) * (1 - shade_total) * H_sw
        return H_s

    def riparian_veg_longwave_radiation_heat(
            self, T_a, iGrid, shade_fraction_riparian, args
    ):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(
            make_tensor(10), (-8.0)
        )  # (J/s*m^2 * K^4)
        emissivity_veg = make_tensor(args["STemp_default_emissivity_veg"])
        H_v = (
                emissivity_veg
                * St_Boltzman_ct
                * shade_fraction_riparian
                * torch.pow((T_a + 273.16), 4)
        )
        # H_v = emissivity_veg * St_Boltzman_ct * shade2[iGrid, :] * torch.pow((T_a + 273.16), 4)
        return H_v

    def ABCD_equations(
            self,
            T_a,
            swrad,
            e_a,
            E,
            elev,
            slope,
            top_width,
            up_inflow,
            T_g,
            iGrid,
            shade_fraction_riparian,
            albedo,
            shade_total,
            args,
            cloud_fraction,
    ):
        """

        :param T_a: average daily air temperature
        :param swrad: solar radiation
        :param e_a: vapor pressure
        :param E: Free-water surface-evaporation rate (assumed to be PET, potet in PRMS)
        :param elev: average basin elevation
        :param slope: average stream slope (seg_slope)
        :param top_width: average top width of the stream
        :param up_inflow: is the discharge (variable seg_inflow) which is from upstream
        :return:
        """
        e_s = 6.108 * torch.exp((17.26939 * T_a) / (237.3 + T_a))
        # e_s = 6.108 * torch.exp((17.26939 * T_0) / (237.3 + T_0))
        P = self.atm_pressure(
            elev
        )  # calculating atmosphere pressure based on elevation
        # chacking vapor pressure with saturation vapor pressure
        denom = e_s - e_a
        mask_denom = denom.ge(0)
        # converting negative values to zero
        denom1 = denom * mask_denom.int().float()
        # adding 0.01 to zero values as it is denominator
        mask_denom2 = denom1.eq(0)
        denom2 = denom1 + 0.01 * mask_denom2.int().float()

        B_c = 0.00061 * P / denom2
        B_c1 = 0.00061 * P / (e_s - e_a)
        K_g = make_tensor(1.65)
        delta_Z = make_tensor(args["STemp_default_delta_Z"])
        # we don't need H_a, because we hae swrad directly from inputs
        H_a = self.atm_longwave_radiation_heat(
            T_a, e_a, shade_total, cloud_fraction, args=args
        )
        ###############
        H_f = self.stream_friction_heat(top_width=top_width, slope=slope, Q=up_inflow)
        H_s = self.shortwave_solar_radiation_heat(
            albedo=albedo, H_sw=swrad, shade_total=shade_total
        )  # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(
            T_a, iGrid, shade_fraction_riparian, args=args
        )

        A = 5.4 * torch.pow(make_tensor(np.full((T_a.shape), 10)), (-8))
        B = torch.pow(make_tensor(10), 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (
                K_g / delta_Z
        )
        C = torch.pow(make_tensor(10), 6) * E * B_c * 2.36
        # Todo: need to check 10**6. it is in fortran code but it is not in the document
        D = (
                H_f
                + H_a
                + H_s
                + H_v
                + 2495 * torch.pow(make_tensor(10), 6) * E * ((B_c * T_a) - 1)
                + (T_g * K_g / delta_Z)
        )
        # D = H_a + swrad + H_v + 2495 * E * ((B_c * T_a) - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e, iter=50):
        def F(T_e):
            return (
                    A * torch.pow((T_e + 273.16), 4) - C * torch.pow(T_e, 2) + B * T_e - D
            )

        def Fprime(T_e):
            return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B

        ## solving the equation with Newton's method
        for i in range(iter):
            next_geuss = T_e - (F(T_e) / Fprime(T_e))
            T_e = next_geuss

        return T_e

    def finding_K1_K2(self, A, B, C, D, T_e, NEARZERO, T_0):
        """
        :param A: Constant coming from equilibrium temp equation
        :param B: Constant coming from equilibrium temp equation
        :param C: Constant coming from equilibrium temp equation
        :param T_e: equilibrium temperature
        :param H_i: initial net heat flux at temperature T_o, of the upstream inflow
        :param T_o: initial water temperature
        :return: K1 (first order thermal exchange coefficient), K2 (second order coefficient)
        """
        H_i = A * torch.pow((T_0 + 273.16), 4) - C * torch.pow(T_0, 2) + B * T_0 - D
        K1 = 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B
        denom = torch.pow((T_e - T_0), 2)
        mask_denom = denom.le(NEARZERO)
        denom1 = denom + mask_denom.int().float()
        # K2 = (H_i - (K1 * (T_e - T_0))) / denom1
        K2 = (-H_i + (K1 * (T_0 - T_e))) / denom1

        return K1, K2

    def srflow_ssflow_gwflow_portions(
            self,
            discharge,
            srflow_factor=make_tensor(0.40),
            ssflow_factor=make_tensor(0.3),
            gwlow_factor=make_tensor(0.3),
    ):
        srflow = srflow_factor * discharge
        ssflow = ssflow_factor * discharge
        gwflow = gwlow_factor * discharge
        return srflow, ssflow, gwflow

    def ave_temp_res_time(
            self, ave_air_temp, x, res_time, iGrid, iT, args, x_total_raw
    ):
        rho = x.shape[1]  # args['hyperparameters']['rho']
        tArray_Total = tRange2Array(args["tRange"])
        tArray_train = tRange2Array(args["t_train"])
        _, ind1, _ = intersect(tArray_Total, tArray_train)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        temp_res = res_time
        with torch.no_grad():
            temp_res1 = temp_res.int()
        A = res_time.repeat(1, rho)
        B = torch.reshape(A, (res_time.shape[0], rho, res_time.shape[1]))
        ave_air = torch.zeros(
            (
                args["batch_size"],
                args["rho"],
                res_time.shape[1],
            ),
            device=args["device"],
        )
        for i in range(res_time.shape[1]):
            for s, station in enumerate(iGrid):
                array = np.zeros((x.shape[1], temp_res1[s, i].item()), dtype=np.int32)
                for j in range(temp_res1[s, i].item()):
                    array[:, j] = np.arange(
                        (ind1_tensor[0] + iT_tensor[s] - j).item(),
                        (ind1_tensor[0] + iT_tensor[s] - j + x.shape[1]).item(),
                    )
                tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
                max_add = torch.sum(tmax_temp, dim=1)
                tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
                min_add = torch.sum(tmin_temp, dim=1)
                ave_air[s, :, i] = (max_add + min_add) / 2  # (2 * res_time[station, i])
        ave_air_temp = ave_air / B
        # return ave_air
        return ave_air_temp

    def x_sample_air_temp(self, iGrid, iT, lenF, args, x_total_raw, time_range):
        """
        :param iGrid:
        :param iT:
        :param lenF: maximum number of days that it is needed to be considered in average
        :return:
        """
        rho = args["rho"]
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        _, ind1, _ = intersect(tArray_Total, tArray_sample)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        ave_air = torch.zeros(
            (
                args["batch_size"],
                args["rho"],
                lenF,
            ),
            device=args["device"],
        )
        for s, station in enumerate(iGrid):
            array = np.zeros((rho, lenF), dtype=np.int32)
            for j in range(lenF):
                array[:, j] = np.arange(
                    (ind1_tensor[0] + iT_tensor[s] - j).item(),
                    (ind1_tensor[0] + iT_tensor[s] - j + rho).item(),
                )
            # array = np.flip(array, 1).copy()
            tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp
        return ave_air

    def x_sample_air_temp2(self, iGrid, iT, lenF, args, ave_air_total):
        rho = args["rho"]
        # ave_air = torch.zeros((len(iGrid), args["hyperparameters"]["rho"],
        #                        lenF),
        #                       device=args["device"])
        a = min(ave_air_total.shape[1], args["rho"])
        ave_air = torch.zeros((len(iGrid), a, lenF), device=args["device"])
        # array = np.array([np.arange(x, y) for x, y in zip(iT, iT + rho)])
        ave_air_temp = ave_air_total[iGrid, :, 0:lenF]
        for i in range(len(iGrid)):
            ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + a), :]
            # ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + ave_air_total.shape[1]), :]

        # ave_air_temp = ave_air_total[:, iT : iT + rho, 0:lenF]
        # ave_air = ave_air_temp[iGrid, :, :]
        return ave_air

    def ave_temp_general(self, args, x_total_raw_tensor, time_range):
        vars = args["varT_NN"] + args["varC_NN"]
        lenF_max = np.maximum(
            args["res_time_lenF_srflow"],
            np.maximum(
                args["res_time_lenF_ssflow"],
                args["res_time_lenF_gwflow"],
            ),
        )
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        c, ind1, ind2 = np.intersect1d(tArray_sample, tArray_Total, return_indices=True)
        ave_air = torch.zeros(
            (x_total_raw_tensor.shape[0], len(tArray_sample), lenF_max), device="cpu"
        )

        array = np.zeros((len(ind2), lenF_max), dtype=np.int32)
        for j in range(lenF_max):
            array[:, j] = np.arange(
                (ind2[0] - j).item(), (ind2[0] - j + len(ind2)).item()
            )

        for s in range(x_total_raw_tensor.shape[0]):
            tmax_temp = x_total_raw_tensor[s, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw_tensor[s, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp

        return ave_air

    def res_time_gamma(self, a, b, lenF):
        # UH. a [time (same all time steps), batch, var]
        # a = torch.abs(a)
        if a.dim() == 2:
            m = a.shape
            a1 = a.repeat(1, lenF)
            b1 = b.repeat(1, lenF)
            alpha = F.relu(a1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.1
            beta = F.relu(b1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 3:
            m = a.shape
            a1 = a.repeat(1, 1, lenF)
            b1 = b.repeat(1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.1
            beta = F.relu(b1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 4:
            m = a.shape
            a1 = a.repeat(1, 1, 1, lenF)
            b1 = b.repeat(1, 1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.1
            beta = F.relu(b1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.5
            x = (
                torch.linspace(0.001, 20, lenF)
                .view(lenF, 1, 1, 1)
                .repeat(1, m[0], m[1], m[2])
            )
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        return www

    def UH_gamma(self, a, b, lenF=10):
        # UH. a [time (same all time steps), batch, var]
        m = a.shape
        lenF = min(a.shape[0], lenF)
        w = torch.zeros([lenF, m[1], m[2]])
        aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
        theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
        # t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
        # t = t.cuda(aa.device)
        t = (torch.linspace(0.001, 20, lenF).view(lenF, 1, 1).repeat(1, m[1], 1))
        t = t.to(aa.device)
        denom = (aa.lgamma().exp()) * (theta ** aa)
        mid = t ** (aa - 1)
        right = torch.exp(-t / theta)
        w = 1 / denom * mid * right
        ww = torch.cumsum(w, dim=0)
        www = ww / ww.sum(0)
        # w = w / w.sum(0)  # scale to 1 for each UH
        return www

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

        # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
        y = F.conv1d(xx, w, groups=groups, padding=padd, stride=1, bias=None)  # we don't need flip
        if padd != 0:
            y = y[:, :, 0:-padd]
        return y.view(mm)

    def res_time_conv(self, x_sample, UH, bias, viewmode=1):
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
        # batch needs to be accommodated by channels and we make use of gr
        # ++++---------------------------------+
        #
        # oups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html
        if UH.shape[1] == 1:
            x = x_sample[:, 0:1, :]
            mm = x.shape
            nb = mm[0]
            m = UH.shape[-1]
            padd = m - 1
            if viewmode == 1:
                xx = x.view([1, nb, mm[-1]])
                w = UH.view([nb, 1, m])
                groups = nb

                # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
                # y = y[:, :, 0:-padd]

            x_sample1 = x_sample.permute(1, 0, 2)
            a = torch.arange(x_sample.shape[1])
            y = F.conv1d(
                x_sample1[a],
                torch.flip(w, [2]),
                groups=groups,
                padding=0,
                stride=1,
                bias=bias,
            )
            y = y.permute(1, 0, 2)
        elif UH.shape[1] > 1:
            w = torch.flip(UH, [2])
            y = x_sample * w
            y = y.sum(2)
            if bias is not None:
                y = y + bias
            y = y.unsqueeze(3)

        return y

    def lateral_flow_temperature(
            self, srflow, ssflow, gwflow, ave_air_temp, args, lat_temp_adj, NEARZERO=1e-6
    ):
        """
        :param srflow: surface runoff
        :param ssflow: subsurface runoff
        :param gwflow: qroundwaterflow
        :param res_time_srflow: residense time for surface runoff
        :param res_time_ssflow: residence time for subsurface runoff
        :param res_time_gwflow: residence time for groundwater flow
        :return: temperature of lateral flow
        """
        # with torch.no_grad():
        if args["res_time_type"] == "SNTEMP":
            mask_ave_air_temp = ave_air_temp.ge(0)
            ave_air_temp = ave_air_temp * mask_ave_air_temp.int().float()

            srflow_temp = ave_air_temp[:, :, :, 0]  # .clone().detach()
            ssflow_temp = ave_air_temp[:, :, :, 1]  # .clone().detach()
            gwflow_temp = ave_air_temp[:, :, :, 2]  # .clone().detach()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        elif args["res_time_type"] == "van Vliet":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0] - 1.5
            mask_srflow_temp = srflow_temp.ge(0.0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 1]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(5.0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )
        #
        # elif args["res_time_params"]["type"] is "Meisner":
        elif args["res_time_type"] == "Meisner":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0]
            mask_srflow_temp = srflow_temp.ge(0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 1]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        denom = gwflow + ssflow + srflow
        mask_denom = denom.eq(0.0)
        denom = denom + mask_denom.int().float()

        if args["lat_temp_adj"] == True:
            gwflow_temp = gwflow_temp + lat_temp_adj

        T_l = (
                      (gwflow * gwflow_temp) + (srflow * srflow_temp) + (ssflow * ssflow_temp)
              ) / denom

        mask_less_zero = T_l.le(NEARZERO)
        T_l[mask_less_zero] = 0.0
        return T_l, srflow_temp, ssflow_temp, gwflow_temp, lat_flow_temp

    def solving_SNTEMP_ODE_second_order(
            self,
            K1,
            K2,
            T_e,
            ave_width,
            q_l,
            L,
            args,
            T_0=make_tensor(0),
            Q_0=make_tensor(0.01),
            NEARZERO=1e-10,
    ):
        # # Note: as we assume that Q_0 is 0.01, we are always gaining flow with positive lateral flow or
        # # with zero lateral flow
        density = args["params_water_density"]
        c_w = args["params_C_w"]
        mask_q_l = q_l.eq(0)
        q_l_pos = q_l + mask_q_l.int().float()
        b = q_l + ((K1 * ave_width) / (density * c_w))
        mask_Q_0 = Q_0.eq(0)
        Q_0_pos = Q_0 + mask_Q_0.int().float()
        R_0 = torch.exp(-(b * L) / Q_0_pos)
        mask_K1 = K1.eq(0)
        K1_masked = K1 + mask_K1.int().float()
        denom = 1 + ((K2 / K1_masked) * (T_e - T_0) * (1 - R_0))
        mask_denom = denom.eq(0)
        denom_masked = denom + mask_denom.int().float()
        Tw = T_e - ((T_e - T_0) * R_0 / denom_masked)
        return Tw

    def parameter_bounds(self, params, num, args):
        if params.dim() == 3:
            if num in args["static_params_list"]:
                out_temp = (
                        params[:, -1, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
                out = (
                    out_temp.view(-1, 1)
                    .repeat(1, params.shape[1])
                    .view(params.shape[0], params.shape[1])
                )
            elif num in args["semi_static_params_list"]:
                out_temp = self.semi_static_params(
                    params,
                    num,
                    interval=args["interval_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                    method=args["method_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                )
                out = (
                        out_temp
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
            else:  # dynamic
                out = (
                        params[:, :, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
        elif params.dim() == 2:
            out_temp = (
                    params[:, num]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = (
                out_temp.view(-1, 1)
                .repeat(1, args["hyperparameters"]["rho"])
                .view(
                    args["hyperparameters"]["batch_size"],
                    args["hyperparameters"]["rho"],
                )
            )
        return out

    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
                method=args["method_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
            )
            out = (
                    out_temp * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
        return out

    def shade_modification(self, w1_shade, w2_shade, w3_shade, args):

        w1 = torch.ones((365, 1, 40), device=args["device"]) / 40

        w1_shade = w1_shade.unsqueeze(-1).repeat((1, 1, 365)).permute(0, 2, 1)
        B = F.conv1d(w1_shade, w1, padding=40, groups=365)
        w1_shade_mov = B[:, 0, 20:385]

        w2_shade = w2_shade.unsqueeze(-1).repeat((1, 1, 365)).permute(0, 2, 1)
        B = F.conv1d(w2_shade, w1, padding=40, groups=365)
        w2_shade_mov = B[:, 0, 20:385]

        w3_shade = w3_shade.unsqueeze(-1).repeat((1, 1, 365)).permute(0, 2, 1)
        B = F.conv1d(w3_shade, w1, padding=40, groups=365)
        no_shade_mov = B[:, 0, 20:385]  # [:, 0, 5:370]

        shade_fraction_riparian = w1_shade_mov
        # shade_fraction_riparian = w1_shade_mov / (w1_shade_mov + w2_shade_mov + no_shade_mov)  # + no_shade_mov
        shade_fraction_riparian = torch.clamp(
            shade_fraction_riparian, min=0.01, max=1.0
        )

        # shade_fraction_topo = w2_shade_mov / (w1_shade_mov + w2_shade_mov + no_shade_mov)
        shade_fraction_topo = (
                (1 - shade_fraction_riparian) * w2_shade_mov / (w2_shade_mov + no_shade_mov)
        )  # + no_shade_mov
        shade_fraction_topo = torch.clamp(shade_fraction_topo, min=0.01, max=1.0)
        shade_total = shade_fraction_riparian + shade_fraction_topo
        shade_total = torch.clamp(shade_total, min=0.01, max=1.0)

        return shade_fraction_riparian, shade_fraction_topo, shade_total

    def frac_modification(
            self, srflow_portion, ssflow_portion, gwflow_portion, Q_T, args
    ):
        nmul = args["nmul"]
        A = list()
        Q = gwflow_portion * Q_T
        for i in range(nmul):
            Q_gw = Q[:, :, i]
            gw_filter_size = args["frac_smoothening"]["gw_filter_size"]
            wgw = (
                    torch.ones(
                        (gwflow_portion.shape[1], 1, gw_filter_size), device=args["device"]
                    )
                    / gw_filter_size
            )

            Q_gw_por = (
                Q_gw.unsqueeze(-1)
                .repeat((1, 1, gwflow_portion.shape[1]))
                .permute(0, 2, 1)
            )
            B = F.conv1d(
                Q_gw_por, wgw, padding=gw_filter_size, groups=gwflow_portion.shape[1]
            )
            Q_gw_por_mov = B[
                           :,
                           0,
                           math.floor(gw_filter_size / 2): math.floor(gw_filter_size / 2)
                                                           + gwflow_portion.shape[1],
                           ]
            # Q_gw_por_mov = torch.clamp(Q_gw_por_mov, min=0.0, max=Q_T)
            Q_gw_por_mov = torch.max(
                torch.min(Q_gw_por_mov, Q_T[:, :, 0]), make_tensor(0.0)
            )
            gwflow_portion_new = Q_gw_por_mov / (
                    Q_T[:, :, 0] + 0.001
            )  # 0.001 is for not having nan values
            gwflow_portion_new = torch.clamp(gwflow_portion_new, min=0.01, max=1.0)
            A.append(gwflow_portion_new.unsqueeze(-1))
        gwflow_portion_new = torch.cat(A, dim=2)
        remain_frac = 1 - gwflow_portion_new

        if args["res_time_params"]["type"] != "Meisner":
            srflow_portion_new = (
                    srflow_portion * remain_frac / (srflow_portion + ssflow_portion + 0.001)
            )
            ssflow_portion_new = (
                    ssflow_portion * remain_frac / (srflow_portion + ssflow_portion + 0.001)
            )
        else:
            srflow_portion_new = remain_frac
            ssflow_portion_new = ssflow_portion * 0.0 + 0.01
        srflow_percentage = torch.clamp(srflow_portion_new, min=0.01, max=1.0)
        ssflow_percentage = torch.clamp(ssflow_portion_new, min=0.01, max=1.0)
        gwflow_percentage = torch.clamp(gwflow_portion_new, min=0.01, max=1.0)

        return srflow_percentage, ssflow_percentage, gwflow_percentage

    def semi_static_params(self, params, param_no, interval=30, method="average"):
        # seperate the piece for each interval
        param = params[:, :, param_no]
        no_basins, no_days = param.shape
        interval_no = math.floor(no_days / interval)
        remainder = no_days % interval
        param_name_list = list()
        if method == "average":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = torch.mean(param[:, 0:remainder], 1, keepdim=True).repeat(
                        (1, remainder)
                    )
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = torch.mean(
                    param[
                    :,
                    ((i * interval) + remainder): (
                            ((i + 1) * interval) + remainder
                    ),
                    ],
                    1,
                    keepdim=True,
                ).repeat((1, interval))
                param_name_list.append(param_name)
        elif method == "single_val":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = (param[:, 0:1]).repeat((1, remainder))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = (
                    param[
                    :,
                    (((i) * interval) + remainder): (((i) * interval) + remainder)
                                                    + 1,
                    ]
                ).repeat((1, interval))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

    def param_bounds(self, params, num, args, bounds):
        nmul = args["nmul"]
        if num in args["static_params_list_SNTEMP"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list_SNTEMP"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param_SNTEMP"][
                    args["semi_static_params_list_SNTEMP"].index(num)
                ],
                method=args["method_for_semi_static_param_SNTEMP"][
                    args["semi_static_params_list_SNTEMP"].index(num)
                ],
            )
            out = (
                    out_temp * (bounds[1] - bounds[0])
                    + bounds[0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
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

    def forward(
            self, x, params, iGrid, iT, args, air_sample_sr, air_sample_ss, air_sample_gw,
            srflow, ssflow, gwflow
    ):
        # restricting the params
        NEARZERO = args["NEARZERO"]

        # for all a and b

        w1_shade = self.param_bounds(params, 0, args, bounds=args["SNTEMP_paramCalLst"][0])
        w2_shade = self.param_bounds(params, 1, args, bounds=args["SNTEMP_paramCalLst"][1])
        w3_shade = self.param_bounds(params, 2, args, bounds=args["SNTEMP_paramCalLst"][2])
        width_coef_nom = self.param_bounds(params, 3, args, bounds=args["SNTEMP_paramCalLst"][3])
        width_coef_denom = self.param_bounds(params, 4, args, bounds=args["SNTEMP_paramCalLst"][4])
        hamon_coef = self.param_bounds(params, 5, args, bounds=args["SNTEMP_paramCalLst"][5])
        lat_temp_adj = self.param_bounds(params, 6, args, bounds=args["SNTEMP_paramCalLst"][6])
        if args["routing_SNTEMP"] == True:
            No_params = len(args["SNTEMP_paramCalLst"])
            a_srflow = self.param_bounds(params, No_params, args, bounds=args["conv_SNTEMP"][0])
            b_srflow = self.param_bounds(params, No_params + 1, args, bounds=args["conv_SNTEMP"][1])
            a_ssflow = self.param_bounds(params, No_params + 2, args, bounds=args["conv_SNTEMP"][2])
            b_ssflow = self.param_bounds(params, No_params + 3, args, bounds=args["conv_SNTEMP"][3])
            a_gwflow = self.param_bounds(params, No_params + 4, args, bounds=args["conv_SNTEMP"][4])
            b_gwflow = self.param_bounds(params, No_params + 5, args, bounds=args["conv_SNTEMP"][5])
        nmul = args["nmul"]
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        # with torch.no_grad():
        flow_tot = (
            (srflow + ssflow + gwflow)
        )  # converting cfs to cms
        precip = (
            x[:, :, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        up_inflow = make_tensor(torch.zeros(flow_tot.size()))
        mean_air_temp = (
            ((x[:, :, vars.index("tmax(C)")] + x[:, :, vars.index("tmin(C)")]) / 2)
            .unsqueeze(-1)
            .repeat(1, 1, nmul)
        )
        dayl = x[:, :, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        vp = 0.01 * x[:, :, vars.index("vp(Pa)")].unsqueeze(-1).repeat(
            1, 1, nmul
        )  # converting to mbar
        swrad = (
            (
                    x[:, :, vars.index("srad(W/m2)")]
                    * x[:, :, vars.index("dayl(s)")]
                    / 86400
            )
            .unsqueeze(-1)
            .repeat(1, 1, nmul)
        )
        elev = (
            x[:, :, vars.index("ELEV_MEAN_M_BASIN")]
            .unsqueeze(-1)
            .repeat(1, 1, nmul)
        )
        slope = 0.01 * x[:, :, vars.index("SLOPE_PCT")].unsqueeze(-1).repeat(
            1, 1, nmul
        )  # adding the percentage, it is a watershed slope not a stream slope
        # stream_density = x[:, :, vars.index("STREAMS_KM_SQ_KM")]
        # stream_length = 1000 * (stream_density * x[:, :, vars.index("DRAIN_SQKM")]).unsqueeze(-1).repeat(1,1,nmul)
        # stream_length = x[:, :, vars.index("stream_length_artificial")]
        # stream_length = x[:, :, vars.index("NHDlength_tot(m)")].unsqueeze(-1).repeat(1,1,nmul)
        stream_length = (
            x[:, :, vars.index("stream_length_artificial")]
            .unsqueeze(-1)
            .repeat(1, 1, nmul)
        )
        # basin_area = x[:, :, vars.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1,1,nmul)
        cloud_fraction = x[:, :, vars.index("ccov")].unsqueeze(-1).repeat(1, 1, nmul)
        # t_monthly = x[:, :, vars.index("t_monthly(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        albedo = args["STemp_default_albedo"]

        # hamon PET equation. We can add other methods too, such as Penman monteith
        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        )

        # d = torch.pow(q * n * (q + 1) / (p * torch.pow(slope, 0.5)), (3 / (5 + 3 * q)))

        # d = torch.pow(width_coef_nom * obsQ / (width_coef_denom * torch.pow(basin_area, width_A_coef)), width_exp)
        # top_width = 5 + p * torch.pow(d, q)
        # top_width = width_coef_nom * obsQ + width_coef_denom
        # top_width = width_coef_nom * (obsQ ** width_coef_denom) + 0.5
        top_width = (
                torch.abs(width_coef_nom)
                * torch.pow(torch.abs(flow_tot + 0.0001), torch.abs(width_coef_denom))
                + 0.5
        )
        # top_width = make_tensor(torch.ones(width_coef_nom.shape) * 10.0, device=args["device"])
        # if p.dim() == 3:
        #     top_width = p * torch.pow(basin_area, q)
        # elif p.dim() == 2:
        #     top_width = p * torch.pow(basin_area, q)

        # masking surface runoff fraction with precipitation.
        # if there is not any precipitaton, it cannot be more than 0.01

        # total shade (solar shade) is accumulative shade of vegetation and topography
        if args["shade_smoothening"] == True:
            (
                shade_fraction_riparian,
                shade_fraction_topo,
                shade_total,
            ) = self.shade_modification(w1_shade, w2_shade, w3_shade, args)
        else:
            shade_fraction_riparian = w1_shade
            shade_fraction_topo = (
                    (1 - shade_fraction_riparian) * w2_shade / (w2_shade + w3_shade)
            )
            shade_total = shade_fraction_riparian + shade_fraction_topo
            # shade_fraction_riparian = w1_shade / (w1_shade + w2_shade + w3_shade)
            # shade_fraction_topo = w2_shade / (w1_shade + w2_shade + w3_shade)
            # shade_total = shade_fraction_riparian + shade_fraction_topo
        a_srflow_new = a_srflow.mean(-1, keepdim=True).permute([1, 0, 2])
        b_srflow_new = b_srflow.mean(-1, keepdim=True).permute([1, 0, 2])
        w_srflow = self.UH_gamma(a=a_srflow_new, b=b_srflow_new,
                                 lenF=args["res_time_lenF_srflow"])
        air_sample_sr = air_sample_sr.permute([0, 2, 1])  # dim:gage*var*time
        w_srflow = w_srflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_sr = self.UH_conv(air_sample_sr, w_srflow)[:, :, args["res_time_lenF_srflow"]:].permute(
            [0, 2, 1]).repeat(1, 1, nmul)

        a_ssflow_new = a_ssflow.mean(-1, keepdim=True).permute([1, 0, 2])
        b_ssflow_new = b_ssflow.mean(-1, keepdim=True).permute([1, 0, 2])
        w_ssflow = self.UH_gamma(a=a_ssflow_new, b=b_ssflow_new,
                                 lenF=args["res_time_lenF_ssflow"])
        air_sample_ss = air_sample_ss.permute([0, 2, 1])  # dim:gage*var*time
        w_ssflow = w_ssflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_ss = self.UH_conv(air_sample_ss, w_ssflow)[:, :, args["res_time_lenF_ssflow"]:].permute(
            [0, 2, 1]).repeat(1, 1, nmul)

        a_gwflow_new = a_gwflow.mean(-1, keepdim=True).permute([1, 0, 2])
        b_gwflow_new = b_gwflow.mean(-1, keepdim=True).permute([1, 0, 2])
        w_gwflow = self.UH_gamma(a=a_gwflow_new, b=b_gwflow_new,
                                 lenF=args["res_time_lenF_gwflow"])
        air_sample_gw = air_sample_gw.permute([0, 2, 1])  # dim:gage*var*time
        w_gwflow = w_gwflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_gw = self.UH_conv(air_sample_gw, w_gwflow)[:, :, args["res_time_lenF_gwflow"]:].permute(
            [0, 2, 1]).repeat(1, 1, nmul)
        # w_srflow = self.res_time_gamma(
        #     a=a_srflow.unsqueeze(-1),
        #     b=b_srflow.unsqueeze(-1),
        #     lenF=args["res_time_lenF_srflow"],
        # )

        # air_sample_sr = self.x_sample_air_temp(iGrid, iT, lenF=args['res_time_params']['lenF_srflow'],
        #                                        args=args, x_total_raw=x_total_raw,
        #                                        time_range=time_range)
        # air_sample_sr = self.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_srflow'],
        #                                         args=args, ave_air_total=ave_air_total)
        # air_sample_sr = air_sample_sr.unsqueeze(-1).repeat(1, 1, 1, nmul)
        # w_srflow = w_srflow.permute(1, 2, 0, 3)
        # ave_air_sr = self.res_time_conv(
        #     air_sample_sr, w_srflow, bias=None
        # )  # bias=None, sr_conv_bias

        # subsurface flow
        # w_ssflow = self.res_time_gamma(
        #     a=a_ssflow.unsqueeze(-1),
        #     b=b_ssflow.unsqueeze(-1),
        #     lenF=args["res_time_params"]["lenF_ssflow"],
        # )

        # air_sample_ss = self.x_sample_air_temp(iGrid, iT, lenF=args['res_time_params']['lenF_ssflow'],
        #                                        args=args, x_total_raw=x_total_raw,
        #                                        time_range=time_range)
        # air_sample_ss = self.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_ssflow'],
        #                                         args=args, ave_air_total=ave_air_total)
        # air_sample_ss = air_sample_ss.unsqueeze(-1).repeat(1, 1, 1, nmul)
        # w_ssflow = w_ssflow.permute(1, 2, 0, 3)
        # ave_air_ss = self.res_time_conv(
        #     air_sample_ss, w_ssflow, bias=None
        # )  # ss_conv_bias
        #
        # # groundwater flow
        # w_gwflow = self.res_time_gamma(
        #     a=a_gwflow.unsqueeze(-1),
        #     b=b_gwflow.unsqueeze(-1),
        #     lenF=args["res_time_params"]["lenF_gwflow"],
        # )
        #
        # # air_sample_gw = self.x_sample_air_temp(iGrid, iT, lenF=args['res_time_params']['lenF_gwflow'],
        # #                                        args=args, x_total_raw=x_total_raw,
        # #                                        time_range=time_range)
        # # air_sample_gw = self.x_sample_air_temp2(iGrid, iT, lenF=args['res_time_params']['lenF_gwflow'],
        # #                                         args=args, ave_air_total=ave_air_total)
        # air_sample_gw = air_sample_gw.unsqueeze(-1).repeat(1, 1, 1, nmul)
        # w_gwflow = w_gwflow.permute(1, 2, 0, 3)
        # ave_air_gw = self.res_time_conv(
        #     air_sample_gw, w_gwflow, bias=None
        # )  # gw_conv_bias

        ave_air_temp = torch.cat((ave_air_sr.unsqueeze(-1), ave_air_ss.unsqueeze(-1), ave_air_gw.unsqueeze(-1)), dim=3)

        (
            T_0,
            srflow_temp,
            ssflow_temp,
            gwflow_temp,
            ave_air_temp_new,
        ) = self.lateral_flow_temperature(
            srflow=srflow,
            ssflow=ssflow,
            gwflow=gwflow,
            ave_air_temp=ave_air_temp,
            args=args,
            lat_temp_adj=lat_temp_adj,
        )

        # 'Correction factor to adjust the bias of the temperature of the lateral inflow'
        # Fortran code:
        # elseif(fs.le.NEARZERO)    (fs: surface runoff)
        # then
        # ! if this is true, then there is no flow from upstream, but there is lateral inflow
        # t_o = seg_tave_lat(i) + lat_temp_adj(i, Nowmonth)
        # if there is upstream flow, it should be weighted average temperature of all flows

        # if args['lat_temp_adj'] == "True":
        #     T_0 = T_l + lat_temp_adj
        # else:
        #     T_0 = T_l
        A, B, C, D = self.ABCD_equations(
            T_a=T_0,
            swrad=swrad,
            e_a=vp,
            elev=elev,
            slope=slope,
            top_width=top_width,
            up_inflow=0.0,
            E=PET,  # up_inflow
            T_g=gwflow_temp,
            iGrid=iGrid,
            shade_fraction_riparian=shade_fraction_riparian,
            albedo=albedo,
            shade_total=shade_total,
            args=args,
            cloud_fraction=cloud_fraction,
        )
        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D, T_e=T_0)
        K1, K2 = self.finding_K1_K2(
            A=A, B=B, C=C, D=D, T_e=T_e, NEARZERO=NEARZERO, T_0=T_0
        )

        Q_0 = make_tensor(np.full((flow_tot.shape), 0.000001))
        # Q_0 = make_tensor(np.full((obsQ.shape[0], obsQ.shape[1]), 0))

        # T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, T_l, T_e, ave_width=top_width,
        #                                            q_l=obsQ, L=stream_length, args=args,
        #                                            T_0=T_0, Q_0=Q_0)

        # writing the original fortran code here
        # they assumed if Q_upstream==0 and q_lat > 0, they assume Q_upstream=q_lat, and q_lat=0
        # it prevents from dividing to zero
        T_w = self.solving_SNTEMP_ODE_second_order(
            K1,
            K2,
            T_e,
            ave_width=top_width,
            q_l=Q_0,
            L=stream_length,
            args=args,
            T_0=T_0,
            Q_0=flow_tot,
        )
        # get rid of negative values:
        T_w = torch.relu(T_w)

        source_temps = torch.cat((srflow_temp.mean(-1, keepdim=True),
                                  ssflow_temp.mean(-1, keepdim=True),
                                  gwflow_temp.mean(-1, keepdim=True)), dim=2)
        if args["lat_temp_adj"] == True:
            SNTEMP_outs = torch.cat((PET.mean(-1, keepdim=True),
                                     shade_fraction_riparian.mean(-1, keepdim=True),
                                     shade_fraction_topo.mean(-1, keepdim=True),
                                     top_width.mean(-1, keepdim=True),
                                     hamon_coef.mean(-1, keepdim=True),
                                     lat_temp_adj.mean(-1, keepdim=True)), dim=2)
            return (T_w.mean(-1, keepdim=True),
                    ave_air_temp_new.mean(2, keepdim=True).squeeze(2),
                    w_gwflow.permute([0, 2, 1]),
                    w_ssflow.permute([0, 2, 1]),
                    source_temps,
                    SNTEMP_outs
                    )
        else:
            SNTEMP_outs = torch.cat((PET.mean(-1, keepdim=True),
                                     shade_fraction_riparian.mean(-1, keepdim=True),
                                     shade_fraction_topo.mean(-1, keepdim=True),
                                     top_width.mean(-1, keepdim=True),
                                     hamon_coef.mean(-1, keepdim=True),
                                     0.0 * hamon_coef.mean(-1, keepdim=True)), dim=2)
            return (T_w.mean(-1, keepdim=True),
                    ave_air_temp_new.mean(2, keepdim=True).squeeze(2),
                    w_gwflow.permute([0, 2, 1]),
                    w_ssflow.permute([0, 2, 1]),
                    source_temps,
                    SNTEMP_outs)


# this class needs observed flow.
class SNTEMP_only(nn.Module):
    def __init__(self):
        super(SNTEMP_only, self).__init__()

    def atm_pressure(self, elev):
        ## from Jake's document
        # mmHg2mb = make_tensor(0.75061683)  # Unit conversion
        # mmHg2inHg = make_tensor(25.3970886)  # Unit conversion
        # P_sea = make_tensor(29.92126)  # Standard pressure ar sea level
        # A_g = make_tensor(9.80665)  # Acceleration due to gravity
        # M_a = make_tensor(0.0289644)  # Molar mass of air
        # R = make_tensor(8.31447)  # universal gas constant
        # T_sea = make_tensor(288.16)  # the standard temperature at sea level
        # P = (1 / mmHg2mb) * (mmHg2inHg) * (P_sea) * torch.exp(-A_g * M_a * elev / (R * T_sea))

        ## the code from stream_temp.f90
        P = 1013 - (0.1055 * elev)

        ## Note: both jakes and stream_temp are close to each other (error less than 0.013 for 99 basins)
        return P

    def atm_longwave_radiation_heat(self, T_a, e_a, shade_total, cloud_fraction, args):
        """
        :param T_a: air temperature in degree Celsius
        :param e_a: vapor pressure
        :return: Atmospheric longwave radiation
        """
        H_a = (
                (3.354939e-8 + 2.74995e-9 * e_a ** 0.5)
                * (1 - shade_total)
                * (1 + 0.17 * cloud_fraction ** 2)
                * (T_a + 273.16) ** 4
        )
        return H_a

    def stream_friction_heat(self, top_width, slope, Q):
        H_f = (
                9805 * Q * slope / top_width
        )  # Q is the seg_inflow (total flow entering a segment)
        return H_f

    def shortwave_solar_radiation_heat(self, albedo, H_sw, shade_total):
        """
        :param albedo: albedo or fraction reflected by stream , dimensionless
        :param H_sw: the clear sky solar radiation in watt per sq meter (seginc_swrad)
        :return: daily average clear sky, shortwave solar radiation for each segment
        """
        # solar_shade_fraction = make_tensor(args['STemp_default_params']['shade_fraction'])
        H_s = (1 - albedo) * (1 - shade_total) * H_sw
        return H_s

    def riparian_veg_longwave_radiation_heat(
            self, T_a, shade_fraction_riparian, args
    ):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(
            make_tensor(10), (-8.0)
        )  # (J/s*m^2 * K^4)
        emissivity_veg = make_tensor(args["STemp_default_emissivity_veg"])
        H_v = (
                emissivity_veg
                * St_Boltzman_ct
                * shade_fraction_riparian
                * torch.pow((T_a + 273.16), 4)
        )
        # H_v = emissivity_veg * St_Boltzman_ct * shade2[iGrid, :] * torch.pow((T_a + 273.16), 4)
        return H_v

    def ABCD_equations(
            self,
            T_a,
            swrad,
            e_a,
            E,
            elev,
            slope,
            top_width,
            up_inflow,
            T_g,
            shade_fraction_riparian,
            albedo,
            shade_total,
            args,
            cloud_fraction,
    ):
        """

        :param T_a: average daily air temperature
        :param swrad: solar radiation
        :param e_a: vapor pressure
        :param E: Free-water surface-evaporation rate (assumed to be PET, potet in PRMS)
        :param elev: average basin elevation
        :param slope: average stream slope (seg_slope)
        :param top_width: average top width of the stream
        :param up_inflow: is the discharge (variable seg_inflow) which is from upstream
        :return:
        """
        e_s = 6.108 * torch.exp((17.26939 * T_a) / (237.3 + T_a))
        # e_s = 6.108 * torch.exp((17.26939 * T_0) / (237.3 + T_0))
        P = self.atm_pressure(
            elev
        )  # calculating atmosphere pressure based on elevation
        # chacking vapor pressure with saturation vapor pressure
        denom = e_s - e_a
        mask_denom = denom.ge(0)
        # converting negative values to zero
        denom1 = denom * mask_denom.int().float()
        # adding 0.01 to zero values as it is denominator
        mask_denom2 = denom1.eq(0)
        denom2 = denom1 + 0.01 * mask_denom2.int().float()

        B_c = 0.00061 * P / denom2
        B_c1 = 0.00061 * P / (e_s - e_a)
        K_g = make_tensor(1.65)
        delta_Z = make_tensor(args["STemp_default_delta_Z"])
        # we don't need H_a, because we hae swrad directly from inputs
        H_a = self.atm_longwave_radiation_heat(
            T_a, e_a, shade_total, cloud_fraction, args=args
        )
        ###############
        H_f = self.stream_friction_heat(top_width=top_width, slope=slope, Q=up_inflow)
        H_s = self.shortwave_solar_radiation_heat(
            albedo=albedo, H_sw=swrad, shade_total=shade_total
        )  # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(
            T_a, shade_fraction_riparian, args=args
        )

        A = 5.4 * torch.pow(make_tensor(np.full((T_a.shape), 10)), (-8))
        B = torch.pow(make_tensor(10), 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (
                K_g / delta_Z
        )
        C = torch.pow(make_tensor(10), 6) * E * B_c * 2.36
        # Todo: need to check 10**6. it is in fortran code but it is not in the document
        D = (
                H_a
                + H_s
                + H_v
                + 2495 * torch.pow(make_tensor(10), 6) * E * ((B_c * T_a) - 1)
                + (T_g * K_g / delta_Z)
        )
        # D = H_a + swrad + H_v + 2495 * E * ((B_c * T_a) - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e, iter=50):
        def F(T_e):
            return (
                    A * torch.pow((T_e + 273.16), 4) - C * torch.pow(T_e, 2) + B * T_e - D
            )

        def Fprime(T_e):
            return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B

        ## solving the equation with Newton's method
        for i in range(iter):
            next_geuss = T_e - (F(T_e) / Fprime(T_e))
            T_e = next_geuss.clone()

        return T_e

    def finding_K1_K2(self, A, B, C, D, T_e, NEARZERO, T_0):
        """
        :param A: Constant coming from equilibrium temp equation
        :param B: Constant coming from equilibrium temp equation
        :param C: Constant coming from equilibrium temp equation
        :param T_e: equilibrium temperature
        :param H_i: initial net heat flux at temperature T_o, of the upstream inflow
        :param T_o: initial water temperature
        :return: K1 (first order thermal exchange coefficient), K2 (second order coefficient)
        """
        H_i = A * torch.pow((T_0 + 273.16), 4) - C * torch.pow(T_0, 2) + B * T_0 - D
        K1 = 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B
        delt = T_0 - T_e
        denom = torch.pow(delt, 2)
        mask_denom = denom.le(NEARZERO)
        denom1 = denom + mask_denom.int().float()
        K2 = ((K1 * delt) - H_i) / denom1
        K2 = torch.where(torch.abs(delt) < NEARZERO,
                         torch.zeros(K2.shape).to(K2),
                         K2)
        # K2 = (-H_i + (K1 * (T_0 - T_e))) / denom1

        return K1, K2

    def srflow_ssflow_gwflow_portions(
            self,
            discharge,
            srflow_factor=make_tensor(0.40),
            ssflow_factor=make_tensor(0.3),
            gwlow_factor=make_tensor(0.3),
    ):
        srflow = srflow_factor * discharge
        ssflow = ssflow_factor * discharge
        gwflow = gwlow_factor * discharge
        return srflow, ssflow, gwflow

    def ave_temp_res_time(
            self, ave_air_temp, x, res_time, iGrid, iT, args, x_total_raw
    ):
        rho = x.shape[1]  # args['hyperparameters']['rho']
        tArray_Total = tRange2Array(args["tRange"])
        tArray_train = tRange2Array(args["t_train"])
        _, ind1, _ = intersect(tArray_Total, tArray_train)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        temp_res = res_time
        with torch.no_grad():
            temp_res1 = temp_res.int()
        A = res_time.repeat(1, rho)
        B = torch.reshape(A, (res_time.shape[0], rho, res_time.shape[1]))
        ave_air = torch.zeros(
            (
                args["batch_size"],
                args["rho"],
                res_time.shape[1],
            ),
            device=args["device"],
        )
        for i in range(res_time.shape[1]):
            for s, station in enumerate(iGrid):
                array = np.zeros((x.shape[1], temp_res1[s, i].item()), dtype=np.int32)
                for j in range(temp_res1[s, i].item()):
                    array[:, j] = np.arange(
                        (ind1_tensor[0] + iT_tensor[s] - j).item(),
                        (ind1_tensor[0] + iT_tensor[s] - j + x.shape[1]).item(),
                    )
                tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
                max_add = torch.sum(tmax_temp, dim=1)
                tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
                min_add = torch.sum(tmin_temp, dim=1)
                ave_air[s, :, i] = (max_add + min_add) / 2  # (2 * res_time[station, i])
        ave_air_temp = ave_air / B
        # return ave_air
        return ave_air_temp

    def x_sample_air_temp(self, iGrid, iT, lenF, args, x_total_raw, time_range):
        """
        :param iGrid:
        :param iT:
        :param lenF: maximum number of days that it is needed to be considered in average
        :return:
        """
        rho = args["rho"]
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        _, ind1, _ = intersect(tArray_Total, tArray_sample)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        ave_air = torch.zeros(
            (
                args["batch_size"],
                args["rho"],
                lenF,
            ),
            device=args["device"],
        )
        for s, station in enumerate(iGrid):
            array = np.zeros((rho, lenF), dtype=np.int32)
            for j in range(lenF):
                array[:, j] = np.arange(
                    (ind1_tensor[0] + iT_tensor[s] - j).item(),
                    (ind1_tensor[0] + iT_tensor[s] - j + rho).item(),
                )
            # array = np.flip(array, 1).copy()
            tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp
        return ave_air

    def x_sample_air_temp2(self, iGrid, iT, lenF, args, ave_air_total):
        rho = args["rho"]
        # ave_air = torch.zeros((len(iGrid), args["hyperparameters"]["rho"],
        #                        lenF),
        #                       device=args["device"])
        a = min(ave_air_total.shape[1], args["rho"])
        ave_air = torch.zeros((len(iGrid), a, lenF), device=args["device"])
        # array = np.array([np.arange(x, y) for x, y in zip(iT, iT + rho)])
        ave_air_temp = ave_air_total[iGrid, :, 0:lenF]
        for i in range(len(iGrid)):
            ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + a), :]
            # ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + ave_air_total.shape[1]), :]

        # ave_air_temp = ave_air_total[:, iT : iT + rho, 0:lenF]
        # ave_air = ave_air_temp[iGrid, :, :]
        return ave_air

    def ave_temp_general(self, args, x_total_raw_tensor, time_range):
        vars = args["varT_NN"] + args["varC_NN"]
        lenF_max = np.maximum(
            args["res_time_lenF_srflow"],
            np.maximum(
                args["res_time_lenF_ssflow"],
                args["res_time_lenF_gwflow"],
            ),
        )
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        c, ind1, ind2 = np.intersect1d(tArray_sample, tArray_Total, return_indices=True)
        ave_air = torch.zeros(
            (x_total_raw_tensor.shape[0], len(tArray_sample), lenF_max), device="cpu"
        )

        array = np.zeros((len(ind2), lenF_max), dtype=np.int32)
        for j in range(lenF_max):
            array[:, j] = np.arange(
                (ind2[0] - j).item(), (ind2[0] - j + len(ind2)).item()
            )

        for s in range(x_total_raw_tensor.shape[0]):
            tmax_temp = x_total_raw_tensor[s, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw_tensor[s, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp

        return ave_air

    def res_time_gamma(self, a, b, lenF):
        # UH. a [time (same all time steps), batch, var]
        # a = torch.abs(a)
        if a.dim() == 2:
            m = a.shape
            a1 = a.repeat(1, lenF)
            b1 = b.repeat(1, lenF)
            alpha = F.relu(a1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.1
            beta = F.relu(b1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 3:
            m = a.shape
            a1 = a.repeat(1, 1, lenF)
            b1 = b.repeat(1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.1
            beta = F.relu(b1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 4:
            m = a.shape
            a1 = a.repeat(1, 1, 1, lenF)
            b1 = b.repeat(1, 1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.1
            beta = F.relu(b1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.5
            x = (
                torch.linspace(0.001, 20, lenF)
                .view(lenF, 1, 1, 1)
                .repeat(1, m[0], m[1], m[2])
            )
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        return www

    def UH_gamma(self, a, b, lenF=10):
        # UH. a [time (same all time steps), batch, var]
        m = a.shape
        lenF = min(a.shape[0], lenF)
        w = torch.zeros([lenF, m[1], m[2]])
        aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
        theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
        # t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
        # t = t.cuda(aa.device)
        t = (torch.linspace(0.001, 20, lenF).view(lenF, 1, 1).repeat(1, m[1], 1))
        t = t.to(aa.device)
        denom = (aa.lgamma().exp()) * (theta ** aa)
        mid = t ** (aa - 1)
        right = torch.exp(-t / theta)
        w = 1 / denom * mid * right
        ww = torch.cumsum(w, dim=0)
        www = ww / ww.sum(0)
        # w = w / w.sum(0)  # scale to 1 for each UH
        return www

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

        # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
        y = F.conv1d(xx, w, groups=groups, padding=padd, stride=1, bias=None)  # we don't need flip
        if padd != 0:
            y = y[:, :, 0:-padd]
        return y.view(mm)

    def res_time_conv(self, x_sample, UH, bias, viewmode=1):
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
        # batch needs to be accommodated by channels and we make use of gr
        # ++++---------------------------------+
        #
        # oups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html
        if UH.shape[1] == 1:
            x = x_sample[:, 0:1, :]
            mm = x.shape
            nb = mm[0]
            m = UH.shape[-1]
            padd = m - 1
            if viewmode == 1:
                xx = x.view([1, nb, mm[-1]])
                w = UH.view([nb, 1, m])
                groups = nb

                # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
                # y = y[:, :, 0:-padd]

            x_sample1 = x_sample.permute(1, 0, 2)
            a = torch.arange(x_sample.shape[1])
            y = F.conv1d(
                x_sample1[a],
                torch.flip(w, [2]),
                groups=groups,
                padding=0,
                stride=1,
                bias=bias,
            )
            y = y.permute(1, 0, 2)
        elif UH.shape[1] > 1:
            w = torch.flip(UH, [2])
            y = x_sample * w
            y = y.sum(2)
            if bias is not None:
                y = y + bias
            y = y.unsqueeze(3)

        return y

    def lateral_flow_temperature(
            self, srflow, ssflow, gwflow, ave_air_temp, args, lat_temp_adj, NEARZERO=1e-6
    ):
        """
        :param srflow: surface runoff
        :param ssflow: subsurface runoff
        :param gwflow: qroundwaterflow
        :param res_time_srflow: residense time for surface runoff
        :param res_time_ssflow: residence time for subsurface runoff
        :param res_time_gwflow: residence time for groundwater flow
        :return: temperature of lateral flow
        """
        # with torch.no_grad():
        if args["res_time_type"] == "SNTEMP":
            mask_ave_air_temp = ave_air_temp.ge(0)
            ave_air_temp = ave_air_temp * mask_ave_air_temp.int().float()

            srflow_temp = ave_air_temp[:, :, :, 0]  # .clone().detach()
            ssflow_temp = ave_air_temp[:, :, :, 1]  # .clone().detach()
            gwflow_temp = ave_air_temp[:, :, :, 2]  # .clone().detach()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        elif args["res_time_type"] == "van Vliet":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0] - 1.5
            mask_srflow_temp = srflow_temp.ge(0.0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 0]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(5.0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()
            gwflow_temp = torch.clamp(gwflow_temp, min=5.0)
            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )
        #
        # elif args["res_time_params"]["type"] is "Meisner":
        elif args["res_time_type"] == "Meisner":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0]
            mask_srflow_temp = srflow_temp.ge(0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 1]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        denom = gwflow + ssflow + srflow
        mask_denom = denom.eq(0.0)
        denom = denom + mask_denom.int().float()

        if args["lat_temp_adj"] == True:
            gwflow_temp = gwflow_temp + lat_temp_adj

        T_l = (
                      (gwflow * gwflow_temp) + (srflow * srflow_temp) + (ssflow * ssflow_temp)
              ) / denom

        mask_less_zero = T_l.le(NEARZERO)
        T_l[mask_less_zero] = 0.0
        return T_l, srflow_temp, ssflow_temp, gwflow_temp, lat_flow_temp

    def solving_SNTEMP_ODE_second_order(
            self,
            K1,
            K2,
            T_e,
            ave_width,
            q_l,
            L,
            args,
            T_0=make_tensor(0),
            Q_0=make_tensor(0.01),
            NEARZERO=1e-10,
    ):
        # # Note: as we assume that Q_0 is 0.01, we are always gaining flow with positive lateral flow or
        # # with zero lateral flow
        density = args["params_water_density"]
        c_w = args["params_C_w"]
        b = K1 * ave_width / (density * c_w)
        mask_q_l = q_l.eq(0)
        q_l_pos = q_l + mask_q_l.int().float()
        rexp = -1.0 * (b * L) / q_l_pos
        r = torch.exp(rexp)  # No idea why it is torch.exp in stream_Temp.f90 (the headwater part)

        delt = T_e - T_0
        mask_K1 = K1.eq(0)
        K1_masked = K1 + mask_K1.int().float()
        denom = 1 + ((K2 / K1_masked) * delt * (1 - r))
        denom = torch.where(torch.abs(denom) < NEARZERO,
                            torch.sign(denom) * NEARZERO,
                            denom)
        Tw = T_e - (delt * r / denom)
        Tw = torch.clamp(Tw, min=0.0)
        return Tw

        #
        # density = args["params_water_density"]
        # c_w = args["params_C_w"]
        # mask_q_l = q_l.eq(0)
        # q_l_pos = q_l + mask_q_l.int().float()
        # b = q_l + ((K1 * ave_width) / (density * c_w))
        # mask_Q_0 = Q_0.eq(0)
        # Q_0_pos = Q_0 + mask_Q_0.int().float()
        # R_0 = torch.exp(-(b * L) / Q_0_pos)
        # mask_K1 = K1.eq(0)
        # K1_masked = K1 + mask_K1.int().float()
        # denom = 1 + ((K2 / K1_masked) * (T_e - T_0) * (1 - R_0))
        # mask_denom = denom.eq(0)
        # denom_masked = denom + mask_denom.int().float()
        # Tw = T_e - ((T_e - T_0) * R_0 / denom_masked)
        # return Tw

    def parameter_bounds(self, params, num, args):
        if params.dim() == 3:
            if num in args["static_params_list"]:
                out_temp = (
                        params[:, -1, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
                out = (
                    out_temp.view(-1, 1)
                    .repeat(1, params.shape[1])
                    .view(params.shape[0], params.shape[1])
                )
            elif num in args["semi_static_params_list"]:
                out_temp = self.semi_static_params(
                    params,
                    num,
                    interval=args["interval_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                    method=args["method_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                )
                out = (
                        out_temp
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
            else:  # dynamic
                out = (
                        params[:, :, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
        elif params.dim() == 2:
            out_temp = (
                    params[:, num]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = (
                out_temp.view(-1, 1)
                .repeat(1, args["hyperparameters"]["rho"])
                .view(
                    args["hyperparameters"]["batch_size"],
                    args["hyperparameters"]["rho"],
                )
            )
        return out

    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
                method=args["method_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
            )
            out = (
                    out_temp * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
        return out

    def shade_modification(self, w1_shade, w2_shade, w3_shade, args):
        nmul = args["nmul"]
        A = list()
        C = list()
        # w1 = torch.ones((365, 1, 40), device=args["device"]) / 40
        w1 = torch.ones((w1_shade.shape[0], 1, 10), device=args["device"]) / 10
        for i in range(nmul):
            sh1 = w1_shade[:, :, i]
            sh2 = w2_shade[:, :, i]
            B1 = self.UH_conv(sh1.unsqueeze(-1).permute([0, 2, 1]), w1).permute([0, 2, 1])
            w1_shade_mov = torch.clamp(B1, min=0.0001, max=1.0)
            A.append(w1_shade_mov)

            B2 = self.UH_conv(sh2.unsqueeze(-1).permute([0, 2, 1]), w1).permute([0, 2, 1])
            w2_shade_mov = torch.clamp(B2, min=0.0001, max=1.0)
            C.append(w2_shade_mov)
        shade_fraction_riparian = torch.cat(A, dim=2)
        w2_shade_por_mov = torch.cat(C, dim=2)

        shade_fraction_riparian = torch.clamp(
            shade_fraction_riparian, min=0.0001, max=1.0
        )

        shade_fraction_topo = (1 - shade_fraction_riparian) * w2_shade_por_mov
        shade_fraction_topo = torch.clamp(shade_fraction_topo, min=0.0001, max=1.0)
        shade_total = shade_fraction_riparian + shade_fraction_topo
        shade_total = torch.clamp(shade_total, min=0.0001, max=1.0)

        return shade_fraction_riparian, shade_fraction_topo, shade_total

    def frac_modification(
            self, srflow_portion, ssflow_portion, gwflow_portion, Q_T, args,
    ):
        nmul = args["nmul"]
        A = list()
        Q = gwflow_portion * Q_T
        gw_filter_size = args["frac_smoothening_gw_filter_size"]
        wgw = (torch.ones(
            (gwflow_portion.shape[0], 1, gw_filter_size), device=args["device"]
        )
               / gw_filter_size
               )
        for i in range(nmul):
            Q_gw = Q[:, :, i]
            B = self.UH_conv(Q_gw.unsqueeze(-1).permute([0, 2, 1]), wgw).permute([0, 2, 1])
            # B = torch.flip(self.UH_conv(torch.flip(Q_gw.unsqueeze(-1).permute([0, 2, 1]), [2]), wgw).permute([0, 2, 1]), [1])
            Q_gw_por_mov = torch.max(
                torch.min(B, Q_T[:, :, 0].unsqueeze(-1)), make_tensor(0.0)
            )
            gwflow_portion_new = Q_gw_por_mov / (
                    Q_T[:, :, 0].unsqueeze(-1) + 0.001
            )  # 0.001 is for not having nan values
            gwflow_portion_new = torch.clamp(gwflow_portion_new, min=0.001, max=1.0)
            A.append(gwflow_portion_new)
        gwflow_portion_new = torch.cat(A, dim=2)
        remain_frac = 1 - gwflow_portion_new

        if args["res_time_type"] != "Meisner":
            srflow_portion_new = srflow_portion * remain_frac
            ssflow_portion_new = remain_frac - srflow_portion_new
        else:
            srflow_portion_new = remain_frac
            ssflow_portion_new = ssflow_portion * 0.0 + 0.0001
        srflow_percentage = torch.clamp(srflow_portion_new, min=0.0001, max=1.0)
        ssflow_percentage = torch.clamp(ssflow_portion_new, min=0.0001, max=1.0)
        gwflow_percentage = torch.clamp(gwflow_portion_new, min=0.0001, max=1.0)

        return srflow_percentage, ssflow_percentage, gwflow_percentage

    def frac_modification2(self, portion, Q_T, filter_size, args):
        nmul = args["nmul"]
        A = list()
        w = torch.ones((portion.shape[0], 1, filter_size), device=args["device"]) / filter_size
        # a = torch.arange(filter_size, device=args["device"], dtype=torch.float32) * (1.0 - 0.1) / filter_size + 0.1
        # w = (torch.flip(a / a.sum(), [0])).repeat(portion.shape[0], 1).unsqueeze(-1).permute([0, 2, 1])
        Q = portion * Q_T
        for i in range(nmul):
            Q_w = Q[:, :, i]
            B = self.UH_conv(Q_w.unsqueeze(-1).permute([0, 2, 1]), w).permute([0, 2, 1])
            # B = torch.flip(self.UH_conv(torch.flip(Q_gw.unsqueeze(-1).permute([0, 2, 1]), [2]), wgw).permute([0, 2, 1]), [1])
            Q_w_por_mov = torch.max(
                torch.min(B, Q_T[:, :, i].unsqueeze(-1)), make_tensor(0.0)
            )
            wflow_portion_new = Q_w_por_mov / (
                    Q_T[:, :, i].unsqueeze(-1) + 0.001
            )  # 0.001 is for not having nan values
            wflow_portion_new = torch.clamp(wflow_portion_new, min=0.0001, max=1.0)
            A.append(wflow_portion_new)
        wflow_portion_new = torch.cat(A, dim=2)

        wflow_percentage = torch.clamp(wflow_portion_new, min=0.0001, max=1.0)

        return wflow_percentage

    def semi_static_params(self, params, param_no, interval=30, method="average"):
        # seperate the piece for each interval
        param = params[:, :, param_no]
        no_basins, no_days = param.shape
        interval_no = math.floor(no_days / interval)
        remainder = no_days % interval
        param_name_list = list()
        if method == "average":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = torch.mean(param[:, 0:remainder], 1, keepdim=True).repeat(
                        (1, remainder)
                    )
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = torch.mean(
                    param[
                    :,
                    ((i * interval) + remainder): (
                            ((i + 1) * interval) + remainder
                    ),
                    ],
                    1,
                    keepdim=True,
                ).repeat((1, interval))
                param_name_list.append(param_name)
        elif method == "single_val":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = (param[:, 0:1]).repeat((1, remainder))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = (
                    param[
                    :,
                    (((i) * interval) + remainder): (((i) * interval) + remainder)
                                                    + 1,
                    ]
                ).repeat((1, interval))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

    def param_bounds(self, params, num, args, bounds):
        nmul = args["nmul"]
        if num in args["static_params_list_SNTEMP"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list_SNTEMP"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param_SNTEMP"][
                    args["semi_static_params_list_SNTEMP"].index(num)
                ],
                method=args["method_for_semi_static_param_SNTEMP"][
                    args["semi_static_params_list_SNTEMP"].index(num)
                ],
            )
            out = (
                    out_temp * (bounds[1] - bounds[0])
                    + bounds[0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
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

    def forward(
            self, x, params, iGrid, iT, args, air_sample_sr, air_sample_ss, air_sample_gw):
        # restricting the params
        NEARZERO = args["NEARZERO"]
        warm_up = args["warm_up"]
        w1_shade = self.param_bounds(params, 0, args, bounds=args["SNTEMP_paramCalLst"][0])
        w2_shade = self.param_bounds(params, 1, args, bounds=args["SNTEMP_paramCalLst"][1])
        # w3_shade = self.param_bounds(params, 2, args, bounds=args["SNTEMP_paramCalLst"][2])
        width_coef_nom = self.param_bounds(params[:, warm_up:, :], 2, args, bounds=args["SNTEMP_paramCalLst"][2])
        width_coef_denom = self.param_bounds(params[:, warm_up:, :], 3, args, bounds=args["SNTEMP_paramCalLst"][3])
        hamon_coef = self.param_bounds(params[:, warm_up:, :], 4, args, bounds=args["SNTEMP_paramCalLst"][4])
        gwflow_portion = self.param_bounds(params, 5, args, bounds=args["SNTEMP_paramCalLst"][5])
        ssflow_portion = self.param_bounds(params, 6, args, bounds=args["SNTEMP_paramCalLst"][6])
        # gwflow_portion = self.param_bounds(params, 8, args, bounds=args["SNTEMP_paramCalLst"][8])
        if args["routing_SNTEMP"] == True:
            No_params = len(args["SNTEMP_paramCalLst"])
            # a_srflow = self.param_bounds(params[:, warm_up:, :], No_params, args, bounds=args["conv_SNTEMP"][0])
            # b_srflow = self.param_bounds(params[:, warm_up:, :], No_params + 1, args, bounds=args["conv_SNTEMP"][1])
            a_ssflow = self.param_bounds(params[:, warm_up:, :], No_params , args, bounds=args["conv_SNTEMP"][0])
            b_ssflow = self.param_bounds(params[:, warm_up:, :], No_params + 1, args, bounds=args["conv_SNTEMP"][1])
            a_gwflow = self.param_bounds(params[:, warm_up:, :], No_params + 2, args, bounds=args["conv_SNTEMP"][2])
            b_gwflow = self.param_bounds(params[:, warm_up:, :], No_params + 3, args, bounds=args["conv_SNTEMP"][3])
            if args["lat_temp_adj"] == True:
                lat_temp_adj = self.param_bounds(params[:, warm_up:, :], No_params + 4, args,
                                                 bounds=args["SNTEMP_lat_adj_paramCallLst"][0])
        if args["lat_temp_adj"] == False:
            lat_temp_adj = 0.0 * hamon_coef
        nmul = args["nmul"]
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        obsQ = x[:, :, vars.index("00060_Mean")].unsqueeze(-1).repeat(1, 1, nmul) * 0.028316  # converting cfs to cms
        precip = (
            x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        up_inflow = make_tensor(torch.zeros(obsQ.size()))
        mean_air_temp = (
            ((x[:, warm_up:, vars.index("tmax(C)")] + x[:, warm_up:, vars.index("tmin(C)")]) / 2)
            .unsqueeze(-1)
            .repeat(1, 1, nmul)
        )
        dayl = x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        vp = 0.01 * x[:, warm_up:, vars.index("vp(Pa)")].unsqueeze(-1).repeat(1, 1, nmul)  # converting to mbar
        swrad = (x[:, warm_up:, vars.index("srad(W/m2)")] * x[:, warm_up:, vars.index("dayl(s)")] / 86400).unsqueeze(-1).repeat(1, 1,
                                                                                                                  nmul)
        elev = x[:, warm_up:, vars.index("ELEV_MEAN_M_BASIN")].unsqueeze(-1).repeat(1, 1, nmul)
        slope = 0.01 * x[:, warm_up:, vars.index("SLOPE_PCT")].unsqueeze(-1).repeat(1, 1, nmul)  # adding the percentage, it is a watershed slope not a stream slope
        # stream_density = x[:, :, vars.index("STREAMS_KM_SQ_KM")]
        # stream_length = 1000 * (stream_density * x[:, :, vars.index("DRAIN_SQKM")]).unsqueeze(-1).repeat(1,1,nmul)
        # stream_length = x[:, :, vars.index("stream_length_artificial")]
        # stream_length = x[:, :, vars.index("NHDlength_tot(m)")].unsqueeze(-1).repeat(1,1,nmul)
        stream_length = x[:, warm_up:, vars.index("stream_length_square")].unsqueeze(-1).repeat(1, 1, nmul) * 1000.0  # km to meter
        # basin_area = x[:, :, vars.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1,1,nmul)
        cloud_fraction = x[:, warm_up:, vars.index("ccov")].unsqueeze(-1).repeat(1, 1, nmul)
        # t_monthly = x[:, :, vars.index("t_monthly(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        albedo = args["STemp_default_albedo"]

        # hamon PET equation. We can add other methods too, such as Penman monteith
        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        )

        # d = torch.pow(q * n * (q + 1) / (p * torch.pow(slope, 0.5)), (3 / (5 + 3 * q)))

        # d = torch.pow(width_coef_nom * obsQ / (width_coef_denom * torch.pow(basin_area, width_A_coef)), width_exp)
        # top_width = 5 + p * torch.pow(d, q)
        # top_width = width_coef_nom * obsQ + width_coef_denom
        # top_width = width_coef_nom * (obsQ ** width_coef_denom) + 0.5
        top_width = width_coef_nom * torch.pow(obsQ[:, warm_up:,:] + 0.0001, width_coef_denom) + 0.2

        # top_width = make_tensor(torch.ones(width_coef_nom.shape) * 10.0, device=args["device"])
        # if p.dim() == 3:
        #     top_width = p * torch.pow(basin_area, q)
        # elif p.dim() == 2:
        #     top_width = p * torch.pow(basin_area, q)

        # masking surface runoff fraction with precipitation.
        # if there is not any precipitaton, it cannot be more than 0.01

        if args["frac_smoothening_mode"] == True:
            gwflow_percentage = self.frac_modification2(gwflow_portion, obsQ,
                                                        args["frac_smoothening_gw_filter_size"], args)
            if args["res_time_type"] == "Meisner":
                ssflow_percentage = 0.0 * ssflow_portion + 0.0001
            else:
                ssflow_por_remain = self.frac_modification2(ssflow_portion,
                                                            obsQ * (1 - gwflow_percentage),
                                                            args["frac_smoothening_ss_filter_size"], args)
                ssflow_percentage = ssflow_por_remain * (1 - gwflow_percentage)
        else:
            gwflow_percentage = gwflow_portion
            if args["res_time_type"] == "Meisner":
                ssflow_percentage = 0.0 * ssflow_portion + 0.0001
            else:
                ssflow_percentage = (1.0 - gwflow_percentage) * ssflow_portion
        srflow_percentage = 1.0 - gwflow_percentage - ssflow_percentage


        # making the flow percentages:
        # if args["res_time_type"] == "Meisner":
        #     ssflow_portion = 0.0001 * ssflow_portion
        # srflow_percentage = srflow_portion
        # ssflow_percentage = (1 - srflow_percentage) * ssflow_portion
        # gwflow_percentage = 1 - srflow_percentage - ssflow_percentage
        #
        # if args["frac_smoothening_mode"] == True:
        #     srflow_percentage, ssflow_percentage, gwflow_percentage = self.frac_modification(srflow_percentage,
        #                                                                                      ssflow_percentage,
        #                                                                                      gwflow_percentage,
        #                                                                                      obsQ,
        #                                                                                      args)



        # total shade (solar shade) is accumulative shade of vegetation and topography
        shade_fraction_riparian =  w1_shade
        # shade_fraction_riparian = 0.01 * x[:, :, vars.index("RIP100_FOREST")].unsqueeze(-1).repeat(1, 1, nmul)
        shade_fraction_topo = w2_shade * (1 - shade_fraction_riparian)
        shade_total = shade_fraction_riparian + shade_fraction_topo
        if args["shade_smoothening"] == True:
            (
                shade_fraction_riparian,
                shade_fraction_topo,
                shade_total,
            ) = self.shade_modification(shade_fraction_riparian,
                                        shade_fraction_topo,
                                        shade_total,
                                        args)

        # get rid of the warm-up period which was used in flow and shade smoothening
        obsQ = obsQ[:, warm_up:, :]
        srflow_percentage = srflow_percentage[:, warm_up:, :]
        ssflow_percentage = ssflow_percentage[:, warm_up:, :]
        gwflow_percentage = gwflow_percentage[:, warm_up:, :]
        shade_fraction_riparian = shade_fraction_riparian[:, warm_up:, :]
        shade_fraction_topo = shade_fraction_topo[:, warm_up:, :]
        shade_total = shade_total[:, warm_up:, :]

        srflow, ssflow, gwflow = self.srflow_ssflow_gwflow_portions(discharge=obsQ,
                                                                    srflow_factor=srflow_percentage,
                                                                    ssflow_factor=ssflow_percentage,
                                                                    gwlow_factor=gwflow_percentage)

        # a_srflow_new = a_srflow.mean(-1, keepdim=True).permute([1, 0, 2])
        # b_srflow_new = b_srflow.mean(-1, keepdim=True).permute([1, 0, 2])
        # w_srflow = self.UH_gamma(a=a_srflow_new, b=b_srflow_new,
        #                          lenF=args["res_time_lenF_srflow"])
        # air_sample_sr = air_sample_sr.permute([0, 2, 1])  # dim:gage*var*time
        # w_srflow = w_srflow.permute([1, 2, 0])  # dim: gage*var*time
        # ave_air_sr = self.UH_conv(air_sample_sr, w_srflow)[:, :, args["res_time_lenF_srflow"]:].permute(
        #     [0, 2, 1]).repeat(1, 1, nmul)
        ave_air_sr = (air_sample_sr[:, args["res_time_lenF_srflow"]:, :]).repeat(1, 1, nmul)
        a_ssflow_new = a_ssflow.mean(-1, keepdim=True).permute([1, 0, 2])
        b_ssflow_new = b_ssflow.mean(-1, keepdim=True).permute([1, 0, 2])
        w_ssflow = self.UH_gamma(a=a_ssflow_new, b=b_ssflow_new,
                                 lenF=args["res_time_lenF_ssflow"])
        air_sample_ss = air_sample_ss.permute([0, 2, 1])  # dim:gage*var*time
        w_ssflow = w_ssflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_ss = self.UH_conv(air_sample_ss, w_ssflow)[:, :, args["res_time_lenF_ssflow"]:].permute(
            [0, 2, 1]).repeat(1, 1, nmul)

        a_gwflow_new = a_gwflow.mean(-1, keepdim=True).permute([1, 0, 2])
        b_gwflow_new = b_gwflow.mean(-1, keepdim=True).permute([1, 0, 2])
        w_gwflow = self.UH_gamma(a=a_gwflow_new, b=b_gwflow_new,
                                 lenF=args["res_time_lenF_gwflow"])
        air_sample_gw = air_sample_gw.permute([0, 2, 1])  # dim:gage*var*time
        w_gwflow = w_gwflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_gw = self.UH_conv(air_sample_gw, w_gwflow)[:, :, args["res_time_lenF_gwflow"]:].permute(
            [0, 2, 1]).repeat(1, 1, nmul)

        ave_air_temp = torch.cat((ave_air_sr.unsqueeze(-1), ave_air_ss.unsqueeze(-1), ave_air_gw.unsqueeze(-1)), dim=3)

        T_0, srflow_temp, ssflow_temp, gwflow_temp, ave_air_temp_new = self.lateral_flow_temperature(
            srflow=srflow,
            ssflow=ssflow,
            gwflow=gwflow,
            ave_air_temp=ave_air_temp,
            args=args,
            lat_temp_adj=lat_temp_adj,
        )

        # 'Correction factor to adjust the bias of the temperature of the lateral inflow'
        # Fortran code:
        # elseif(fs.le.NEARZERO)    (fs: surface runoff)
        # then
        # ! if this is true, then there is no flow from upstream, but there is lateral inflow
        # t_o = seg_tave_lat(i) + lat_temp_adj(i, Nowmonth)
        # if there is upstream flow, it should be weighted average temperature of all flows

        # if args['lat_temp_adj'] == "True":
        #     T_0 = T_l + lat_temp_adj
        # else:
        #     T_0 = T_l
        A, B, C, D = self.ABCD_equations(
            T_a=T_0,
            swrad=swrad,
            e_a=vp,
            elev=elev,
            slope=slope,
            top_width=top_width,
            up_inflow=0.0,  # should be obsQ
            E=PET,  # up_inflow
            T_g=gwflow_temp,
            shade_fraction_riparian=shade_fraction_riparian,
            albedo=albedo,
            shade_total=shade_total,
            args=args,
            cloud_fraction=cloud_fraction,
        )
        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D, T_e=T_0)
        K1, K2 = self.finding_K1_K2(
            A=A, B=B, C=C, D=D, T_e=T_e, NEARZERO=NEARZERO, T_0=T_0
        )

        Q_0 = make_tensor(np.full((obsQ.shape), 0.000001))
        # Q_0 = make_tensor(np.full((obsQ.shape[0], obsQ.shape[1]), 0))

        # T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, T_l, T_e, ave_width=top_width,
        #                                            q_l=obsQ, L=stream_length, args=args,
        #                                            T_0=T_0, Q_0=Q_0)

        # writing the original fortran code here
        # they assumed if Q_upstream==0 and q_lat > 0, they assume Q_upstream=q_lat, and q_lat=0
        # it prevents from dividing to zero
        T_w = self.solving_SNTEMP_ODE_second_order(
            K1,
            K2,
            T_e,
            ave_width=top_width,
            q_l=obsQ,
            L=stream_length,
            args=args,
            T_0=T_0,  # because there is no upstream flow, it is lateral flow temp
            Q_0=0.0,
        )
        # get rid of negative values:
        # T_w = torch.relu(T_w)

        source_temps = torch.cat((srflow_temp.mean(-1, keepdim=True),
                                  ssflow_temp.mean(-1, keepdim=True),
                                  gwflow_temp.mean(-1, keepdim=True)), dim=2)
        if args["lat_temp_adj"] == False:
            lat_temp_adj = 0.0 * hamon_coef
        SNTEMP_outs = torch.cat((PET.mean(-1, keepdim=True),
                                 shade_fraction_riparian.mean(-1, keepdim=True),
                                 shade_fraction_topo.mean(-1, keepdim=True),
                                 top_width.mean(-1, keepdim=True),
                                 width_coef_nom.mean(-1, keepdim=True),
                                 width_coef_denom.mean(-1, keepdim=True),
                                 hamon_coef.mean(-1, keepdim=True),
                                 lat_temp_adj.mean(-1, keepdim=True),
                                 srflow_percentage.mean(-1, keepdim=True),
                                 ssflow_percentage.mean(-1, keepdim=True),
                                 gwflow_percentage.mean(-1, keepdim=True),
                                 a_ssflow_new.permute([1,0,2]).mean(-1, keepdim=True),
                                 b_ssflow_new.permute([1,0,2]).mean(-1, keepdim=True),
                                 a_gwflow_new.permute([1,0,2]).mean(-1, keepdim=True),
                                 b_gwflow_new.permute([1,0,2]).mean(-1, keepdim=True)), dim=2)
        return (T_w.mean(-1, keepdim=True),
                ave_air_temp_new.mean(2, keepdim=True).squeeze(2),
                w_gwflow.permute([0, 2, 1]),
                w_ssflow.permute([0, 2, 1]),
                source_temps,
                SNTEMP_outs
                )


# this class needs a hydrology flow simulator as a backbone to provide daily flow of sr, ss, gw.
class SNTEMP_flowSim(nn.Module):
    def __init__(self):
        super(SNTEMP_flowSim, self).__init__()
        self.parameters_bound = dict(
            w1_shade=[0.001, 1.0],        # raw rip shade factor [0, 1]
            w2_shade=[0.001, 1.0],              # raw topo shade factor  [0, 1]
            width_coef_factor=[1.0, 25.0],    # width A coefficient, c=9.0
            width_coef_pow=[0.01, 0.25],     # width power coefficient, c=0.2273
            albedo=[0.06, 0.1]     #albedo     [0.06, 0.1]
        )
        self.conv_temp_model_bound = dict(
            a_ssflow=[0.001, 12.0],  # a (k) for ss flow temp
            b_ssflow=[0.001, 12.0],    #b (theta)  for ss flow temp
            a_gwflow=[0.001, 12.0],    # a (k) for gw flow temp
            b_gwflow=[0.001, 12.0],    # b (theta)  for gw flow temp
        )
        self.lat_adj_params_bound = [
             [-4, 7]                            # lateral temp adjusment
        ]
        self.PET_coef_bound = [
            [0.01, 1]  # PET_coef -> for converting PET to AET  ( Farshid added this param to the model)
        ]
        self.activation_sigmoid = torch.nn.Sigmoid()


    def atm_pressure(self, elev):
        ## from Jake's document
        # mmHg2mb = make_tensor(0.75061683)  # Unit conversion
        # mmHg2inHg = make_tensor(25.3970886)  # Unit conversion
        # P_sea = make_tensor(29.92126)  # Standard pressure ar sea level
        # A_g = make_tensor(9.80665)  # Acceleration due to gravity
        # M_a = make_tensor(0.0289644)  # Molar mass of air
        # R = make_tensor(8.31447)  # universal gas constant
        # T_sea = make_tensor(288.16)  # the standard temperature at sea level
        # P = (1 / mmHg2mb) * (mmHg2inHg) * (P_sea) * torch.exp(-A_g * M_a * elev / (R * T_sea))

        ## the code from stream_temp.f90
        P = 1013 - (0.1055 * elev)

        ## Note: both jakes and stream_temp are close to each other (error less than 0.013 for 99 basins)
        return P

    def atm_longwave_radiation_heat(self, T_a, e_a, shade_total, cloud_fraction, args):
        """
        :param T_a: air temperature in degree Celsius
        :param e_a: vapor pressure
        :return: Atmospheric longwave radiation
        """
        H_a = (
                (3.354939e-8 + 2.74995e-9 * e_a ** 0.5)
                * (1 - shade_total)
                * (1 + 0.17 * cloud_fraction ** 2)
                * (T_a + 273.16) ** 4
        )
        return H_a

    def stream_friction_heat(self, top_width, slope, Q):
        H_f = (
                9805 * Q * slope / top_width
        )  # Q is the seg_inflow (total flow entering a segment)
        return H_f

    def shortwave_solar_radiation_heat(self, albedo, H_sw, shade_total):
        """
        :param albedo: albedo or fraction reflected by stream , dimensionless
        :param H_sw: the clear sky solar radiation in watt per sq meter (seginc_swrad)
        :return: daily average clear sky, shortwave solar radiation for each segment
        """
        # solar_shade_fraction = make_tensor(args['STemp_default_params']['shade_fraction'])
        H_s = (1 - albedo) * (1 - shade_total) * H_sw
        return H_s

    def riparian_veg_longwave_radiation_heat(
            self, T_a, shade_fraction_riparian, args
    ):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = 5.670373 * torch.pow(
            make_tensor(10, device=args["device"]), (-8.0)
        ).to(device=args["device"])
        emissivity_veg = make_tensor(args["STemp_default_emissivity_veg"], device=args["device"])
        H_v = (
                emissivity_veg
                * St_Boltzman_ct
                * shade_fraction_riparian
                * torch.pow((T_a + 273.16), 4)
        )
        # H_v = emissivity_veg * St_Boltzman_ct * shade2[iGrid, :] * torch.pow((T_a + 273.16), 4)
        return H_v

    def ABCD_equations(
            self,
            T_a,
            swrad,
            e_a,
            E,
            elev,
            slope,
            top_width,
            up_inflow,
            T_g,
            shade_fraction_riparian,
            albedo,
            shade_total,
            args,
            cloud_fraction,
    ):
        """

        :param T_a: average daily air temperature
        :param swrad: solar radiation
        :param e_a: vapor pressure
        :param E: Free-water surface-evaporation rate (assumed to be PET, potet in PRMS)
        :param elev: average basin elevation
        :param slope: average stream slope (seg_slope)
        :param top_width: average top width of the stream
        :param up_inflow: is the discharge (variable seg_inflow) which is from upstream
        :return:
        """
        e_s = 6.108 * torch.exp((17.26939 * T_a) / (237.3 + T_a))
        # e_s = 6.108 * torch.exp((17.26939 * T_0) / (237.3 + T_0))
        P = self.atm_pressure(
            elev
        )  # calculating atmosphere pressure based on elevation
        # chacking vapor pressure with saturation vapor pressure
        denom = e_s - e_a
        mask_denom = denom.ge(0)
        # converting negative values to zero
        denom1 = denom * mask_denom.int().float()
        # adding 0.01 to zero values as it is denominator
        mask_denom2 = denom1.eq(0)
        denom2 = denom1 + 0.01 * mask_denom2.int().float()

        B_c = 0.00061 * P / denom2
        B_c1 = 0.00061 * P / (e_s - e_a)
        K_g = make_tensor(1.65, device=args["device"])
        delta_Z = make_tensor(args["STemp_default_delta_Z"], device=args["device"])
        # we don't need H_a, because we hae swrad directly from inputs
        H_a = self.atm_longwave_radiation_heat(
            T_a, e_a, shade_total, cloud_fraction, args=args
        )
        ###############
        H_f = self.stream_friction_heat(top_width=top_width, slope=slope, Q=up_inflow)
        H_s = self.shortwave_solar_radiation_heat(
            albedo=albedo, H_sw=swrad, shade_total=shade_total
        )  # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(
            T_a, shade_fraction_riparian, args=args
        )

        A = 5.4 * (10 ** (-8)) #torch.pow(make_tensor(np.full((T_a.shape), 10), device=), (-8))
        B = (10 ** 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (
                K_g / delta_Z
        )
        C = (10 ** 6) * E * B_c * 2.36
        # Todo: need to check 10**6. it is in fortran code but it is not in the document
        D = (
                H_a
                + H_s
                + H_v
                + 2495 * (10 ** 6) * E * ((B_c * T_a) - 1)
                + (T_g * K_g / delta_Z)
        )
        # D = H_a + swrad + H_v + 2495 * E * ((B_c * T_a) - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e, iter=50):
        def F(T_e):
            return (
                    A * torch.pow((T_e + 273.16), 4) - C * torch.pow(T_e, 2) + B * T_e - D
            )

        def Fprime(T_e):
            return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B

        ## solving the equation with Newton's method
        for i in range(iter):
            next_geuss = T_e - (F(T_e) / Fprime(T_e))
            T_e = next_geuss.clone().detach()    # Todo: should it be detach()

        return T_e

    def finding_K1_K2(self, A, B, C, D, T_e, NEARZERO, T_0):
        """
        :param A: Constant coming from equilibrium temp equation
        :param B: Constant coming from equilibrium temp equation
        :param C: Constant coming from equilibrium temp equation
        :param T_e: equilibrium temperature
        :param H_i: initial net heat flux at temperature T_o, of the upstream inflow
        :param T_o: initial water temperature
        :return: K1 (first order thermal exchange coefficient), K2 (second order coefficient)
        """
        H_i = A * torch.pow((T_0 + 273.16), 4) - C * torch.pow(T_0, 2) + B * T_0 - D
        K1 = 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B
        delt = T_0 - T_e
        denom = torch.pow(delt, 2)
        mask_denom = denom.le(NEARZERO)
        denom1 = denom + mask_denom.int().float()
        K2 = ((K1 * delt) - H_i) / denom1
        K2 = torch.where(torch.abs(delt) < NEARZERO,
                         torch.zeros(K2.shape).to(K2),
                         K2)
        # K2 = (-H_i + (K1 * (T_0 - T_e))) / denom1

        return K1, K2

    def srflow_ssflow_gwflow_portions(
            self,
            discharge,
            srflow_factor=make_tensor(0.40),
            ssflow_factor=make_tensor(0.3),
            gwlow_factor=make_tensor(0.3),
    ):
        srflow = srflow_factor * discharge
        ssflow = ssflow_factor * discharge
        gwflow = gwlow_factor * discharge
        return srflow, ssflow, gwflow

    def ave_temp_res_time(
            self, ave_air_temp, x, res_time, iGrid, iT, args, x_total_raw
    ):
        rho = x.shape[1]  # args['hyperparameters']['rho']
        tArray_Total = tRange2Array(args["tRange"])
        tArray_train = tRange2Array(args["t_train"])
        _, ind1, _ = intersect(tArray_Total, tArray_train)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        temp_res = res_time
        with torch.no_grad():
            temp_res1 = temp_res.int()
        A = res_time.repeat(1, rho)
        B = torch.reshape(A, (res_time.shape[0], rho, res_time.shape[1]))
        ave_air = torch.zeros(
            (
                args["batch_size"],
                args["rho"],
                res_time.shape[1],
            ),
            device=args["device"],
        )
        for i in range(res_time.shape[1]):
            for s, station in enumerate(iGrid):
                array = np.zeros((x.shape[1], temp_res1[s, i].item()), dtype=np.int32)
                for j in range(temp_res1[s, i].item()):
                    array[:, j] = np.arange(
                        (ind1_tensor[0] + iT_tensor[s] - j).item(),
                        (ind1_tensor[0] + iT_tensor[s] - j + x.shape[1]).item(),
                    )
                tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
                max_add = torch.sum(tmax_temp, dim=1)
                tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
                min_add = torch.sum(tmin_temp, dim=1)
                ave_air[s, :, i] = (max_add + min_add) / 2  # (2 * res_time[station, i])
        ave_air_temp = ave_air / B
        # return ave_air
        return ave_air_temp

    def x_sample_air_temp(self, iGrid, iT, lenF, args, x_total_raw, time_range):
        """
        :param iGrid:
        :param iT:
        :param lenF: maximum number of days that it is needed to be considered in average
        :return:
        """
        rho = args["rho"]
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        _, ind1, _ = intersect(tArray_Total, tArray_sample)
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = args["varT_SNTEMP"] + args["varC_SNTEMP"]
        ave_air = torch.zeros(
            (
                args["batch_size"],
                args["rho"],
                lenF,
            ),
            device=args["device"],
        )
        for s, station in enumerate(iGrid):
            array = np.zeros((rho, lenF), dtype=np.int32)
            for j in range(lenF):
                array[:, j] = np.arange(
                    (ind1_tensor[0] + iT_tensor[s] - j).item(),
                    (ind1_tensor[0] + iT_tensor[s] - j + rho).item(),
                )
            # array = np.flip(array, 1).copy()
            tmax_temp = x_total_raw[station, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw[station, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp
        return ave_air

    def x_sample_air_temp2(self, iGrid, iT, lenF, args, ave_air_total):
        rho = args["rho"]
        # ave_air = torch.zeros((len(iGrid), args["hyperparameters"]["rho"],
        #                        lenF),
        #                       device=args["device"])
        a = min(ave_air_total.shape[1], args["rho"])
        ave_air = torch.zeros((len(iGrid), a, lenF), device=args["device"])
        # array = np.array([np.arange(x, y) for x, y in zip(iT, iT + rho)])
        ave_air_temp = ave_air_total[iGrid, :, 0:lenF]
        for i in range(len(iGrid)):
            ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + a), :]
            # ave_air[i, :, :] = ave_air_temp[i, np.arange(iT[i], iT[i] + ave_air_total.shape[1]), :]

        # ave_air_temp = ave_air_total[:, iT : iT + rho, 0:lenF]
        # ave_air = ave_air_temp[iGrid, :, :]
        return ave_air

    def ave_temp_general(self, args, x_total_raw_tensor, time_range):
        vars = args["varT_NN"] + args["varC_NN"]
        lenF_max = np.maximum(
            args["res_time_lenF_srflow"],
            np.maximum(
                args["res_time_lenF_ssflow"],
                args["res_time_lenF_gwflow"],
            ),
        )
        tArray_Total = tRange2Array(args["tRange"])
        tArray_sample = tRange2Array(time_range)
        c, ind1, ind2 = np.intersect1d(tArray_sample, tArray_Total, return_indices=True)
        ave_air = torch.zeros(
            (x_total_raw_tensor.shape[0], len(tArray_sample), lenF_max), device="cpu"
        )

        array = np.zeros((len(ind2), lenF_max), dtype=np.int32)
        for j in range(lenF_max):
            array[:, j] = np.arange(
                (ind2[0] - j).item(), (ind2[0] - j + len(ind2)).item()
            )

        for s in range(x_total_raw_tensor.shape[0]):
            tmax_temp = x_total_raw_tensor[s, array, vars.index("tmax(C)")]
            tmin_temp = x_total_raw_tensor[s, array, vars.index("tmin(C)")]
            temp = (tmax_temp + tmin_temp) / 2
            ave_air[s, :, :] = temp

        return ave_air

    def res_time_gamma(self, a, b, lenF):
        # UH. a [time (same all time steps), batch, var]
        # a = torch.abs(a)
        if a.dim() == 2:
            m = a.shape
            a1 = a.repeat(1, lenF)
            b1 = b.repeat(1, lenF)
            alpha = F.relu(a1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.1
            beta = F.relu(b1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 3:
            m = a.shape
            a1 = a.repeat(1, 1, lenF)
            b1 = b.repeat(1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.1
            beta = F.relu(b1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 4:
            m = a.shape
            a1 = a.repeat(1, 1, 1, lenF)
            b1 = b.repeat(1, 1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.1
            beta = F.relu(b1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.5
            x = (
                torch.linspace(0.001, 20, lenF)
                .view(lenF, 1, 1, 1)
                .repeat(1, m[0], m[1], m[2])
            )
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        return www

    def UH_gamma(self, a, b, lenF=10):
        # UH. a [time (same all time steps), batch, var]
        m = a.shape
        # lenF = min(a.shape[0], lenF)
        w = torch.zeros([lenF, m[1], m[2]])
        aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.001  # minimum 0.1. First dimension of a is repeat
        theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.001  # minimum 0.5
        # t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
        # t = t.cuda(aa.device)
        t = (torch.linspace(0.001, 10, lenF).view(lenF, 1, 1).repeat(1, m[1], 1))
        t = t.to(aa.device)
        denom = (aa.lgamma().exp()) * (theta ** aa)
        mid = t ** (aa - 1)
        right = torch.exp(-t / theta)
        w = 1 / denom * mid * right
        ww = torch.cumsum(w, dim=0)
        www = ww / ww.sum(0)
        # w = w / w.sum(0)  # scale to 1 for each UH
        return www

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

        # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
        y = F.conv1d(xx, w, groups=groups, padding=padd, stride=1, bias=None)  # we don't need flip- double checked 08/18/2023
        if padd != 0:
            y = y[:, :, 0:-padd]
        return y.view(mm)

    def res_time_conv(self, x_sample, UH, bias, viewmode=1):
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
        # batch needs to be accommodated by channels and we make use of gr
        # ++++---------------------------------+
        #
        # oups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html
        if UH.shape[1] == 1:
            x = x_sample[:, 0:1, :]
            mm = x.shape
            nb = mm[0]
            m = UH.shape[-1]
            padd = m - 1
            if viewmode == 1:
                xx = x.view([1, nb, mm[-1]])
                w = UH.view([nb, 1, m])
                groups = nb

                # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
                # y = y[:, :, 0:-padd]

            x_sample1 = x_sample.permute(1, 0, 2)
            a = torch.arange(x_sample.shape[1])
            y = F.conv1d(
                x_sample1[a],
                torch.flip(w, [2]),
                groups=groups,
                padding=0,
                stride=1,
                bias=bias,
            )
            y = y.permute(1, 0, 2)
        elif UH.shape[1] > 1:
            w = torch.flip(UH, [2])
            y = x_sample * w
            y = y.sum(2)
            if bias is not None:
                y = y + bias
            y = y.unsqueeze(3)

        return y

    def ave_air_temp_calculation(self, args, x, a_ssflow, b_ssflow, a_gwflow, b_gwflow):
        varT = args["varT_temp_model"]
        nmul = args["nmul"]
        warm_up = args["warm_up"]
        air_sample_sr = x[args["res_time_lenF_gwflow"] - args["res_time_lenF_srflow"]:, :, 0:1]
        air_sample_ss = x[args["res_time_lenF_gwflow"] - args["res_time_lenF_ssflow"]:, :, 0:1]
        air_sample_bas_shallow = x[args["res_time_lenF_gwflow"] - args["res_time_lenF_bas_shallow"]:, :, 0:1]
        air_sample_gw = x[args["res_time_lenF_gwflow"] - args["res_time_lenF_gwflow"]:, :, 0:1]

        ave_air_sr = (air_sample_sr[args["res_time_lenF_srflow"]:, :, :]).repeat(1, 1, nmul)

        w_ssflow = self.UH_gamma(a=a_ssflow, b=b_ssflow,
                                 lenF=args["res_time_lenF_ssflow"])
        air_sample_ss = air_sample_ss.permute([1, 2, 0])  # dim:gage*var*time
        w_ssflow = w_ssflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_ss = self.UH_conv(air_sample_ss, w_ssflow)[:, :, args["res_time_lenF_ssflow"]:].permute(
            [2, 0, 1]).repeat(1, 1, nmul)

        w_gwflow = self.UH_gamma(a=a_gwflow, b=b_gwflow,
                                 lenF=args["res_time_lenF_gwflow"])
        air_sample_gw = air_sample_gw.permute([1, 2, 0])  # dim:gage*var*time
        w_gwflow = w_gwflow.permute([1, 2, 0])  # dim: gage*var*time
        ave_air_gw = self.UH_conv(air_sample_gw, w_gwflow)[:, :, args["res_time_lenF_gwflow"]:].permute(
            [2, 0, 1]).repeat(1, 1, nmul)

        ave_air_temp = torch.cat((ave_air_sr.unsqueeze(-1), ave_air_ss.unsqueeze(-1), ave_air_gw.unsqueeze(-1),
                                  ), dim=3)
        return ave_air_temp, w_ssflow, w_gwflow

    def lateral_flow_temperature(
            self, srflow, ssflow, gwflow, ave_air_temp, args, lat_temp_adj, NEARZERO=1e-6
    ):
        """
        :param srflow: surface runoff
        :param ssflow: subsurface runoff
        :param gwflow: qroundwaterflow
        :param res_time_srflow: residense time for surface runoff
        :param res_time_ssflow: residence time for subsurface runoff
        :param res_time_gwflow: residence time for groundwater flow
        :return: temperature of lateral flow
        """
        # with torch.no_grad():
        if args["res_time_type"] == "SNTEMP":
            ave_air_temp = torch.clamp(ave_air_temp, min=NEARZERO)

            srflow_temp = ave_air_temp[:, :, :, 0]  # .clone().detach()
            ssflow_temp = ave_air_temp[:, :, :, 1]  # .clone().detach()
            gwflow_temp = ave_air_temp[:, :, :, 2]  # .clone().detach()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        elif args["res_time_type"] == "van Vliet":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0] - 1.5
            mask_srflow_temp = srflow_temp.ge(0.0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 0]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(5.0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()
            gwflow_temp = torch.clamp(gwflow_temp, min=5.0)
            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )
        #
        # elif args["res_time_params"]["type"] is "Meisner":
        elif args["res_time_type"] == "Meisner":
            # look at http://dx.doi.org/10.1029/2018WR023250 page 4
            srflow_temp = ave_air_temp[:, :, :, 0]
            mask_srflow_temp = srflow_temp.ge(0)
            srflow_temp = srflow_temp * mask_srflow_temp.int().float()

            ssflow_temp = ave_air_temp[:, :, :, 1]
            mask_ssflow_temp = ssflow_temp.ge(0.0)
            ssflow_temp = ssflow_temp * mask_ssflow_temp.int().float()

            gwflow_temp = ave_air_temp[:, :, :, 2]
            mask_gwflow_temp = gwflow_temp.ge(0)
            gwflow_temp = gwflow_temp * mask_gwflow_temp.int().float()

            lat_flow_temp = torch.cat(
                (
                    srflow_temp.unsqueeze(-1),
                    ssflow_temp.unsqueeze(-1),
                    gwflow_temp.unsqueeze(-1),
                ),
                dim=3,
            )

        denom = gwflow + ssflow + srflow
        mask_denom = denom.eq(0.0)
        denom = denom + mask_denom.int().float()

        if args["lat_temp_adj"] == True:
            gwflow_temp = gwflow_temp + lat_temp_adj

        T_l = (
                      (gwflow * gwflow_temp) + (srflow * srflow_temp) + (ssflow * ssflow_temp)
              ) / denom

        mask_less_zero = T_l.le(NEARZERO)
        T_l[mask_less_zero] = 0.0
        return T_l, srflow_temp, ssflow_temp, gwflow_temp, lat_flow_temp

    def solving_SNTEMP_ODE_second_order(
            self,
            K1,
            K2,
            T_e,
            ave_width,
            q_l,
            L,
            args,
            T_0=make_tensor(0),
            Q_0=make_tensor(0.01),
            NEARZERO=1e-10,
    ):
        # # Note: as we assume that Q_0 is 0.01, we are always gaining flow with positive lateral flow or
        # # with zero lateral flow
        density = args["params_water_density"]
        c_w = args["params_C_w"]
        b = K1 * ave_width / (density * c_w)
        mask_q_l = q_l.eq(0)
        q_l_pos = q_l + mask_q_l.int().float()
        rexp = -1.0 * (b * L) / q_l_pos
        r = torch.exp(rexp)  # No idea why it is torch.exp in stream_Temp.f90 (the headwater part)

        delt = T_e - T_0
        mask_K1 = K1.eq(0)
        K1_masked = K1 + mask_K1.int().float()
        denom = 1 + ((K2 / K1_masked) * delt * (1 - r))
        denom = torch.where(torch.abs(denom) < NEARZERO,
                            torch.sign(denom) * NEARZERO,
                            denom)
        Tw = T_e - (delt * r / denom)
        Tw = torch.clamp(Tw, min=NEARZERO)
        return Tw

        #
        # density = args["params_water_density"]
        # c_w = args["params_C_w"]
        # mask_q_l = q_l.eq(0)
        # q_l_pos = q_l + mask_q_l.int().float()
        # b = q_l + ((K1 * ave_width) / (density * c_w))
        # mask_Q_0 = Q_0.eq(0)
        # Q_0_pos = Q_0 + mask_Q_0.int().float()
        # R_0 = torch.exp(-(b * L) / Q_0_pos)
        # mask_K1 = K1.eq(0)
        # K1_masked = K1 + mask_K1.int().float()
        # denom = 1 + ((K2 / K1_masked) * (T_e - T_0) * (1 - R_0))
        # mask_denom = denom.eq(0)
        # denom_masked = denom + mask_denom.int().float()
        # Tw = T_e - ((T_e - T_0) * R_0 / denom_masked)
        # return Tw

    def parameter_bounds(self, params, num, args):
        if params.dim() == 3:
            if num in args["static_params_list"]:
                out_temp = (
                        params[:, -1, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
                out = (
                    out_temp.view(-1, 1)
                    .repeat(1, params.shape[1])
                    .view(params.shape[0], params.shape[1])
                )
            elif num in args["semi_static_params_list"]:
                out_temp = self.semi_static_params(
                    params,
                    num,
                    interval=args["interval_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                    method=args["method_for_semi_static_param"][
                        args["semi_static_params_list"].index(num)
                    ],
                )
                out = (
                        out_temp
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
            else:  # dynamic
                out = (
                        params[:, :, num]
                        * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                        + args["paramCalLst"][num][0]
                )
        elif params.dim() == 2:
            out_temp = (
                    params[:, num]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = (
                out_temp.view(-1, 1)
                .repeat(1, args["hyperparameters"]["rho"])
                .view(
                    args["hyperparameters"]["batch_size"],
                    args["hyperparameters"]["rho"],
                )
            )
        return out

    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
                method=args["method_for_semi_static_param"][
                    args["semi_static_params_list"].index(num)
                ],
            )
            out = (
                    out_temp * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (args["paramCalLst"][num][1] - args["paramCalLst"][num][0])
                    + args["paramCalLst"][num][0]
            )
        return out

    def shade_modification(self, w1_shade, w2_shade, w3_shade, args):
        nmul = args["nmul"]
        A = list()
        C = list()
        # w1 = torch.ones((365, 1, 40), device=args["device"]) / 40
        w1 = torch.ones((w1_shade.shape[0], 1, 10), device=args["device"]) / 10
        for i in range(nmul):
            sh1 = w1_shade[:, :, i]
            sh2 = w2_shade[:, :, i]
            B1 = self.UH_conv(sh1.unsqueeze(-1).permute([0, 2, 1]), w1).permute([0, 2, 1])
            w1_shade_mov = torch.clamp(B1, min=0.0001, max=1.0)
            A.append(w1_shade_mov)

            B2 = self.UH_conv(sh2.unsqueeze(-1).permute([0, 2, 1]), w1).permute([0, 2, 1])
            w2_shade_mov = torch.clamp(B2, min=0.0001, max=1.0)
            C.append(w2_shade_mov)
        shade_fraction_riparian = torch.cat(A, dim=2)
        w2_shade_por_mov = torch.cat(C, dim=2)

        shade_fraction_riparian = torch.clamp(
            shade_fraction_riparian, min=0.0001, max=1.0
        )

        shade_fraction_topo = (1 - shade_fraction_riparian) * w2_shade_por_mov
        shade_fraction_topo = torch.clamp(shade_fraction_topo, min=0.0001, max=1.0)
        shade_total = shade_fraction_riparian + shade_fraction_topo
        shade_total = torch.clamp(shade_total, min=0.0001, max=1.0)

        return shade_fraction_riparian, shade_fraction_topo, shade_total

    def frac_modification(
            self, srflow_portion, ssflow_portion, gwflow_portion, Q_T, args,
    ):
        nmul = args["nmul"]
        A = list()
        Q = gwflow_portion * Q_T
        gw_filter_size = args["frac_smoothening_gw_filter_size"]
        wgw = (torch.ones(
            (gwflow_portion.shape[0], 1, gw_filter_size), device=args["device"]
        )
               / gw_filter_size
               )
        for i in range(nmul):
            Q_gw = Q[:, :, i]
            B = self.UH_conv(Q_gw.unsqueeze(-1).permute([0, 2, 1]), wgw).permute([0, 2, 1])
            # B = torch.flip(self.UH_conv(torch.flip(Q_gw.unsqueeze(-1).permute([0, 2, 1]), [2]), wgw).permute([0, 2, 1]), [1])
            Q_gw_por_mov = torch.max(
                torch.min(B, Q_T[:, :, 0].unsqueeze(-1)), make_tensor(0.0)
            )
            gwflow_portion_new = Q_gw_por_mov / (
                    Q_T[:, :, 0].unsqueeze(-1) + 0.001
            )  # 0.001 is for not having nan values
            gwflow_portion_new = torch.clamp(gwflow_portion_new, min=0.001, max=1.0)
            A.append(gwflow_portion_new)
        gwflow_portion_new = torch.cat(A, dim=2)
        remain_frac = 1 - gwflow_portion_new

        if args["res_time_type"] != "Meisner":
            srflow_portion_new = srflow_portion * remain_frac
            ssflow_portion_new = remain_frac - srflow_portion_new
        else:
            srflow_portion_new = remain_frac
            ssflow_portion_new = ssflow_portion * 0.0 + 0.0001
        srflow_percentage = torch.clamp(srflow_portion_new, min=0.0001, max=1.0)
        ssflow_percentage = torch.clamp(ssflow_portion_new, min=0.0001, max=1.0)
        gwflow_percentage = torch.clamp(gwflow_portion_new, min=0.0001, max=1.0)

        return srflow_percentage, ssflow_percentage, gwflow_percentage

    def frac_modification2(self, portion, Q_T, filter_size, args):
        nmul = args["nmul"]
        A = list()
        w = torch.ones((portion.shape[0], 1, filter_size), device=args["device"]) / filter_size
        # a = torch.arange(filter_size, device=args["device"], dtype=torch.float32) * (1.0 - 0.1) / filter_size + 0.1
        # w = (torch.flip(a / a.sum(), [0])).repeat(portion.shape[0], 1).unsqueeze(-1).permute([0, 2, 1])
        Q = portion * Q_T
        for i in range(nmul):
            Q_w = Q[:, :, i]
            B = self.UH_conv(Q_w.unsqueeze(-1).permute([0, 2, 1]), w).permute([0, 2, 1])
            # B = torch.flip(self.UH_conv(torch.flip(Q_gw.unsqueeze(-1).permute([0, 2, 1]), [2]), wgw).permute([0, 2, 1]), [1])
            Q_w_por_mov = torch.max(
                torch.min(B, Q_T[:, :, i].unsqueeze(-1)), make_tensor(0.0)
            )
            wflow_portion_new = Q_w_por_mov / (
                    Q_T[:, :, i].unsqueeze(-1) + 0.001
            )  # 0.001 is for not having nan values
            wflow_portion_new = torch.clamp(wflow_portion_new, min=0.0001, max=1.0)
            A.append(wflow_portion_new)
        wflow_portion_new = torch.cat(A, dim=2)

        wflow_percentage = torch.clamp(wflow_portion_new, min=0.0001, max=1.0)

        return wflow_percentage

    def semi_static_params(self, params, param_no, interval=30, method="average"):
        # seperate the piece for each interval
        param = params[:, :, param_no]
        no_basins, no_days = param.shape
        interval_no = math.floor(no_days / interval)
        remainder = no_days % interval
        param_name_list = list()
        if method == "average":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = torch.mean(param[:, 0:remainder], 1, keepdim=True).repeat(
                        (1, remainder)
                    )
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = torch.mean(
                    param[
                    :,
                    ((i * interval) + remainder): (
                            ((i + 1) * interval) + remainder
                    ),
                    ],
                    1,
                    keepdim=True,
                ).repeat((1, interval))
                param_name_list.append(param_name)
        elif method == "single_val":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = (param[:, 0:1]).repeat((1, remainder))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = (
                    param[
                    :,
                    (((i) * interval) + remainder): (((i) * interval) + remainder)
                                                    + 1,
                    ]
                ).repeat((1, interval))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

    def param_bounds(self, params, num, args, bounds):
        nmul = args["nmul"]
        if num in args["static_params_list_SNTEMP"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list_SNTEMP"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param_SNTEMP"][
                    args["semi_static_params_list_SNTEMP"].index(num)
                ],
                method=args["method_for_semi_static_param_SNTEMP"][
                    args["semi_static_params_list_SNTEMP"].index(num)
                ],
            )
            out = (
                    out_temp * (bounds[1] - bounds[0])
                    + bounds[0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
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

    def change_param_range(self, param, bounds):
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

    def forward(self, x, airT_memory, c, params_raw, conv_params_temp, args, PET, source_flows):
        NEARZERO = args["NEARZERO"]
        warm_up = args["warm_up"]
        nmul = args["nmul"]
        varT = args["varT_temp_model"]
        varC = args["varC_temp_model"]
        # source flow components
        ## to make sure there are only three flow sources:
        if len(source_flows.keys()) != 3:
            print("inconsistency between hydrology model and temp model")
            exit()
        srflow = source_flows["srflow"]
        ssflow = source_flows["ssflow"]
        gwflow = source_flows["gwflow"]

        Q_tot = srflow + ssflow + gwflow

        # initialization of the params
        Nstep = x.shape[0] - warm_up

        ## scale the parameters
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = self.change_param_range(param=params_raw[warm_up:, :, num, :],
                                                         bounds=self.parameters_bound[param])

        if args["lat_temp_adj"] == True:
            lat_temp_params_raw = params_raw[warm_up:, :, -1, :]
            # else:
            #     lat_temp_params_raw = params_raw[-1, :, -1, :]

            params_dict_raw["lat_temp_adj"] = self.change_param_range(param=lat_temp_params_raw,
                                                                  bounds=self.lat_adj_params_bound[0])
        else:
            params_dict_raw["lat_temp_adj"] = 0.0 * params_dict_raw["w1_shade"]

        # implementing static and dynamic parametrization
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key in args["dyn_params_list_temp"]:  ## it is a static parameter
                params_dict[key] = params_dict_raw[key]
            else:
                params_dict[key] = params_dict_raw[key][-1, :, :]

        if args["routing_temp_model"] == True:   # makes it consistent with airT_memory dimension
            for num, param in enumerate(self.conv_temp_model_bound.keys()):
                rep = max(args["res_time_lenF_ssflow"],
                          args["res_time_lenF_gwflow"],
                          Nstep)
                params_dict[param] = self.change_param_range(param=conv_params_temp[:, num],
                                                             bounds=self.conv_temp_model_bound[param]).repeat(rep,
                                                                                                              1).unsqueeze(
                    -1)



        # Tmaxf = x[warm_up:, :, varT.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # Tminf = x[warm_up:, :, varT.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # mean_air_temp = (Tmaxf + Tminf) / 2
        # dayl = x[warm_up:, :, varT.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        # up_inflow = torch.zeros_like(Tmaxf)
        vp = 0.01 * x[warm_up:, :, varT.index("vp(Pa)")].unsqueeze(-1).repeat(1, 1, nmul)  # converting to mbar
        # swrad = (x[warm_up:, :, varT.index("srad(W/m2)")] * x[warm_up:, :, varT.index("dayl(s)")] / 86400).unsqueeze(
        #     -1).repeat(1, 1, nmul)   # this one is when srad(W/m2) is directly from daymet  which is only for daylight
        swrad = x[warm_up:, :, varT.index("srad(W/m2)")].unsqueeze(-1).repeat(1, 1, nmul)
        cloud_fraction = x[warm_up:, :, varT.index("ccov")].unsqueeze(-1).repeat(1, 1, nmul)
        elev = c[:, varC.index("ELEV_MEAN_M_BASIN")].unsqueeze(-1).repeat(swrad.shape[0], 1, nmul)
        slope = 0.01 * c[:, varC.index("SLOPE_PCT")].unsqueeze(-1).repeat(swrad.shape[0], 1, nmul)  # adding the percentage, it is a watershed slope not a stream slope
        # stream_density = x[:, :, vars.index("STREAMS_KM_SQ_KM")]
        # stream_length = 1000 * (stream_density * x[:, :, vars.index("DRAIN_SQKM")]).unsqueeze(-1).repeat(1,1,nmul)
        # stream_length = x[:, :, vars.index("stream_length_artificial")]
        # stream_length = x[:, :, vars.index("NHDlength_tot(m)")].unsqueeze(-1).repeat(1,1,nmul)
        stream_length = c[:, varC.index("stream_length_square")].unsqueeze(-1).repeat(swrad.shape[0], 1, nmul) * 1000.0   # km to meter
        # basin_area = x[:, :, vars.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1,1,nmul)

        # t_monthly = x[:, :, vars.index("t_monthly(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # albedo = args["STemp_default_albedo"]

        # hamon PET equation. We can add other methods too, such as Penman monteith
        # if args["potet_module"] == "potet_hamon":
        #     # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.004, 0.008], ndays=No_days, nmul=args["nmul"])
        #     PET = get_potet(
        #         args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=0.008 #PET_coef
        #     )
        # elif args["potet_module"] == "potet_hargreaves":
        #     day_of_year = x[warm_up:, :, varT.index("dayofyear")].unsqueeze(-1).repeat(1, 1, nmul)
        #     lat = c[:, varC.index("lat")].unsqueeze(0).unsqueeze(-1).repeat(dayl.shape[0], 1, nmul)
        #     # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
        #     #                                 nmul=args["nmul"])
        #
        #     PET = get_potet(
        #         args=args, tmin=Tminf, tmax=Tmaxf,
        #         tmean=mean_air_temp, lat=lat,
        #         day_of_year=day_of_year
        #     )
        # elif args["potet_module"] == "dataset":
        #     PET = x[warm_up:, :, varT.index(args["potet_dataset_name"])].unsqueeze(-1).repeat(1, 1, nmul)
        #
        # #converting PET to AET and converting mm/day to m/sec
        # AET = params_dict["PET_coef"] * PET * (1 / (1000 * 86400))   # converting mm/day to m/sec



        # d = torch.pow(q * n * (q + 1) / (p * torch.pow(slope, 0.5)), (3 / (5 + 3 * q)))

        # d = torch.pow(width_coef_factor * obsQ / (width_coef_pow * torch.pow(basin_area, width_A_coef)), width_exp)
        # top_width = 5 + p * torch.pow(d, q)
        # top_width = width_coef_factor * obsQ + width_coef_pow
        # top_width = width_coef_factor * (obsQ ** width_coef_pow) + 0.5
        # top_width = width_coef_factor * torch.pow(obsQ[:, warm_up:,:] + 0.0001, width_coef_pow) + 0.2
        top_width = params_dict["width_coef_factor"] * torch.pow(Q_tot + 0.0001, params_dict["width_coef_pow"]) + 0.2
        # top_width = make_tensor(torch.ones(width_coef_factor.shape) * 10.0, device=args["device"])
        # if p.dim() == 3:
        #     top_width = p * torch.pow(basin_area, q)
        # elif p.dim() == 2:
        #     top_width = p * torch.pow(basin_area, q)

        # masking surface runoff fraction with precipitation.
        # if there is not any precipitaton, it cannot be more than 0.01

        # if args["frac_smoothening_mode"] == True:
        #     gwflow_percentage = self.frac_modification2(gwflow_portion, obsQ,
        #                                                 args["frac_smoothening_gw_filter_size"], args)
        #     if args["res_time_type"] == "Meisner":
        #         ssflow_percentage = 0.0 * ssflow_portion + 0.0001
        #     else:
        #         ssflow_por_remain = self.frac_modification2(ssflow_portion,
        #                                                     obsQ * (1 - gwflow_percentage),
        #                                                     args["frac_smoothening_ss_filter_size"], args)
        #         ssflow_percentage = ssflow_por_remain * (1 - gwflow_percentage)
        # else:
        #     gwflow_percentage = gwflow_portion
        #     if args["res_time_type"] == "Meisner":
        #         ssflow_percentage = 0.0 * ssflow_portion + 0.0001
        #     else:
        #         ssflow_percentage = (1.0 - gwflow_percentage) * ssflow_portion
        # srflow_percentage = 1.0 - gwflow_percentage - ssflow_percentage


        # making the flow percentages:
        # if args["res_time_type"] == "Meisner":
        #     ssflow_portion = 0.0001 * ssflow_portion
        # srflow_percentage = srflow_portion
        # ssflow_percentage = (1 - srflow_percentage) * ssflow_portion
        # gwflow_percentage = 1 - srflow_percentage - ssflow_percentage
        #
        # if args["frac_smoothening_mode"] == True:
        #     srflow_percentage, ssflow_percentage, gwflow_percentage = self.frac_modification(srflow_percentage,
        #                                                                                      ssflow_percentage,
        #                                                                                      gwflow_percentage,
        #                                                                                      obsQ,
        #                                                                                      args)



        # total shade (solar shade) is accumulative shade of vegetation and topography
        params_dict["shade_fraction_riparian"] = params_dict["w1_shade"]
        # shade_fraction_riparian = 0.01 * x[:, :, vars.index("RIP100_FOREST")].unsqueeze(-1).repeat(1, 1, nmul)
        params_dict["shade_fraction_topo"] = params_dict["w2_shade"] * (1 - params_dict["shade_fraction_riparian"])
        params_dict["shade_total"] = params_dict["shade_fraction_riparian"] + params_dict["shade_fraction_topo"]
        if args["shade_smoothening"] == True:
            (
                params_dict["shade_fraction_riparian"],
                params_dict["shade_fraction_topo"],
                params_dict["shade_total"],
            ) = self.shade_modification(params_dict["shade_fraction_riparian"],
                                        params_dict["shade_fraction_topo"],
                                        params_dict["shade_total"],
                                        args)

        # calculating the temperature of source flow using convolution
        ave_air_temp, w_ssflow, w_gwflow = self.ave_air_temp_calculation(
            args, airT_memory, params_dict["a_ssflow"], params_dict["b_ssflow"], params_dict["a_gwflow"], params_dict["b_gwflow"])
        # modifying the source flow temperature with physical constraints
        T_0, srflow_temp, ssflow_temp, gwflow_temp, ave_air_temp_new = self.lateral_flow_temperature(
            srflow=srflow,
            ssflow=ssflow,
            gwflow=gwflow,
            ave_air_temp=ave_air_temp,
            args=args,
            lat_temp_adj=params_dict["lat_temp_adj"],
        )

        A, B, C, D = self.ABCD_equations(
            T_a=T_0,
            swrad=swrad,
            e_a=vp,
            elev=elev,
            slope=slope,
            top_width=top_width,
            up_inflow=0.0,  # should be obsQ
            E=PET,  # up_inflow
            T_g=gwflow_temp,
            shade_fraction_riparian=params_dict["shade_fraction_riparian"],
            albedo=params_dict["albedo"],
            shade_total=params_dict["shade_total"],
            args=args,
            cloud_fraction=cloud_fraction,
        )
        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D, T_e=T_0)
        K1, K2 = self.finding_K1_K2(
            A=A, B=B, C=C, D=D, T_e=T_e, NEARZERO=NEARZERO, T_0=T_0
        )

        Q_0 = make_tensor(np.full((swrad.shape), 0.000001))
        # Q_0 = make_tensor(np.full((obsQ.shape[0], obsQ.shape[1]), 0))

        # T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, T_l, T_e, ave_width=top_width,
        #                                            q_l=obsQ, L=stream_length, args=args,
        #                                            T_0=T_0, Q_0=Q_0)

        # writing the original fortran code here
        # they assumed if Q_upstream==0 and q_lat > 0, they assume Q_upstream=q_lat, and q_lat=0
        # it prevents from dividing to zero
        T_w = self.solving_SNTEMP_ODE_second_order(
            K1,
            K2,
            T_e,
            ave_width=top_width,
            q_l=Q_tot,
            L=stream_length,
            args=args,
            T_0=T_0,  # because there is no upstream flow, it is lateral flow temp
            Q_0=0.0,
        )
        # get rid of negative values:
        # T_w = torch.relu(T_w)

        return dict(temp_sim=T_w.mean(-1, keepdim=True),
                    srflow_temp=srflow_temp.mean(-1, keepdim=True),
                    ssflow_temp=ssflow_temp.mean(-1, keepdim=True),
                    gwflow_temp=gwflow_temp.mean(-1, keepdim=True),
                    w_gwflow=w_gwflow.permute([2, 0, 1]),
                    w_ssflow=w_ssflow.permute([2, 0, 1]),
                    # AET_temp=AET.mean(-1, keepdim=True),
                    # PET_temp=PET.mean(-1, keepdim=True) * (1 / (1000 * 86400)),   # converting to m/sec, same as AET
                    shade_fraction_riparian=params_dict["shade_fraction_riparian"].mean(-1, keepdim=True),
                    shade_fraction_topo=params_dict["shade_fraction_topo"].mean(-1, keepdim=True),
                    top_width=top_width.mean(-1, keepdim=True),
                    width_coef_factor=params_dict["width_coef_factor"].mean(-1, keepdim=True),
                    width_coef_pow=params_dict["width_coef_pow"].mean(-1, keepdim=True),
                    # PET_coef_temp=params_dict["PET_coef"].mean(-1, keepdim=True),
                    lat_temp_adj=params_dict["lat_temp_adj"].mean(-1, keepdim=True),
                    a_ssflow=params_dict["a_ssflow"].mean(-1, keepdim=True),
                    b_ssflow=params_dict["b_ssflow"].mean(-1, keepdim=True),
                    a_gwflow=params_dict["a_gwflow"].mean(-1, keepdim=True),
                    b_gwflow=params_dict["b_gwflow"].mean(-1, keepdim=True)
                    )

# from dataclasses import dataclass
# @dataclass
# class PGMLOutput:
#     T_w: torch.Tensor
#     ave_air_temp: torch.Tensor
#     w_gwflow: torch.Tensor
#     w_ssflow: torch.Tensor
#     source_temps: torch.Tensor
#     SNTEMP_outs: torch.Tensor
#
#     def write(self):
#
#
# PGMLOutput(T_w.mean(-1, keepdim=True),
#                 ave_air_temp_new.mean(2, keepdim=True).squeeze(2),
#                 w_gwflow.permute([2, 0, 1]),
#                 w_ssflow.permute([2, 0, 1]),
#                 source_temps,
#                 SNTEMP_outs
#                 )
#
# PGMLOutput.write()
