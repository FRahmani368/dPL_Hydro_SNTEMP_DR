import numpy as np
import torch
import torch.nn as nn
import datetime
from core.read_configurations import config
#from rnn import CudnnLstmModel
from core.small_codes import make_tensor, tRange2Array, intersect

class MLP(nn.Module):
    def __init__(self,
                 args):
        super(MLP, self).__init__()
        self.seq_lin_layers = nn.Sequential(
            nn.Linear(len(args['optData']['varC']), \
                      args['seq_lin_layers']['hidden_size']),
            # nn.ReLU(),
            nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size']),
            # nn.ReLU(),
            nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size']),
            nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size']),
            # nn.ReLU(),
            nn.Linear(args['seq_lin_layers']['hidden_size'], len(args['one_param_target'])),
            # nn.ReLU()
        )
        self.L1 = nn.Linear(len(args['optData']['varC']), \
                      args['seq_lin_layers']['hidden_size'])
        self.L2 = nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size'])
        self.L3 = nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size'])
        self.L4 = nn.Linear(args['seq_lin_layers']['hidden_size'], len(args['one_param_target']))

        # self.lstm = CudnnLstmModel(
        #     nx=input.shape[2],
        #     ny=len(args['params_target']),
        #     hiddenSize=args['hyperprameters']['hidden_size'],
        #     dr=args['hyperprameters']['dropout'])
        #
        # self.stream_temp_eq = stream_temp_eq(args)
        self.activation = torch.nn.Sigmoid()
        self.activation_tanh = torch.nn.Tanh()
    def forward(self, x):
        # out = self.seq_lin_layers(x)
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        # out1 = torch.abs(out)
        out = self.activation(out)
        return out



def str_to_datetime(t):
    if isinstance(t, datetime.datetime):
        return t
    elif isinstance(t, str):
        year, month, day = t.split("-")
        t_datetime_format = datetime.datetime(int(year), int(month), int(day))
        return t_datetime_format

class STREAM_TEMP_EQ(nn.Module):
    def __init__(self,
                 args,
                 x_total_raw):
        super(STREAM_TEMP_EQ, self).__init__()
        self.args = args
        self.x_total_raw = x_total_raw
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
        self.res_time_srflow = nn.Parameter(torch.zeros((99, 1)) + 1)
        self.res_time_ssflow = nn.Parameter(torch.zeros((99, 1)) + 1)
        self.res_time_gwflow = nn.Parameter(torch.zeros((99, 1)) + 1)
        # self.shade = nn.Parameter(torch.load("/home/fzr5082/PGML_STemp_results/data/shade.pt"))
    def atm_pressure(self, elev):
        mmHg2mb = make_tensor(0.75061683)  # Unit conversion
        mmHg2inHg = make_tensor(25.3970886)  # Unit conversion
        P_sea = make_tensor(29.92126)  # Standard pressure ar sea level
        A_g = make_tensor(9.80665)  # Acceleration due to gravity
        M_a = make_tensor(0.0289644)  # Molar mass of air
        R = make_tensor(8.31447)  # universal gas constant
        T_sea = make_tensor(288.16)  # the standard temperature at sea level
        P = (1 / mmHg2mb) * (mmHg2inHg) * (P_sea) * torch.exp(-A_g * M_a * elev / (R * T_sea))
        return P

    def atm_longwave_radiation_heat(self, T_a, e_a):
        """
        :param T_a: air temperature in degree Celsius
        :param e_a: vapor pressure
        :return: Atmospheric longwave radiation
        """
        emissivity_air = 0.61 + 0.05 * torch.pow(e_a, 0.5)
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(make_tensor(10), (-8.0))  # (J/s*m^2 * K^4)
        longwave_reflect_frac = make_tensor(self.args['STemp_default_params']['longwave_reflect_fraction'])
        shade_fraction = make_tensor(self.args['STemp_default_params']['shade_fraction'])
        cloud_fraction = make_tensor(self.args['STemp_default_params']['cloud_fraction'])
        H_a = (
                (1 - longwave_reflect_frac) * (1 - shade_fraction) * (1 + 0.17 * cloud_fraction) *
                emissivity_air * St_Boltzman_ct * torch.pow((T_a + 273.16), 4)
        )
        return H_a

    def stream_friction_heat(self, top_width, slope, Q):
        H_f = 9805 * Q * slope /top_width    # Q is the seg_inflow (total flow entering a segment)
        return H_f

    def shortwave_solar_radiation_heat(self, albedo, H_sw):
        """
        :param albedo: albedo or fraction reflected by stream , dimensionless
        :param H_sw: the clear sky solar radiation in watt per sq meter (seginc_swrad)
        :return: daily average clear sky, shortwave solar radiation for each segment
        """
        shade_fraction = make_tensor(self.args['STemp_default_params']['shade_fraction'])
        H_s = (1 - albedo) * (1 - shade_fraction) * H_sw
        return H_s

    def riparian_veg_longwave_radiation_heat(self, T_a, iGrid):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(make_tensor(10), (-8.0))  # (J/s*m^2 * K^4)
        emissivity_veg = make_tensor(self.args['STemp_default_params']['emissivity_veg'])
        shade_fraction_riparian = make_tensor(self.args['STemp_default_params']['shade_fraction_riparian'])
        # for i in range(iGrid.shape[0]):
        #     shade_fraction_riparian[i, 0] = self.shade_fraction[iGrid[i]][0]
        # shade2 = self.shade.repeat(1, T_a.shape[1])
        # shade2 = torch.sigmoid(shade2)
        # shade_fraction_riparian = self.shade_fraction.clone().repeat(1, T_a.shape[1])
        # shade_fraction_riparian_2 = shade_fraction_riparian.repeat(1, T_a.shape[1])
        # shade_fraction_riparian = torch.empty((self.args['hyperparameters']['batch_size'],
        #                                        self.args['hyperparameters']['rho']), device=self.args['device'])
        H_v = emissivity_veg * St_Boltzman_ct * shade_fraction_riparian * torch.pow((T_a + 273.16), 4)
        # H_v = emissivity_veg * St_Boltzman_ct * shade2[iGrid, :] * torch.pow((T_a + 273.16), 4)
        return H_v

    def ABCD_equations(self, T_a, swrad, e_a, E, elev, slope, top_width, inflow, T_g, iGrid):
        """

        :param T_a: average daily air temperature
        :param swrad: solar radiation
        :param e_a: vapor pressure
        :param E: Free-water surface-evaporation rate (assumed to be PET, potet in PRMS)
        :param elev: average basin elevation
        :param slope: average stream slope (seg_slope)
        :param top_width: average top width of the stream
        :param inflow: is the discharge (variable seg_inflow)
        :return:
        """
        e_s = 6.11 * torch.exp((17.27 * T_a) / (273.16 + T_a))
        P = self.atm_pressure(elev)  # calculating atmosphere pressure based on elevation
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
        delta_Z = make_tensor(self.args['STemp_default_params']['delta_Z'])
        # we don't need H_a, because we hae swrad directly from inputs
        H_a = self.atm_longwave_radiation_heat(T_a, e_a)
        ###############
        H_f = self.stream_friction_heat(top_width=top_width, slope=slope, Q=inflow)
        H_s = self.shortwave_solar_radiation_heat(albedo=self.args['STemp_default_params']['albedo'], H_sw=swrad) # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(T_a, iGrid)

        A = 5.4 * torch.pow(make_tensor(np.full((T_a.shape[0], T_a.shape[1]), 10)), (-8))
        B = torch.pow(make_tensor(10), 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (K_g / delta_Z)
        C = torch.pow(make_tensor(10), 6) * E * B_c * 2.36
        # D = H_a + H_f + H_s + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)
        # D = H_a + H_f + H_s + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)
        D = H_a + swrad + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e=make_tensor(20), iter=50):
        def F(T_e):
            return A * torch.pow((T_e + 273.16), 4) - C * torch.pow(T_e, 2) + B * T_e - D
        def Fprime(T_e):
            return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B
        ## solving the equation with Newton's method
        for i in range(iter):
            next_geuss = T_e - (F(T_e)/Fprime(T_e))
            T_e = next_geuss
        return T_e

    def finding_K1_K2(self, A, B, C, D, T_e, T_0=make_tensor(0)):
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
        denom = (torch.pow((T_e - T_0), 2))
        mask_denom = denom.eq(0)
        denom1 = denom + mask_denom.int().float()
        K2 = (H_i - (K1 * (T_e - T_0))) / denom1
        return K1, K2

    def srflow_ssflow_gwflow_portions(self,
                             discharge,
                             srflow_factor=make_tensor(0.40),
                             ssflow_factor=make_tensor(0.3),
                             gwlow_factor=make_tensor(0.3)):
        srflow = srflow_factor * discharge
        ssflow = ssflow_factor * discharge
        gwflow = gwlow_factor * discharge
        return srflow, ssflow, gwflow

    def ave_temp_res_time(self, ave_air_temp, x, res_time, iGrid, iT):
        rho = x.shape[1]  # args['hyperparameters']['rho']
        tArray_Total = tRange2Array(self.args['optData']['tRange'])
        tArray_train = tRange2Array(self.args['optData']['t_train'])
        _, ind1, _ = (intersect(tArray_Total, tArray_train))
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = self.args['optData']['varT'] + self.args['optData']['varC']
        temp_res = res_time
        with torch.no_grad():
            temp_res1 = temp_res.int()
        A = res_time.repeat(1, rho)
        B = torch.reshape(A, (res_time.shape[0], rho, res_time.shape[1]))
        ave_air = torch.zeros((self.args['hyperparameters']['batch_size'], self.args["hyperparameters"]["rho"],
                               res_time.shape[1]),
                               device=self.args["device"])
        for i in range(res_time.shape[1]):
            for s, station in enumerate(iGrid):
                array = np.zeros((x.shape[1], temp_res1[s, i].item()), dtype=np.int32)
                for j in range(temp_res1[s, i].item()):
                    array[:, j] = np.arange((ind1_tensor[0] + iT_tensor[s] - j).item(),
                                            (ind1_tensor[0] + iT_tensor[s] - j + x.shape[1]).item())
                tmax_temp = self.x_total_raw[station, array, vars.index("tmax(C)")]
                max_add = torch.sum(tmax_temp, dim=1)
                tmin_temp = self.x_total_raw[station, array, vars.index("tmin(C)")]
                min_add = torch.sum(tmin_temp, dim=1)
                ave_air[s, :, i] = (max_add + min_add) / 2  # (2 * res_time[station, i])
        ave_air_temp = ave_air / B
        # return ave_air
        return ave_air_temp

    def lateral_flow_temperature(self, srflow, ssflow, gwflow, ave_air_temp):
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
        srflow_temp = ave_air_temp[:, :, 0]  #.clone().detach()
        ssflow_temp = ave_air_temp[:, :, 1]  #.clone().detach()
        gwflow_temp = ave_air_temp[:, :, 2]  #.clone().detach()

        T_l = ((gwflow * ave_air_temp[:, :, 2] + srflow * ave_air_temp[:, :, 0] +
                ssflow * ave_air_temp[:, :, 1]) / (gwflow + ssflow + srflow))
        return T_l, srflow_temp, ssflow_temp, gwflow_temp

    def solving_SNTEMP_ODE_second_order(self, K1, K2, T_l, T_e, ave_width, q_l, L,
                                        T_0=make_tensor(0), Q_0=make_tensor(0.01)):
        # Note: as we assume that Q_0 is 0.01, we are always gaining flow with positive lateral flow or
        # with zero lateral flow
        density = self.args['params']['water_density']
        c_w = self.args['params']['C_w']
        # a = (q_l * T_l) + ((K1 * ave_width)/(self.args['params']['water_density'] * self.args['params']['C_w'])) * T_e
        # b = q_l + (K1 * ave_width) / (self.args['params']['water_density'] * self.args['params']['C_w'])


        # for positive lateral flow
        mask_pos = q_l.ge(0)
        a_pos = (q_l * mask_pos.int().float() * T_l * mask_pos.int().float()) + (((K1 * mask_pos.int().float()
                                                                                   * ave_width * mask_pos.int().float())
                                                   / (density * c_w)) * T_e * mask_pos.int().float())
        b_pos1 = q_l * mask_pos.int().float() + ((K1 * mask_pos.int().float() * ave_width * mask_pos.int().float()
                                                 ) / (density * c_w))
        # to get rid of having zero denominator in the next line
        mask_b_pos = b_pos1.eq(0)
        b_pos2 = mask_b_pos.int().float() * b_pos1 * 0.01 + b_pos1
        # b_pos[b_pos == 0] = 0.01

        Tprime_e_pos = a_pos / b_pos2
        R_pos = torch.pow((1 + (q_l * mask_pos.int().float() * L * mask_pos.int().float() /
                                Q_0 * mask_pos.int().float())),
                                -(ave_width * mask_pos.int().float() / q_l * mask_pos.int().float()))
        # with torch.no_grad():
        #     denom = (1 + ((K2 * mask_pos.int().float()/K1 * mask_pos.int().float()) *
        #                   (Tprime_e_pos - T_0 * mask_pos.int().float()) * (1 - R_pos)))
        #     denom[denom==0] = 0.01
        # Tw_pos = Tprime_e_pos - ((Tprime_e_pos - T_0 * mask_pos.int().float()) * R_pos / denom)

        denom_pos = (1 + ((K2 * mask_pos.int().float() / K1 * mask_pos.int().float()) *
                      (Tprime_e_pos - T_0 * mask_pos.int().float()) * (1 - R_pos)))
        mask_denom = denom_pos.eq(0)
        denom_pos2 = denom_pos + mask_denom.int().float() * 0.01
        Tw_pos = Tprime_e_pos - ((Tprime_e_pos - T_0 * mask_pos.int().float()) * R_pos / denom_pos2)

        # for zero lateral flow
        mask_zero = q_l.eq(0)
        Tprime_e_zero = T_e * mask_zero.int().float()
        R_zero = torch.exp(-ave_width * mask_zero.int().float() * L * mask_zero.int().float() /
                           Q_0 * mask_zero.int().float())
        # with torch.no_grad():
        #     denom = (1 + ((K2 * mask_zero.int().float()/K1 * mask_zero.int().float()) *
        #                   (Tprime_e_zero - T_0 * mask_zero.int().float()) * (1 - R_zero)))
        #     denom[denom==0] = 0.01
        # Tw_zero = Tprime_e_zero - ((Tprime_e_zero - T_0 * mask_zero.int().float()) * R_zero / denom)

        denom_zero = (1 + ((K2 * mask_zero.int().float() / K1 * mask_zero.int().float()) *
                      (Tprime_e_zero - T_0 * mask_zero.int().float()) * (1 - R_zero)))
        mask_denom_zero = denom_zero.eq(0)
        denom_zero2 = denom_zero + mask_denom_zero.int().float() * 0.01
        Tw_zero = Tprime_e_zero - ((Tprime_e_zero - T_0 * mask_zero.int().float()) * R_zero / denom_zero2)



        # for negative q_l
        mask_q_l = q_l < 0
        if mask_q_l.sum() > 0:
            print("negative q_l")
            exit()
        # Tw_pos and Tw_zero do not have any common cell which means, in each cell,
        # at least one of them is zero. That's why we can add them up.
        T_w = Tw_pos + Tw_zero
        return T_w

    def forward(self, x, iGrid, iT, ave_air_temp):
        vars = self.args['optData']['varT'] + self.args['optData']['varC']
        with torch.no_grad():
            obsQ = x[:, :, vars.index("00060_Mean")]
            T_0 = ((x[:, :, vars.index("tmax(C)")] + x[:, :, vars.index("tmin(C)")]) / 2)
            vp = 0.01 * x[:, :, vars.index('vp(Pa)')]  # converting to mbar
            swrad = (x[:, :, vars.index('srad(W/m2)')] * x[:, :, vars.index('dayl(s)')] / 86400)
            elev = x[:, :, vars.index("ELEV_MEAN_M_BASIN")]
            slope = 0.01 * x[:, :, vars.index("SLOPE_PCT")] # adding the percentage
            stream_density = x[:, :, vars.index("STREAMS_KM_SQ_KM")]
            stream_length = 1000 * stream_density * x[:, :, vars.index("DRAIN_SQKM")]

            top_width = make_tensor(np.full((x.shape[0], x.shape[1]), 10), has_grad=False)
            PET = make_tensor(np.full((x.shape[0], x.shape[1]), 0.010/86400), has_grad=False)




        srflow, ssflow, gwflow = self.srflow_ssflow_gwflow_portions(discharge=obsQ)

        res_time = torch.cat(((torch.sigmoid(self.res_time_srflow[iGrid])) * 25,
                              (torch.sigmoid(self.res_time_ssflow[iGrid])) * 50,
                              (torch.sigmoid(self.res_time_gwflow[iGrid])) * 500), dim=1)
        ave_air_temp = self.ave_temp_res_time(ave_air_temp, x, res_time, iGrid, iT)
        T_l, srflow_temp, ssflow_temp, gwflow_temp = self.lateral_flow_temperature(srflow=srflow,
                                                                                   ssflow=ssflow,
                                                                                   gwflow=gwflow,
                                                                                   ave_air_temp=ave_air_temp)

        A, B, C, D = self.ABCD_equations(T_a=T_0, swrad=swrad, e_a=vp, elev=elev,
                                                                  slope=slope, top_width=top_width, inflow=obsQ, E=PET,
                                                                  T_g=gwflow_temp, iGrid=iGrid)

        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D)

        K1, K2 = self.finding_K1_K2(A=A, B=B, C=C, D=D, T_e=T_e, T_0=T_0)
        Q_0 = make_tensor(np.full((obsQ.shape[0], obsQ.shape[1]), 0.01))

        T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, T_l, T_e, ave_width=top_width,
                                                                q_l=obsQ/stream_length, L=stream_length,
                                                                T_0=T_0, Q_0=Q_0)

        return T_w, ave_air_temp













