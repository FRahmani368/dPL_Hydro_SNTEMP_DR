import numpy as np
import torch
import torch.nn as nn
import datetime
from rnn import CudnnLstmModel
from core.small_codes import make_tensor


class MLP(nn.Module):
    def __init__(self,
                 args):
        super(MLP, self).__init__()
        self.seq_lin_layers = nn.Sequential(
            nn.Linear(len(args['optData']['varT']) + len(args['optData']['varC']), \
                      args['seq_lin_layers']['hidden_size']),
            nn.ReLU(),
            nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size']),
            nn.ReLU(),
            nn.Linear(args['seq_lin_layers']['hidden_size'], len(args['params_target']))
        )

        self.lstm = CudnnLstmModel(
            nx=input.shape[2],
            ny=len(args['params_target']),
            hiddenSize=args['hyperprameters']['hidden_size'],
            dr=args['hyperprameters']['dropout'])
        #
        # self.stream_temp_eq = stream_temp_eq(args)

    def forward(self, x):
        out = self.seq_lin_layers(x)
        return out


def str_to_datetime(t):
    if isinstance(t, datetime.datetime):
        return t
    elif isinstance(t, str):
        year, month, day = t.split("-")
        t_datetime_format = datetime.datetime(int(year), int(month), int(day))
        return t_datetime_format

class STREAM_TEMP_EQ:
    def __init__(self,
                 args,
                 forcings,
                 attr,
                 t,
                 site_no):
        self.args = args
        self.forcings = forcings
        self.attr = attr
        self.t = t
        self.site_no = site_no


    def atm_pressure(self, elev):
        mmHg2mb = make_tensor(0.75061683)  # Unit conversion
        mmHg2inHg = make_tensor(25.3970886)  # Unit conversion
        P_sea = make_tensor(29.92126)  # Standard pressure ar sea level
        A_g = make_tensor(9.80665)  # Acceleration due to gravity
        M_a = make_tensor(0.0289644)  # Molar mass of air
        R = make_tensor(8.31447)  # universal gas constant
        T_sea = make_tensor(288.16)  # the standard temperature at sea level
        P = (1 / mmHg2mb) * (mmHg2inHg) * (P_sea) * torch.exp(A_g * M_a * elev / (R * T_sea))
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
                emissivity_air * St_Boltzman_ct * torch.pow(make_tensor(T_a + 273.16), 4)
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

    def riparian_veg_longwave_radiation_heat(self, T_a):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(make_tensor(10), (-8.0))  # (J/s*m^2 * K^4)
        emissivity_veg = make_tensor(self.args['STemp_default_params']['emissivity_veg'])
        shade_fraction_riparian = make_tensor(self.args['STemp_default_params']['shade_fraction_riparian'])
        H_v = emissivity_veg * St_Boltzman_ct * shade_fraction_riparian * torch.pow((T_a + 273.16), 4)
        return H_v

    def ABCD_equations(self, T_a, swrad, e_a, E, elev, slope, top_width, inflow):
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
        B_c = 0.00061 * P / (e_s - e_a)
        K_g = make_tensor(1.65)
        delta_Z = make_tensor(1.0)
        H_a = self.atm_longwave_radiation_heat(T_a, e_a)
        H_f = self.stream_friction_heat(top_width=top_width, slope=slope, Q=inflow)
        H_s = self.shortwave_solar_radiation_heat(albedo=self.args['STemp_default_params']['albedo'], H_sw=swrad) # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(T_a=T_a)

        A = make_tensor(5.40 * 10 ^ (-8))
        B = (10 ^ 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (K_g / Z)
        C = (10 ^ 6) * E * R_c * 2.36
        D = H_a + H_f + H_s + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e = make_tensor(10), iter=100):
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
        K2 = (H_i - (K1 * (T_e - T_0)))/(torch.pow((T_e - T_0), 2))
        return K1, K2

    def srflow_ssflow_gwflow_portions(self,
                             discharge,
                             srflow_factor=make_tensor(0.85),
                             ssflow_factor=make_tensor(0.1),
                             gwlow_factor=make_tensor(0.05)):
        srflow = srflow_factor * discharge
        ssflow = ssflow_factor * discharge
        gwflow = gwlow_factor * discharge
        return srflow, ssflow, gwflow

    def residence_time_temperature(self, res_time):
        t_datetime_format = str_to_datetime(self.t)
        t1 = t_datetime_format - datetime.timedelta(res_time)
        t1_str = str(t_datetime_format.year) + "-" + str(t_datetime_format.month).zfill(2) + "-" + str(t_datetime_format.day).zfill(2)
        tmax = self.forcings.loc[(self.forcings['datetime'] <= self.t) &
                                         (self.forcings['datetime']> t1_str), "tmax(C)"]
        tmin = self.forcings.loc[(self.forcings['datetime'] <= self.t) &
                                 (self.forcings['datetime'] > t1_str ), "tmin(C)"]
        ave_air_temp = make_tensor(tmax.sum() + tmin.sum())/ (2 * len(tmax))
        return ave_air_temp

    def lateral_flow_temperature(self, srflow, ssflow, gwflow,
                                 res_time_srflow=1, res_time_ssflow=30, res_time_gwflow=365):
        """
        :param srflow: surface runoff
        :param ssflow: subsurface runoff
        :param gwflow: qroundwaterflow
        :param res_time_srflow: residense time for surface runoff
        :param res_time_ssflow: residence time for subsurface runoff
        :param res_time_gwflow: residence time for groundwater flow
        :return: temperature of lateral flow
        """
        gwflow_Temp = self.residence_time_temperature(res_time_gwflow)
        srflow_temp = self.residence_time_temperature(res_time_srflow)
        ssflow_temp = self.residence_time_temperature(res_time_ssflow)

        T_l = (gwflow * gwflow_Temp + srflow * srflow_temp + ssflow * ssflow_temp / (gwflow + ssflow + srflow))
        return T_l, srflow_temp, ssflow_temp, gwflow_Temp

    def solving_SNTEMP_ODE_second_order(self, K1, K2, ave_width, q_l, T_l, T_e, T_0=make_tensor(0), L, Q_0=make_tensor(0.01)):
        a = (q_l * T_l) + ((K1 * ave_width)/(args['params']['water_density'] * args['params']['C_w'])) * T_e
        b = q_l + (K1 * ave_width)/ (args['params']['water_density'] * args['params']['C_w'])
        if q_l > 0:
            Tprime_e = a/b
            R = torch.pow((1 + (q_l * make_tensor(L)/ Q_0)), -(ave_width/q_l))
        elif q_l == 0:
            Tprime_e = T_e
            R = torch.exp(-(ave_width * L) / Q_0)
        else:   # negative q_l
            print("negative q_l")
            exit()
        T_w = Tprime_e - ((Tprime_e - T_0) * R / (1 + ((K2/K1) * (Tprime_e - T_0) * (1 - R))))
        return T_w

    def forward(self):
        obsQ = make_tensor(self.forcings.loc[(self.forcings['datetime'] == self.t) &
                                             (self.forcings['site_no'] == self.site_no), '00060_Mean'])

        T_0 = make_tensor((self.forcings.loc[(self.forcings['datetime'] == self.t) &
                                             (self.forcings['site_no'] == self.site_no), 'tmax(C)'] +
                           (self.forcings.loc[(self.forcings['datetime'] == self.t) &
                                              (self.forcings['site_no'] == self.site_no), 'tmin(C)'])) / 2)

        vp = make_tensor(self.forcings.loc[(self.forcings['datetime'] == self.t) &
                                             (self.forcings['site_no'] == self.site_no), 'vp(Pa)'])

        swrad = make_tensor(self.forcings.loc[(self.forcings['datetime'] == self.t) &
                               (self.forcings['site_no'] == self.site_no), 'srad(W/m2)'])

        elev = make_tensor(self.attr.loc[self.attr['site_no'] == self.site_no, "ELEV_MEAN_M_BASIN"])

        slope = make_tensor(self.attr.loc[self.attr['site_no'] == self.site_no, "SLOPE_PCT"])

        stream_density = make_tensor(self.attr.loc[self.attr['site_no'] == self.site_no, "STREAMS_KM_SQ_KM"])

        stream_length = stream_density * make_tensor(self.attr.loc[self.attr['site_no'] == self.site_no, "DRAIN_SQKM"])

        top_width=make_tensor(10)

        A, B, C, D = self.ABCD_equations(T_a=T_0, swrad=swrad, e_a=vp, elev=elev,
                                         slope=slope, top_width=top_width, inflow=obsQ)

        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D)

        K1 , K2 = self.finding_K1_K2(A=A, B=B, C=C, D=D, T_e=T_e, T_0=T_0)



        srflow, ssflow, gw_flow = self.srflow_ssflow_gwflow_portions(discharge=obsQ)

        T_l = self.lateral_flow_temperature(srflow=srflow, ssflow=ssflow, gwflow=gwflow,
                                            res_time_gwflow=res_time_gwflow,
                                            res_time_srflow=res_time_srflow,
                                            res_time_ssflow=res_time_ssflow)

        T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, ave_width, q_l=obsQ/stream_length, T_l, T_e, T_0=T_0, L=stream_length, Q_0=make_tensor(0.01))
        return T_w













