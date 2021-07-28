import numpy as np
import torch
import torch.nn as nn
from rnn import CudnnLstmModel
from core.small_codes import make_tensor


class PGML_STemp(nn.Module):
    def __init__(self,
                 args, vars):
        super(PGML_STemp, self).__init__()
        self.seq_lin_layers = nn.Sequential(
            nn.Linear(vars, args['seq_lin_layers']['hidden_size']),
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


class stream_temp_eq:
    def __init__(self,
                 args):
        self.args = args

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

    def equations(self, T_a, swrad, e_a, E, elev, slope, top_width, inflow):
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

    def finding_K1(self, A, B, C, T_e):
        return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B

    def finding_K2(self, H_i=0, ):





