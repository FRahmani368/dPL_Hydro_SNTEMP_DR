import numpy as np
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from torch.nn import Parameter
import math
from core.read_configurations import config
# from rnn import CudnnLstmModel
import matplotlib.pyplot as plt
from core.small_codes import make_tensor, tRange2Array, intersect
from .dropout import createMask, DropMask

class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(  #torch._C._VariableFunctions._cudnn_rnn(
            input.cuda(), weight, 4, None, hx.cuda(), cx.cuda(), 2,  # 2 means LSTM
            self.hiddenSize, 1, False, 0, self.training, False, (), None)   # 4 was False before
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(                                           # LSTMcell_untied CudnnLstm farshid
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)          # for LSTM-untied:    inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)
        #self.lstm = LSTMcell_untied(
         #   inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)

        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.activation_sigmoid = torch.nn.Sigmoid()
    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        out = self.linearOut(outLSTM)
        ### Farshid added this line:
        out = self.activation_sigmoid(out)
        return out




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
            nn.Linear(args['seq_lin_layers']['hidden_size'], args['res_time_params']['lenF_srflow'] +
                      args['res_time_params']['lenF_ssflow'] +
                      args['res_time_params']['lenF_gwflow']),
            # nn.ReLU()
        )
        self.L1 = nn.Linear(len(args['optData']['varC']), \
                            args['seq_lin_layers']['hidden_size'])
        self.L2 = nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size'])
        self.L3 = nn.Linear(args['seq_lin_layers']['hidden_size'], args['seq_lin_layers']['hidden_size'])

        self.L4 = nn.Linear(args['seq_lin_layers']['hidden_size'], 23)

        # 6 for alpha and beta of surface/subsurface/groundwater flow
        # 3 for conv bias,
        # 2 for scaling and bias of final answer,
        # 1 for shade_factor_riparian
        # 3 for surface/subsurface/groundwater flow percentage
        # 1 for albedo
        # 1 for solar shade factor
        # 4 for width coefficient nominator, width coefficient denominator, width A coefficient, and width exponent
        # 2 for p & q

        # self.lstm = CudnnLstmModel(
        #     nx=input.shape[2],
        #     ny=len(args['params_target']),
        #     hiddenSize=args['hyperprameters']['hidden_size'],
        #     dr=args['hyperprameters']['dropout'])
        #
        # self.stream_temp_eq = stream_temp_eq(args)
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.activation_tanh = torch.nn.Tanh()

    def forward(self, x):
        # out = self.seq_lin_layers(x)
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        # out1 = torch.abs(out)
        out = self.activation_sigmoid(out)
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
        H_f = 9805 * Q * slope / top_width  # Q is the seg_inflow (total flow entering a segment)
        return H_f

    def shortwave_solar_radiation_heat(self, albedo, H_sw, solar_shade_factor):
        """
        :param albedo: albedo or fraction reflected by stream , dimensionless
        :param H_sw: the clear sky solar radiation in watt per sq meter (seginc_swrad)
        :return: daily average clear sky, shortwave solar radiation for each segment
        """
        # solar_shade_fraction = make_tensor(self.args['STemp_default_params']['shade_fraction'])
        H_s = (1 - albedo) * (1 - solar_shade_factor) * H_sw
        return H_s

    def riparian_veg_longwave_radiation_heat(self, T_a, iGrid, shade_fraction_riparian):
        """
        Incoming shortwave solar radiation is often intercepted by surrounding riparian vegetation.
        However, the vegetation will emit some longwave radiation as a black body
        :param T_a: average daily air temperature
        :return: riparian vegetation longwave radiation
        """
        St_Boltzman_ct = make_tensor(5.670373) * torch.pow(make_tensor(10), (-8.0))  # (J/s*m^2 * K^4)
        emissivity_veg = make_tensor(self.args['STemp_default_params']['emissivity_veg'])
        # shade_fraction_riparian = make_tensor(self.args['STemp_default_params']['shade_fraction_riparian'])
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

    def ABCD_equations(self, T_a, swrad, e_a, E, elev, slope,
                       top_width, inflow, T_g, iGrid, shade_fraction_riparian, albedo, solar_shade_factor):
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
        H_s = self.shortwave_solar_radiation_heat(albedo=albedo,
                                                  H_sw=swrad,
                                                  solar_shade_factor=solar_shade_factor)  # shortwave solar radiation heat
        H_v = self.riparian_veg_longwave_radiation_heat(T_a, iGrid, shade_fraction_riparian)

        A = 5.4 * torch.pow(make_tensor(np.full((T_a.shape[0], T_a.shape[1]), 10)), (-8))
        B = torch.pow(make_tensor(10), 6) * E * (B_c * (2495 + 2.36 * T_a) - 2.36) + (K_g / delta_Z)
        C = torch.pow(make_tensor(10), 6) * E * B_c * 2.36
        # D = H_a + H_f + H_s + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)
        # D = H_a + H_f + H_s + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)
        D = H_a + H_s + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)
        # D = H_a + swrad + H_v + 2495 * E * (B_c * T_a - 1) + (T_g * K_g / delta_Z)

        return A, B, C, D

    def Equilibrium_temperature(self, A, B, C, D, T_e=make_tensor(20), iter=50):
        def F(T_e):
            return A * torch.pow((T_e + 273.16), 4) - C * torch.pow(T_e, 2) + B * T_e - D

        def Fprime(T_e):
            return 4 * A * torch.pow((T_e + 273.16), 3) - 2 * C * T_e + B

        ## solving the equation with Newton's method
        for i in range(iter):
            next_geuss = T_e - (F(T_e) / Fprime(T_e))
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

    def x_sample_air_temp(self, iGrid, iT, lenF):
        """
        :param iGrid:
        :param iT:
        :param lenF: maximum number of days that it is needed to be considered in average
        :return:
        """
        rho = self.args['hyperparameters']['rho']
        tArray_Total = tRange2Array(self.args['optData']['tRange'])
        tArray_train = tRange2Array(self.args['optData']['t_train'])
        _, ind1, _ = (intersect(tArray_Total, tArray_train))
        ind1_tensor = make_tensor(ind1, has_grad=False)
        iT_tensor = make_tensor(iT, has_grad=False)
        vars = self.args['optData']['varT'] + self.args['optData']['varC']
        ave_air = torch.zeros((self.args['hyperparameters']['batch_size'], self.args["hyperparameters"]["rho"],
                               lenF),
                              device=self.args["device"])
        for s, station in enumerate(iGrid):
            array = np.zeros((rho, lenF), dtype=np.int32)
            for j in range(lenF):
                array[:, j] = np.arange((ind1_tensor[0] + iT_tensor[s] - j).item(),
                                        (ind1_tensor[0] + iT_tensor[s] - j + rho).item())
            # array = np.flip(array, 1).copy()
            tmax_temp = self.x_total_raw[station, array, vars.index("tmax(C)")]
            tmin_temp = self.x_total_raw[station, array, vars.index("tmin(C)")]
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
            x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            # m = a.shape
            # w = torch.zeros([lenF, m[1], m[2]])
            # aa = F.relu(a[0:lenF, :, 0]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
            # theta = F.relu(b[0:lenF, :, 0]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
            # t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
            # t = t.cuda(aa.device)
            # denom = (aa.lgamma().exp()) * (theta ** aa)
            # mid = t ** (aa - 1)
            # right = torch.exp(-t / theta)
            # w = 1 / denom * mid * right
            ww = torch.cumsum(w, dim=0)
            www = ww / ww.sum(0)  # scale to 1 for each UH
        elif a.dim() == 3:
            m = a.shape
            a1 = a.repeat(1, 1, lenF)
            b1 = b.repeat(1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.1
            beta = F.relu(b1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.5
            x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
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
            y = F.conv1d(x_sample1[a], torch.flip(w, [2]), groups=groups, padding=0, stride=1, bias=bias)
            y = y.permute(1, 0, 2)
        elif UH.shape[1] > 1:
            w = torch.flip(UH, [2])
            y = x_sample * w
            y = y.sum(2)
            if bias is not None:
                y = y + bias
            y = y.unsqueeze(2)


        return y

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
        srflow_temp = ave_air_temp[:, :, 0]  # .clone().detach()
        ssflow_temp = ave_air_temp[:, :, 1]  # .clone().detach()
        gwflow_temp = ave_air_temp[:, :, 2]  # .clone().detach()

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
                                                                                  / (
                                                                                          density * c_w)) * T_e * mask_pos.int().float())
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

    def forward(self, x, params, iGrid, iT, ave_air_temp):
        # restricting the params
        paramCalLst = [
            [0.01, 5], [0.01, 5], [0.01, 8], [0.01, 8], [0.01, 10], [0.01, 10],      # a and b
            [0.01, 1],                                                    # shade factor
            [0.01, 1], [0.01, 1], [0.01, 1],                                    # flow portions
            [-2, 2], [-2, 2], [-2, 2],                                 # conv bias
            [0, 3], [-4, 4],                                            # final scale and final bias
            [0, 1],                                                        # albedo
            [0, 1],                                                           # solar shade factor
            [0, 1],                                                       # width coefficient nominator
            [0, 1],                                                         # width exponent
            [0, 1],                                                            # width A coefficient
            [0, 1],                                                           # width coefficient denominator
            [0, 40],                                                    # p
            [0, 2]                                                        # q
        ]
        # for all a and b
        if params.dim() == 3:
            a_srflow = params[:, :, 0: 1] * (paramCalLst[0][1] - paramCalLst[0][0]) + paramCalLst[0][0]
            b_srflow = params[:, :, 1: 2] * (paramCalLst[1][1] - paramCalLst[1][0]) + paramCalLst[1][0]
            a_ssflow = params[:, :, 2: 3] * (paramCalLst[2][1] - paramCalLst[2][0]) + paramCalLst[2][0]
            b_ssflow = params[:, :, 3: 4] * (paramCalLst[3][1] - paramCalLst[3][0]) + paramCalLst[3][0]
            a_gwflow = params[:, :, 4: 5] * (paramCalLst[4][1] - paramCalLst[4][0]) + paramCalLst[4][0]
            b_gwflow = params[:, :, 5: 6] * (paramCalLst[5][1] - paramCalLst[5][0]) + paramCalLst[5][0]

            shade_fraction_riparian = params[:, :, 6] * (paramCalLst[6][1] - paramCalLst[6][0]) + paramCalLst[6][0]

            srflow_portion = params[:, :, 7] * (paramCalLst[7][1] - paramCalLst[7][0]) + paramCalLst[7][0]
            ssflow_portion = params[:, :, 8] * (paramCalLst[8][1] - paramCalLst[8][0]) + paramCalLst[8][0]
            gwflow_portion = params[:, :, 9] * (paramCalLst[9][1] - paramCalLst[9][0]) + paramCalLst[9][0]

            sr_conv_bias = ((params[:, :, 10: 11]).squeeze()) * (paramCalLst[10][1] - paramCalLst[10][0]) + \
                           paramCalLst[10][0]
            ss_conv_bias = ((params[:, :, 11: 12]).squeeze()) * (paramCalLst[11][1] - paramCalLst[11][0]) + \
                           paramCalLst[11][0]
            gw_conv_bias = ((params[:, :, 12: 13]).squeeze()) * (paramCalLst[12][1] - paramCalLst[12][0]) + \
                           paramCalLst[12][0]
            # for scaling and bias of final y_Sim
            final_scale = params[:, :, 13] * (paramCalLst[13][1] - paramCalLst[13][0]) + paramCalLst[13][0]
            final_bias = params[:, :, 14] * (paramCalLst[14][1] - paramCalLst[14][0]) + paramCalLst[14][0]

            albedo = params[:, :, 15] * (paramCalLst[15][1] - paramCalLst[15][0]) + paramCalLst[15][0]
            solar_shade_factor = params[:, :, 16] * (paramCalLst[16][1] - paramCalLst[16][0]) + paramCalLst[16][0]

            width_coef_nom = params[:, :, 17: 18] * (paramCalLst[17][1] - paramCalLst[17][0]) + paramCalLst[17][0]
            width_exp = params[:, :, 18: 19] * (paramCalLst[18][1] - paramCalLst[18][0]) + paramCalLst[18][0]
            width_A_coef = params[:, :, 19: 20] * (paramCalLst[19][1] - paramCalLst[19][0]) + paramCalLst[19][0]
            width_coef_denom = params[:, :, 20: 21] * (paramCalLst[20][1] - paramCalLst[20][0]) + paramCalLst[20][0]
            p = params[:, :, 21] * (paramCalLst[21][1] - paramCalLst[21][0]) + paramCalLst[21][0]
            q = params[:, :, 22] * (paramCalLst[22][1] - paramCalLst[22][0]) + paramCalLst[22][0]

        if params.dim() == 2:
            a_srflow = params[:, 0: 1] * (paramCalLst[0][1] - paramCalLst[0][0]) + paramCalLst[0][0]
            b_srflow = params[:, 1: 2] * (paramCalLst[1][1] - paramCalLst[1][0]) + paramCalLst[1][0]
            a_ssflow = params[:, 2: 3] * (paramCalLst[2][1] - paramCalLst[2][0]) + paramCalLst[2][0]
            b_ssflow = params[:, 3: 4] * (paramCalLst[3][1] - paramCalLst[3][0]) + paramCalLst[3][0]
            a_gwflow = params[:, 4: 5] * (paramCalLst[4][1] - paramCalLst[4][0]) + paramCalLst[4][0]
            b_gwflow = params[:, 5: 6] * (paramCalLst[5][1] - paramCalLst[5][0]) + paramCalLst[5][0]

            shade_fraction_riparian = params[:, 6:7] * (paramCalLst[6][1] - paramCalLst[6][0]) + paramCalLst[6][0]

            srflow_portion = params[:, 7: 8] * (paramCalLst[7][1] - paramCalLst[7][0]) + paramCalLst[7][0]
            ssflow_portion = params[:, 8: 9] * (paramCalLst[8][1] - paramCalLst[8][0]) + paramCalLst[8][0]
            gwflow_portion = params[:, 9: 10] * (paramCalLst[9][1] - paramCalLst[9][0]) + paramCalLst[9][0]

            sr_conv_bias = ((params[:, 10: 11]).squeeze()) * (paramCalLst[10][1] - paramCalLst[10][0]) + \
                           paramCalLst[10][0]
            ss_conv_bias = ((params[:, 11: 12]).squeeze()) * (paramCalLst[11][1] - paramCalLst[11][0]) + \
                           paramCalLst[11][0]
            gw_conv_bias = ((params[:, 12: 13]).squeeze()) * (paramCalLst[12][1] - paramCalLst[12][0]) + \
                           paramCalLst[12][0]
            # for scaling and bias of final y_Sim
            final_scale = params[:, 13:14] * (paramCalLst[13][1] - paramCalLst[13][0]) + paramCalLst[13][0]
            final_bias = params[:, 14:15] * (paramCalLst[14][1] - paramCalLst[14][0]) + paramCalLst[14][0]

            albedo = params[:, 15:16] * (paramCalLst[15][1] - paramCalLst[15][0]) + paramCalLst[15][0]
            solar_shade_factor = params[:, 16:17] * (paramCalLst[16][1] - paramCalLst[16][0]) + paramCalLst[16][0]

            width_coef_nom = params[:, 17:18] * (paramCalLst[17][1] - paramCalLst[17][0]) + paramCalLst[17][0]
            width_exp = params[:, 18:19] * (paramCalLst[18][1] - paramCalLst[18][0]) + paramCalLst[18][0]
            width_A_coef = params[:, 19:20] * (paramCalLst[19][1] - paramCalLst[19][0]) + paramCalLst[19][0]
            width_coef_denom = params[:, 20:21] * (paramCalLst[20][1] - paramCalLst[20][0]) + paramCalLst[20][0]
            p = params[:, 21:22] * (paramCalLst[21][1] - paramCalLst[21][0]) + paramCalLst[21][0]
            q = params[:, 22:] * (paramCalLst[22][1] - paramCalLst[22][0]) + paramCalLst[22][0]


        srflow_percentage = srflow_portion / (srflow_portion + ssflow_portion + gwflow_portion)
        ssflow_percentage = ssflow_portion / (srflow_portion + ssflow_portion + gwflow_portion)
        gwflow_percentage = gwflow_portion / (srflow_portion + ssflow_portion + gwflow_portion)




        vars = self.args['optData']['varT'] + self.args['optData']['varC']
        with torch.no_grad():
            obsQ = x[:, :, vars.index("00060_Mean")]
            T_0 = ((x[:, :, vars.index("tmax(C)")] + x[:, :, vars.index("tmin(C)")]) / 2)
            vp = 0.01 * x[:, :, vars.index('vp(Pa)')]  # converting to mbar
            swrad = (x[:, :, vars.index('srad(W/m2)')] * x[:, :, vars.index('dayl(s)')] / 86400)
            elev = x[:, :, vars.index("ELEV_MEAN_M_BASIN")]
            slope = 0.01 * x[:, :, vars.index("SLOPE_PCT")]  # adding the percentage
            stream_density = x[:, :, vars.index("STREAMS_KM_SQ_KM")]
            stream_length = 1000 * stream_density * x[:, :, vars.index("DRAIN_SQKM")]


            basin_area = x[:, :, vars.index("DRAIN_SQKM")]

            # top_width = make_tensor(np.full((x.shape[0], x.shape[1]), 10), has_grad=False)

            PET = make_tensor(np.full((x.shape[0], x.shape[1]), 0.010 / 86400), has_grad=False)

        # d = torch.pow(width_coef_nom * obsQ / (width_coef_denom * torch.pow(basin_area, width_A_coef)), width_exp)
        # top_width = p * torch.pow(d, q)
        # if p.dim() == 3:
        top_width = p * torch.pow(basin_area, q)
        # elif p.dim() == 2:
        #     top_width = p * torch.pow(basin_area, q)

        srflow, ssflow, gwflow = self.srflow_ssflow_gwflow_portions(discharge=obsQ,
                                                                    srflow_factor=srflow_percentage,
                                                                    ssflow_factor=ssflow_percentage,
                                                                    gwlow_factor=gwflow_percentage)

        # res_time = torch.cat(((torch.sigmoid(self.res_time_srflow[iGrid])) * 25,
        #                       (torch.sigmoid(self.res_time_ssflow[iGrid])) * 50,
        #                       (torch.sigmoid(self.res_time_gwflow[iGrid])) * 500), dim=1)
        # ave_air_temp = self.ave_temp_res_time(ave_air_temp, x, res_time, iGrid, iT)

        # surface flow
        w_srflow = self.res_time_gamma(a=a_srflow, b=b_srflow, lenF=self.args['res_time_params']['lenF_srflow'])

        # w_srflow[0:10, :, :] = 1
        # w_srflow[10:, :, :] = 0
        # w_srflow = w_srflow/10

        air_sample_sr = self.x_sample_air_temp(iGrid, iT, lenF=self.args['res_time_params']['lenF_srflow'])
        w_srflow = w_srflow.permute(1, 2, 0)
        ave_air_sr = self.res_time_conv(air_sample_sr, w_srflow, bias=sr_conv_bias)    # sr_conv_bias

        # subsurface flow
        w_ssflow = self.res_time_gamma(a=a_ssflow, b=b_ssflow, lenF=self.args['res_time_params']['lenF_ssflow'])

        # w_ssflow = torch.empty(self.args['res_time_params']['lenF_ssflow'],
        #                        self.args['hyperparameters']['batch_size'], 1, device=self.args['device'])

        # w_ssflow[0:30, :, :] = 1
        # w_ssflow[30:, :, :] = 0
        # w_ssflow = w_ssflow / 30

        air_sample_ss = self.x_sample_air_temp(iGrid, iT, lenF=self.args['res_time_params']['lenF_ssflow'])
        w_ssflow = w_ssflow.permute(1, 2, 0)
        ave_air_ss = self.res_time_conv(air_sample_ss, w_ssflow, bias=ss_conv_bias)  # ss_conv_bias

        # groundwater flow
        w_gwflow = self.res_time_gamma(a=a_gwflow, b=b_gwflow, lenF=self.args['res_time_params']['lenF_gwflow'])

        # w_gwflow = torch.empty(self.args['res_time_params']['lenF_gwflow'],
        #                        self.args['hyperparameters']['batch_size'], 1, device=self.args['device'])
        # w_gwflow[0:365, :, :] = 1
        # w_gwflow[365:, :, :] = 0
        # w_gwflow = w_gwflow / 365

        air_sample_gw = self.x_sample_air_temp(iGrid, iT, lenF=self.args['res_time_params']['lenF_gwflow'])
        w_gwflow = w_gwflow.permute(1, 2, 0)
        ave_air_gw = self.res_time_conv(air_sample_gw, w_gwflow, bias=gw_conv_bias)  # gw_conv_bias

        ave_air_temp = torch.cat((ave_air_sr, ave_air_ss, ave_air_gw), dim=2)

        T_l, srflow_temp, ssflow_temp, gwflow_temp = self.lateral_flow_temperature(srflow=srflow,
                                                                                   ssflow=ssflow,
                                                                                   gwflow=gwflow,
                                                                                   ave_air_temp=ave_air_temp)

        A, B, C, D = self.ABCD_equations(T_a=T_0, swrad=swrad, e_a=vp, elev=elev,
                                         slope=slope, top_width=top_width, inflow=obsQ, E=PET,
                                         T_g=gwflow_temp, iGrid=iGrid, shade_fraction_riparian=shade_fraction_riparian,
                                         albedo=albedo,
                                         solar_shade_factor=solar_shade_factor)

        T_e = self.Equilibrium_temperature(A=A, B=B, C=C, D=D)

        K1, K2 = self.finding_K1_K2(A=A, B=B, C=C, D=D, T_e=T_e, T_0=T_0)
        Q_0 = make_tensor(np.full((obsQ.shape[0], obsQ.shape[1]), 0.01))

        T_w = self.solving_SNTEMP_ODE_second_order(K1, K2, T_l, T_e, ave_width=top_width,
                                                   q_l=obsQ / stream_length, L=stream_length,
                                                   T_0=T_0, Q_0=Q_0)

        # scaling and bias
        T_w = final_scale * T_w + final_bias

        return T_w, ave_air_temp
