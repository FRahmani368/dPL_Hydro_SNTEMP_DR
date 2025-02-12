###
# Written by Farshid Rahmani,Penn State University
import torch
from MODELS.PET_models.potet import get_potet
import torch.nn.functional as F

class NWM_SACSMA_Mul(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(NWM_SACSMA_Mul, self).__init__()
        self.parameters_bound = dict(pctim=[0.0, 1.0],
                                     smax=[1, 3000],    #[1, 3000]
                                     f1=[0.005, 0.995],
                                     f2=[0.005, 0.995],
                                     kuz=[0.0, 1],
                                     rexp=[0.0, 8],      #[0.0, 8]
                                     f3=[0.005, 0.995],
                                     f4=[0.005, 0.995],
                                     pfree=[0, 1],
                                     klzp=[0, 1],
                                     klzs=[0, 1],
                                     parTT=[-3.0, 3.0],   #[-2.5, 2.5]
                                     parCFMAX=[0.4, 12],   # [0.4, 12]
                                     parCFR=[0, 0.1],
                                     parCWH=[0, 0.2]
                                     )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]  # routing parameter b
        ]
    def split_1(self, p1, In):
        """
        Split flow (returns flux [mm/d])

        :param p1: fraction of flux to be diverted [-]
        :param In: incoming flux [mm/d]
        :return: divided flux
        """
        out = p1 * In
        return out

    def soilmoisture_1(self, S1, S1max, S2, S2max):
        """
        Water rebalance to equal relative storage (2 stores)

        :param S1: current storage in S1 [mm]
        :param S1max: maximum storage in S1 [mm]
        :param S2: current storage in S2 [mm]
        :param S2max: maximum storage in S2 [mm]
        :return: rebalanced water storage
        """
        mask = S1/S1max < S2/S2max
        mask = mask.type(torch.cuda.FloatTensor)
        out = ((S2 * S1max - S1 * S2max) / (S1max + S2max)) * mask
        return out

    def evap_7(self, S, Smax, Ep, dt):
        """
        Evaporation scaled by relative storage

        :param S: current storage [mm]
        :param Smax: maximum contributing storage [mm]
        :param Ep: potential evapotranspiration rate [mm/d]
        :param dt: time step size [d]
        :return: evaporation [mm]
        """
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def saturation_1(self, In, S, Smax):
        """
        Saturation excess from a store that has reached maximum capacity

        :param In: incoming flux [mm/d]
        :param S: current storage [mm]
        :param Smax: maximum storage [mm]
        :param args: smoothing variables (optional)
        :return: saturation excess
        """
        mask = S >= Smax
        mask = mask.type(torch.cuda.FloatTensor)
        out = In * mask

        return out

    def interflow_5(self, p1, S):
        """
        Linear interflow

        :param p1: time coefficient [d-1]
        :param S: current storage [mm]
        :return: interflow output
        """
        out = p1 * S
        return out

    def evap_1(self, S, Ep, dt):
        """
        Evaporation at the potential rate

        :param S: current storage [mm]
        :param Ep: potential evaporation rate [mm/d]
        :param dt: time step size
        :return: evaporation output
        """
        out = torch.min(S / dt, Ep)
        return out

    def percolation_4(self, p1, p2, p3, p4, p5, S, Smax, dt):
        """
        Demand-based percolation scaled by available moisture

        :param p1: base percolation rate [mm/d]
        :param p2: percolation rate increase due to moisture deficiencies [mm/d]
        :param p3: non-linearity parameter [-]
        :param p4: summed deficiency across all model stores [mm]
        :param p5: summed capacity of model stores [mm]
        :param S: current storage in the supplying store [mm]
        :param Smax: maximum storage in the supplying store [mm]
        :param dt: time step size [d]
        :return: percolation output
        """
        # Prevent negative S values and ensure non-negative percolation demands
        S_rel = torch.max(torch.tensor(1e-8).cuda(), S / Smax)

        percolation_demand = p1 * (torch.tensor(1.0).cuda() + p2 * (p4 / p5) ** (torch.tensor(1.0).cuda() + p3))
        out = torch.max(torch.tensor(1e-8).cuda(), torch.min(S / dt, S_rel * percolation_demand))
        return out

    def soilmoisture_2(self, S1, S1max, S2, S2max, S3, S3max):
        """
        Water rebalance to equal relative storage (3 stores)

        :param S1: current storage in S1 [mm]
        :param S1max: maximum storage in S1 [mm]
        :param S2: current storage in S2 [mm]
        :param S2max: maximum storage in S2 [mm]
        :param S3: current storage in S3 [mm]
        :param S3max: maximum storage in S3 [mm]
        :return: rebalanced water storage
        """
        part1 = S2 * (S1 * (S2max + S3max) + S1max * (S2 + S3)) / ((S2max + S3max) * (S1max + S2max + S3max))
        mask = S1 / S1max < (S2 + S3) / (S2max + S3max)
        mask = mask.type(torch.cuda.FloatTensor)
        out = part1 * mask
        return out

    def baseflow_1(self,K,S):
        return K * S



    def deficitBasedDistribution_pytorch(self, S1, S1max, S2, S2max):
        # Calculate relative deficits
        rd1 = (S1max-S1 ) / S1max
        rd2 = (S2max-S2 ) / S2max

        # Calculate fractional split
        total_rd = rd1 + rd2
        mask = total_rd != torch.tensor(0.0).cuda()
        mask = mask.type(torch.cuda.FloatTensor)
        total_max = S1max + S2max
        f1 = rd1 / total_rd * mask + S1max / total_max*(torch.tensor(1.0)-mask)

        return f1

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


    def change_param_range(self, param, bounds):
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

    def source_flow_calculation(self, args, flow_out, c_NN, after_routing=True):
        varC_NN = args["varC_NN"]
        if "DRAIN_SQKM" in varC_NN:
            area_name = "DRAIN_SQKM"
        elif "area_gages2" in varC_NN:
            area_name = "area_gages2"
        else:
            print("area of basins are not available among attributes dataset")
        area = c_NN[:, varC_NN.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
            flow_out["flow_sim"].shape[
                0], 1, 1)
        # flow calculation. converting mm/day to m3/sec
        if after_routing == True:
            srflow = (1000 / 86400) * area * (flow_out["srflow"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out["ssflow"]).repeat(1, 1, args["nmul"])  # ras
            gwflow = (1000 / 86400) * area * (flow_out["gwflow"]).repeat(1, 1, args["nmul"])
        else:
            srflow = (1000 / 86400) * area * (flow_out["srflow_no_rout"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out["ssflow_no_rout"]).repeat(1, 1, args["nmul"])  # ras
            gwflow = (1000 / 86400) * area * (flow_out["gwflow_no_rout"]).repeat(1, 1, args["nmul"])
        # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
        # ssflow = torch.clamp(ssflow, min=0.0)
        # gwflow = torch.clamp(gwflow, min=0.0)
        return srflow, ssflow, gwflow

    def forward(self, x_hydro_model, c_hydro_model, params_raw, args, muwts=None, warm_up=0, init=False, routing=False, comprout=False, conv_params_hydro=None):
        """ from fortran code https://github.com/NOAA-OWP/sac-sma/blob/master/src/sac/sac1.f
        C    ----- VARIABLES -----
C    DT       Computational time interval
C    IFRZE    Frozen ground module switch.  0 = No frozen ground module,
C                 1 = Use frozen ground module
C    EDMND    ET demand for the time interval
C    E1       ET from the upper zone tension water content (UZTWC)
C    RED      Residual ET demand
C    E2       ET from upper zone free water content (UZFWC)
C    UZRAT    Upper zone ratio used to transfer water from free to
C                 tension water store
C    E3       ET from the lower zone tension water content (LZTWC)
C    RATLZT   Ratio of the lower zone tension water content to the
C                 maximum tension water.  AKA: percent saturation of
C                 the lower zone tension water
C    DEL      Used for multiple calculations in the code:
C                 1. Amount of water moved from lower zone free water
C                    content to the tension water content
C                 2. Incremental interflow
C    E5       ET from ADIMP area
C    TWX      Time interval available moisture in excess of UZTW
C                 requirements
C    SIMPVT   Sum of ROIMP
C    SPERC    Sum of incremental percolation
C    SPBF     Sum of the incremental LZ primary baseflow component only
C    NINC     Number of time sub-increments that the time interval is
C                 diveded into for further soil moisture accounting
C    DINC     Length of each sub-increment (calculated by NINC) in days
C    PINC     Amount of available moisture for each time sub-increment
C    DUZ      Depletion in the upper zone
C    DLZP     Depletion in the lower zone, primary
C    DLZS     Depletion in the lower zone, secondary
C    PAREA    Pervious area
C    I        Loop counter
C    ADSUR    Surface runoff from portion of ADIMP not currently
C                 generating direct runoff (ADDRO)
C    RATIO    Ratio of excess water in the upper zone from ADIMC to the
C                 maximum lower zone tension water. Used to calculate
C                 ADDRO
C    ADDRO    Additional "impervious" direct runoff from ADIMP.
C                 Essentially saturation excess runoff from ADIMP area
C    BF       Used for multiple baseflow calculations in the code
C                 1. Incremental baseflow, lower zone primary
C                 2. Incremental baseflow, lower zone secondary
C    SBF      Sum of the incremental baseflow components (LZ primary,
C                 secondary).
C    PERCM    Limiting percolation value (aka maximum percolation). In
C                 some documentation it is referred to as PBASE
C    PERC     Percolation
C    DEFR     Lower zone moisture deficiency ratio
C    FR       Change in percolation withdrawal due to frozen ground
C    FI       Change in interflow withdrawal due to frozen ground
C    UZDEFR   Calculated, but not used. RECOMMEND removing
C    CHECK    A check to see if percolation exceeds the lower zone
C                 deficiency
C    SPERC    Sum of interval percolation
C    PERCT    Percolation to tension water
C    PERCF    Percolation to free water
C    HPL      Relative size of the lower zone max free water, primary
C                 storage to the lower zone total max free water storage
C    RATLP    Content capacity ratio (LZ, primary) (i.e. relative
C                 fullness)
C    RATLS    Content capacity ratio (LZ, secondary) (i.e. relative
C                 fullness)
C    FRACP    Fraction going to primary store during each interval
C    PERCP    Amount of excess percolation going to the LZ primary store
C    PERCS    Amount of excess percolation going to the LZ secondary
C                 store
C    EXCESS   LZ free water in excess of the maximum to be removed from
C                 LZFPC and added to LZTWC
C    SUR      Incremental surface runoff.  Not multiplied by PAREA until
C                 added to the sum (SSUR)
C    EUSED    Total ET from the pervious area (PAREA) = E1+E2+E3
C    TBF      Total baseflow
C    BFCC     Baseflow channel component (reduces TBF by fraction SIDE)
C    SINTFT   Monthly sum of SIF (NOT USED)
C    SGWFP    Monthly sum of BFP (NOT USED)
C    SGWFS    Monthly sum of BFS (NOT USED)
C    SRECHT   Monthly sum of BFNCC (NOT USED)
C    SROST    Monthly sum of SSUR (NOT USED)
C    SRODT    Monthly sum of SDRO (NOT USED)
C    E4       ET from riparian vegetation using RIVA
C    SROT     Assuming this is the monthly sum of TCI (NOT USED)
C    TET      Total evapotranspiration
C    SETT     Assuming this is the monthly sum of TET (NOT USED)
C    SE1      Assuming this is the monthly sum of E1 (NOT USED)
C    SE2      Assuming this is the monthly sum of E2 (NOT USED)
C    SE3      Assuming this is the monthly sum of E3 (NOT USED)
C    SE4      Assuming this is the monthly sum of E4 (NOT USED)
C    SE5      Assuming this is the monthly sum of E5 (NOT USED)
C    RSUM(7)  Sums of (1) TCI, (2) ROIMP, (3) SDRO, (4) SSUR, (5) SIF,
C                 (6) BFS, (7) BFP. (NOT USED)
        """

        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        ### TODO: need to check DT value
        DT = 1  # Daily time interval
        warm_up = 0

        # Initialization
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                warm_up_model = NWM_SACSMA_Mul().to(args["device"])
                Qsrout, SNOWPACK_storage, MELTWATER_storage, UZTWC, UZFWC, LZTWC, \
                LZFPC, LZFSC = warm_up_model(xinit, c_hydro_model, params_raw, args,
                                                                      muwts=None, warm_up=0, init=True, routing=False,
                                                                      comprout=False, conv_params_hydro=None)
        else:
            # Without buff time, initialize state variables with zeros
            Ngrid = x_hydro_model.shape[1]
            SNOWPACK_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            MELTWATER_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            UZTWC = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            UZFWC = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            LZTWC = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            LZFPC = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            LZFSC = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            ADIMC = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        ## parameters for prms_marrmot. there are 18 parameters in it. we take all params and make the changes
        # inside the for loop
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = self.change_param_range(param=params_raw[:, :, num, :],
                                                             bounds=self.parameters_bound[param])

        vars = args["varT_hydro_model"]
        vars_c = args["varC_hydro_model"]
        P = x_hydro_model[warm_up:, :, vars.index("prcp(mm/day)")]
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        mean_air_temp = x_hydro_model[warm_up:, :, vars.index('tmean(C)')].unsqueeze(2).repeat(1, 1, nmul)

        if args["potet_module"] == "potet_hamon":
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.004, 0.008], ndays=No_days, nmul=args["nmul"])
            PET = get_potet(
                args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=PET_coef
            )  # mm/day
        elif args["potet_module"] == "potet_hargreaves":
            day_of_year = x_hydro_model[warm_up:, :, vars.index("dayofyear")].unsqueeze(-1).repeat(1, 1, nmul)
            lat = c_hydro_model[:, vars_c.index("lat")].unsqueeze(0).unsqueeze(-1).repeat(day_of_year.shape[0], 1, nmul)
            Tmaxf = x_hydro_model[warm_up:, :, vars.index("tmax(C)")].unsqueeze(2).repeat(1, 1, nmul)
            Tminf = x_hydro_model[warm_up:, :, vars.index("tmin(C)")].unsqueeze(2).repeat(1, 1, nmul)
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
        Q_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        srflow_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        ssflow_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        gwflow_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        AET = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        tosoil_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        PC_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        pcfw_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        pctw_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        pcfws_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        twexls_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        twexlp_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        Rls_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        Rlp_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        Elztw_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        Euzfw_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        Twexu_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        Ru_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        SWE_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        # do static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in args["dyn_params_list_hydro"]:  ## it is a static parameter
                params_dict[key] = params_dict_raw[key][-1, :, :]

        Nstep, Ngrid = P.size()

        ### Doing dynamic parameters based on dydrop ratio
        # basically, it drops dynamic parameters for some basins (based on dydrop ratio), and substitute them
        # for a static parameter, which is the last day parameter
        if len(args["dyn_params_list_hydro"]) > 0:
            params_dict_raw_dyn = dict()
            pmat = torch.ones([Ngrid, 1]) * args["dydrop"]
            for i, key in enumerate(args["dyn_params_list_hydro"]):
                drmask = torch.bernoulli(pmat).detach_().to(args["device"])
                dynPar = params_dict_raw[key]
                staPar = params_dict_raw[key][-1, :, :].unsqueeze(0).repeat([dynPar.shape[0], 1, 1])
                params_dict_raw_dyn[key] = dynPar * (1 - drmask) + staPar * drmask
        ###

        for t in range(Nstep):
            # do dynamic parameters
            for key in params_dict_raw.keys():
                if key in args["dyn_params_list_hydro"]:  ## it is a dynamic parameter
                    # params_dict[key] = params_dict_raw[key][warm_up + t, :, :]
                    # to drop dynamic parameters as static in some basins
                    params_dict[key] = params_dict_raw_dyn[key][warm_up + t, :, :]

            uztwm = params_dict["f1"] * params_dict["smax"]
            uzfwm = torch.clamp(params_dict["f2"] * (params_dict["smax"] - uztwm), min=0.005 / 4)
            lztwm = torch.clamp(params_dict["f3"] * (params_dict["smax"] - uztwm - uzfwm), min=0.005 / 4)
            lzfpm = torch.clamp(params_dict["f4"] * (params_dict["smax"] - uztwm - uzfwm - lztwm), min=0.005 / 4)
            lzfsm = torch.clamp((1 - params_dict["f4"]) * (params_dict["smax"] - uztwm - uzfwm - lztwm), min=0.005 / 4)
            pbase = lzfpm * params_dict["klzp"] + lzfsm * params_dict["klzs"]
            zperc = torch.clamp((lztwm + lzfsm * (1 - params_dict["klzs"])) / (lzfsm * params_dict["klzs"] +
                                                                                lzfpm * params_dict["klzp"]) +
                                (lzfpm * (1 - params_dict["klzp"])) / (lzfsm * params_dict["klzs"] +
                                                                        lzfpm * params_dict["klzp"]), max=100000.0)

            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (mean_air_temp[t, :, :] >= params_dict["parTT"]).type(torch.float32))
            SNOW = torch.mul(PRECIP, (mean_air_temp[t, :, :] < params_dict["parTT"]).type(torch.float32))

            # Snow
            SNOWPACK_storage = SNOWPACK_storage + SNOW
            melt = params_dict["parCFMAX"] * (mean_air_temp[t, :, :] - params_dict["parTT"])
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK_storage)
            MELTWATER_storage = MELTWATER_storage + melt
            SNOWPACK_storage = torch.clamp(SNOWPACK_storage - melt, min=NEARZERO)
            refreezing = params_dict["parCFR"] * params_dict["parCFMAX"] * (
                        params_dict["parTT"] - mean_air_temp[t, :, :])
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER_storage)
            SNOWPACK_storage = SNOWPACK_storage + refreezing
            MELTWATER_storage = torch.clamp(MELTWATER_storage - refreezing - NEARZERO, min=NEARZERO)
            tosoil = MELTWATER_storage - (params_dict["parCWH"] * SNOWPACK_storage)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER_storage = torch.clamp(MELTWATER_storage - tosoil - NEARZERO, min=NEARZERO)
            ####
            ## TODO: tosoil is basically the melted water ready to join the network. Please check this statement
            PXV = RAIN + tosoil  # PXV: Input moisture (e.g. precip, precip+melt)
            EDMND = PET[t, :, :]

            E1 = EDMND * UZTWC / uztwm   # E1 = flux_Euztw in marrmot
            RED = torch.clamp(EDMND-E1, min=NEARZERO)   ## RED IS RESIDUAL EVAP DEMAND
            UZTWC = torch.clamp(UZTWC-E1, min=NEARZERO)

            E2 = torch.where((UZFWC > RED),
                             RED,
                             UZFWC
                             )

            UZFWC = torch.clamp(UZFWC - E2, min=NEARZERO)
            RED = torch.clamp(RED - E2, min=NEARZERO)

            E3 = torch.where((UZTWC / uztwm) > (UZFWC / uzfwm),
                             RED * (LZTWC / (uztwm + lztwm)),
                             torch.zeros(E1.shape, dtype=torch.float32, device=args["device"]))

            # UPPER ZONE FREE WATER RATIO EXCEEDS UPPER ZONE
            # TENSION WATER RATIO, THUS TRANSFER FREE WATER TO TENSION
            UZRAT = torch.where((UZTWC / uztwm) < (UZFWC / uzfwm),
                                RED * (LZTWC / (uztwm + lztwm)),
                                torch.zeros(E1.shape, dtype=torch.float32, device=args["device"]))

            UZTWC = torch.where((UZTWC / uztwm) < (UZFWC / uzfwm),
                                uztwm * UZRAT,
                                UZTWC)

            UZFWC = torch.where((UZTWC / uztwm) < (UZFWC / uzfwm),
                                        uzfwm * UZRAT,
                                UZFWC)

            # E3 implementation, E3 cannot exceed LZTWC
            E3 = torch.min(LZTWC, E3)
            LZTWC = torch.clamp(LZTWC - E3, min=NEARZERO)
            RATLZT = LZTWC / lztwm   # for dynamic parametrization, this line doesn't work

            SAVED = params_dict["RSERV"] * (lzfpm + lzfsm)

            RATLZ = (LZTWC + LZFPC  + LZFSC - SAVED) / (lztwm + lzfpm + lzfsm - SAVED)

            E5 = torch.where((RATLZT > RATLZ),
                              E1 + (RED + E2) *((ADIMC - E1 - )),
                              torch.zeros(E1.shape, dtype=torch.float32, device=args["device"]))
            E5 = torch.min(E5, ADIMC)
            ADIMC = torch.clamp(ADIMC - E5, min=NEARZERO)
            # ADIMP is the additional impervious area, not sure why we multiply it to ADIMP
            E5 = E5 * ADIMP
            ## DEL equals to Rls in marrmot version
            DEL = torch.where((RATLZT < RATLZ),
                              (RATLZ - RATLZT) * lztwm,
                              torch.zeros(E1.shape, dtype=torch.float32, device=args["device"]))
            LZTWC = LZTWC + DEL
            LZFSC = torch.clamp(LZFSC - DEL, min=NEARZERO)
            ### TODO: I think the above two lines are not correct, although they are from NWM fortran code.
            ### The order matters. it should be something like:
            # DEL = torch.min(LZFSC, DEL)
            # LZFSC = torch.clamp(LZFSC - DEL, min=NEARZERO)
            # LZTWC = LZTWC + DEL
            #----------------------------
            # COMPUTE PERCOLATION AND RUNOFF AMOUNTS.
            # PXV is input moisture i.e. Precip, or Precip + melt
            TWX = UZTWC + PXV - uztwm
            UZTWC = torch.where((TWX > torch.zeros(TWX.shape, dtype=torch.float32, device=args["device"])),
                               uztwm,
                               UZTWM + PXV)
            TWX = torch.clamp(TWX, min=NEARZERO)
            # MOISTURE AVAILABLE IN EXCESS OF UZTW STORAGE
            ADIMC = ADMIC + PXV - TWX

            # COMPUTE IMPERVIOUS AREA RUNOFF.
            ROIMP = PXV * params_dict["pctim"]   # ROIMP    Impervious runoff from the permanent impervious area
            # ROIMP IS RUNOFF FROM THE MINIMUM IMPERVIOUS AREA.
            SIMPVT = SIMPVT + ROIMP    # sum of ROIMP

            # DETERMINE COMPUTATIONAL TIME INCREMENTS FOR THE BASIC TIME INTERVAL
            NINC = 1 + 0.2 * (UZFWC + TWX)     # NINC=NUMBER OF TIME INCREMENTS THAT THE TIME INTERVAL
            DINC = (1 / NINC) * DT   # DT: computational time interval
            PINC = TWX / NINC    # PINC=AMOUNT OF AVAILABLE MOISTURE FOR EACH INCREMENT

            # COMPUTE FREE WATER DEPLETION FRACTIONS FOR
            # THE TIME INCREMENT BEING USED-BASIC DEPLETIONS
            # ARE FOR ONE DAY
            DUZ = 1.0 - ((1.0 - params_dict["UZK"]) ** DINC)   # UZK in NWM version = Kuz in marrmot version
            DLZP = 1.0 - ((1.0 - params_dict["LZPK"]) ** DINC)   # LZPK in NWM version = Klzp in marrmot version
            DLZS = 1.0 - ((1.0 - params_dict["LZSK"]) ** DINC)   # LZSK in NWM version = Klzs in marrmot version

            ## INFERRED PARAMETER (ADDED BY Q DUAN ON 3/6/95)
            PAREA = 1.0 - ADIMP - params_dict["pctim"]

            ## ---------------------------------
            # START INCREMENTAL DO LOOP FOR THE TIME INTERVAL.
            SPBF = torch.zeros(UZTWC.shape, dtype=torch.float32, device=args["device"])
            SBF = torch.zeros(UZTWC.shape, dtype=torch.float32, device=args["device"])
            SPERC = torch.zeros(UZTWC.shape, dtype=torch.float32, device=args["device"])
            SIF = torch.zeros(UZTWC.shape, dtype=torch.float32, device=args["device"])
            SSUR = torch.zeros(UZTWC.shape, dtype=torch.float32, device=args["device"])
            for i in range(1, NINC + 1):
                ## COMPUTE DIRECT RUNOFF (FROM ADIMP AREA)
                RATIO = torch.clamp((ADIMC - UZTWC) / lztwm, min=0.0)
                # ADDRO IS THE AMOUNT OF DIRECT RUNOFF FROM THE AREA ADIMP.
                ADDRO = PINC * (RATIO ** 2)
                # COMPUTE BASEFLOW AND KEEP TRACK OF TIME INTERVAL SUM.
                BF = torch.clamp(LZFPC * DLZP, min=0.0)   # baseflow primary and secondary
                BF = torch.min(BF, LZFPC)
                ## TODO: since it is a for loop, I need to chack min=NEARZERO is not adding to much error to the system
                LZFPC = torch.clamp(LZFPC - BF, min=NEARZERO)
                SPBF = SPBF + BF

                BF = torch.clamp(LZFSC * DLZS, min=0.0)
                BF = torch.min(BF, LZFSC)
                ## TODO: since it is a for loop, I need to chack min=NEARZERO is not adding to much error to the system
                LZFSC = torch.clamp(LZFSC - BF, min=NEARZERO)
                SBF = SBF + BF

                # COMPUTE PERCOLATION-IF NO WATER AVAILABLE THEN SKIP
                # to do it in parallel, we need to keep this PINC + UZFWC <> 0.01 to handle GOTO commands in fortran
                UZFWC = torch.where((PINC + UZFWC < 0.01 ),
                                    PINC + UZFWC,
                                    UZFWC)
                PERCM = torch.where((PINC + UZFWC > 0.01 ),
                                    lzfpm * DLZP + lzfsm *DLZS,
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                PERC = torch.where((PINC + UZFWC > 0.01 ),
                                    PERCM * (UZFWC / uzfwm),
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                ## equivalent to LZ_deficiency / LZ_capacity in marrmot
                DEFR = torch.where((PINC + UZFWC > 0.01 ),
                                   1.0 - ((LZTWC + LZFPC LZFSC) / (lztwm + lzfpm + lzfsm)),
                                   torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))

                ## TODO: need to add a function for considering the effect of frzen ground on PERC
                ## For now IFRZE = 0.0 --> no need to call FGFR1
                ## COMPUTES THE CHANGE IN THE PERCOLATION AND INTERFLOW WITHDRAWAL RATES DUE TO FROZEN GROUND
                # FR, FI = self.FGFR1(self, DEFR, LZTWC, LZFSC, LZFPC, UZTWC, UZFWC, LZFPC,
                #                     ADIMC)
                FR = torch.tensor(1.0, dtype=torch.float32, device=args["device"])
                FI = torch.tensor(1.0, dtype=torch.float32, device=args["device"])

                PERC = torch.where((PINC + UZFWC > 0.01 ),
                                   PERC * (1.0 + ZPERC * (DEFR ** params_dict["rexp"])) * FR,
                                   torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                # NOTE...PERCOLATION OCCURS FROM UZFWC BEFORE PAV IS ADDED.
                PERC = torch.where((PINC + UZFWC > 0.01 ) & (PERC > UZFWC),
                                   UZFWC,
                                   torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                UZFWC= torch.where((PINC + UZFWC > 0.01 ),
                                    UZFWC - PERC,
                                    UZFWC)
                ## TODO: not sure if I need this line without having the condition PINC + UZFWC > 0.01
                UZFWC = torch.clamp(UZFWC, min=NEARZERO)
                # CHECK TO SEE IF PERCOLATION EXCEEDS LOWER ZONE DEFICIENCY.
                CHECK = torch.where((PINC + UZFWC > 0.01 ),
                                    LZTWC + LZFPC + LZFSC + PERC - lztwm - lzfpm - lzfsm,
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                PERC = torch.where((PINC + UZFWC > 0.01 ) & (CHECK > 0.0),
                                    PERC - CHECK,
                                    PERC)
                UZFWC = torch.where((PINC + UZFWC > 0.01 ) & (CHECK > 0.0),
                                     UZFWC + CHECK,
                                     UZFWC)
                # SPERC IS THE TIME INTERVAL SUMMATION OF PERC
                SPERC = torch.where((PINC + UZFWC > 0.01 ),
                                     SPERC + PERC,
                                     SPERC)

                # COMPUTE INTERFLOW AND KEEP TRACK OF TIME INTERVAL SUM.
                # NOTE...PINC HAS NOT YET BEEN ADDED
                DEL_uz = torch.where((PINC + UZFWC > 0.01 ),
                                      UZFWC * DUZ * FI,
                                      torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                # SIF: sum of interflow
                SIF = torch.where((PINC + UZFWC > 0.01 ),
                                     SIF + DEL_uz,
                                     torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                UZFWC = torch.where((PINC + UZFWC > 0.01 ),
                                     UZFWC - DEL_uz,
                                     UZFWC)

                # DISTRIBE PERCOLATED WATER INTO THE LOWER ZONES
                # TENSION WATER MUST BE FILLED FIRST EXCEPT FOR THE PFREE AREA.
                # PERCT IS PERCOLATION TO TENSION WATER AND PERCF IS PERCOLATION GOING TO FREE WATER.
                # PERCT is equivalent to PCtw in marrmot
                PERCT = torch.where((PINC + UZFWC > 0.01 ),
                                    PERC * (1.0 - params_dict["pfree"]),
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                # if (PERCT + LZTWC > lztwm)
                PERCF = torch.where((PINC + UZFWC > 0.01 ) & (PERCT + LZTWC > lztwm),
                                    PERCT + LZTWC - lztwm,
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                LZTWC = torch.where((PINC + UZFWC > 0.01 ) & (PERCT + LZTWC > lztwm),
                                    lztwm,
                                    LZTWC)

                # if (PERCT + LZTWC < lztwm)
                # for PERCF with (PERCT + LZTWC < lztwm) : it has been taken care of on the upper lines
                LZTWC = torch.where((PINC + UZFWC > 0.01 ) & (PERCT + LZTWC < lztwm),
                                    LZTWC + PERCT,
                                    LZTWC)

                # line 426 in sac.f with label 244
                # DISTRIBUTE PERCOLATION IN EXCESS OF TENSION
                # REQUIREMENTS AMONG THE FREE WATER STORAGES.
                PERCF = torch.where((PINC + UZFWC > 0.01 ),
                                    PERCF + PERC * PFREE,
                                    PERCF)

                # HPL IS THE RELATIVE SIZE OF THE PRIMARY STORAGE
                # AS COMPARED WITH TOTAL LOWER ZONE FREE WATER STORAGE.
                HPL = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                  lzfpm / (lzfpm + lzfsm),
                                  torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))

                # RATLP AND RATLS ARE CONTENT TO CAPACITY RATIOS, OR
                # IN OTHER WORDS, THE RELATIVE FULLNESS OF EACH STORAGE
                RATLP = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                  LZFPC / (lzfpm),
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))

                RATLS = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                    LZFSC / lzfsm,
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                # FRACP IS THE FRACTION GOING TO PRIMARY.
                FRACP = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                    (HPL * 2.0 * (1..0 - RATLP)) / ((1.0 - RATLP) + (1.0 - RATLS)),
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))

                FRACP = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0) & (FRACP > 1.0),
                                    torch.ones(FRACP.shape, dtype=torch.float32, device=args["device"]),
                                    FRACP)

                # PERCP AND PERCS ARE THE AMOUNT OF THE EXCESS
                # PERCOLATION GOING TO PRIMARY AND SUPPLEMENTAL STORGES,RESPECTIVELY.
                PERCP = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                    PERCF * FRACP,
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                PERCS = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                    PERCF - PERCP,
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))

                # PERCS = torch.min(PERCS, torch.clamp(lzrfsm - LZFSC, min=0.0))
                LZFSC = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                    LZFSC + PERCS,
                                    LZFSC)
                PERCS = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0) & (LZFSC > lzfsm),
                                    PERCS - LZFSC + lzfsm,
                                    PERCS)
                LZFSC = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0) & (LZFSC > lzfsm),
                                    lzfsm,
                                    LZFSC)
                #label 246 in Fortran code sac1.f
                LZFPC = torch.where((PINC + UZFWC > 0.01 ) & (PERCF != 0.0),
                                    LZFPC + (PERCF - PERCS),
                                    LZFPC)
                # CHECK TO MAKE SURE LZFPC DOES NOT EXCEED LZFPM
                # TODO: need to recheck if the following line needs to be written by torch.where()
                EXCESS = torch.clamp(LZFPC - lzfpm, min=NEARZERO)
                LZTWC = torch.where(((PINC + UZFWC) > 0.01 ) & (PERCF != 0.0) & (LZFPC > lzfpm),
                                    LZTWC + EXCESS,
                                    LZTWC)
                LZFPC = torch.where(((PINC + UZFWC) > 0.01 ) & (PERCF != 0.0) & (LZFPC > lzfpm),
                                    lzfpm,
                                    LZFPC)

                ## line 455 in Fortran sac1.f, label 245
                # DISTRIBUTE PINC BETWEEN UZFWC AND SURFACE RUNOFF.
                # if (PINC + UZFWC) < uzfwm
                UZFWC = torch.where(((PINC + UZFWC) > 0.01 ) &
                                    (PINC > 0.0) & ((PINC + UZFWC) < uzfwm),
                                    UZFWC + PINC,
                                    UZFWC)

                ## Label 248
                # if (PINC + UZFWC) > uzfwm
                # COMPUTE SURFACE RUNOFF (SUR) AND KEEP TRACK OF TIME INTERVAL SUM.
                SUR = torch.where(((PINC + UZFWC) > 0.01 ) & (PINC > 0.0) & ((PINC + UZFWC) > uzfwm),
                                    PINC + UZFWC - uzfwm,
                                  torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                UZFWC = torch.where(((PINC + UZFWC) > 0.01 ) & (PINC > 0.0) & ((PINC + UZFWC) > uzfwm),
                                  uzfwm,
                                    UZFWC)

                # SSUR: Sum of the indirect surface runoff components from ADIMP and PAREA
                SSUR = torch.where(((PINC + UZFWC) > 0.01 ) & (PINC > 0.0) & ((PINC + UZFWC) > uzfwm),
                                   SSUR + SUR * PAREA,
                                   SSUR)

                # ADSUR IS THE AMOUNT OF SURFACE RUNOFF WHICH COMES
                # FROM THAT PORTION OF ADIMP WHICH IS NOT
                # CURRENTLY GENERATING DIRECT RUNOFF.  ADDRO/PINC
                # IS THE FRACTION OF ADIMP CURRENTLY GENERATING
                # DIRECT RUNOFF.
                ADSUR = torch.where(((PINC + UZFWC) > 0.01 ) & (PINC > 0.0) & ((PINC + UZFWC) > uzfwm),
                                    SUR * (1.0 - ADDRO / PINC),
                                    torch.zeros(UZFWC.shape, dtype=torch.float32, device=args["device"]))
                SSUR = torch.where(((PINC + UZFWC) > 0.01 ) & (PINC > 0.0) & ((PINC + UZFWC) > uzfwm),
                                   SSUR + ADSUR * ADIMP,
                                   SSUR)

                # line 476 in Fortran sac1.f, label 249 --> All conditions are gone here.
                ADIMC = ADIMC + PINC - ADDRO - ADSUR
                ADDRO = torch.where()

















































































            flux_qdir = self.split_1(params_dict["pctim"], RAIN)
            flux_peff = self.split_1(1 - params_dict["pctim"], RAIN)
            UZTWC = UZTWC + flux_peff + tosoil
            ## to make sure UZTW_storage < uztwm
            flux_Twexu = torch.clamp(UZTWC - uztwm, min=0.0)
            UZTWC = torch.clamp(UZTWC - flux_Twexu - NEARZERO, min=NEARZERO)
            ## to make sure UZFW_storage < uzfwm
            UZFWC = UZFWC + flux_Twexu
            flux_Qsur = torch.clamp(UZFWC - uzfwm, min=0.0)
            UZFWC = torch.clamp(UZFWC - flux_Qsur - NEARZERO, min=NEARZERO)

            flux_Ru = torch.where((UZTWC / uztwm) < (UZFWC / uzfwm),
                                  (uztwm * UZFWC - uzfwm * UZTWC) / (uztwm + uzfwm),
                                  torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Ru = torch.min(flux_Ru, UZFWC)
            UZFWC = torch.clamp(UZFWC - flux_Ru - NEARZERO, min=NEARZERO)
            UZTWC = UZTWC + flux_Ru

            # redo UZTW_storage and UZFW_storage
            ## to make sure UZTW_storage < uztwm
            extra_flux_UZTW = torch.clamp(UZTWC - uztwm, min=0.0)
            UZTWC = torch.clamp(UZTWC - extra_flux_UZTW - NEARZERO, min=NEARZERO)
            flux_Twexu = flux_Twexu + extra_flux_UZTW
            ## to make sure UZFW_storage < uzfwm
            UZFWC = UZFWC + extra_flux_UZTW
            extra_flux_UZFW = torch.clamp(UZFWC - uzfwm, min=0.0)
            flux_Qsur = flux_Qsur + extra_flux_UZFW
            UZFWC = torch.clamp(UZFWC - extra_flux_UZFW - NEARZERO, min=NEARZERO)

            flux_Euztw = Ep * UZTWC / uztwm
            flux_Euztw = torch.min(flux_Euztw, UZTWC)
            UZTWC = torch.clamp(UZTWC - flux_Euztw, min=NEARZERO)
            flux_Qint = params_dict["kuz"] * UZFWC
            UZFWC = torch.clamp(UZFWC - flux_Qint, min=NEARZERO)


            # to make sure LZTW_storage and LZFWP_storage and LZFWS_storage are lower than max volume
            # This is important for the case of dynamic smax. if smax is smaller than smax in previous days,
            # the water should be released first, before going through any equation.
            flux_twexl = torch.clamp(LZTWC - lztwm, min=0.0)
            LZTWC = torch.clamp(LZTWC - flux_twexl - NEARZERO, min=NEARZERO)
            flux_Qbfp = torch.clamp(LZFPC - lzfpm, min=0.0)
            LZFPC = torch.clamp(LZFPC - flux_Qbfp - NEARZERO, min=NEARZERO)
            flux_Qbfs = torch.clamp(LZFSC - lzfsm, min=0.0)
            LZFSC = torch.clamp(LZFSC - flux_Qbfs - NEARZERO, min=NEARZERO)

            # go to the equations
            LZ_deficiency = (lztwm - LZTWC) + (lzfpm - LZFPC) + (lzfsm - LZFSC)
            LZ_deficiency = torch.clamp(LZ_deficiency, min=0.0)    # just to make sure there is no negative values
            LZ_capacity = lztwm + lzfsm + lzfpm
            Pc_demand = pbase * (1 + (zperc * ((LZ_deficiency / (LZ_capacity+0.0001)) ** (1 + params_dict["rexp"]))))
            flux_Pc = Pc_demand * UZFWC / uzfwm
            flux_Pc = torch.min(flux_Pc, UZFWC)
            UZFWC = torch.clamp(UZFWC - flux_Pc, min=NEARZERO)
            flux_Euzfw = torch.clamp(Ep - flux_Euztw, min=0.0)
            flux_Euzfw = torch.min(flux_Euzfw, UZFWC)
            UZFWC = torch.clamp(UZFWC - flux_Euzfw, min=NEARZERO)


            Rl_nominator = -LZTWC * (lzfpm + lzfsm) + lztwm * (LZFPC + LZFSC)
            Rl_denominator = (lzfpm + lzfsm) * (lztwm + lzfpm + lzfsm)
            flux_Rlp = torch.where((LZTWC / lztwm) < ((LZFPC + LZFSC) / (lzfpm + lzfsm)),
                                   lzfpm * (Rl_nominator / Rl_denominator),
                                   torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Rlp = torch.min(flux_Rlp, LZFPC)
            LZFPC = torch.clamp(LZFPC - flux_Rlp, min=NEARZERO)
            LZTWC = LZTWC + flux_Rlp
            ## if LZTW_storage > lztwm, we add the extra to flux_twexl
            extra_LZTW_storage1 = torch.clamp(LZTWC - lztwm, min=0.0)
            flux_twexl = flux_twexl + extra_LZTW_storage1
            LZTWC = torch.clamp(LZTWC - extra_LZTW_storage1 - NEARZERO, min=NEARZERO)

            flux_Rls = torch.where((LZTWC / lztwm) < ((LZFPC + LZFSC) / (lzfpm + lzfsm)),
                                   lzfsm * (Rl_nominator / Rl_denominator),
                                   torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Rls = torch.min(flux_Rls, LZFSC)
            LZFSC = torch.clamp(LZFSC - flux_Rls, min=NEARZERO)
            LZTWC = LZTWC + flux_Rls

            flux_Pctw = (1 - params_dict["pfree"]) * flux_Pc
            flux_Pcfw = params_dict["pfree"] * flux_Pc
            LZTWC = LZTWC + flux_Pctw
            ## this is the second time I added the extra water to flux_twexl
            extra_LZTW_storage2 = torch.clamp(LZTWC - lztwm, min=0.0)
            flux_twexl = flux_twexl + extra_LZTW_storage2
            LZTWC = torch.clamp(LZTWC - extra_LZTW_storage2 - NEARZERO, min=NEARZERO)

            flux_Elztw = torch.where((LZTWC > 0.0) & (Ep > flux_Euztw + flux_Euzfw),
                                     (Ep - flux_Euztw - flux_Euzfw) * (LZTWC / (uztwm + lztwm)),
                                     torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Elztw = torch.min(flux_Elztw, LZTWC)
            LZTWC = torch.clamp(LZTWC - flux_Elztw, min=NEARZERO)

            flux_Pcfwp = torch.clamp(((lzfpm - LZFPC) / (lzfpm * (((lzfpm - LZFPC) / lzfpm) + (
                        (lzfsm - LZFSC) / lzfsm)) + NEARZERO)) * flux_Pcfw, min=0.0)
            flux_twexlp = torch.clamp(((lzfpm - LZFPC) / (lzfpm * (((lzfpm - LZFPC) / lzfpm) + (
                        (lzfsm - LZFSC) / lzfsm)) + NEARZERO)) * flux_twexl, min=0.0)
            LZFPC = LZFPC + flux_Pcfwp + flux_twexlp

            flux_Qbfp2 = params_dict["klzp"] * LZFPC
            flux_Qbfp = flux_Qbfp + flux_Qbfp2
            LZFPC = torch.clamp(LZFPC - flux_Qbfp2, min=NEARZERO)
            # checking extra water for the second time
            extra_LZFWP = torch.clamp(LZFPC - lzfpm, min=0.0)
            LZFPC = torch.clamp(LZFPC - extra_LZFWP - NEARZERO,
                                        min=NEARZERO)  # I added this to make the storage not to exceed the max
            # just to make sure LZFWP_storage is always smaller than lzfwpm. we need it to calculate flux_twexls
            LZFPC = torch.where(LZFPC >= lzfpm,
                                        lzfpm - 0.0001,
                                        LZFPC)
            flux_Qbfp = flux_Qbfp + extra_LZFWP
            # This line needs to be rechecked with the documents (flux_Pcfws + flux_Pcfwp != flux_Pcfw
            # flux_Pcfws = ((lzfwsm - LZFWS_storage) / (
            #             lzfwsm * ((lzfwsm - LZFWP_storage) / lzfwpm) + ((lzfwsm - LZFWS_storage) / lzfwsm))) * flux_Pcfw
            flux_Pcfws = torch.clamp(flux_Pcfw - flux_Pcfwp, min=0.0)
            flux_twexls = ((lzfsm - LZFSC) / (lzfsm * (((lzfpm - LZFPC) / lzfpm) + (
                        (lzfsm - LZFSC) / lzfsm)) + NEARZERO)) * flux_twexl
            LZFSC = LZFSC + flux_Pcfws + flux_twexls

            flux_Qbfs2 = params_dict["klzs"] * LZFSC
            LZFSC = torch.clamp(LZFSC - flux_Qbfs2, min=NEARZERO)
            flux_Qbfs = flux_Qbfs + flux_Qbfs2
            extra_LZFWS = torch.clamp(LZFSC - lzfsm, min=0.0)
            LZFSC = torch.clamp(LZFSC - extra_LZFWS - NEARZERO,
                                        min=NEARZERO)  # I added this to make the storage not to exceed the max
            # just to make sure LZFWS_storage is always smaller than lzfwsm
            LZFSC = torch.where(LZFSC >= lzfsm,
                                        lzfsm - 0.0001,
                                        LZFSC)
            flux_Qbfs = flux_Qbfs + extra_LZFWS
            Q_sim[t, :, :] = flux_qdir + flux_Qsur + flux_Qint + flux_Qbfp + flux_Qbfs
            srflow_sim[t, :, :] = flux_qdir + flux_Qsur
            ssflow_sim[t, :, :] = flux_Qint
            gwflow_sim[t, :, :] = flux_Qbfp + flux_Qbfs
            AET[t, :, :] = flux_Euztw + flux_Euzfw + flux_Elztw
            tosoil_sim[t, :, :] = tosoil
            PC_sim[t, :, :] = flux_Pc
            pcfw_sim[t, :, :] = flux_Pcfw
            pctw_sim[t, :, :] = flux_Pctw
            pcfws_sim[t, :, :] = flux_Pcfws
            twexls_sim[t, :, :] = flux_twexls
            twexlp_sim[t, :, :] = flux_twexlp
            Rlp_sim[t, :, :] = flux_Rlp
            Rls_sim[t, :, :] = flux_Rls
            Elztw_sim[t, :, :] = flux_Elztw
            Euzfw_sim[t, :, :] = flux_Euzfw
            Twexu_sim[t, :, :] = flux_Twexu
            Ru_sim[t, :, :] = flux_Ru
            SWE_sim[t, :, :] = SNOWPACK_storage

        if routing == True:
            tempa = self.change_param_range(param=conv_params_hydro[:, 0],
                                            bounds=self.conv_routing_hydro_model_bound[0])
            tempb = self.change_param_range(param=conv_params_hydro[:, 1],
                                            bounds=self.conv_routing_hydro_model_bound[1])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            # Q_sim_new = Q_sim.mean(-1, keepdim=True).permute(1,0,2)
            UH = self.UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = Q_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = self.UH_conv(rf, UH).permute([2, 0, 1])

            rf_srflow = srflow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            srflow_rout = self.UH_conv(rf_srflow, UH).permute([2, 0, 1])

            rf_ssflow = ssflow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            ssflow_rout = self.UH_conv(rf_ssflow, UH).permute([2, 0, 1])

            rf_gwflow = gwflow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            gwflow_rout = self.UH_conv(rf_gwflow, UH).permute([2, 0, 1])
        else:
            Qsrout = Q_sim.mean(-1, keepdim=True)
            srflow_rout = srflow_sim.mean(-1, keepdim=True)
            ssflow_rout = ssflow_sim.mean(-1, keepdim=True)
            gwflow_rout = gwflow_sim.mean(-1, keepdim=True)

        if init:  # means we are in warm up. here we just return the storages to be used as initial values
            return Qsrout, SNOWPACK_storage, MELTWATER_storage, UZTWC, UZFWC, LZTWC, \
                LZFPC, LZFSC

        else:
            return dict(flow_sim=Qsrout,
                        srflow=srflow_rout,
                        ssflow=ssflow_rout,
                        gwflow=gwflow_rout,
                        PET_hydro=PET.mean(-1, keepdim=True),
                        AET_hydro=AET.mean(-1, keepdim=True),
                        flow_sim_no_rout=Q_sim.mean(-1, keepdim=True),
                        srflow_no_rout=srflow_sim.mean(-1, keepdim=True),
                        ssflow_no_rout=ssflow_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=gwflow_sim.mean(-1, keepdim=True),
                        tosoil=tosoil_sim.mean(-1, keepdim=True),
                        flux_pc=PC_sim.mean(-1, keepdim=True),
                        flux_pcfw=pcfw_sim.mean(-1, keepdim=True),
                        flux_pctw=pctw_sim.mean(-1, keepdim=True),
                        flux_pcfws=pcfws_sim.mean(-1, keepdim=True),
                        flux_twexlp=twexlp_sim.mean(-1, keepdim=True),
                        flux_twexls=twexls_sim.mean(-1, keepdim=True),
                        flux_Rlp=Rlp_sim.mean(-1, keepdim=True),
                        flux_Rls=Rls_sim.mean(-1, keepdim=True),
                        flux_Euzfw=Euzfw_sim.mean(-1, keepdim=True),
                        flux_Elztw=Elztw_sim.mean(-1, keepdim=True),
                        flux_Twexu=Twexu_sim.mean(-1, keepdim=True),
                        flux_Ru=Ru_sim.mean(-1, keepdim=True),
                        SWE_sim=SWE_sim.mean(-1, keepdim=True)
                        )
