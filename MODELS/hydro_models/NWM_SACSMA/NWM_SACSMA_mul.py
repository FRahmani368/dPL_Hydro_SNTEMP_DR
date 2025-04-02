###
# Written by Farshid Rahmani,Penn State University
import torch
from MODELS.PET_models.potet import get_potet
import torch.nn.functional as F

class NWM_SACSMA_Mul(torch.nn.Module):
    """ from fortran code https://github.com/NOAA-OWP/sac-sma/blob/master/src/sac/sac1.f
    C    ----- VARIABLES -----
C    DT       Computational time interval
C    IFRZE    Frozen ground module switch.  0 = No frozen ground module,
C                 1 = Use frozen ground module
C    EDMND    ET demand for the time interval, Farshid's note: It is the potential evapotranspiration for time interval
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
C    SPBF     Sum of the incremental LZ primary baseflow component onlydef split_1(self, p1, In):

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

    def __init__(self):
        """Initiate a HBV instance"""
        super(NWM_SACSMA_Mul, self).__init__()
        self.parameters_bound = dict(uztwm=[1, 20],
                                     uzfwm =[0.25, 3000],    #0.0025
                                     lztwm=[0.25, 3000],
                                     lzfsm=[0.25, 3000],
                                     lzfpm=[0.25, 3000],
                                     UZK=[0.0, 1.0],
                                     pctim=[0.0, 1.0],
                                     ADIMP=[0, 1],    #[1, 3000]
                                     RIVA=[0.0, 0.001],   ## it's basically zero
                                     ZPERC=[0.005, 10000],
                                     rexp=[0.0, 8],      #[0.0, 8]
                                     LZSK=[0, 1],
                                     LZPK=[0, 1],
                                     PFREE=[0, 1],
                                     rserv=[0.3, 0.3001],   ## always 0.3
                                     SIDE=[0.0, 0.0001],   # The atio of unobserved to observed flow
                                     parTT=[-3.0, 3.0],   #[-2.5, 2.5]   ###not in SACSMA. TO handle snow to precip
                                     parCFMAX=[0.4, 12],   # [0.4, 12]
                                     parCFR=[0, 0.1],
                                     parCWH=[0, 0.2]
                                     )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]  # routing parameter b
        ]


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
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        dtype = torch.float32
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
            SNOWPACK_storage = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            MELTWATER_storage = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            UZTWC = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            UZFWC = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            LZTWC = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            LZFPC = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            LZFSC = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            ADIMC = (torch.zeros([Ngrid, nmul], dtype=dtype) + 0.0001).to(args["device"])
            # ETact = (torch.zeros([Ngrid,mu], dtype=dtype) + 0.001).cuda()

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
        Q_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        srflow_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        TCI_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        gwflow_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        AET = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        ROIMP_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        SDRO_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        interflow_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        direct_runoff_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        channel_bf_primary_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        channel_bf_secondary_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        nonchannel_bf_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        ssflow_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
        SWE_sim = torch.zeros(Pm.shape, dtype=dtype, device=args["device"])
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

            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (mean_air_temp[t, :, :] >= params_dict["parTT"]).type(dtype))
            SNOW = torch.mul(PRECIP, (mean_air_temp[t, :, :] < params_dict["parTT"]).type(dtype))

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
            """
            print(f"PXV = {PXV[0,0]}")
            print(f"EDMND = {EDMND[0, 0]}")
            """
            E1 = EDMND * UZTWC / params_dict["uztwm"]   # E1 = flux_Euztw in marrmot
            RED = torch.clamp(EDMND-E1, min=0.0)   ## RED IS RESIDUAL EVAP DEMAND
            UZTWC = torch.clamp(UZTWC-E1, min=0.0)
            """
            print(f"E1 = {E1[0, 0]}")
            print(f"RED = {RED[0, 0]}")
            print(f"UZTWC = {UZTWC[0, 0]}")
            """
            E2 = torch.zeros(E1.shape, dtype=dtype, device=args["device"])

            E2 = torch.where((UZFWC > RED) & (UZTWC == 0.0),
                             RED,
                             E2
                             )
            E2 = torch.where((UZFWC <= RED) & (UZTWC == 0.0),
                             UZFWC,
                             E2
                             )

            UZFWC = torch.clamp(UZFWC - E2, min=NEARZERO)
            RED = torch.clamp(RED - E2, min=0.0)

            # UPPER ZONE FREE WATER RATIO EXCEEDS UPPER ZONE
            # TENSION WATER RATIO, THUS TRANSFER FREE WATER TO TENSION -->
            # it means we redistribute water between UZTWC and UZFWC with ratio of UZRAT
            UZRAT = torch.where((UZTWC / params_dict["uztwm"]) <= (UZFWC / params_dict["uzfwm"]),
                                (UZTWC +UZFWC) / (params_dict["uztwm"] + params_dict["uzfwm"]),
                                torch.zeros(E1.shape, dtype=dtype, device=args["device"]))

            UZTWC = torch.where((UZTWC / params_dict["uztwm"]) <= (UZFWC / params_dict["uzfwm"]),
                                params_dict["uztwm"] * UZRAT,
                                UZTWC)

            UZFWC = torch.where((UZTWC / params_dict["uztwm"]) <= (UZFWC / params_dict["uzfwm"]),
                                params_dict["uzfwm"] * UZRAT,
                                UZFWC)


            # E3 implementation, E3 cannot exceed LZTWC
            E3 = RED * (LZTWC / (params_dict["uztwm"] + params_dict["lztwm"]))

            E3 = torch.min(LZTWC, E3)
            LZTWC = torch.clamp(LZTWC - E3, min=NEARZERO)
            """

            print("------Before RATLZT--------")
            print(f"E3 = {E3[0, 0]}")
            print(f"UZRAT = {UZRAT[0, 0]}")
            print(f"UZTWC = {UZTWC[0, 0]}")
            print(f"UZFWC = {UZFWC[0, 0]}")
            print(f"LZTWC = {LZTWC[0, 0]}")
            """

            RATLZT = LZTWC / params_dict["lztwm"]   # for dynamic parametrization, this line doesn't work

            SAVED = params_dict["rserv"] * (params_dict["lzfpm"] + params_dict["lzfsm"])

            RATLZ = (LZTWC + LZFPC  + LZFSC - SAVED) / (params_dict["lztwm"] + params_dict["lzfpm"] + params_dict["lzfsm"] - SAVED)
            """

            print("------label 226 and before GOTO 230--------")
            print(f"SAVED = {SAVED[0, 0]}")
            print(f"RATLZT = {RATLZT[0, 0]}")
            print(f"RATLZ = {RATLZ[0, 0]}")
            print(f"LZTWC = {LZTWC[0, 0]}")
            print(f"LZFSC = {LZFSC[0, 0]}")
            print("------------------------------------")
            """
            E5 = torch.where((RATLZT > RATLZ),
                              E1 + (RED + E2) *((ADIMC - E1 - UZTWC) / (params_dict["uztwm"] + params_dict["lztwm"])),
                              torch.zeros(E1.shape, dtype=dtype, device=args["device"]))
            """
            print("------label 230 and before GOTO 231--------")
            print(f"E5 = {E5[0, 0]}")
            print(f"ADIMC = {ADIMC[0, 0]}")
            print(f"E1 = {E1[0, 0]}")
            print(f"E2 = {E2[0, 0]}")
            print(f"RED = {RED[0, 0]}")
            print(f"UZTWC = {UZTWC[0, 0]}")
            a = params_dict["uztwm"][0, 0]
            b = params_dict["lztwm"][0, 0]
            print(f"UZTWM+LZTWM = {a + b}")
            print("-----------------------------")
            """
            E5 = torch.min(E5, ADIMC)
            ADIMC = torch.clamp(ADIMC - E5, min=0.0)
            """
            print("------after label 231 and before GOTO 232--------")
            print(f"ADIMC = {ADIMC[0, 0]}")
            print("-------------------------------------------------")
            """
            # ADIMP is the additional impervious area, not sure why we multiply it to ADIMP
            #update: TODO: still not sure why the following line happens after all the calculations related to E5
            E5 = E5 * params_dict["ADIMP"]
            """
            print("------After label 230 and before GOTO 232--------")
            print(f"SAVED = {SAVED[0, 0]}")
            print(f"E5 = {E5[0, 0]}")
            print("-------------------------------------------------")
            """
            ## DEL equals to Rls in marrmot version
            DEL = torch.where((RATLZT <= RATLZ),
                              (RATLZ - RATLZT) * params_dict["lztwm"],
                              torch.zeros(E1.shape, dtype=dtype, device=args["device"]))
            LZTWC = LZTWC + DEL
            LZFSC = LZFSC - DEL
            ## My understanding is that whenever DEl is large and there is not enough water in LZFSC,
            # then LZFPC provides the deficit to LZTWC. The question is what if LZFPC doesn't have enough water
            LZFPC = torch.where(LZFSC <= NEARZERO,
                                LZFPC + LZFSC,
                                LZFPC)
            ## if water is not available in LZFPC either, then DEL cannot be fully payed. therefore,
            # we return the water from LZTWC to LZFPC
            LZTWC = torch.where(LZFPC < NEARZERO,
                                LZTWC + LZFPC,
                                LZTWC)
            LZFPC = torch.clamp(LZFPC, min=NEARZERO)
            LZFSC = torch.clamp(LZFSC, min=NEARZERO)   #

            ### TODO: The above lines are not in the NWM fortran code and I added them,
            #         because I think the original code was not right!!.
            ### The order matters. it should be something like:
            # DEL = torch.min(LZFSC, DEL)
            # LZFSC = torch.clamp(LZFSC - DEL, min=NEARZERO)
            # LZTWC = LZTWC + DEL
            #----------------------------
            # COMPUTE PERCOLATION AND RUNOFF AMOUNTS.
            # PXV is input moisture i.e. Precip, or Precip + melt
            TWX = UZTWC + PXV - params_dict["uztwm"]
            """
            print("------After label 231 and before GOTO 232--------")
            print(f"E5 = {E5[0, 0]}")
            print(f"ADIMC = {ADIMC[0, 0]}")
            print(f"TWX = {TWX[0, 0]}")
            print(f"PXV = {PXV[0, 0]}")
            print(f"UZTWC = {UZTWC[0, 0]}")
            a = params_dict["uztwm"]
            print(f"uztwm = {a[0, 0]}")
            print("-----------------------------")
            """
            UZTWC = torch.where((TWX > 0.0),
                                params_dict["uztwm"],
                                UZTWC + PXV)
            TWX = torch.clamp(TWX, min=0.0)
            """
            print("------ GOTO 233--------")
            print(f"TWX = {TWX[0, 0]}")
            print(f"UZTWC = {UZTWC[0, 0]}")
            print("-------------------------------------------------")
            """
            # MOISTURE AVAILABLE IN EXCESS OF UZTW STORAGE
            ADIMC = ADIMC + PXV - TWX

            # COMPUTE IMPERVIOUS AREA RUNOFF.
            ROIMP = PXV * params_dict["pctim"]   # ROIMP    Impervious runoff from the permanent impervious area
            # ROIMP IS RUNOFF FROM THE MINIMUM IMPERVIOUS AREA.
            # SIMPVT = SIMPVT + ROIMP    # sum of ROIMP

            # DETERMINE COMPUTATIONAL TIME INCREMENTS FOR THE BASIC TIME INTERVAL
            NINC = torch.trunc(1 + 0.2 * (UZFWC + TWX))     # NINC=NUMBER OF TIME INCREMENTS THAT THE TIME INTERVAL
            DINC = (1 / NINC) * DT   # DT: computational time interval
            ## TODO: not sure if torch.round  is ok to be used in terms of gradient descent
            PINC = TWX / int(NINC.max().clamp(min=1.0)) #    # PINC=AMOUNT OF AVAILABLE MOISTURE FOR EACH INCREMENT

            # COMPUTE FREE WATER DEPLETION FRACTIONS FOR
            # THE TIME INCREMENT BEING USED-BASIC DEPLETIONS
            # ARE FOR ONE DAY
            DUZ = 1.0 - ((1.0 - params_dict["UZK"]) ** DINC)   # UZK in NWM version = Kuz in marrmot version
            DLZP = 1.0 - ((1.0 - params_dict["LZPK"]) ** DINC)   # LZPK in NWM version = Klzp in marrmot version
            DLZS = 1.0 - ((1.0 - params_dict["LZSK"]) ** DINC)   # LZSK in NWM version = Klzs in marrmot version

            ## INFERRED PARAMETER (ADDED BY Q DUAN ON 3/6/95)
            PAREA = 1.0 - params_dict["ADIMP"] - params_dict["pctim"]
            """
            print("------ GOTO 233--------")
            print(f"ROIMP = {ROIMP[0, 0]}")
            print(f"ADIMC = {ADIMC[0, 0]}")
            print(f"NINC = {NINC[0, 0]}")
            print(f"DINC = {DINC[0, 0]}")
            print(f"PINC = {PINC[0, 0]}")
            print(f"DUZ = {DUZ[0, 0]}")
            print(f"DLZP = {DLZP[0, 0]}")
            print(f"DLZS = {DLZS[0, 0]}")
            print(f"PAREA = {PAREA[0, 0]}")
            print("-------------------------------------------------")
            """
            ## ---------------------------------
            # START INCREMENTAL DO LOOP FOR THE TIME INTERVAL.
            SPBF = torch.zeros(UZTWC.shape, dtype=dtype, device=args["device"])
            SBF = torch.zeros(UZTWC.shape, dtype=dtype, device=args["device"])
            SPERC = torch.zeros(UZTWC.shape, dtype=dtype, device=args["device"])
            SIF = torch.zeros(UZTWC.shape, dtype=dtype, device=args["device"])
            SSUR = torch.zeros(UZTWC.shape, dtype=dtype, device=args["device"])
            SDRO = torch.zeros(UZTWC.shape, dtype=dtype, device=args["device"])
            for i in range(1, int(NINC.max()) + 1):
                ## COMPUTE DIRECT RUNOFF (FROM ADIMP AREA)
                RATIO = torch.clamp((ADIMC - UZTWC) / params_dict["lztwm"], min=0.0)
                # ADDRO IS THE AMOUNT OF DIRECT RUNOFF FROM THE AREA ADIMP.
                ADDRO = PINC * (RATIO ** 2)
                # COMPUTE BASEFLOW AND KEEP TRACK OF TIME INTERVAL SUM.
                # baseflow from LZFSC
                BF_LZFPC = torch.clamp(LZFPC * DLZP, min=0.0)   # baseflow primary and secondary
                LZFPC = LZFPC - BF_LZFPC

                ## TODO: since it is a for loop, I need to chack min=NEARZERO is not adding too much error to the system
                # LZFPC = torch.clamp(LZFPC - BF, min=NEARZERO)
                BF_LZFPC = torch.where((LZFPC <= 0.0001),
                                    BF_LZFPC + LZFPC,
                                    BF_LZFPC)
                # Basically, it means if the LZFPC is smaller than 0.0001--> all of  it  is BF and LZFPC is zero.
                LZFPC = torch.where((LZFPC <= 0.0001),
                                 torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]) + NEARZERO,
                                 LZFPC)


                SPBF = SPBF + BF_LZFPC
                # baseflow from LZFSC
                BF_LZFSC = torch.clamp(LZFSC * DLZS, min=0.0)
                LZFSC = LZFSC - BF_LZFSC
                BF_LZFSC = torch.where((LZFSC <= 0.0001),
                                       BF_LZFSC + LZFSC,
                                       BF_LZFSC)
                LZFSC = torch.where((LZFSC <= 0.0001),
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]) + NEARZERO,
                                    LZFSC)

                # BF_LZFSC = torch.min(BF_LZFSC, LZFSC)
                # ## TODO: since it is a for loop, I need to chack min=NEARZERO is not adding to much error to the system
                # LZFSC = torch.clamp(LZFSC - BF_LZFSC, min=NEARZERO)
                """
                print("------ in label 234 before if and before GOTO 235--------")
                print(f"LZFSC = {LZFSC[0, 0]}")
                print("---------------------------------")
                """
                # Label 235
                SBF = SBF + BF_LZFSC

                # COMPUTE PERCOLATION-IF NO WATER AVAILABLE THEN SKIP
                # to do it in parallel, we need to keep this PINC + UZFWC <> 0.01 to handle GOTO commands in fortran
                mask_PINC_UZFWC = (PINC + UZFWC) > 0.01
                UZFWC = torch.where(~mask_PINC_UZFWC,
                                    PINC + UZFWC,
                                    UZFWC)
                PERCM = torch.where(mask_PINC_UZFWC,
                                    params_dict["lzfpm"] * DLZP + params_dict["lzfsm"] *DLZS,
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                PERC = torch.where(mask_PINC_UZFWC,
                                    PERCM * (UZFWC / params_dict["uzfwm"]),
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                ## equivalent to LZ_deficiency / LZ_capacity in marrmot
                DEFR = torch.where(mask_PINC_UZFWC,
                                   1.0 - ((LZTWC + LZFPC + LZFSC) / (params_dict["lztwm"] + params_dict["lzfpm"] + params_dict["lzfsm"])),
                                   torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))

                ## TODO: need to add a function for considering the effect of frzen ground on PERC
                ## For now IFRZE = 0.0 --> no need to call FGFR1
                ## COMPUTES THE CHANGE IN THE PERCOLATION AND INTERFLOW WITHDRAWAL RATES DUE TO FROZEN GROUND
                # FR, FI = self.FGFR1(self, DEFR, LZTWC, LZFSC, LZFPC, UZTWC, UZFWC, LZFPC,
                #                     ADIMC)
                FR = torch.tensor(1.0, dtype=dtype, device=args["device"])
                FI = torch.tensor(1.0, dtype=dtype, device=args["device"])

                PERC = torch.where(mask_PINC_UZFWC,
                                   PERC * (1.0 + params_dict["ZPERC"] * (DEFR ** params_dict["rexp"])) * FR,
                                   torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                # NOTE...PERCOLATION OCCURS FROM UZFWC BEFORE PAV IS ADDED.
                PERC = torch.where(mask_PINC_UZFWC & (PERC >= UZFWC),
                                   UZFWC,
                                   PERC)
                UZFWC= torch.where(mask_PINC_UZFWC,
                                    UZFWC - PERC,
                                    UZFWC)
                ## TODO: not sure if I need this line without having the condition PINC + UZFWC > 0.01
                UZFWC = torch.clamp(UZFWC, min=NEARZERO)
                # CHECK TO SEE IF PERCOLATION EXCEEDS LOWER ZONE DEFICIENCY.
                CHECK = torch.where(mask_PINC_UZFWC,
                                    LZTWC + LZFPC + LZFSC + PERC - params_dict["lztwm"] - params_dict["lzfpm"] - params_dict["lzfsm"],
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                """
                print("------ after ifcond GOTO 239 before GOTO 241--------")
                print(f"UZFWC = {UZFWC[0, 0]}")
                print(f"CHECK = {CHECK[0, 0]}")
                print(f"LZTWC = {LZTWC[0, 0]}")
                print(f"LZFPC = {LZFPC[0, 0]}")
                print(f"LZFSC = {LZFSC[0, 0]}")
                print(f"PERC = {PERC[0, 0]}")
                print("---------------------------------")
                """
                PERC = torch.where(mask_PINC_UZFWC & (CHECK > 0.0),
                                    PERC - CHECK,
                                    PERC)
                UZFWC = torch.where(mask_PINC_UZFWC & (CHECK > 0.0),
                                     UZFWC + CHECK,
                                     UZFWC)
                # SPERC IS THE TIME INTERVAL SUMMATION OF PERC
                SPERC = torch.where(mask_PINC_UZFWC,
                                     SPERC + PERC,
                                     SPERC)

                # COMPUTE INTERFLOW AND KEEP TRACK OF TIME INTERVAL SUM.
                # NOTE...PINC HAS NOT YET BEEN ADDED
                DEL_uz = torch.where(mask_PINC_UZFWC,
                                      UZFWC * DUZ * FI,
                                      torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                # SIF: sum of interflow
                SIF = torch.where(mask_PINC_UZFWC,
                                     SIF + DEL_uz,
                                     torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                UZFWC = torch.where(mask_PINC_UZFWC,
                                     UZFWC - DEL_uz,
                                     UZFWC)

                # DISTRIBE PERCOLATED WATER INTO THE LOWER ZONES
                # TENSION WATER MUST BE FILLED FIRST EXCEPT FOR THE PFREE AREA.
                # PERCT IS PERCOLATION TO TENSION WATER AND PERCF IS PERCOLATION GOING TO FREE WATER.
                # PERCT is equivalent to PCtw in marrmot
                PERCT = torch.where(mask_PINC_UZFWC,
                                    PERC * (1.0 - params_dict["PFREE"]),
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                mask_PERCT_LZTWC_low = (PERCT + LZTWC) <= params_dict["lztwm"]
                LZTWC = torch.where(mask_PINC_UZFWC & mask_PERCT_LZTWC_low,
                                    LZTWC + PERCT,
                                    LZTWC)
                # if (PERCT + LZTWC > lztwm)
                PERCF_LZTWC = torch.where((mask_PINC_UZFWC) & (~mask_PERCT_LZTWC_low),
                                    PERCT + LZTWC - params_dict["lztwm"],
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                LZTWC = torch.where((mask_PINC_UZFWC) & (~mask_PERCT_LZTWC_low),
                                    params_dict["lztwm"],
                                    LZTWC)

                # if (PERCT + LZTWC < lztwm)
                # for PERCF with (PERCT + LZTWC < lztwm) : it has been taken care of on the upper lines
                # LZTWC = torch.where(mask_PINC_UZFWC & mask_PERCT_LZTWC_low,
                #                     LZTWC + PERCT,
                #                     LZTWC)

                # line 426 in sac.f with label 244
                # DISTRIBUTE PERCOLATION IN EXCESS OF TENSION
                # REQUIREMENTS AMONG THE FREE WATER STORAGES.
                PERCF = torch.where(mask_PINC_UZFWC,
                                    PERCF_LZTWC + PERC * params_dict["PFREE"],
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))

                # HPL IS THE RELATIVE SIZE OF THE PRIMARY STORAGE
                # AS COMPARED WITH TOTAL LOWER ZONE FREE WATER STORAGE.
                HPL = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                  params_dict["lzfpm"] / (params_dict["lzfpm"] + params_dict["lzfsm"]),
                                  torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))

                # RATLP AND RATLS ARE CONTENT TO CAPACITY RATIOS, OR
                # IN OTHER WORDS, THE RELATIVE FULLNESS OF EACH STORAGE
                RATLP = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                  LZFPC / (params_dict["lzfpm"]),
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))

                RATLS = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                    LZFSC / params_dict["lzfsm"],
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                # FRACP IS THE FRACTION GOING TO PRIMARY.
                FRACP = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                    (HPL * 2.0 * (1.0 - RATLP)) / ((1.0 - RATLP) + (1.0 - RATLS)),
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))

                FRACP = torch.where(mask_PINC_UZFWC & (PERCF != 0.0) & (FRACP > 1.0),
                                    torch.ones(FRACP.shape, dtype=dtype, device=args["device"]),
                                    FRACP)

                # PERCP AND PERCS ARE THE AMOUNT OF THE EXCESS
                # PERCOLATION GOING TO PRIMARY AND SUPPLEMENTAL STORGES,RESPECTIVELY.
                PERCP = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                    PERCF * FRACP,
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                PERCS = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                    PERCF - PERCP,
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))

                # PERCS = torch.min(PERCS, torch.clamp(lzrfsm - LZFSC, min=0.0))
                LZFSC = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                    LZFSC + PERCS,
                                    LZFSC)
                mask_LZFSC_high = LZFSC > params_dict["lzfsm"]
                PERCS = torch.where(mask_PINC_UZFWC & (PERCF != 0.0) & mask_LZFSC_high,
                                    PERCS - LZFSC + params_dict["lzfsm"],
                                    PERCS)
                LZFSC = torch.where(mask_PINC_UZFWC & (PERCF != 0.0) & mask_LZFSC_high,
                                    params_dict["lzfsm"],
                                    LZFSC)
                #label 246 in Fortran code sac1.f
                LZFPC = torch.where(mask_PINC_UZFWC & (PERCF != 0.0),
                                    LZFPC + (PERCF - PERCS),
                                    LZFPC)
                # CHECK TO MAKE SURE LZFPC DOES NOT EXCEED LZFPM. I f it exceeds, exess water goes to LZTWC
                # TODO: need to recheck if the following line needs to be written by torch.where()
                #       it can be written as:
                # EXCESS = torch.where(((PINC + UZFWC) > 0.01 ) & (LZFPC > params_dict["lzfpm"]),
                #                      LZFPC - params_dict["lzfpm"],
                #                      torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                EXCESS = torch.clamp(LZFPC - params_dict["lzfpm"], min=0.0)
                mask_LZFPC_high = LZFPC > params_dict["lzfpm"]
                LZTWC = torch.where(mask_PINC_UZFWC & (PERCF != 0.0) & mask_LZFPC_high,
                                    LZTWC + EXCESS,
                                    LZTWC)
                # TODO: don't we need to check if LZTWC is larger than lztwm, when we add EXCESS?
                LZFPC = torch.where(mask_PINC_UZFWC & (PERCF != 0.0) & mask_LZFPC_high,
                                    params_dict["lzfpm"],
                                    LZFPC)

                ## line 455 in Fortran sac1.f, label 245
                # DISTRIBUTE PINC BETWEEN UZFWC AND SURFACE RUNOFF.
                # if (PINC + UZFWC) < uzfwm
                ## Note: we make a mask for UZFWC because  we need the exact same mask
                # for calculating SUR, SSUR, ADSUR in label 248
                mask_UZFWC_extra = mask_PINC_UZFWC & (PINC > 0.0) & ((PINC + UZFWC) > params_dict["uzfwm"])
                mask_UZFWC_extra_low = mask_PINC_UZFWC & (PINC > 0.0) & ((PINC + UZFWC) <= params_dict["uzfwm"])
                UZFWC = torch.where(mask_UZFWC_extra_low,
                                    UZFWC + PINC,
                                    UZFWC)

                ## Label 248
                # if (PINC + UZFWC) > uzfwm
                # COMPUTE SURFACE RUNOFF (SUR) AND KEEP TRACK OF TIME INTERVAL SUM.
                SUR = torch.where(mask_UZFWC_extra,
                                    PINC + UZFWC - params_dict["uzfwm"],
                                  torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                ## Note: mask_UZFWC_extra may not cover all elements in UZFWC larger than uzfwm.
                # It covers only the elements that were higher than uzfwm before adding PINC to it
                UZFWC = torch.where(mask_UZFWC_extra,
                                    params_dict["uzfwm"],
                                    UZFWC)

                # SSUR: Sum of the indirect surface runoff components from ADIMP and PAREA
                SSUR = torch.where(mask_UZFWC_extra,
                                   SSUR + SUR * PAREA,
                                   SSUR)

                # ADSUR IS THE AMOUNT OF SURFACE RUNOFF WHICH COMES
                # FROM THAT PORTION OF ADIMP WHICH IS NOT
                # CURRENTLY GENERATING DIRECT RUNOFF.  ADDRO/PINC
                # IS THE FRACTION OF ADIMP CURRENTLY GENERATING
                # DIRECT RUNOFF.
                ADSUR = torch.where(mask_UZFWC_extra,
                                    SUR * (1.0 - ADDRO / (PINC + NEARZERO)),
                                    torch.zeros(UZFWC.shape, dtype=dtype, device=args["device"]))
                SSUR = torch.where(mask_UZFWC_extra,
                                   SSUR + ADSUR * params_dict["ADIMP"],
                                   SSUR)

                # line 476 in Fortran sac1.f, label 249 --> All conditions are gone here.
                # ADIMP AREA WATER BALANCE -- SDRO IS THE 6 HR SUM OF DIRECT RUNOFF.
                ADIMC = ADIMC + PINC - ADDRO - ADSUR
                ADDRO = torch.where(ADIMC >= (params_dict["uztwm"] + params_dict["lztwm"]),
                                    ADDRO + ADIMC - (params_dict["uztwm"] + params_dict["lztwm"]),
                                    ADDRO)
                ADIMC = torch.where(ADIMC >= (params_dict["uztwm"] + params_dict["lztwm"]),
                                    params_dict["uztwm"] + params_dict["lztwm"],
                                    ADIMC)
                ## line 480, label 247 --> all conditions are gone again
                SDRO = SDRO + ADDRO * params_dict["ADIMP"]

            # COMPUTE SUMS AND ADJUST RUNOFF AMOUNTS BY THE AREA OVER WHICH THEY ARE GENERATED.
            EUSED  = E1 + E2 + E3
            # EUSED IS THE ET FROM PAREA WHICH IS 1.0-ADIMP-PCTIM
            SIF = SIF * PAREA

            #  SEPARATE CHANNEL COMPONENT OF BASEFLOW FROM THE NON-CHANNEL COMPONENT
            # TBF IS TOTAL BASEFLOW
            TBF = SBF * PAREA
            # BFCC IS BASEFLOW, CHANNEL COMPONENT
            BFCC = TBF * (1.0 / (1.0 + params_dict["SIDE"]))
            BFP = SPBF * PAREA / (1.0 + params_dict["SIDE"])
            BFS = BFCC - BFP
            BFS = torch.clamp(BFS, min=0.0)
            # BFNCC IS BASEFLOW,NON-CHANNEL COMPONENT
            BFNCC = TBF - BFCC

            ### TODO: not sure what this "ADD TO MONTHLY SUMS" mean in Fortran code
            # like wy do we need that?
            # SINTFT = SINTFT + SIF
            # SGWFP = SGWFP + BFP
            # SGWFS = SGWFS + BFS
            # SRECHT = SRECHT + BFNCC
            # SROST = SROST+SSUR
            # SRODT = SRODT + SDRO

            ## COMPUTE TOTAL CHANNEL INFLOW FOR THE TIME INTERVAL.
            TCI = ROIMP + SDRO + SSUR + SIF + BFCC

            #COMPUTE E4-ET FROM RIPARIAN VEGETATION.
            E4 = (EDMND - EUSED) * params_dict["RIVA"]

            ## SUBTRACT E4 FROM CHANNEL INFLOW
            E4 = torch.min(E4, TCI)
            TCI = torch.clamp(TCI - E4, min=0.0)    ## TODO: or maybe NEARZERO?
            """
            print("------ after GOTO 240 before if cond to go to 250--------")
            print(f"E4 = {E4[0, 0]}")
            print(f"TCI = {TCI[0, 0]}")
            print(f"EDMND = {EDMND[0, 0]}")
            print(f"EUSED = {EUSED[0, 0]}")
            a = params_dict["RIVA"][0, 0]
            print(f"RIVA = {a}")
            print("-------------------------------------------------")
            """
            ## line 525 Fortran, label 250
            # SROT = SROT + TCI  #SROT is monthly. and we may not need it

            ## COMPUTE TOTAL EVAPOTRANSPIRATION-TET
            EUSED = EUSED * PAREA   ### EUSED Total ET rom pervious area
            TET = EUSED + E5 + E4   ## total evapotranspiration
            # SETT = SETT + TET   ## SETT monthly
            # SE1 = SE1 + E1 * PAREA   ## SE1 is monthly
            # SE3 = SE3 + E3 * PAREA  ## SE# is onthly
            # SE4 = SE4 + E4      ## SE4 is monthly
            # SE5 = SE5 + E5     # SE5 is monthly

            ## CHECK THAT ADIMC.GE.UZTWC  , TODO: not sure why we are doing it again
            ADIMC = torch.where(ADIMC < UZTWC,
                                UZTWC,
                                ADIMC)

            ### COMPUTE NEW FROST INDEX AND MOISTURE TRANSFER.
            ## we need to call TFRZE subroutine

         # COMPUTE FINAL TOTAL STORAGE AND WATER BALANCE
         # sdro:  direct runoff
         # ROIMP: impervious area runoff
         # SSUR:  surface runoff
         # SIF:   interflow
         # BFS:   non-channel baseflow
         # BFP:   some kind of baseflow...
         # TCI:   Total channel inflow
            """
            print("***********************")
            print(f"TIME:     {t}")
            print(f"UZTWC = {UZTWC[0, 0]}")
            print(f"UZFWC = {UZFWC[0, 0]}")
            print(f"LZTWC = {LZTWC[0, 0]}")
            print(f"LZFSC = {LZFSC[0, 0]}")
            print(f"LZFPC = {LZFPC[0, 0]}")
            print("***********************")
            """

            QS = ROIMP + SDRO + SSUR + SIF
            QG = BFS + BFP   # what about nonchannel baseflow
            Q  = TCI

            Q_sim[t, :, :] = TCI
            ## TODO: Think about ssflow. it looks like water is either QS or QG, thee is nothing in between
            ### TODO: srflow_sim, ssflow_sim, and gwflow_sim are not correct.
            # for instance E4 is not implemented on any of them, but TCI
            srflow_sim[t, :, :] = ROIMP + SDRO + SSUR
            ssflow_sim[t, :, :] = SIF
            gwflow_sim[t, :, :] = BFS + BFP
            AET[t, :, :] = TET
            SWE_sim[t, :, :] = SNOWPACK_storage
            TCI_sim[t, :, :] = TCI
            ROIMP_sim[t, :, :] = ROIMP
            SDRO_sim[t, :, :] = SDRO
            interflow_sim[t, :, :] = SIF
            direct_runoff_sim[t, :, :] = ROIMP + SDRO + SSUR + SIF
            channel_bf_primary_sim[t, :, :] = BFP
            channel_bf_secondary_sim[t, :, :] = BFS
            nonchannel_bf_sim[t, :, :] = BFNCC

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
                        SWE_sim=SWE_sim.mean(-1, keepdim=True),
                        TCI_sim=TCI_sim.mean(-1, keepdim=True),
                        ROIMP_sim=ROIMP_sim.mean(-1, keepdim=True),
                        SDRO_sim=SDRO_sim.mean(-1, keepdim=True),
                        interflow_sim=interflow_sim.mean(-1, keepdim=True),
                        direct_runoff_sim=direct_runoff_sim.mean(-1, keepdim=True),
                        channel_bf_primary_sim=channel_bf_primary_sim.mean(-1, keepdim=True),
                        channel_bf_secondary_sim=channel_bf_secondary_sim.mean(-1, keepdim=True),
                        nonchannel_bf_sim=nonchannel_bf_sim.mean(-1, keepdim=True),
                        )
