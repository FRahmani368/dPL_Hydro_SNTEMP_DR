# This code is written by Yalan Song from MHPI group, Penn State Univerity
# Purpose: This code solves ODEs Hydrological models with AD
# Farshid changed it to a complete model with PRMS
import torch
from ODEsolver import NRBacksolveFunction
from HydroModels.PRMS import PRMS
from HydroModels.config.read_configurations import config
torch.cuda.set_device(0)
device = torch.device("cuda")
dtype=torch.double



# from core.read_configurations import config
from core.utils.randomseed_config import randomseed_config
from core.load_data.data_prep import (
    load_df,
    scaling,
    train_val_test_split,
    randomIndex,
    selectSubset,
)
from core.utils.small_codes import create_output_dirs
from MODELS.temp_models.PGML_STemp import MLP, CudnnLstmModel
from MODELS.loss_functions import crit
from core import hydroDL
from core.utils.small_codes import make_tensor
import torch
import time
import os
from sklearn import preprocessing


def main():
    mode_type = ["SNTEMP", "SNTEMP"]  # ["van Vliet","Meisner","SNTEMP"]
    lenF_gwflow_list = [365]
    lenF_ssflow_list = [30, 30]
    lat_temp_adj_list = ["True", "True"]
    frac_smoothening_list = ["True", "False"]
    s = [0, 0]
    # seeds = args['randomseed']
    for seed, typ, LenF_gw, LenF_ss, adj, frac_smooth in zip(
            s,
            mode_type,
            lenF_gwflow_list,
            lenF_ssflow_list,
            lat_temp_adj_list,
            frac_smoothening_list,
    ):
        args = config
        args["res_time_params"]["type"] = typ
        args["res_time_params"]["lenF_gwflow"] = LenF_gw
        args["res_time_params"]["lenF_ssflow"] = LenF_ss
        args["lat_temp_adj"] = adj
        # args["shade_smoothening"] = shade_smooth
        args["frac_smoothening"]["mode"] = frac_smooth
        # args['randomseed'] = seed
        # torch.cuda.set_per_process_memory_fraction(0.9)   # work for torch > 1.4
        randomseed_config(seed)
        # Creating output directories
        args = create_output_dirs(args, seed)
        min_max_scaler = preprocessing.MinMaxScaler()
        # getting the data
        x_total_temp, y_raw, c_raw, c_PRMS, x_PRMS = load_df(args)

        x_total_raw = x_total_temp.copy()
        x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
        c_scaled_0_1 = min_max_scaler.fit_transform(c_raw)
        time1 = hydroDL.utils.time.tRange2Array(args["optData"]["tRange"])
        #
        vars = args["optData"]["varT"] + args["optData"]["varC"]
        x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False, device="cpu")

        no_tot_params = len(args["marrmot_paramCalLst"])
        ny = args["nmul"] * no_tot_params
        model = CudnnLstmModel(
            nx=len(args["optData"]["varT"] + args["optData"]["varC"]),
            ny=ny,
            hiddenSize=args["hyperparameters"]["hidden_size"],
            dr=args["hyperparameters"]["dropout"],
        )
        # loss function
        lossFun = crit.RmseLoss()
        optim = torch.optim.Adadelta(model.parameters())  # , lr=0.1
        if torch.cuda.is_available():
            model = model.to(args["device"])
            # mlp = mlp.cuda()
            # PRMS = PRMS.to(args["device"])
            # Ts = Ts.to(args["device"])
            # lossFun = lossFun.cuda()
            lossFun = lossFun.to(args["device"])
            torch.backends.cudnn.deterministic = True
            CUDA_LAUNCH_BLOCKING = 1

        c_tensorTrain = make_tensor(c_scaled_0_1, has_grad=False, device="cpu")
        if 0 in args["Action"]:
            (
                x_train,
                y_train,
                ngrid_train,
                nIterEp,
                nt,
                batchSize,
            ) = train_val_test_split("t_train", args, time1, x_total_raw, y_raw)
            x_train_scaled, y_train_scaled, _, _, _, _ = train_val_test_split(
                "t_train", args, time1, x_total_scaled, y_scaled
            )
            x_PRMS_train, _, _, _, _, _ = train_val_test_split(
                "t_train", args, time1, x_PRMS, y_raw
            )

            # vars = args["optData"]["varT"] + args["optData"]["varC"]
            # x_train_scaled_noccov = np.delete(
            #     x_train_scaled, vars.index("ccov"), axis=2
            # )
            #
            # ave_air_total = Ts.ave_temp_general(
            #     args, x_total_raw_tensor, time_range=args["optData"]["t_train"]
            # )

            rho = args["hyperparameters"]["rho"]
            warm_up = args["warm_up"]
            bs = args["hyperparameters"]["batch_size"]
            # model.zero_grad()
            model.train()
            for epoch in range(1, args["hyperparameters"]["EPOCHS"] + 1):
                lossEp = 0
                t0 = time.time()
                for iIter in range(1, nIterEp + 1):
                    iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho + warm_up])

                    xTrain_sample = selectSubset(
                        args, x_train, iGrid, iT, rho + warm_up, has_grad=False
                    )
                    xTrain_sample_scaled = selectSubset(
                        args,
                        x_train_scaled,
                        iGrid,
                        iT,
                        rho + warm_up,
                        has_grad=False,
                    )  # x_train_scaled

                    x_PRMS_sample = selectSubset(
                        args, x_PRMS_train, iGrid, iT, rho + warm_up, has_grad=False
                    )
                    c_PRMS_sample = torch.tensor(
                        c_PRMS[iGrid], device=args["device"], dtype=torch.float32
                    )

                    ### MLP
                    if type(model) in [MLP]:
                        params = model(c_tensorTrain[iGrid])
                    ### CudnnLstm
                    if type(model) in [CudnnLstmModel]:
                        params = model(xTrain_sample_scaled.permute(1, 0, 2))
                    theta = params[:,-1,:]  # just using the last one
                    # theta = nn.Parameter(
                    #     torch.tensor(
                    #         params[:,-1,:],
                    #         dtype=torch.double,
                    #         requires_grad=True,
                    #         device=device,
                    #     )
                    # )
                    yObs = selectSubset(
                        args, y_train, iGrid, iT, rho + warm_up, has_grad=False
                    )


                    nflux = 25
                    delta_t = torch.tensor(1.0).to(device=args['device'], dtype=dtype)
                    input_s00 = [15.0, 7.0, 3.0, 8.0, 22.0, 10.0, 10.0]  # initial values for storages in hydro model
                    input_s0 = torch.tensor(input_s00).repeat(bs, 1).to(delta_t)
                    precip = x_PRMS_sample[:, :, args["optData"]["varT_PRMS"].index("prcp(mm/day)")]
                    T = (x_PRMS_sample[:, :, args["optData"]["varT_PRMS"].index("tmax(C)")] +
                         x_PRMS_sample[:, :, args["optData"]["varT_PRMS"].index("tmin(C)")]) / 2
                    PET = x_PRMS_sample[:, :, args["optData"]["varT_PRMS"].index("pet_nldas")]
                    climate_data = torch.cat([precip.unsqueeze(-1), T.unsqueeze(-1), PET.unsqueeze(-1)], dim=2)  ##nt*bs*ny
                    y0 = torch.tensor(input_s0, device=device, dtype=dtype)  # bs*ny

                    NR = NRBacksolveFunction()
                    fluxSolution_new = torch.zeros((rho, bs, nflux)).to(y0)
                    NR = NRBacksolveFunction()
                    fluxSolution_new = torch.zeros((rho, bs, nflux)).to(y0)

                    def meany(y0, theta, delta_t, climate_data, nflux, args):
                        f = PRMS(theta, delta_t, climate_data, args)
                        ySolution, fluxSolution, Residual, NRFlag = NR(y0, theta, delta_t, climate_data, nflux, f)
                        for day in range(rho):
                            _, flux = f(torch.tensor(day).to(ySolution), ySolution[day, :, :])
                            fluxSolution_new[day, :, :] = flux * delta_t
                        flow_pred = (fluxSolution_new[:, :, 10] + fluxSolution_new[:, :, 11] +
                                     fluxSolution_new[:, :, 17] + fluxSolution_new[:, :, 18])
                        return flow_pred

                    y_mean = meany(y0, theta, delta_t, climate_data, nflux, args)
                    # y_mean = torch.mean(params, dim=2).transpose(1, 0)
                    loss = lossFun(
                        y_mean.unsqueeze(-1), yObs[warm_up:, :, :]
                    )
                    # loss.register_hook(lambda grad: print(grad))
                    loss.backward()
                    optim.step()
                    model.zero_grad()
                    lossEp = lossEp + loss.item()
                    print(
                        iIter,
                        " from ",
                        nIterEp,
                        " in the ",
                        epoch,
                        "th epoch, and Loss is ",
                        loss.item(),
                    )
                lossEp = lossEp / nIterEp
                # torch.cuda.synchronize()
                logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
                    epoch,
                    lossEp,
                    time.time() - t0,
                    int(torch.cuda.memory_allocated() * 0.001),
                )
                print(logStr)

                if epoch % args["hyperparameters"]["saveEpoch"] == 0:
                    # save model
                    modelFile = os.path.join(
                        args["output"]["out_dir"], "model_Ep" + str(epoch) + ".pt"
                    )
                    torch.save(model, modelFile)






if __name__ == "__main__":
    main()
