from config.read_configurations import config_PRMS as config
from core.utils.randomseed_config import randomseed_config
from core.load_data.data_prep import (
    load_df,
    scaling,
    train_val_test_split,
    No_iter_nt_ngrid,
    randomIndex,
    selectSubset,
)
from core.utils.small_codes import create_output_dirs
from MODELS.temp_models.PGML_STemp import MLP, CudnnLstmModel
from MODELS.hydro_models.marrmot.prms_marrmot_only import prms_marrmot
from MODELS.loss_functions import crit
from core import hydroDL
from core.utils.small_codes import make_tensor
import torch
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from post import stat, plot
import math
from core.hydroDL.data.camels import initcamels


def main_marrmot_PRMS_only(args):
    # setting random seeds
    # randomseed_config(args)
    seed = args["randomseed"][0]
    randomseed_config(seed)
    # Creating output directories
    args = create_output_dirs(args, seed)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # getting the data for the Neural Network, PRMS, and SNTEMP
    x_total_temp_NN, y_raw, c_raw_NN, c_PRMS, x_PRMS, _, _ = load_df(args)

    x_total_raw_NN = x_total_temp_NN.copy()   # making a copy of x for doing a normalization in future

    # making the stats for normalization only based on the training part
    time1 = hydroDL.utils.time.tRange2Array(args["tRange"])
    x_NN_train, y_NN_train = train_val_test_split("t_train", args, time1, x_total_raw_NN, y_raw)
    initcamels(args, x_NN_train, y_NN_train)
    del (x_NN_train, y_NN_train)   # we just needed them to create a stat file for normalization.

    # normalizing x, y, c
    x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp_NN, y_raw, c_raw_NN)

    #
    vars = args["varT_NN"] + args["varC_NN"]
    x_total_raw_tensor = make_tensor(x_total_raw_NN, has_grad=False, device="cpu")
    c_tensorTrain = make_tensor(c_scaled, has_grad=False, device="cpu")

    # ANN model to simulate parameters
    # all parameters are going to be 3D -> [batchsize, rho, nmul]
    # number of total parameters we need from NN, including the conv parameters
    #PRMS
    if args["routing_PRMS"] == True:  # needs a and b for routing with conv method
        ny_prms = args["nmul"] * (len(args["marrmot_paramCalLst"]) + len(args["conv_PRMS"]))
    else:
        ny_prms = args["nmul"] * len(args["marrmot_paramCalLst"])

    #SNTEMP  # needs a and b for calculating different source flow temperatures with conv method
    # if args["routing_SNTEMP"] == True:
    #     ny_sntemp = args["nmul"] * (len(args["SNTEMP_paramCalLst"]) + len(args["conv_SNTEMP"]))
    # else:
    #     ny_sntemp = args["nmul"] * len(args["SNTEMP_paramCalLst"])
    # if args["lat_temp_adj"] == True:
    #     ny_sntemp = ny_sntemp + args["nmul"]

    model = CudnnLstmModel(
        nx=len(args["varT_NN"] + args["varC_NN"]),
        ny=ny_prms,
        hiddenSize=args["hidden_size"],
        dr=args["dropout"],
    )
    # model = MLP(args)
    # model = torch.load("/mnt/sdc/fzr5082/PGML_STemp_results/models/415_sites/SNTEMP_gw_365_ss_30_adj_T_fr_T40_stat__semi__nmul_16_s_0/model_Ep30.pt")
    PRMS = prms_marrmot()  # this is for discrete version of marrmot
    # PRMS = PRMS_pytorch()   # This is for Newton-Raphson method version of marrmot
    # Ts = SNTEMP_EQ()    # SNTEMP model
    # Ts = SNTEMP_only()

    # loss function
    # lossFun = crit.RmseLoss()    # simple rmse loss function
    # lossFun = crit.RmseLoss_temp_flow(w=0.75)   #0.25 for streamflow
    lossFun = crit.RmseLossComb(alpha=0.25)
    optim = torch.optim.Adadelta(model.parameters())  # , lr=0.1
    # optim = torch.optim.SGD(model.parameters(), lr=10)

    if torch.cuda.is_available():
        model = model.to(args["device"])
        PRMS = PRMS.to(args["device"])
        # Ts = Ts.to(args["device"])
        lossFun = lossFun.to(args["device"])
        torch.backends.cudnn.deterministic = True
        CUDA_LAUNCH_BLOCKING = 1

    del x_total_temp_NN

    if 0 in args["Action"]:
        # preparing training dataset for NN, PRMS, SNTEMP
        x_train, y_train = train_val_test_split("t_train", args, time1, x_total_raw_NN, y_raw)
        x_train_scaled, y_train_scaled = train_val_test_split("t_train", args, time1, x_total_scaled, y_scaled)
        x_PRMS_train, _ = train_val_test_split("t_train", args, time1, x_PRMS, y_raw)
        # x_SNTEMP_train, _ = train_val_test_split("t_train", args, time1, x_SNTEMP, y_raw)
        # ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args["t_train"])
        # mean_air_temp_train = (x_SNTEMP_train[:,:, args["varT_SNTEMP"].index("tmax(C)")] +
        #                        x_SNTEMP_train[:,:, args["varT_SNTEMP"].index("tmin(C)")]) / 2
        ngrid_train, nIterEp, nt, batchSize = No_iter_nt_ngrid("t_train", args, x_train)
        rho = args["rho"]
        warm_up = args["warm_up"]
        nmul = args["nmul"]
        model.zero_grad()
        model.train()
        # training
        for epoch in range(1, args["EPOCHS"] + 1):
            lossEp = 0
            t0 = time.time()
            for iIter in range(1, nIterEp + 1):
                iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho + warm_up])

                # NN sampling
                xTrain_sample_scaled = selectSubset(
                    args, x_train_scaled, iGrid, iT, rho + warm_up, has_grad=False,
                ).permute([1, 0, 2])
                # PRMS sampling
                x_PRMS_sample = selectSubset(
                    args, x_PRMS_train, iGrid, iT, rho + warm_up, has_grad=False
                ).permute([1, 0, 2])
                c_PRMS_sample = torch.tensor(
                    c_PRMS[iGrid], device=args["device"], dtype=torch.float32
                )
                # SNTEMP sampling
                # x_SNTEMP_sample = selectSubset(
                #     args, x_SNTEMP_train, iGrid, iT, rho + warm_up, has_grad=False
                # ).permute([1, 0, 2])    # [warm_up:,:, :]there is no need for warm up in temp section yet
                # c_SNTEMP_sample = torch.tensor(
                #     c_SNTEMP[iGrid], device=args["device"], dtype=torch.float32
                # )
                # here, warm_up should not be smaller than res_time_lenF_gwflow
                # air_sample_sr = selectSubset(
                #     args, np.expand_dims(mean_air_temp_train, axis=2), iGrid, iT,
                #     rho + warm_up, has_grad=False
                # )[warm_up - args["res_time_lenF_srflow"]:, :, :].permute([1, 0, 2])
                # air_sample_ss = selectSubset(
                #     args, np.expand_dims(mean_air_temp_train, axis=2), iGrid, iT,
                #     rho + warm_up, has_grad=False
                # )[warm_up - args["res_time_lenF_ssflow"]:, :, :].permute([1, 0, 2])
                # air_sample_gw = selectSubset(
                #     args, np.expand_dims(mean_air_temp_train, axis=2), iGrid, iT,
                #     rho + warm_up, has_grad=False
                # )[warm_up - args["res_time_lenF_gwflow"]:, :, :].permute([1, 0, 2])
                # observations
                targets = args["target"]
                flowObs = selectSubset(
                    args, np.expand_dims(y_train[:, :, targets.index("00060_Mean")], axis=2),
                    iGrid, iT, rho + warm_up, has_grad=False
                )
                # tempObs = selectSubset(
                #     args, np.expand_dims(y_train[:, :, targets.index("00010_Mean")], axis=2),
                #     iGrid, iT, rho + warm_up, has_grad=False
                # )

                ### MLP
                if type(model) in [MLP]:
                    params = model(c_tensorTrain[iGrid])
                ### CudnnLstm
                if type(model) in [CudnnLstmModel]:
                    params = model(xTrain_sample_scaled)

                params_PRMS = params[:, :, 0:ny_prms]
                # params_SNTEMP = params[:, warm_up:, ny_prms:]
                # params_PRMS = torch.load(r"G:\Farshid\fzr5082\params_PRMS.pt")
                # params_SNTEMP = torch.load(r"G:\Farshid\fzr5082\params_SNTEMP.pt")

                flowSim_total = PRMS(
                    x_PRMS_sample,
                    c_PRMS_sample,
                    params_PRMS,
                    args,
                    # Hamon_coef=params_SNTEMP[:, :, 5 * nmul: 6 * nmul],  # PET is in both temp and flow model
                    Hamon_coef=None,
                    warm_up=warm_up,
                )

                # converting mm/day to m3/ day
                # varC_PRMS = args["varC_PRMS"]
                # area = c_PRMS_sample[:, varC_PRMS.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1, flowSim_total.shape[1])
                # # flow calculation. converting mm/day to m3/sec
                # srflow = (1000 / 86400) * area * (
                #             flowSim_total[:, :, 0] - flowSim_total[:, :, 3] - flowSim_total[:, :, 4])   # Q_t - gw - ss
                # ssflow = (1000 / 86400) * area * (flowSim_total[:, :, 4])   # ras
                # gwflow = (1000 / 86400) * area * (flowSim_total[:, :, 3])

                # temp_sim, _, _, _, _, _ = Ts.forward(x_SNTEMP_sample,
                #                                                  params, iGrid, iT,   #params[:, warm_up:, :]
                #                                                  args=args, air_sample_sr=air_sample_sr,
                #                                                  air_sample_ss=air_sample_ss,
                #                                                  air_sample_gw=air_sample_gw)
                flowSim = flowSim_total[:, :, 0:1]
                mask_yp = flowSim.ge(1e-6)
                flow_sim = flowSim * mask_yp.int().float()
                flowObs = flowObs[warm_up:, :, :].permute([1, 0 , 2])  # to make it in flowSim format
                varC_PRMS = args["varC_PRMS"]
                if "DRAIN_SQKM" in varC_PRMS:
                    A = "DRAIN_SQKM"
                elif "area_gages2" in varC_PRMS:
                    A = "area_gages2"
                area = c_PRMS_sample[:, varC_PRMS.index(A)].unsqueeze(-1).repeat(1, flowObs.shape[1]).unsqueeze(-1)
                flowObs = (10 ** 3) * flowObs * 0.0283168 * 3600 * 24 / (area * (10 ** 6))  #convert ft3/s to mm/day
                # flowSim = flowSim * 0.001 * area * (10 ** 6) * 0.000408735   #converting mm/day to ft3/s
                # loss = lossFun(flowObs, tempObs[warm_up:, :, :].permute([1, 0, 2]),
                #     flowSim_total[:, :, 0].unsqueeze(-1), temp_sim
                # )
                # loss = lossFun(temp_sim, tempObs[warm_up:, :, :].permute([1, 0, 2]))
                loss = lossFun(flowSim, flowObs)
                loss.backward()  # retain_graph=True
                optim.step()
                model.zero_grad()
                lossEp = lossEp + loss.item()
                # if (iIter % 1 == 0) or (iIter == nIterEp):
                #     print(iIter," from ", nIterEp, " in the ", epoch,
                #         "th epoch, and Loss is ", loss.item())
            lossEp = lossEp / nIterEp
            logStr = "Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory".format(
                epoch, lossEp, time.time() - t0,
                int(torch.cuda.memory_allocated(device=args["device"]) * 0.001))
            print(logStr)

            if epoch % args["saveEpoch"] == 0:
                # save model
                modelFile = os.path.join(args["out_dir"], "model_Ep" + str(epoch) + ".pt")
                torch.save(model, modelFile)
            if epoch == args["EPOCHS"]:
                print("last epoch")
        print("end")
        del (x_train_scaled, y_train_scaled, y_train,  x_train)      # ave_air_total,

    if 1 in args["Action"]:
        # to free up some GPU memory
        del c_raw_NN, c_scaled

        warm_up = args["warm_up"]
        nmul = args["nmul"]
        modelFile = os.path.join(args["out_dir"], "model_Ep" + str(args["EPOCHS"]) + ".pt")
        # modelFile = r"/home/fzr5082/PGML_STemp_results/models/415_sites/E_900_R_365_B_208_H_256_dr_0.5_0R1/model_Ep900.pt"

        model = torch.load(modelFile)
        model.eval()

        time1 = hydroDL.utils.time.tRange2Array(args["tRange"])
        # getting the raw and normalized inputs
        x_test, y_test = train_val_test_split("t_test", args, time1, x_total_raw_NN, y_raw)
        x_test_scaled, y_test_scaled = train_val_test_split("t_test", args, time1, x_total_scaled, y_scaled)
        x_PRMS_test, _ = train_val_test_split("t_test", args, time1, x_PRMS, y_raw)
        # x_SNTEMP_test, _ = train_val_test_split("t_test", args, time1, x_SNTEMP, y_raw)
        # ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args["t_test"])
        # mean_air_temp_test = (x_SNTEMP_test[:, :, args["varT_SNTEMP"].index("tmax(C)")] +
        #                        x_SNTEMP_test[:, :, args["varT_SNTEMP"].index("tmin(C)")]) / 2

        del x_total_raw_NN, y_raw, x_total_scaled, y_scaled,  x_PRMS    # x_SNTEMP,
        # x_test_scaled_noccov = np.delete(x_test_scaled, vars.index("ccov"), axis=2)

        np.save(os.path.join(args["out_dir"], "x.npy"), x_test)  # saves with the overlap in the beginning
        x_test_tensor = make_tensor(x_test, has_grad=False)
        x_test_scaled_tensor = make_tensor(x_test_scaled, device=args["device"], has_grad=False)
        x_PRMS_test_tensor = make_tensor(x_PRMS_test,device=args["device"], has_grad=False)
        # x_SNTEMP_test_tensor = make_tensor(x_SNTEMP_test, device=args["device"], has_grad=False)
        y_test_tensor = make_tensor(y_test, device=args["device"], has_grad=False)

        args_mod = args.copy()
        args_mod["batch_size"] = args["no_basins"]
        # args_mod["hyperparameters"]["rho"] = x_test_scaled_tensor.shape[1]
        ngrid, nt, nx = x_test_scaled_tensor.shape
        rho = args["rho"]
        nrows = math.ceil((nt - warm_up) / rho)   # need to reduce the warm_up from beginning
        batch_size = args_mod["batch_size"]
        iS = np.arange(0, ngrid, batch_size)
        iE = np.append(iS[1:], ngrid)

        for i in range(0, len(iS)):
            for j in range(nrows):
                if j != (nrows - 1):
                    cut = (j + 1) * rho + warm_up
                else:
                    cut = y_test_tensor.shape[1]

                # yTemp = torch.tensor(
                #     y_test_tensor[iS[i]: iE[i], j * rho: cut, :]
                # )
                xTemp_scaled = x_test_scaled_tensor[
                               iS[i]: iE[i], j * rho: cut, :
                               ]
                x_PRMS_sample = x_PRMS_test_tensor[iS[i]: iE[i], j * rho: cut, :].type(torch.float32)
                c_PRMS_sample = torch.tensor(
                    c_PRMS[iS[i]: iE[i], :], device=args["device"], dtype=torch.float32
                )

                # x_SNTEMP_sample = x_SNTEMP_test_tensor[iS[i]: iE[i], j * rho: cut, :].type(
                #     torch.float32)#[:, warm_up:, :]
                # c_SNTEMP_sample = torch.tensor(
                #     c_SNTEMP[iS[i]: iE[i], :], device=args["device"], dtype=torch.float32
                # )
                #
                # air_sample_sr = make_tensor(
                #     np.expand_dims(mean_air_temp_test[iS[i]: iE[i], j * rho: cut],
                #                    axis=2), device=args["device"], has_grad=False)[
                #                 :, warm_up - args["res_time_lenF_srflow"]:, :]
                # air_sample_ss = make_tensor(
                #     np.expand_dims(mean_air_temp_test[iS[i]: iE[i], j * rho: cut],
                #                    axis=2), device=args["device"], has_grad=False)[
                #                 :, warm_up - args["res_time_lenF_ssflow"]:, :]
                # air_sample_gw = make_tensor(
                #     np.expand_dims(mean_air_temp_test[iS[i]: iE[i], j * rho: cut],
                #                    axis=2), device=args["device"], has_grad=False)[
                #                 :, warm_up - args["res_time_lenF_gwflow"]:, :]


                if type(model) in [MLP]:
                    params = model(xTemp_scaled)
                ### CudnnLstm
                if type(model) in [CudnnLstmModel]:
                    params = model(xTemp_scaled.float())

                params_PRMS = params[:, :, 0:ny_prms]
                # params_SNTEMP = params[:, warm_up:, ny_prms:]

                iGrid = np.arange(xTemp_scaled.shape[0])
                iT = np.zeros((len(iGrid)))
                flowSim_total = PRMS(
                    x_PRMS_sample,
                    c_PRMS_sample,
                    params_PRMS,
                    args,
                    Hamon_coef=None, #params_SNTEMP[:, :, 5 * nmul: 6 * nmul],  # PET is in both temp and flow model
                    warm_up=warm_up,
                )
                # varC_PRMS = args["varC_PRMS"]
                # area = c_PRMS_sample[:, varC_PRMS.index("DRAIN_SQKM")].unsqueeze(-1).repeat(1,
                #                                                                             flowSim_total.shape[
                #                                                                                 1])
                # # flow calculation. converting mm/day to m3/sec to be fed to SNTEMP
                # srflow = (1000 / 86400) * area * (
                #         flowSim_total[:, :, 0] - flowSim_total[:, :, 3] - flowSim_total[:, :,
                #                                                           4])  # Q_t - gw - ss
                # ssflow = (1000 / 86400) * area * (flowSim_total[:, :, 4])  # ras
                # gwflow = (1000 / 86400) * area * (flowSim_total[:, :, 3])  # bas

                # temp_sim, ave_air_temp, w_gwflow, w_ssflow, source_temps, SNTEMP_outs = Ts.forward(x_SNTEMP_sample,
                #                                                      params, iGrid, iT,   # [:, warm_up:, :]
                #                                                      args=args, air_sample_sr=air_sample_sr,
                #                                                      air_sample_ss=air_sample_ss,
                #                                                      air_sample_gw=air_sample_gw)

                if j == 0:
                    Q_sim = torch.clone(flowSim_total.detach().cpu())
                    # T_sim = torch.clone(temp_sim.detach().cpu())
                    # air_T = torch.clone(ave_air_temp.detach().cpu())
                    # w_gw = torch.clone(w_gwflow.detach().cpu())
                    # w_ss = torch.clone(w_ssflow.detach().cpu())
                    # source_T = torch.clone(source_temps.detach().cpu())
                    # outs = torch.clone(SNTEMP_outs.detach().cpu())
                else:
                    Q_sim = torch.cat((Q_sim, flowSim_total.detach().cpu()), dim=1)
                    # T_sim = torch.cat((T_sim, temp_sim.detach().cpu()), dim=1)
                    # air_T = torch.cat((air_T, ave_air_temp.detach().cpu()), dim=1)
                    # w_gw = torch.cat((w_gw, w_gwflow.detach().cpu()), dim=1)
                    # w_ss = torch.cat((w_ss, w_ssflow.detach().cpu()), dim=1)
                    # source_T = torch.cat((source_T, source_temps.detach().cpu()), dim=1)
                    # outs = torch.cat((outs, SNTEMP_outs.detach().cpu()), dim=1)
            if i == 0:
                flow_pred = torch.clone(Q_sim)
                # temp_pred = torch.clone(T_sim)
                # air_t = torch.clone(air_T)
                # weight_gw = torch.clone(w_gw)
                # weight_ss = torch.clone(w_ss)
                # source_temp = torch.clone(source_T)
                # SN_outs = torch.clone(outs)

            else:
                flow_pred = torch.cat((flow_pred, Q_sim), dim=0)
                # temp_pred = torch.cat((temp_pred, T_sim), dim=0)
                # air_t = torch.cat((air_t, air_T), dim=0)
                # weight_gw = torch.cat((weight_gw, w_gw), dim=0)
                # weight_ss = torch.cat((weight_ss, w_ss), dim=0)
                # source_temp = torch.cat((source_temp, source_T), dim=0)
                # SN_outs = torch.cat((SN_outs, outs), dim=0)
        #
        varC_PRMS = args["varC_PRMS"]
        if "DRAIN_SQKM" in varC_PRMS:
            A = "DRAIN_SQKM"
        elif "area_gages2" in varC_PRMS:
            A = "area_gages2"
        flow_obs = y_test[:, warm_up:, args["target"].index("00060_Mean")]
        area = np.repeat(np.expand_dims(c_PRMS[:, varC_PRMS.index(A)], 1),
                         flow_obs.shape[1]).reshape(c_PRMS.shape[0], flow_obs.shape[1])
        flow_obs = (10 ** 3) * flow_obs * 0.0283168 * 3600 * 24 / (
                area * (10 ** 6))  # convert ft3/s to mm/day
        # temp_obs = y_test[:, warm_up:, args["target"].index("00010_Mean")]
        q_pred = flow_pred[:,:,0].unsqueeze(-1)
        # loss_flow = lossFun(q_pred.detach().cpu(),
        #                np.expand_dims(flow_obs, 2))
        # loss_temp = lossFun(temp_pred.detach().cpu(),
        #                np.expand_dims(temp_obs, 2))

        # print("loss_flow", loss_flow, "\n")
        # print("loss_temp", loss_temp, "\n")

        np.save(os.path.join(args["out_dir"], "flowSim_tot.npy"), flow_pred.cpu().detach().numpy())
        np.save(os.path.join(args["out_dir"], "flow_obs.npy"), np.expand_dims(flow_obs, 2))
        # np.save(os.path.join(args["out_dir"], "temp_pred.npy"), temp_pred.cpu().detach().numpy())
        # np.save(os.path.join(args["out_dir"], "temp_obs.npy"), np.expand_dims(temp_obs, 2))
        # np.save(os.path.join(args["out_dir"], "air_t.npy"), air_t.cpu().detach().numpy())
        # np.save(os.path.join(args["out_dir"], "weight_gw.npy"), weight_gw.cpu().detach().numpy())
        # np.save(os.path.join(args["out_dir"], "weight_ss.npy"), weight_ss.cpu().detach().numpy())
        # np.save(os.path.join(args["out_dir"], "source_temp.npy"), source_temp.cpu().detach().numpy())
        # np.save(os.path.join(args["out_dir"], "SN_outs.npy"), SN_outs.cpu().detach().numpy())

        predLst_flow = list()
        obsLst_flow = list()
        predLst_flow.append(flow_pred[:, :, 0: 1].cpu().detach().numpy())
        obsLst_flow.append(np.expand_dims(flow_obs, 2))
        statDictLst = [
            stat.statError(x.squeeze(), y.squeeze())
            for (x, y) in zip(predLst_flow, obsLst_flow)
        ]

        # predLst_temp = list()
        # obsLst_temp = list()
        # predLst_temp.append(temp_pred.cpu().detach().numpy())
        # obsLst_temp.append(np.expand_dims(temp_obs, 2))
        # statDictLst = [
        #     stat.statError(x.squeeze(), y.squeeze())
        #     for (x, y) in zip(predLst_temp, obsLst_temp)
        # ]

        ### save this file too
        # median and STD calculation
        count = 0
        mdstd = np.zeros([len(statDictLst[0]), 3])
        for i in statDictLst[0].values():
            median = np.nanmedian((i))  # abs(i)
            STD = np.nanstd((i))  # abs(i)
            mean = np.nanmean((i))  # abs(i)
            k = np.array([[median, STD, mean]])
            mdstd[count] = k
            count = count + 1
        mdstd = pd.DataFrame(
            mdstd, index=statDictLst[0].keys(), columns=["median", "STD", "mean"]
        )
        mdstd.to_csv((os.path.join(args["out_dir"], "mdstd_temp.csv")))


        # Show boxplots of the results
        plt.rcParams["font.size"] = 14
        keyLst = ["Bias", "RMSE", "ubRMSE", "NSE", "Corr"]
        dataBox = list()
        for iS in range(len(keyLst)):
            statStr = keyLst[iS]
            temp = list()
            for k in range(len(statDictLst)):
                data = statDictLst[k][statStr]
                data = data[~np.isnan(data)]
                temp.append(data)
            dataBox.append(temp)
        labelname = [
            "PGML"
        ]  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

        xlabel = ["Bias ($\mathregular{deg}$C)", "RMSE", "ubRMSE", "NSE", "Corr"]
        fig = plot.plotBoxFig(
            dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8)
        )
        fig.patch.set_facecolor("white")
        boxPlotName = "PGML"
        fig.suptitle(boxPlotName, fontsize=12)
        plt.rcParams["font.size"] = 12
        plt.savefig(
            os.path.join(args["out_dir"], "Boxplot.png")
        )  # , dpi=500
        fig.show()
        plt.close()
        print("END testing")
        del x_total_raw_tensor,  x_test_scaled  # ,x_test_scaled_noccov

        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = config
    main_marrmot_PRMS_only(args)
    print("END")
