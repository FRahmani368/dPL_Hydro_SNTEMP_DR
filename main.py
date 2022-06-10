from core.read_configurations import config
from core.randomseed_config import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split, randomIndex, selectSubset
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ, CudnnLstm, CudnnLstmModel
from MODELS import crit
from core import hydroDL
from core.small_codes import make_tensor, tRange2Array, intersect
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from post import stat, plot
import math


def syntheticP(args):
    x_total_temp, y_raw, c_raw = load_df(args)
    x_total_raw = x_total_temp.copy()
    x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
    time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
    x_train, y_train, ngrid_train, nIterEp, nt, rho, batchSize = train_val_test_split("t_train", args, time1,
                                                                                      x_total_raw, y_raw)
    x_train_scaled, y_train_scaled, _, _, _, _, _ = train_val_test_split("t_train", args, time1,
                                                                         x_total_scaled, y_scaled)
    x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False)
    model = STREAM_TEMP_EQ(args, x_total_raw_tensor)
    if torch.cuda.is_available():
        model = model.cuda()
    # initializing the variables
    ave_air_temp = torch.empty((args['hyperparameters']['batch_size'],
                                x_train.shape[1],
                                len(args['params_target'])), device=args['device'], requires_grad=False)

    iGrid = np.arange(99)
    iT = np.zeros(99)
    Yp, ave_air_temp = model.forward(torch.from_numpy(x_train).cuda().float(), iGrid, iT, ave_air_temp)
    # Yp = model.forward(torch.from_numpy(x_train).cuda().float(), ave_air_temp)
    print('end')




def main(args):
    # setting random seeds
    randomseed_config(args)
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        args['randomseed'] = seed
        # Creating output directories mn
        args = create_output_dirs(args)
        min_max_scaler = preprocessing.MinMaxScaler()
        # getting the data
        x_total_temp, y_raw, c_raw = load_df(args)
        x_total_raw = x_total_temp.copy()
        x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
        c_scaled_0_1 = min_max_scaler.fit_transform(c_raw)
        time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
        x_train, y_train, ngrid_train, nIterEp, nt, batchSize = train_val_test_split("t_train", args, time1,
                                                                                          x_total_raw, y_raw)
        x_train_scaled, y_train_scaled, _, _, _, _ = train_val_test_split("t_train", args, time1,
                                                                                          x_total_scaled, y_scaled)

        vars = args["optData"]["varT"] + args["optData"]["varC"]
        x_train_scaled_noccov = np.delete(x_train_scaled, vars.index("ccov"), axis=2)
        # changing the numpy to tensor
        # (x_total_raw_tensor, y_raw_tensor, c_raw_tensor,
        #  x_total_scaled_tensor, y_scaled_tensor, c_scaled_tensor,
        #  x_train_tensor, y_train_tensor) = make_tensor(x_total_raw, y_raw, c_raw,
        #                                                x_total_scaled, y_scaled, c_scaled,
        #                                                x_train, y_train)
        x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False)


        # ANN model to simulate parameters
        # model = MLP(args)
        model = CudnnLstmModel(nx=len(args["optData"]["varT"] + args["optData"]["varC"])-1,
                            ny=15,
                            hiddenSize=args["hyperparameters"]["hidden_size"],
                            dr=args["hyperparameters"]["dropout"])
        Ts = STREAM_TEMP_EQ()
        # model = torch.load(r"/home/fzr5082/PGML_STemp_results/models/E_560_R_365_B_50_H_256_dr_0.5/model_Ep560.pt")
        #
        # loss function
        lossFun = crit.RmseLoss()
        optim = torch.optim.Adadelta(model.parameters())   # , lr=0.1
        # optim = torch.optim.SGD(model.parameters(), lr=10)

        if torch.cuda.is_available():
            model = model.cuda()
            # mlp = mlp.cuda()
            Ts = Ts.cuda()
            lossFun = lossFun.cuda()
            torch.backends.cudnn.deterministic = True
            CUDA_LAUNCH_BLOCKING = 1
            # moving dataset to CUDA

        c_tensorTrain = make_tensor(c_scaled_0_1, has_grad=False)
        # init_shade = mlp(c_tensorTrain)
        # model.shade_fraction[:, 0] = init_shade[:, 0]
        # model.shade_fraction = nn.ParameterList([nn.Parameter(x) for x in init_shade])
        # model.shade = nn.Parameter(init_shade)
        # model.a_srflow = nn.Parameter(torch.ones(args['no_basins'], args["res_time_params"]["lenF_srflow"], 1))
        # model.bias_srflow[:] = 0
        # model.bias_ssflow[:] = 0
        # model.bias_gwflow[:] = 0

        # initializing the variables
        ave_air_temp = torch.zeros(args['hyperparameters']['batch_size'], args["hyperparameters"]["rho"],
                                   device=args["device"], dtype=torch.float32, requires_grad=False)
        shade_fraction_riparian = torch.zeros(args['hyperparameters']['batch_size'], 1,
                                              device=args["device"], dtype=torch.float32, requires_grad=False)

        gwflow_percentage = torch.zeros(args['hyperparameters']['batch_size'], args["hyperparameters"]["rho"],
                                   device=args["device"], dtype=torch.float32, requires_grad=False)
        ssflow_percentage = torch.zeros(args['hyperparameters']['batch_size'], args["hyperparameters"]["rho"],
                                        device=args["device"], dtype=torch.float32, requires_grad=False)
        # ss_tau = torch.zeros(args['hyperparameters']['batch_size'], args["hyperparameters"]["rho"],
        #                                 device=args["device"], dtype=torch.float32, requires_grad=False)
        # gw_tau = torch.zeros(args['hyperparameters']['batch_size'], args["hyperparameters"]["rho"],
        #                                 device=args["device"], dtype=torch.float32, requires_grad=False)

        if 0 in args['Action']:
            ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args['optData']['t_train'])
            rho = args["hyperparameters"]["rho"]
            model.zero_grad()
            model.train()
            # training
            # shade_fraction_riparian = init_shade.clone()
            for epoch in range(1, args['hyperparameters']['EPOCHS'] + 1):
                lossEp = 0
                t0 = time.time()
                for iIter in range(1, nIterEp + 1):
                    iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])
                    # iGrid = np.arange(99)
                    # iT = np.zeros(99, dtype=np.int32)
                    xTrain_sample = selectSubset(x_train, iGrid, iT, rho, has_grad=False)
                    xTrain_sample_scaled = selectSubset(x_train_scaled_noccov, iGrid, iT, rho, has_grad=False) #x_train_scaled
                    ### MLP
                    if type(model) in [MLP]:
                        params = model(c_tensorTrain[iGrid])
                    ### CudnnLstm
                    if type(model) in [CudnnLstmModel]:
                        params = model(xTrain_sample_scaled.permute(1, 0, 2))
                    yObs = selectSubset(y_train, iGrid, iT, rho, has_grad=False)
                    # Yp, ave_air_temp = model.forward(xTrain_sample.transpose(0, 1), iGrid, iT, ave_air_temp)
                    # Yp, ave_air_temp = Ts.forward(xTrain_sample.transpose(0, 1), params, iGrid, iT, ave_air_temp,
                    #                               args=args, x_total_raw=x_total_raw_tensor,
                    #                               time_range=args['optData']['t_train'])
                    Yp, ave_air_temp, gwflow_percentage, ssflow_percentage, gw_tau, ss_tau, pet, \
                    shade_fraction_riparian, shade_fraction_topo, \
                    top_width, cloud_fraction, hamon_coef = Ts.forward(xTrain_sample.transpose(0, 1),
                                                                     params, iGrid, iT, ave_air_temp,
                                                                     args=args, ave_air_total=ave_air_total,
                                                                     gwflow_percentage=gwflow_percentage,
                                                                     ssflow_percentage=ssflow_percentage)

                    mask_yp = Yp.ge(1e-6)
                    y_sim = Yp * mask_yp.int().float()
                    loss = lossFun(y_sim.unsqueeze(-1), yObs.transpose(1, 0))
                    # loss = lossFun(test_sim, test)
                    # c = list(model.parameters())[0].clone()
                    loss.backward()  # retain_graph=True
                    # for param in model.parameters():
                    #     print(param.grad.data.sum())
                    optim.step()
                    # b = list(model.parameters())[0].clone()
                    # print(torch.equal(c.data, b.data))
                    model.zero_grad()
                    lossEp = lossEp + loss.item()
                    # del loss
                    # del Yp
                    # print(iIter, " from ", nIterEp, " in the ", epoch, "th epoch, and Loss is ", loss.item())
                lossEp = lossEp / nIterEp
                # torch.cuda.synchronize()
                logStr = 'Epoch {} Loss {:.6f}, time {:.2f} sec, {} Kb allocated GPU memory'.format(
                    epoch, lossEp,
                    time.time() - t0, int(torch.cuda.memory_allocated() * 0.001))
                print(logStr)
                # if lossEp < 0.05:
                #     torch.save(res_time, os.path.join(args['output']['data'], 'res_time' + str(lossEp) + ".pt"))
                #     torch.save(res_time_mod, os.path.join(args['output']['data'], 'res_time_mod' + str(lossEp) + ".pt"))
                # elif lossEp != lossEp:
                #     print("stop")
                if epoch % args['hyperparameters']['saveEpoch'] == 0:
                    # save model
                    modelFile = os.path.join(args['output']['out_dir'],
                                             'model_Ep' + str(epoch) + '.pt')
                    torch.save(model, modelFile)
                if epoch == args['hyperparameters']['EPOCHS']:
                    print('last epoch')
            print('end')

        if 1 in args['Action']:
            modelFile = os.path.join(args['output']['out_dir'],
                                      'model_Ep' + str(args['hyperparameters']['EPOCHS']) + '.pt')
            # modelFile = r"/home/fzr5082/PGML_STemp_results/models/415_sites/E_900_R_365_B_208_H_256_dr_0.5_0R1/model_Ep900.pt"
            # modelFile = r"/home/fzr5082/PGML_STemp_results/models/415_sites/E_1000_R_365_B_208_H_256_dr_0.5_4/model_Ep650.pt"
            # modelFile = r"/home/fzr5082/PGML_STemp_results/models/E_300_R_365_B_50_H_100_dr_0.5/model_Ep20.pt"\
            # modelFile = r"/home/fzr5082/PGML_STemp_results/models/E_300_R_365_B_99_H_100_dr_0.5/model_Ep80.pt"\
            # modelFile = r"/home/fzr5082/PGML_STemp_results/models/E_1650_R_730_B_50_H_100_dr_0.5/model_Ep1700.pt"
            model = torch.load(modelFile)
            model.eval()
            # iGrid = np.arange(99)

            time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
            x_test, y_test, ngrid_test, nIterEp, nt, batchSize = train_val_test_split("t_test", args, time1,
                                                                                              x_total_raw, y_raw)
            #Normalizing the inputs for ML part
            x_test_scaled, _, _, _, _, _ = train_val_test_split("t_test", args, time1,
                                                                                           x_total_scaled, y_raw)
            x_test_scaled_noccov = np.delete(x_test_scaled, vars.index("ccov"), axis=2)

            np.save(os.path.join(args['output']['out_dir'], 'x.npy'), x_test)
            x_test_tensor = make_tensor(x_test, has_grad=False)
            x_test_scaled_tensor = make_tensor(x_test_scaled_noccov, has_grad=False) #x_test_scaled
            y_test_tensor = make_tensor(y_test, has_grad=False)

            # iGrid = np.arange(x_test_scaled_tensor.shape[0])
            # iT = np.zeros(x_test_scaled_tensor.shape[1], dtype=np.int32)
            iGrid = np.array([0])
            iT = np.zeros((1))
            args_mod = args.copy()
            # args_mod["hyperparameters"]["batch_size"] = args['no_basins']
            # args_mod["hyperparameters"]["rho"] = x_test_scaled_tensor.shape[1]
            ngrid, nt, nx = x_test_scaled_tensor.shape
            rho = args['hyperparameters']['rho']
            nrows = math.ceil(nt/rho)
            batch_size = args_mod["hyperparameters"]["batch_size"]
            iS = np.arange(0, ngrid, batch_size)
            iE = np.append(iS[1:], ngrid)
            ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args['optData']['t_test'])
            for i in range(0, len(iS)):
                # print('batch {}'.format(i))

                for j in range(nrows):
                    if j != (nrows - 1):
                        yTemp = torch.tensor(y_test_tensor[iS[i]:iE[i], j * rho: (j + 1) * rho, :])
                        xTemp_scaled = x_test_scaled_tensor[iS[i]:iE[i], j * rho: (j + 1) * rho, :]
                        xTemp = x_test_tensor[iS[i]:iE[i], j * rho: (j + 1) * rho, :]
                        ave_air_test = ave_air_total[iS[i]:iE[i], j * rho: (j + 1) * rho, :]
                        if type(model) in [MLP]:
                            params = model(xTemp_scaled)
                        ### CudnnLstm
                        if type(model) in [CudnnLstmModel]:
                            params = model(xTemp_scaled)
                        iGrid = np.arange(xTemp.shape[0])
                        iT = np.zeros((len(iGrid)))
                        Yp, ave_air_temp, gwflow_percentage, ssflow_percentage, gw_tau, ss_tau, pet,\
                        shade_fraction_riparian, shade_fraction_topo, \
                        top_width, cloud_fraction, hamon_coef = Ts.forward(xTemp, params, iGrid,
                                                      iT, ave_air_temp,
                                                      args=args_mod, ave_air_total=ave_air_test,
                                                      gwflow_percentage=gwflow_percentage,
                                                      ssflow_percentage=ssflow_percentage)

                    else:
                        yTemp = torch.tensor(y_test_tensor[iS[i]:iE[i], j * rho:, :])
                        xTemp_scaled = x_test_scaled_tensor[iS[i]:iE[i], j * rho:, :]
                        xTemp = x_test_tensor[iS[i]:iE[i], j * rho:, :]
                        ave_air_test = ave_air_total[iS[i]:iE[i], j * rho:, :]
                        if type(model) in [MLP]:
                            params = model(xTemp_scaled)
                        ### CudnnLstm
                        if type(model) in [CudnnLstmModel]:
                            params = model(xTemp_scaled)
                        iGrid = np.arange(xTemp.shape[0])
                        iT = np.zeros((len(iGrid)))
                        Yp, ave_air_temp, gwflow_percentage, ssflow_percentage, gw_tau, ss_tau, pet,\
                        shade_fraction_riparian, shade_fraction_topo, \
                        top_width, cloud_fraction, hamon_coef = Ts.forward(xTemp, params, iGrid,
                                                    iT, ave_air_temp,
                                                    args=args_mod, ave_air_total=ave_air_test,
                                                    gwflow_percentage=gwflow_percentage,
                                                    ssflow_percentage=ssflow_percentage)
                        # yP, _ = model(torch.tensor(xTemp).float().cuda())
                        # yP = model(torch.tensor(xTemp).float().cuda(), yTemp)

                    if (j == 0):
                        out = Yp.detach().cpu()
                        obstemp = yTemp
                        gw = gwflow_percentage.unsqueeze(-1).detach().cpu()
                        ss = ssflow_percentage.unsqueeze(-1).detach().cpu()
                        w_gw_tau = gw_tau.unsqueeze(-1).detach().cpu()
                        w_ss_tau = ss_tau.unsqueeze(-1).detach().cpu()
                        PET = pet.unsqueeze(-1).detach().cpu()
                        shade_frac_rip = shade_fraction_riparian.unsqueeze(-1).detach().cpu()
                        shade_frac_top = shade_fraction_topo.unsqueeze(-1).detach().cpu()
                        top_w = top_width.unsqueeze(-1).detach().cpu()
                        cloud = cloud_fraction.unsqueeze(-1).detach().cpu()
                        hamon_co = hamon_coef.unsqueeze(-1).detach().cpu()
                        lat_temp = ave_air_temp.unsqueeze(-1).detach().cpu()
                    else:
                        out = torch.cat((out, Yp.detach().cpu()), dim=1)  # Farshid: should dim be 1 or 2?
                        obstemp = torch.cat((obstemp, yTemp), dim=1)
                        gw = torch.cat((gw, gwflow_percentage.unsqueeze(-1).detach().cpu()), dim=1)
                        ss = torch.cat((ss, ssflow_percentage.unsqueeze(-1).detach().cpu()), dim=1)
                        ww_gw_tau = torch.cat((w_gw_tau, gw_tau.unsqueeze(-1).detach().cpu()), dim=1)
                        ww_ss_tau = torch.cat((w_ss_tau, ss_tau.unsqueeze(-1).detach().cpu()), dim=1)
                        PET_m = torch.cat((PET, pet.unsqueeze(-1).detach().cpu()), dim=1)
                        shade_frac_rip_m = torch.cat((shade_frac_rip, shade_fraction_riparian.unsqueeze(-1).detach().cpu()), dim=1)
                        shade_frac_top_m = torch.cat((shade_frac_top, shade_fraction_topo.unsqueeze(-1).detach().cpu()), dim=1)
                        top_w_m = torch.cat((top_w, top_width.unsqueeze(-1).detach().cpu()), dim=1)
                        cloud_m = torch.cat((cloud, cloud_fraction.unsqueeze(-1).detach().cpu()), dim=1)
                        hamon_co_m = torch.cat((hamon_co, hamon_coef.unsqueeze(-1).detach().cpu()), dim=1)
                        lat_temp_m = torch.cat((lat_temp, ave_air_temp.unsqueeze(-1).detach().cpu()), dim=1)
                if i == 0:
                    pred = out
                    obs = obstemp
                    gw_p = gw
                    ss_p = ss
                    weight_gw = ww_gw_tau
                    weight_ss = ww_ss_tau
                    PET_mm = PET_m
                    shade_frac_rip_mm = shade_frac_rip_m
                    shade_frac_top_mm = shade_frac_top_m
                    top_w_mm = top_w_m
                    cloud_mm = cloud_m
                    hamon_co_mm = hamon_co_m
                    lat_temp_mm = lat_temp_m
                else:
                    pred = torch.cat((pred, out), dim=0)
                    obs = torch.cat((obs, obstemp), dim=0)
                    gw_p = torch.cat((gw_p, gw), dim=0)
                    ss_p = torch.cat((ss_p, ss), dim=0)
                    weight_gw = torch.cat((weight_gw, ww_gw_tau), dim=0)
                    weight_ss = torch.cat((weight_ss, ww_ss_tau), dim=0)
                    PET_mm = torch.cat((PET_mm, PET_m), dim=0)
                    shade_frac_rip_mm = torch.cat((shade_frac_rip_mm, shade_frac_rip_m), dim=0)
                    shade_frac_top_mm = torch.cat((shade_frac_top_mm, shade_frac_top_m), dim=0)
                    top_w_mm = torch.cat((top_w_mm, top_w_m), dim=0)
                    cloud_mm = torch.cat((cloud_mm, cloud_m), dim=0)
                    hamon_co_mm = torch.cat((hamon_co_mm, hamon_co_m), dim=0)
                    lat_temp_mm = torch.cat((lat_temp_mm, lat_temp_m), dim=0)


            # if type(model) in [MLP]:
            #     params = model(c_tensorTrain[iGrid])
            # ### CudnnLstm
            # if type(model) in [CudnnLstmModel]:
            #     params = model(x_test_scaled_tensor[iGrid])
            #
            # # params = model(c_tensorTrain)
            # # yObs = selectSubset(y_test, iGrid, iT, rho, has_grad=False)
            # ave_air_total = Ts.ave_temp_general(args, x_total_raw_tensor, time_range=args['optData']['t_test'])
            # # Yp, ave_air_temp = Ts.forward(x_test_tensor[iGrid], params, iGrid, iT, ave_air_temp,
            # #                               args=args_mod, x_total_raw=x_total_raw_tensor,
            # #                               time_range=args['optData']['t_test'])
            # Yp, ave_air_temp = Ts.forward(x_test_tensor[iGrid], params, iGrid, iT, ave_air_temp,
            #                               args=args_mod, ave_air_total=ave_air_total)

            mask_pred = pred.ge(0)
            y_sim = (pred * mask_pred.int().float()).unsqueeze(-1)
            loss = lossFun(y_sim.detach().cpu(), obs.detach().cpu())
            print(loss)


            predLst = list()
            obsLst = list()
            y_sim_np = y_sim.detach().cpu().numpy()
            y_obs_np = obs.detach().cpu().numpy()
            gw_p_np = gw_p.detach().cpu().numpy()
            ss_p_np = ss_p.detach().cpu().numpy()
            weight_gw_np = weight_gw.detach().cpu().numpy()
            weight_ss_np = weight_ss.detach().cpu().numpy()
            PET_mm_np = PET_mm.detach().cpu().numpy()
            shade_frac_rip_mm_np = shade_frac_rip_mm.detach().cpu().numpy()
            shade_frac_top_mm_np = shade_frac_top_mm.detach().cpu().numpy()
            top_w_mm_np = top_w_mm.detach().cpu().numpy()
            cloud_mm_np = cloud_mm.detach().cpu().numpy()
            hamon_co_mm_np = hamon_co_mm.detach().cpu().numpy()
            lat_temp_mm_np = lat_temp_mm.detach().cpu().numpy()
            predLst.append(y_sim_np)  # the prediction list for all the models
            obsLst.append(y_obs_np)
            np.save(os.path.join(args['output']['out_dir'], 'pred.npy'), y_sim_np)
            np.save(os.path.join(args['output']['out_dir'], 'obs.npy'), y_obs_np)
            np.save(os.path.join(args['output']['out_dir'], 'gw_p.npy'), gw_p_np)
            np.save(os.path.join(args['output']['out_dir'], 'ss_p.npy'), ss_p_np)
            np.save(os.path.join(args['output']['out_dir'], 'weight_gw.npy'), weight_gw_np)
            np.save(os.path.join(args['output']['out_dir'], 'weight_ss.npy'), weight_ss_np)
            np.save(os.path.join(args['output']['out_dir'], 'PET.npy'), PET_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'shade_frac_rip.npy'), shade_frac_rip_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'shade_frac_topo.npy'), shade_frac_top_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'top_width.npy'), top_w_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'cloud_frac.npy'), cloud_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'hamon_coef.npy'), hamon_co_mm_np)
            np.save(os.path.join(args['output']['out_dir'], 'lat_temp.npy'), lat_temp_mm_np)
            statDictLst = [stat.statError(x.squeeze(), y.squeeze()) for (x, y) in zip(predLst, obsLst)]
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
            mdstd = pd.DataFrame(mdstd, index=statDictLst[0].keys(), columns=['median', 'STD', 'mean'])
            mdstd.to_csv((os.path.join(args['output']['out_dir'], "mdstd.csv")))

            # Show boxplots of the results
            plt.rcParams['font.size'] = 14
            keyLst = ['Bias', 'RMSE', 'ubRMSE', 'NSE', 'Corr']
            dataBox = list()
            for iS in range(len(keyLst)):
                statStr = keyLst[iS]
                temp = list()
                for k in range(len(statDictLst)):
                    data = statDictLst[k][statStr]
                    data = data[~np.isnan(data)]
                    temp.append(data)
                dataBox.append(temp)
            labelname = ['PGML']  # ['STA:316,batch158', 'STA:156,batch156', 'STA:1032,batch516']   # ['LSTM-34 Basin']

            xlabel = ['Bias ($\mathregular{deg}$C)', 'RMSE', "ubRMSE", 'NSE', 'Corr']
            fig = plot.plotBoxFig(dataBox, xlabel, label2=labelname, sharey=False, figsize=(16, 8))
            fig.patch.set_facecolor('white')
            boxPlotName = "PGML"
            fig.suptitle(boxPlotName, fontsize=12)
            plt.rcParams['font.size'] = 12
            plt.savefig(os.path.join(args['output']['out_dir'], "Boxplot.png"))  # , dpi=500
            fig.show()
            plt.close()
            print("END testing")





if __name__=='__main__':
    args = config
    # syntheticP(args)
    main(args)
    print('END')












