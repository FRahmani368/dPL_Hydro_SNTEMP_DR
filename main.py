from core.read_configurations import config
from core.randomseed_config import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split, randomIndex, selectSubset
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ
from MODELS import crit
from core import hydroDL
from core.small_codes import make_tensor, tRange2Array, intersect
import torch
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


def ave_temp_res_time(args, ave_air_temp, x, res_time, res_rnd, x_total, iGrid, iT):
    rho = args['hyperparameters']['rho']
    tArray_Total = tRange2Array(args['optData']['tRange'])
    tArray_train = tRange2Array(args['optData']['t_train'])
    _, ind1, _ = (intersect(tArray_Total, tArray_train))
    ind1_tensor = make_tensor(ind1, has_grad=False)
    iT_tensor = make_tensor(iT, has_grad=False)
    vars = args['optData']['varT'] + args['optData']['varC']
    temp_data = torch.empty((x.shape[1], x.shape[0]))
    # for i in range(res_time.shape[1]):
    #     for station in range(x.shape[1]):
    #         for day in range(x.shape[0]):
    #             tmax_temp = x_total[station,
    #                         ind1[day] + iT[station] - int(res_time[station, i]) + 1: ind1[day] + iT[
    #                             station] + 1,
    #                         vars.index("tmax(C)")].sum().item()
    #             tmin_temp = x_total[station,
    #                         ind1[day] + iT[station] - int(res_time[station, i]) + 1: ind1[day] + iT[
    #                             station] + 1,
    #                         vars.index("tmin(C)")].sum().item()
    #             temp_data[station, day] = tmax_temp + tmin_temp
    #             ave_air_temp[station, day, i] = (tmax_temp + tmin_temp) / (2 * res_time[station, i])
        # ave_air_temp[:, day, :] = temp_data / (2 * res_time)

    # #############
    # for i in range(res_time.shape[1]):
    #     for day in range(x.shape[0]):
    #         tmax_temp = x_total[np.arange(99),
    #                     (ind1_tensor[day] + iT_tensor[:] - (res_time[:, i]) + 1).int(): (ind1_tensor[day] + iT_tensor[:] + 1).int(),
    #                     vars.index("tmax(C)")].sum().item()
    #         tmin_temp = x_total[:,
    #                     ind1[day] + iT[:] - int(res_time[:, i]) + 1: ind1[day] + iT[
    #                         :] + 1,
    #                     vars.index("tmin(C)")].sum().item()
    #         temp_data[:, day] = tmax_temp + tmin_temp
    #         ave_air_temp[:, day, i] = (tmax_temp + tmin_temp) / (2 * res_time[:, i])
    # #############
    #############
    for i in range(res_time.shape[1]):
        for station in range(x.shape[1]):
            array = np.zeros((x.shape[0], res_time[station, i].int().item()), dtype=np.int32)
            for j in range(res_time[station, i].int().item()):
                array[:, j] = np.arange((ind1_tensor[0] + iT_tensor[station] - j).item(),
                                           (ind1_tensor[0] + iT_tensor[station] - j + x.shape[0]).item())
            tmax_temp = x_total[station, array, vars.index("tmax(C)")]
            max_add = torch.sum(tmax_temp, dim=1)
            tmin_temp = x_total[station, array, vars.index("tmin(C)")]
            min_add = torch.sum(tmin_temp, dim=1)
            ave_air_temp[station, :, i] = (max_add + min_add) / (2 * res_time[station, i])
    #############
    return ave_air_temp


def main(args):
    # setting random seeds
    randomseed_config(args)

    # Creating output directories
    args = create_output_dirs(args)

    # getting the data
    x_total_temp, y_raw, c_raw = load_df(args)
    x_total_raw = x_total_temp.copy()
    x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
    time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
    x_train, y_train, ngrid_train, nIterEp, nt, rho, batchSize = train_val_test_split("t_train", args, time1,
                                                                                      x_total_raw, y_raw)
    x_train_scaled, y_train_scaled, _, _, _, _, _ = train_val_test_split("t_train", args, time1,
                                                                                      x_total_scaled, y_scaled)

    # changing the numpy to tensor
    # (x_total_raw_tensor, y_raw_tensor, c_raw_tensor,
    #  x_total_scaled_tensor, y_scaled_tensor, c_scaled_tensor,
    #  x_train_tensor, y_train_tensor) = make_tensor(x_total_raw, y_raw, c_raw,
    #                                                x_total_scaled, y_scaled, c_scaled,
    #                                                x_train, y_train)
    x_total_raw_tensor = make_tensor(x_total_raw, has_grad=False)

    # ANN model to simulate parameters
    model = MLP(args)
    # loss function
    lossFun = crit.RmseLoss()
    optim = torch.optim.Adadelta(model.parameters())
    # optim = torch.optim.SGD(model.parameters(), lr=0.0001)

    if torch.cuda.is_available():
        model = model.cuda()
        lossFun = lossFun.cuda()
        # moving dataset to CUDA
    model.zero_grad()
    T_w = STREAM_TEMP_EQ(args, x_total_raw_tensor)

    # initializing the variables
    ave_air_temp = torch.empty((args['hyperparameters']['batch_size'],
                                args['hyperparameters']['rho'],
                                len(args['params_target'])), device=args['device'], requires_grad=False)

    res_time_mod = torch.empty((args['hyperparameters']['batch_size'],
                                len(args['params_target'])), device=args['device'], requires_grad=False)
    Tprime_e = torch.empty((args['hyperparameters']['batch_size'],
                            args['hyperparameters']['rho']), device=args['device'], requires_grad=False)
    R = torch.empty((args['hyperparameters']['batch_size'],
                     args['hyperparameters']['rho']), device=args['device'], requires_grad=False)
    a = torch.empty((args['hyperparameters']['batch_size'],
                     args['hyperparameters']['rho']), device=args['device'], requires_grad=False)

    factor = torch.empty((args['hyperparameters']['batch_size'],
                                len(args['params_target'])), device=args['device'], requires_grad=False)
    factor[:, 0] = 10   # srflow residence time
    factor[:, 1] = 200   # ssflow residence time
    factor[:, 2] = 1000   # gwflow residence factor
    # training

    for epoch in range(1, args['hyperparameters']['EPOCHS'] + 1):
        lossEp = 0
        t0 = time.time()
        for iIter in range(1, nIterEp + 1):
            iGrid, iT = randomIndex(ngrid_train, nt, [batchSize, rho])
            xTrain_sample = selectSubset(x_train, iGrid, iT, rho, has_grad=False)
            xTrain_sample_scaled = selectSubset(x_train_scaled, iGrid, iT, rho, has_grad=False)
            yObs = selectSubset(y_train, iGrid, iT, rho, has_grad=False)
            c_tensorTrain = make_tensor(c_scaled[iGrid], has_grad=False)
            res_time = model(c_tensorTrain)
            res_time_mod = (res_time * factor + 1) #.round()

            # res_time = model(xTrain_sample_scaled)
            # ave_air_temp = ave_temp_res_time(args, ave_air_temp, xTrain_sample, res_time_mod, res_rnd, x_total_raw_tensor[iGrid], iGrid, iT)
            #
            # Yp, Tprime_e, R, a = T_w.forward(xTrain_sample.transpose(0, 1), ave_air_temp, Tprime_e=Tprime_e, R=R, a=a)
            loss = lossFun(res_time_mod[:, 0:1].unsqueeze(-1), yObs[0:1, :, :].transpose(1, 0))
            # loss = lossFun(ave_air_temp[:, :, 0:1], yObs.transpose(1, 0))
            # loss = lossFun(Yp.unsqueeze(-1), yObs.transpose(1, 0))
            c = list(model.parameters())[0].clone()
            loss.backward()  # retain_graph=True
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optim.step()
            b = list(model.parameters())[0].clone()
            print(torch.equal(c.data, b.data))
            model.zero_grad()
            lossEp = lossEp + loss.item()
            print(iIter, " from ", nIterEp, " in the ", epoch, "th epoch, and Loss is ", loss.item())
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.6f} time {:.2f}'.format(
            epoch, lossEp,
            time.time() - t0)
        print(logStr)

        if epoch % args['hyperparameters']['saveEpoch'] == 0:
            # save model
            modelFile = os.path.join(args['output']['out_dir'],
                                     'model_Ep' + str(epoch) + '.pt')
            torch.save(model, modelFile)

    print('end')

if __name__=='__main__':
    args = config
    main(args)
    print('END')
