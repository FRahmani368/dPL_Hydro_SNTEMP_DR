from core.read_configurations import config
from core.randomseed_config import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split, randomIndex, selectSubset
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ
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
from rnn_dp import UH_gamma, UH_conv


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

    # Creating output directories
    args = create_output_dirs(args)
    min_max_scaler = preprocessing.MinMaxScaler()
    # getting the data
    x_total_temp, y_raw, c_raw = load_df(args)
    x_total_raw = x_total_temp.copy()
    x_total_scaled, y_scaled, c_scaled = scaling(args, x_total_temp, y_raw, c_raw)
    c_scaled_0_1 = min_max_scaler.fit_transform(c_raw)
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
    mlp = MLP(args)
    model = STREAM_TEMP_EQ(args, x_total_raw_tensor)
    # model = torch.load("/home/fzr5082/PGML_STemp_results/models/E_400_R_730_B_99_H_60_dr_0.5/model_Ep400.pt")

    # loss function
    lossFun = crit.RmseLoss()
    optim = torch.optim.Adadelta(model.parameters(), lr=400)
    # optim = torch.optim.SGD(model.parameters(), lr=10)

    if torch.cuda.is_available():
        model = model.cuda()
        mlp = mlp.cuda()
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


    model.zero_grad()
    model.train()

    # initializing the variables
    ave_air_temp = torch.zeros(args['hyperparameters']['batch_size'], args["hyperparameters"]["rho"],
                               device=args["device"], dtype=torch.float32, requires_grad=False)
    shade_fraction_riparian = torch.zeros(args['hyperparameters']['batch_size'], 1,
                               device=args["device"], dtype=torch.float32, requires_grad=False)
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
            xTrain_sample_scaled = selectSubset(x_train_scaled, iGrid, iT, rho, has_grad=False)
            yObs = selectSubset(y_train, iGrid, iT, rho, has_grad=False)
            Yp, ave_air_temp = model.forward(xTrain_sample.transpose(0, 1), iGrid, iT, ave_air_temp)
            # mask_yp = Yp.ge(0)
            # y_sim = Yp * mask_yp.int().float()
            loss = lossFun(Yp.unsqueeze(-1), yObs.transpose(1, 0))
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

if __name__=='__main__':
    args = config
    # syntheticP(args)
    main(args)
    print('END')
