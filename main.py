from core.read_configurations import config
from core import randomseed_config
from core.data_prep import load_df, scaling, train_val_test_split
from core.small_codes import create_output_dirs
from MODELS.PGML_STemp import MLP, STREAM_TEMP_EQ
from MODELS import crit
from core import hydroDL
import torch
import pandas as pd

def main(args):
    # setting random seeds
    randomseed_config

    # getting the data
    x_total_raw, y_total_raw, C_total_raw = load_df(args)
    x_total_scaled, y_total_scaled = scaling(args, x_total_raw, y_total_raw)
    time1 = hydroDL.utils.time.tRange2Array(args['optData']["tRange"])
    x, y, ngrid_train, nIterEp, nt = train_val_test_split("t_train", args, time1, x_total_scaled, y_total_scaled)

    # Creating output directories
    args = create_output_dirs(args)
    # ANN model to simulate parameters
    model = MLP(args)
    # loss function
    lossFun = crit.RmseLoss()
    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    if torch.cuda.is_available():
        model = model.cuda()
        lossFun = lossFun.cuda()
    T_w = STREAM_TEMP_EQ(args)

    # training
    batchSize = args['hyperparameters']['batchSize']
    rho = args['hyperparameters']['rho']
    ngrid, nt = x_total_raw.shape[0], x_total_raw.shape[1]
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt)))
    for epoch in range(args['hyperparameters']['EPOCHS']):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            res_time_srflow, res_time_srflow, res_time_srflow = model(C_total_raw)
            Yp = T_w.forward()
            loss = lossFun(Yp, y_total_raw)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
            # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.6f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)







    print('end')




if __name__=='__main__':
    args = config
    main(args)
