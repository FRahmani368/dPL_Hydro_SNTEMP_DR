"""All functions related to loading the data"""
from collections import deque
from itertools import islice
import numpy as np
import torch
from core import hydroDL
from core.hydroDL import master, utils
from core.hydroDL.data.camels import initcamels, transNorm
# from utils import load_dataloader


def load_df(args):
    """
    A function that loads the data into a
    :return:
    """
    df, x, y, c = master.loadData(args)
    nx = x.shape[-1] + c.shape[-1]
    x_total = np.zeros((x.shape[0], x.shape[1], nx))
    ct = np.repeat(c, repeats=x.shape[1], axis=0)
    for k in range(x.shape[0]):
        x_total[k, :, :] = np.concatenate(
            (x[k, :, :], np.tile(c[k], (x.shape[1], 1))), axis=1
        )
    # streamflow values should not be negative
    # vars = args['optData']['varT'] + args['optData']['varC']
    # x_total[x_total[:, :, vars.index("00060_Mean")] < 0] = 0
    return x_total, y, c

def scaling(args, x, y, c):
    """
    creates our datasets
    :param set_name:
    :param args:
    :param time1:
    :param x_total_raw:
    :param y_total_raw:
    :return:  x, y, ngrid, nIterEp, nt
    """
    initcamels(args, x, y)
    #Normalization
    x_total_scaled = transNorm(x, args['optData']['varT']+args['optData']['varC'], toNorm=True)
    y_scaled = transNorm(y, args['optData']['target'][0], toNorm=True)
    c_scaled = transNorm(c, args['optData']['varC'], toNorm=True)
    return x_total_scaled, y_scaled, c_scaled

def train_val_test_split(set_name, args, time1, x_total, y_total):
    c, ind1, ind2 = np.intersect1d(
        time1, hydroDL.utils.time.tRange2Array(args['optData'][set_name]), return_indices=True
    )
    x = x_total[:, ind1, :]
    y = y_total[:, ind1, :]
    ngrid, nt, nx = x.shape
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - args['hyperparameters']['batch_size'] * \
                                      args['hyperparameters']['rho'] / ngrid / nt)))

    return x, y, ngrid, nIterEp, nt, args['hyperparameters']['rho'], args['hyperparameters']['batch_size']

def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
        if nt <= rho:
            iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out


def randomIndex(ngrid, nt, dimSubset):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])     # np.random.randint(0, nt - rho, [batchSize])
    return iGrid, iT




def create_tensor(rho, mini_batch, x, y):
    """
    Creates a data tensor of the input variables and incorporates a sliding window of rho
    :param mini_batch: min batch length
    :param rho: the seq len
    :param x: the x data
    :param y: the y data
    :return:
    """
    j = 0
    k = rho
    _sample_data_x = []
    _sample_data_y = []
    for i in range(x.shape[0]):
        _list_x = []
        _list_y = []
        while k < x[0].shape[0]:
            """In the format: [total basins, basin, days, attributes]"""
            _list_x.append(x[1, j:k, :])
            _list_y.append(y[1, j:k, 0])
            j += mini_batch
            k += mini_batch
        _sample_data_x.append(_list_x)
        _sample_data_y.append(_list_y)
        j = 0
        k = rho
    sample_data_x = torch.tensor(_sample_data_x).float()
    sample_data_y = torch.tensor(_sample_data_y).float()
    return sample_data_x, sample_data_y


def create_tensor_list(x, y):
    """
    we want to return the :
    x_list = [[[basin_1, num_samples_x, num_attr_x], [basin_1, num_samples_y, num_attr_y]]
        .
        .
        .
        [[basin_20, num_samples_x, num_attr_x], [basin_20, num_samples_y, num_attr_y]]]
    :param data:
    :return:
    """
    tensor_list = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            _var = (torch.tensor(x[i][j][:, :]), y[i, j])
            tensor_list.append(_var)
    return tensor_list
def run_farshid(args):
    x_total, y = load_df(args)
    time1 = utils.time.tRange2Array(args.optData["tRange"])
    c, ind1, ind2 = np.intersect1d(
        time1, utils.time.tRange2Array(args.optData["t_train"]), return_indices=True
    )
    x_train, y_train = x_total[:, ind1, :], y[:, ind1, :]

    c, ind1, ind2 = np.intersect1d(
        time1, utils.time.tRange2Array(args.optData["t_val"]), return_indices=True
    )
    x_val, y_val = x_total[:, ind1, :], y[:, ind1, :]


    c, ind1, ind2 = np.intersect1d(
        time1, utils.time.tRange2Array(args.optData["t_test"]), return_indices=True
    )
    x_test, y_test = x_total[:, ind1, :], y[:, ind1, :]
    return (
        x_total, y,
        x_train, y_train,
        x_val, y_val,
        x_test, y_test
    )


# TODO add batch size into calculations here