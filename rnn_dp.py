import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
from . import cnn
import csv
import numpy as np


class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskX_i = createMask(x, self.dr)
        self.maskX_f = createMask(x, self.dr)
        self.maskX_c = createMask(x, self.dr)
        self.maskX_o = createMask(x, self.dr)

        self.maskH_i = createMask(h, self.dr)
        self.maskH_f = createMask(h, self.dr)
        self.maskH_c = createMask(h, self.dr)
        self.maskH_o = createMask(h, self.dr)

        self.maskC = createMask(c, self.dr)

        self.maskW_xi = createMask(self.w_xi, self.dr)
        self.maskW_xf = createMask(self.w_xf, self.dr)
        self.maskW_xc = createMask(self.w_xc, self.dr)
        self.maskW_xo = createMask(self.w_xo, self.dr)
        self.maskW_hi = createMask(self.w_hi, self.dr)
        self.maskW_hf = createMask(self.w_hf, self.dr)
        self.maskW_hc = createMask(self.w_hc, self.dr)
        self.maskW_ho = createMask(self.w_ho, self.dr)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and 'drX' in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and 'drW' in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.mode = mode
        if mode == 'train':
            self.train(mode=True)
        elif mode == 'test':
            self.train(mode=False)
        elif mode == 'drMC':
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.maskX = createMask(x, self.dr)
        self.maskH = createMask(h, self.dr)
        self.maskC = createMask(c, self.dr)
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and 'drX' in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + \
            F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


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
        handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
            input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
            self.hiddenSize, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]

class CNN1dkernel(torch.nn.Module):
    def __init__(self,
                 *,
                 ninchannel=1,
                 nkernel=3,
                 kernelSize=3,
                 stride=1,
                 padding=0):
        super(CNN1dkernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out


class CNN1dLSTMmodel(torch.nn.Module):
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None):
        # two convolutional layer
        super(CNN1dLSTMmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x0 = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x0))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLSTMInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLSTMInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCmodel(torch.nn.Module):
    # add the CNN extracted features into original LSTM input, then pass through linear layer
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1 # need to modify the hardcode: 4 for smap and 1 for FDC
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid = z.shape[0]
        rho, BS, Nvar = x.shape
        if len(z.shape) == 2: # for FDC, else 3 dimension for smap
            z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid, nobs = z.shape
        rho, BS, Nvar = x.shape
        z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CudnnInvLstmModel(torch.nn.Module):
    # using cudnnLstm to extract features from SMAP observations
    def __init__(self, *, nx, ny, hiddenSize, ninv, nfea, hiddeninv, dr=0.5, drinv=0.5):
        # two LSTM
        super(CudnnInvLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv)
        self.lstm = CudnnLstmModel(
            nx=nfea+nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        dim = x.shape;
        nt = dim[0]
        invpara = Gen[-1, :, :].repeat(nt, 1, 1)
        x1 = torch.cat((x, invpara), dim=2)
        out = self.lstm(x1)
        # out = rho/time * batchsize * Ntargetvar
        return out


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW')
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            xt = x[t, :, :]
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx + 1, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnCond, self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = False
        self.hiddenSize = hiddenSize
        self.opt = opt

        self.cnn = cnn.Cnn1d(nx=nx, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        if opt == 3:
            self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        else:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if opt == 2:
            self.linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny)
        else:
            self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, xc):
        # x- [nt,ngrid,nx]
        x1 = xc
        x1 = self.cnn(x1)
        x2 = x
        if self.opt == 1:
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2, hx=x1[None, :, :])
            x2 = self.linearOut(x2)
        elif self.opt == 2:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(torch.cat([x2, x1], 2))
        elif self.opt == 3:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = torch.cat([x2, x1], 2)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnForcast, self).__init__()

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = True
        self.hiddenSize = hiddenSize
        self.opt = opt
        self.cnnSize = cnnSize

        if opt == 1:
            self.cnn = cnn.Cnn1d(
                nx=nx + 1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)
        if opt == 2:
            self.cnn = cnn.Cnn1d(
                nx=1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, y):
        # x- [nt,ngrid,nx]
        nt, ngrid, nx = x.shape
        ct = self.ct
        pt = nt - ct

        if self.opt == 1:
            x1 = torch.cat((y, x), dim=2)
        elif self.opt == 2:
            x1 = y

        x1out = torch.zeros([pt, ngrid, self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k, :, :] = self.cnn(x1[k:k + ct, :, :])

        x2 = x[ct:nt, :, :]
        x2 = torch.cat([x2, x1out], 2)
        x2 = F.relu(self.linearIn(x2))
        x2, (hn, cn) = self.lstm(x2)
        x2 = self.linearOut(x2)

        return x2

class CudnnLstmModel_R2P(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, filename):
        super(CudnnLstmModel_R2P, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        #self.linearR2P = torch.nn.Linear(nx[1], nx[2])
        self.linearR2Pa = torch.nn.Linear(nx[1], hiddenSize)
        self.linearR2Pa1 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.linearR2Pa2 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.linearR2Pb = torch.nn.Linear(hiddenSize, nx[2]+2) #add two for final shift layer
        #self.linearR2Pb = torch.nn.Linear(hiddenSize, nx[2]) #add two for final shift layer 
        self.linearDrop = nn.Dropout(dr)
        #self.bn1 = torch.nn.BatchNorm1d(num_features=hiddenSize)

        #self.lstm = CudnnLstmModel(
        #    nx=nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.lstm = torch.load(filename)
        
        # self.lstm.eval()

        for param in self.lstm.parameters():
            param.requires_grad = False
            
        self.linearRV2S = torch.nn.Linear(nx[1]+ny,ny)
        self.linearR2S = torch.nn.Linear(nx[1],ny)
        self.linearV2S = torch.nn.Linear(ny,ny)  # mapping to SMAP
        #self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False, outModel=None):
        if type(x) is tuple or type(x) is list:
            Forcing, Raw = x

        ##Param = F.relu(self.linearR2P(Raw))
        #Param_a = self.linearDrop(torch.relu(self.linearR2Pa(Raw))) # dropout setting
        #Param_a = torch.relu(self.linearR2Pa(Raw))
        Param_a = self.linearDrop(torch.relu(self.linearR2Pa(Raw)))
        ##Param_bn = self.bn1(Param_a)

        #Param_a1 = torch.relu(self.linearR2Pa1(Param_a))
        Param_a1 = self.linearDrop(torch.relu(self.linearR2Pa1(Param_a)))
        #Param_a2 = self.linearDrop(torch.relu(self.linearR2Pa2(Param_a1)))
        
        # Param = torch.atan(self.linearR2Pb(Param_a1))
        ##Param = torch.relu(self.linearR2Pb(Param_a1))
        
        Param_two = torch.atan(self.linearR2Pb(Param_a1))
        dim = Param_two.shape
        Param = Param_two[:,:,0:dim[2]-2]
        a = Param_two[:,:,dim[2]-2:dim[2]-1]
        b = Param_two[:,:,dim[2]-1:dim[2]]     ##Param = torch.rrelu(self.linearR2Pb(Param_bn))
        
        if outModel is None:
            x1 = torch.cat((Forcing,Param),dim=len(Param.shape)-1) # by default cat along dim=0
            #self.lstm.eval()
            outLSTM_surrogate = self.lstm(x1, doDropMC=doDropMC, dropoutFalse=True)
            #outLSTM_SMAP = torch.atan(self.linearV2S(outLSTM_surrogate))  # mapping to SMAP
            #x2 = torch.cat((outLSTM_surrogate, Raw), dim=len(Raw.shape)-1)
            outLSTM_SMAP = a * outLSTM_surrogate+b  # mapping to SMAP
            #outLSTM_SMAP = outLSTM_surrogate  # mapping to SMAP
            #return outLSTM_surrogate, Param
            return outLSTM_SMAP, Param
        else:
            # outQ_hymod, outE_hymod = self.lstm.advance(Forcing[:,:,0], Forcing[:,:,1], \
            #     Param[:,:,0], Param[:,:,1], Param[:,:,2], Param[:,:,3], Param[:,:,4], Param[:,:,5],\
            #         Param[:,:,6], Param[:,:,7])
            #outQ_hymod = self.hymode.advance(Forcing[:,:,0], Forcing[:,:,1], Param[:,:,:])
            #outQ_hymod = outQ_hymod.unsqueeze(2)
            # return outQ_hymod, Param

            # ====== reasonable hyod parameters ======
            out = '/mnt/sdc/SUR_VIC/multiOutput_CONUSv16f1_VIC/hymod/'
            with open(out+'parameters_hymod') as f:
                reader = csv.reader(f, delimiter=',')
                parameters = list(reader)
                parameters = np.array(parameters).astype(float)
            
            parameters = torch.from_numpy(parameters)
            return parameters
        
        #self.lstm.train()
        #out = self.linearOut(outLSTM)
        #return outLSTM

class CpuLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = -1

    def forward(self, x, doDropMC=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # return out
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        out = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out

class hymod(torch.nn.Module):
    """Simple 5 parameter model"""
    def __init__(self, *, a, b, cmax, rq, rs, s, slow, fast):
        """Initiate a hymod instance"""
        super(hymod, self).__init__()
        self.a = a  # percentage of quickflow
        self.b = b  # shape of Pareto ditribution
        self.cmax = cmax # maximum storage capacity
        self.rq = rq # quickflow time constant
        self.rs = rs # slowflow time constant
        self.smax = self.cmax / (1. + self.b)
        self.s = s # soil moisture
        self.slow = slow # slowflow reservoir
        self.fast = fast # fastflow reservoirs
        self.error = 0

    def __repr__(self):
        bstr = 'a:{!r}'.format(self.a)
        bstr += ' b:{!r}'.format(self.b)
        bstr += ' cmax:{!r}'.format(self.cmax)
        bstr += ' rq:{!r}'.format(self.rq)
        bstr += ' rs:{!r}'.format(self.rs)
        bstr += ' smax:{!r}\n'.format(self.smax)
        bstr += 's:{!r}'.format(self.s)
        bstr += ' slow:{!r}'.format(self.slow)
        bstr += ' fast:{!r}'.format(self.fast)
        bstr += ' error:{!r}'.format(self.error)
        return bstr

    def advance(self, P, PET):
        if self.s > self.smax:
            self.error += self.s - 0.999 * self.smax
            self.s = 0.999 * self.smax

        cprev = self.cmax * (1 - np.power((1-((self.b+1)*self.s/self.cmax)), (1/(self.b+1))))
        ER1 = np.maximum(P + cprev - self.cmax, 0.0) # effective rainfal part 1
        P -= ER1
        dummy = np.minimum(((cprev + P)/self.cmax), 1)
        s1 = (self.cmax/(self.b+1)) * (1 - np.power((1-dummy), (self.b+1))) # new state
        ER2 = np.maximum(P-(s1-self.s), 0) # effective rainfall part 2
        evap = np.minimum(s1, s1/self.smax * PET) # actual ET is linearly related to the soil moisture state
        self.s = s1-evap # update state
        UQ = ER1 + self.a * ER2 # quickflow contribution
        US = (1 - self.a) * ER2 # slowflow contribution
        for i in range(3):
            self.fast[i] = (1-self.rq) * self.fast[i] + (1-self.rq) * UQ # forecast step
            UQ = (self.rq/(1-self.rq)) * self.fast[i]
        self.slow = (1-self.rs) * self.slow + (1-self.rs) * US
        US = (self.rs/(1-self.rs)) * self.slow
        Q = UQ + US
        return Q, evap

    def getfast(self):
        return self.fast

    def getslow(self):
        return self.slow

    def getsoilmoisture(self):
        return self.s

    def getparams(self):
        return self.a, self.b, self.cmax, self.rq. self.rs

    def geterror(self):
        return self.error

    def setfast(self, fast):
        self.fast = fast

    def setslow(self, slow):
        self.slow = slow

    def setsoilmoisture(self, s):
        self.s = s

def UH_conv(x,UH,viewmode=1):
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

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w  = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w,[2]), groups=groups, padding=padd, stride=1, bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1],m[2]])
    aa = F.relu(a[0:lenF,:,0]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,0]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0) # scale to 1 for each UH

    return w

class HBV(torch.nn.Module):
    """HBV Model Pytorch version"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBV, self).__init__()

    def forward(self, x, parameters, bufftime=0, outstate=False, routOpt=False):
        # HBV(P, ETpot, T, parameters)
        #
        # Runs the HBV-light hydrological model (Seibert, 2005). NaN values have to be
        # removed from the inputs.
        #
        # Input:
        #     P = array with daily values of precipitation (mm/d)
        #     ETpot = array with daily values of potential evapotranspiration (mm/d)
        #     T = array with daily values of air temperature (deg C)
        #     parameters = array with parameter values having the following structure and scales:
        #         BETA[1,6]; CET; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         MAXBAS; PERC[0,10]; UZL[0,100]; PCORR; TT[-2.5,2.5]; CFMAX[0.5,10]; SFCF; CFR[0,0.1]; CWH[0,0.2]
        #
        #
        # Output, all in mm:
        #     Qsim = daily values of simulated streamflow
        #     SM = soil storage
        #     SUZ = upper zone storage
        #     SLZ = lower zone storage
        #     SNOWPACK = snow depth
        #     ETact = actual evaporation

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBV()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, bufftime=0, outstate=True, routOpt=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            ETact = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        T = x[bufftime:, :, 1]
        ETpot = x[bufftime:, :, 2]


        ## scale the parameters
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0,2.9], [0,6.5]]

        parBETA = parascaLst[0][0] + parameters[:,0]*(parascaLst[0][1]-parascaLst[0][0])
        # parCET = parameters[:,1]
        parFC = parascaLst[1][0] + parameters[:,1]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5]*(parascaLst[5][1]-parascaLst[5][0])
        # parMAXBAS = parameters[:,7]
        parPERC = parascaLst[6][0] + parameters[:,6]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7]*(parascaLst[7][1]-parascaLst[7][0])
        # parPCORR = parameters[:,10]
        parTT = parascaLst[8][0] + parameters[:,8]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9]*(parascaLst[9][1]-parascaLst[9][0])
        # parSFCF = parameters[:,13]
        parCFR = parascaLst[10][0] + parameters[:,10]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11]*(parascaLst[11][1]-parascaLst[11][0])

        Nstep, Ngrid = P.size()
        # Apply correction factor to precipitation
        # P = parPCORR.repeat(Nstep, 1) * P

        # Initialize time series of model variables
        Qsim = (torch.zeros(P.size(), dtype=torch.float32) + 0.001).cuda()
        # # Debug for the state variables
        # SMlog = np.zeros(P.size())
        logSM = np.zeros(P.size())
        logPS = np.zeros(P.size())
        logswet = np.zeros(P.size())
        logRE = np.zeros(P.size())

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = P[t, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (T[t, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (T[t, :] < parTT).type(torch.float32))
            # SNOW = SNOW * parSFCF

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (T[t, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - T[t, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            ## log for displaying
            logSM[t,:] = SM.detach().cpu().numpy()
            logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            logRE[t, :] = recharge.detach().cpu().numpy()

            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpot[t, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsim[t, :] = Q0 + Q1 + Q2

            # # for debug state variables
            # SMlog[t,:] = SM.detach().cpu().numpy()

        if routOpt is True: # routing
            tempa = parascaLst[-2][0] + parameters[:,-2]*(parascaLst[-2][1]-parascaLst[-2][0])
            tempb = parascaLst[-1][0] + parameters[:,-1]*(parascaLst[-1][1]-parascaLst[-1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb,lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
            UH = UH.permute([1, 2, 0])
            Qs = UH_conv(rf, UH).permute([2, 0, 1])

        else:

            Qs = torch.unsqueeze(Qsim, -1) # add a dimension
        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs


class SimpAnn(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(SimpAnn, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x):
        ht = F.relu(self.i2h(x))
        ht2 = F.relu(self.h2h(ht))
        out = F.relu(self.h2o(ht2))
        return out



class HBVNN(torch.nn.Module):
    """HBV Model with Neural Network Component"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVNN, self).__init__()
        self.ANN = SimpAnn(nx=4, ny=1, hiddenSize=32)

    def forward(self, x, parameters, bufftime=0, outstate=False, routOpt=False):
        # HBV(P, ETpot, T, parameters)
        #
        # Runs the HBV-light hydrological model (Seibert, 2005). NaN values have to be
        # removed from the inputs.
        #
        # Input:
        #     P = array with daily values of precipitation (mm/d)
        #     ETpot = array with daily values of potential evapotranspiration (mm/d)
        #     T = array with daily values of air temperature (deg C)
        #     parameters = array with parameter values having the following structure and scales:
        #         BETA[1,6]; CET; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         MAXBAS; PERC[0,10]; UZL[0,100]; PCORR; TT[-2.5,2.5]; CFMAX[0.5,10]; SFCF; CFR[0,0.1]; CWH[0,0.2]
        #
        #
        # Output, all in mm:
        #     Qsim = daily values of simulated streamflow
        #     SM = soil storage
        #     SUZ = upper zone storage
        #     SLZ = lower zone storage
        #     SNOWPACK = snow depth
        #     ETact = actual evaporation

        PRECS = 1e-5

        # Initialization
        if bufftime > 0:
            with torch.no_grad():
                xinit = x[0:bufftime, :, :]
                initmodel = HBV()
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit, parameters, bufftime=0, outstate=True, routOpt=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            MELTWATER = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            SM = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            SUZ = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            SLZ = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()
            ETact = (torch.zeros(Ngrid, dtype=torch.float32) + 0.001).cuda()

        P = x[bufftime:, :, 0]
        T = x[bufftime:, :, 1]
        ETpot = x[bufftime:, :, 2]


        ## scale the parameters
        parascaLst = [[1,6], [50,1000], [0.05,0.9], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2], [0,2.9], [0,6.5]]

        parBETA = parascaLst[0][0] + parameters[:,0]*(parascaLst[0][1]-parascaLst[0][0])
        # parCET = parameters[:,1]
        parFC = parascaLst[1][0] + parameters[:,1]*(parascaLst[1][1]-parascaLst[1][0])
        parK0 = parascaLst[2][0] + parameters[:,2]*(parascaLst[2][1]-parascaLst[2][0])
        parK1 = parascaLst[3][0] + parameters[:,3]*(parascaLst[3][1]-parascaLst[3][0])
        parK2 = parascaLst[4][0] + parameters[:,4]*(parascaLst[4][1]-parascaLst[4][0])
        parLP = parascaLst[5][0] + parameters[:,5]*(parascaLst[5][1]-parascaLst[5][0])
        # parMAXBAS = parameters[:,7]
        parPERC = parascaLst[6][0] + parameters[:,6]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL = parascaLst[7][0] + parameters[:,7]*(parascaLst[7][1]-parascaLst[7][0])
        # parPCORR = parameters[:,10]
        parTT = parascaLst[8][0] + parameters[:,8]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX = parascaLst[9][0] + parameters[:,9]*(parascaLst[9][1]-parascaLst[9][0])
        # parSFCF = parameters[:,13]
        parCFR = parascaLst[10][0] + parameters[:,10]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH = parascaLst[11][0] + parameters[:,11]*(parascaLst[11][1]-parascaLst[11][0])

        Nstep, Ngrid = P.size()
        # Apply correction factor to precipitation
        # P = parPCORR.repeat(Nstep, 1) * P

        # Initialize time series of model variables
        Qsim = (torch.zeros(P.size(), dtype=torch.float32) + 0.001).cuda()
        # # Debug for the state variables
        logSM = np.zeros(P.size())
        logPS = np.zeros(P.size())
        logswet = np.zeros(P.size())
        logRE = np.zeros(P.size())

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = P[t, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (T[t, :] >= parTT).type(torch.float32))
            SNOW = torch.mul(PRECIP, (T[t, :] < parTT).type(torch.float32))
            # SNOW = SNOW * parSFCF

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = parCFMAX * (T[t, :] - parTT)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - T[t, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (parCWH * SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation
            # replace with ANN
            # soil_wetness = (SM / parFC) ** parBETA
            # soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            # recharge = (RAIN + tosoil) * soil_wetness
            swet = SM/parFC
            xIn = torch.stack((parBETA, SM, swet, RAIN + tosoil), dim=1)
            recharge = self.ANN(xIn).squeeze()
            ## log for displaying
            logSM[t,:] = SM.detach().cpu().numpy()
            logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            logswet[t,:] = swet.detach().cpu().numpy()
            logRE[t, :] = recharge.detach().cpu().numpy()

            # recharge = torch.clamp(recharge, min=0.0)
            recharge = torch.min(recharge, RAIN + tosoil)
            SM = SM + RAIN + tosoil - recharge
            excess = SM - parFC
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = ETpot[t, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=PRECS) # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, parPERC)
            SUZ = SUZ - PERC
            Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
            SUZ = SUZ - Q0
            Q1 = parK1 * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = parK2 * SLZ
            SLZ = SLZ - Q2
            Qsim[t, :] = Q0 + Q1 + Q2

            # # for debug state variables
            # SMlog[t,:] = SM.detach().cpu().numpy()

        if routOpt is True: # routing
            tempa = parascaLst[-2][0] + parameters[:,-2]*(parascaLst[-2][1]-parascaLst[-2][0])
            tempb = parascaLst[-1][0] + parameters[:,-1]*(parascaLst[-1][1]-parascaLst[-1][0])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = UH_gamma(routa, routb,lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])
            UH = UH.permute([1, 2, 0])
            Qs = UH_conv(rf, UH).permute([2, 0, 1])

        else:

            Qs = torch.unsqueeze(Qsim, -1) # add a dimension
        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs


class CudnnInv_HBVModel(torch.nn.Module):
    def __init__(self, *, ninv, nfea, hiddeninv, drinv=0.5, inittime=0, routOpt=False, annOpt=False):
        # LSTM Inv + HBV Forward
        super(CudnnInv_HBVModel, self).__init__()
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv)
        if annOpt is True:
            self.HBV = HBVNN()
        else:
            self.HBV = HBV()
        self.gpu = 1
        self.inittime=inittime
        self.routOpt=routOpt


    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        Params0 = Gen[-1, :, :]
        # print(Params0)
        Params = torch.clamp(Params0, min=0.0, max=1.0) # Make all parameters into range [0,1]
        out = self.HBV(x, Params, bufftime=self.inittime, routOpt=self.routOpt)
        return out


# class HBVNNModel(torch.nn.Module):
#     def __init__(self, *, ninv, nfea, hiddeninv, drinv=0.5, inittime=0, routOpt=False):
#         # LSTM Inv + HBV Forward
#         super(HBVNNModel, self).__init__()
#         self.ninv = ninv
#         self.nfea = nfea
#         self.hiddeninv = hiddeninv
#         self.lstminv = CudnnLstmModel(
#             nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv)
#         self.HBV = HBVNN(nx=3, ny=1, hiddensize=64)
#         self.gpu = 1
#         self.inittime=inittime
#         self.routOpt=routOpt
#
#
#     def forward(self, x, z, doDropMC=False):
#         Gen = self.lstminv(z)
#         Params0 = Gen[-1, :, :]
#         # print(Params0)
#         Params = torch.clamp(Params0, min=0.0, max=1.0) # Make all parameters into range [0,1]
#         out = self.HBV(x, Params, bufftime=self.inittime, routOpt=self.routOpt)
#         return out


'''
def main():
    nsteps = 1000
    model = hymod(0.83, 0.38, 350, 0.46, 0.03, 100, 100, [50, 50, 50])
    P = 20 * np.random.random(nsteps)
    P[P<15] = 0
    PET = 5 * np.random.random(nsteps)
    for t in range(nsteps):
        q, evap = model.advance(P[t], PET[t])
        print '{:.5f} {:.5f} {:.5f} {:.5f}'.format(P[t], PET[t], q, evap)

if __name__ == "__main__":
    main()
'''
