import warnings

import torch.nn as nn
import torch

from models.RNN import BidirectionalGRU
from models.CNN import CNN
from models.tcn import TemporalConvNet
from models.Resnet import resnet18
# from RNN import BidirectionalGRU
# from CNN import CNN
# from tcn import TemporalConvNet

import pdb
# import kornia

class CRNN(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0,
                 cnn_integration=False, learned_post=False, **kwargs):
        super(CRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.rnn_type = rnn_type
        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        # self.cnn = resnet18(pretrained=False)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if self.rnn_type == 'BGRU':
            nb_in = self.cnn.nb_filters[-1]
            # nb_in = 128 
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        elif self.rnn_type =='TCN':
            # Number of [n_RNN_cell] needs to be defined
            self.rnn = TemporalConvNet(self.cnn.nb_filters[-1], [n_RNN_cell] * 2, 3, dropout=0.25)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)
        if rnn_type == 'BGRU':
            self.dense = nn.Linear(n_RNN_cell*2, nclass)
        else:
            self.dense = nn.Linear(n_RNN_cell, nclass)
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            if self.rnn_type == 'BGRU':
                self.dense_softmax = nn.Linear(n_RNN_cell*2, nclass)
            else:
                self.dense_softmax = nn.Linear(n_RNN_cell, nclass)
            self.softmax = nn.Softmax(dim=-1) # attention over class axis, dim=-2 is attention over time axis


    def load_cnn(self, state_dict):
        self.cnn.load_state_dict(state_dict)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict["cnn"])
        self.rnn.load_state_dict(state_dict["rnn"])
        self.dense.load_state_dict(state_dict["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'rnn': self.rnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)
        
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)        
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
        
        # rnn features
        if self.rnn_type == 'BGRU':
            x = self.rnn(x)
        elif self.rnn_type == 'TCN':
            x = self.rnn(x.transpose(1, 2)).transpose(1, 2)

        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
            # weak = (strong * strong).sum(1) / strong.sum(1)   # [bs, nclass]
        else:
            weak = strong.mean(1)
        # pdb.set_trace()
        return strong, weak


if __name__ == '__main__':
    x = torch.rand(24,1,628,128)
    nnet = CRNN(1, 10, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
            attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
            pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    # nnet = CRNN(32, 10, kernel_size=5 * [3], padding=5 * [1], stride=5 * [1], nb_filters=[64,  128,  128, 128, 128],
    #         attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
    #         pooling=[[2, 4], [2, 2], [1, 2], [1, 2], [1, 2]])    
    strong, weak = nnet(x)

