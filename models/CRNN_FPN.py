import warnings

import torch.nn as nn
import torch

from models.RNN import BidirectionalGRU
from models.CNN_FPN import CNN

import pdb

class CRNN_fpn(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0,
                 cnn_integration=False, **kwargs):
        super(CRNN_fpn, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.rnn_type = rnn_type
        n_in_cnn = n_in_channel
        if cnn_integration:
            n_in_cnn = 1
        self.cnn = CNN(n_in_cnn, activation, dropout, **kwargs)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if self.rnn_type == 'BGRU':
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
            self.rnn_2 = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
            self.rnn_4 = BidirectionalGRU(nb_in,
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)

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

        self.upsample_2 = nn.Upsample((157,1), mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample((78,1), mode='bilinear', align_corners=True)
        self.conv1x1_2 = nn.Conv2d(512,256,1) # for x_2
        self.conv1x1_4 = nn.Conv2d(512,256,1) # for x

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # conv features
        x, x_2, x_4 = self.cnn(x)
        bs, chan, frames, freq = x.size()
        bs_2, chan_2, frames_2, freq_2 = x_2.size()
        bs_4, chan_4, frames_4, freq_4 = x_4.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)
        
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)        
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
            x_2 = x_2.squeeze(-1)
            x_2 = x_2.permute(0, 2, 1)  # [bs, frames, chan]        
            x_4 = x_4.squeeze(-1)
            x_4 = x_4.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        if self.rnn_type == 'BGRU':
            x = self.rnn(x)
            x = x.permute(0, 2, 1)
            x_2 = self.rnn_2(x_2)
            x_2 = x_2.permute(0, 2, 1)  # [bs, chan, frames] 
            x_4 = self.rnn_4(x_4)
            x_4 = x_4.permute(0, 2, 1)  # [bs, chan, frames]

        x = self.dropout(x).unsqueeze(-1)
        x_2 = self.dropout(x_2).unsqueeze(-1)
        x_4 = self.dropout(x_4).unsqueeze(-1)
        x_2 = torch.cat((x_2, self.upsample_4(x_4)), 1)
        x_2 = self.conv1x1_2(x_2)
        x = torch.cat((x, self.upsample_2(x_2)), 1)
        x = self.conv1x1_4(x).squeeze(-1)
        x = x.permute(0, 2, 1)

        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak = strong.mean(1)

        # uncomment this part only at inference stage without ensemble
        # for training stage or inference stage with ensemble, you should remain comment
        '''
        check = (weak > 0.5).type(torch.FloatTensor).cuda()
        check = check.unsqueeze(1).repeat(1,157,1)
        strong = strong * check
        '''
        
        return strong, weak


if __name__ == '__main__':
    x = torch.rand(24,1,628,128)
    nnet = CRNN(1, 10, kernel_size=7 * [3], padding=7 * [1], stride=7 * [1], nb_filters=[16,  32,  64,  128,  128, 128, 128],
            attention=True, activation="GLU", dropout=0.5, n_RNN_cell=128, n_layers_RNN=2,
            pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
    strong, weak = nnet(x)

