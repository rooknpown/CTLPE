import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from .NDEEncoder import NCDE

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class PecdePositionalEmbedding(nn.Module):
    def __init__(self, c_in, irrsin, d_model, win_size, max_len=5000):
        super(PecdePositionalEmbedding, self).__init__()
        # print("c_in")
        # print(c_in)
        self.irrsin = irrsin
        if irrsin == 0:
            self.ncde = NCDE(4, d_model)

        else:
            self.ncde = NCDE(512, d_model)
        self.win_size = win_size
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    
        # print("pe")
        # print(pe)
        self.register_buffer('pe', pe)

        self.mapping_size =  1.
        self.rand_key = 2023
        np.random.seed(self.rand_key)
        self.B = np.random.normal(0, self.mapping_size, size = (int(d_model/2), win_size))

    def pos_mapping(self, x):
        x_proj = (2.*np.pi*x.cpu()) @ self.B.T
        return torch.Tensor(np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)).permute(1, 0, 2)
    

    def forward(self, time,adjoint=True,**kwargs):       

        # for comparison
        # return self.pe[:, :time.size(1)]
        # print("time")
        # print(time)
        
        #testing with index
        # print(time[0][0][0])
        # print(time)
        # print(time.shape)
        # t = torch.linspace(0, 1, self.win_size)
        # t = t[..., 0, :].to(time.device)
        # print(t.shape)
        # time = t
        # v0 = self.initial(time[..., 0, :].to(time.device))

        # t = time[0,...,0]
        # t1 = torch.ones(t.shape[0]).to(time.device)
        # t = torch.sub(t, t1, alpha = t[0].item())
        # t = torch.div(t, 1000000)
        
        # print("time window")
        # print(t)
        # print("pecde")
        # print(time.shape)
        # print(t.shape)
        ###### 1. pecde ###########
        if self.irrsin == 0:
            vt = self.ncde(time)
        ###########################
        ###### 2. sin pe to cde #######
        else:
            x2 = self.pe.squeeze().expand(time.size(0), self.pe.size(1), self.pe.size(2))
            # print(torch.gather(x2, 1, x.expand(x.size(0), x.size(1), self.pe.size(2)).type(torch.int64) ).shape)
            x3 = torch.gather(x2, 1, time.expand(time.size(0), time.size(1), self.pe.size(2)).type(torch.int64) )

            vt = self.ncde(x3)
        ###########################
        ###### 3. inr pe to cde #######
        # self.pos_mapping(time.squeeze())

        ###########################
        # print("sin pe")
        # print(self.pe[:, :time.size(1)])

        # print(vt)

        # print("pecde shape")
        # print(vt.shape)

        # return vt.permute(1, 0, 2)
        return vt

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()                                                                              

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)
    
class PecdeDataEmbedding(nn.Module):
    def __init__(self, c_in, irrsin, d_model, dropout=0.0, win_size = 100):
        super(PecdeDataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PecdePositionalEmbedding(c_in=c_in, irrsin = irrsin, d_model=d_model, win_size = win_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, time):
        # print("embedding shapes")
        # print(self.value_embedding(x).shape)
        # print(self.position_embedding(time).shape)
        # lambda_coeff = 0.00001
        # print("value, pos")
        # print(torch.mean(torch.abs(self.value_embedding(x))))
        # print(torch.mean(torch.abs(lambda_coeff *self.position_embedding(time))))
        # x = self.value_embedding(x) + lambda_coeff * self.position_embedding(time)
        x = self.value_embedding(x) + self.position_embedding(time)
        return self.dropout(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)