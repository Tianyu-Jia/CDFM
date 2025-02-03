import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

import random

def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # load parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.G = configs.G
        self.enc_in = configs.enc_in
        self.alpha = configs.alpha
        self.init_type = configs.init_type
        self.decomp_module = series_decomp(configs.kernel_size)

        self.Linear_Seasonal_s = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend_s = nn.Linear(self.seq_len, self.pred_len)
        # Initialize parameters
        self.Linear_Seasonal_s.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Seasonal_s.bias = nn.Parameter(torch.zeros([self.pred_len]))
        self.Linear_Trend_s.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Trend_s.bias = nn.Parameter(torch.zeros([self.pred_len]))
        
        self.Linear_Seasonal_ns = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend_ns = nn.Linear(self.seq_len, self.pred_len)
        # Initialize parameters
        self.Linear_Seasonal_ns.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Seasonal_ns.bias = nn.Parameter(torch.zeros([self.pred_len]))
        self.Linear_Trend_ns.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_Trend_ns.bias = nn.Parameter(torch.zeros([self.pred_len]))
        
        self.Linear_Sigma_tpred = nn.Linear(self.seq_len+1, 1)
        self.Linear_Sigma_tpred.weight = nn.Parameter((1/(self.seq_len+1))*torch.ones([1,self.seq_len+1]))

        if self.init_type == 0:
            self.gamma = nn.Parameter(torch.zeros(self.enc_in))
        else:
            self.gamma = nn.Parameter(torch.ones(self.enc_in))

        
    def forward(self, x, is_shifted): 
        b,l,n = x.shape


        # stationary branch
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        norm_x = (x - mean) /(std+1e-5)
        seasonal_init, trend_init = self.decomp_module(norm_x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
    
        seasonal_output = self.Linear_Seasonal_s(seasonal_init)
        trend_output = self.Linear_Trend_s(trend_init)
        norm_out_s = seasonal_output + trend_output
        norm_out_s = norm_out_s.permute(0, 2, 1)
        out_s = norm_out_s * (std+1e-5) + mean
        
        # non-stationary branch
        seasonal_init, trend_init = self.decomp_module(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output = self.Linear_Seasonal_ns(seasonal_init)
        trend_output = self.Linear_Trend_ns(trend_init)
        out_ns = seasonal_output + trend_output
        out_ns = out_ns.permute(0, 2, 1)
        
        # calculate the fusion weights 
        sigma_t = torch.std(x, dim=1)
        local_input = torch.cat([sigma_t.unsqueeze(1), x], dim=1)
        sigma_tpred = self.Linear_Sigma_tpred(local_input.permute(0, 2, 1)).squeeze(2)  
        gamma = self.gamma.repeat(b, 1)   
        W = gamma* (sigma_t + sigma_tpred)
        W = W.unsqueeze(1).repeat(1, self.pred_len, 1)   

        # fuse predictions 
        top_k_chs= self.G.topk(int(self.G.shape[0]*self.alpha)).indices
        nonshifted_chs = torch.nonzero(~is_shifted).squeeze()
        common_mask = torch.isin(top_k_chs, nonshifted_chs)
        sel_chs = top_k_chs[common_mask]
        # print(sel_chs)
        # print(is_shifted)
        # exit()

        # sel_chs = [0, 2]
        sel_chs = torch.tensor(sel_chs).to(x.device)
        O = torch.ones_like(W).to(W.device)
        mask = torch.zeros([n]).to(x.device)
        mask = torch.scatter(mask, 0, sel_chs, 1) 
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, self.pred_len, 1)   
        W = W * mask
        out = W * out_ns + (O - W) * out_s
        
        return out, out_s, out_ns, sel_chs
       