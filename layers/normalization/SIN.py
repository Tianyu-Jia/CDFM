import numpy as np
import torch
import torch.nn as nn

class SIN(nn.Module):
    def __init__(self, configs):
        super(SIN, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.selected_U = configs.selected_U
        self.selected_V = configs.selected_V
        self.tau = configs.tau
        self.affine_weight = nn.Parameter(torch.ones(self.tau))

    def forward(self, x, mode:str):
        if mode == 'norm':
            x, self.thetax = self._get_norm_x(x)
        elif mode == 'denorm':
            b,l,n = x.shape
            affine_weight = self.affine_weight.reshape(-1, 1, 1).repeat(1,b,n)
            thetay = self.thetax * affine_weight
            x = self._get_y(x, thetay)
        else: raise NotImplementedError
        return x


    def _get_norm_x(self, x):
        b, l, n = x.shape
        removed_x = torch.zeros_like(x)
        thetax = torch.zeros([self.tau, b, n]).to(x.device)
        selected_U = torch.tensor(self.selected_U, dtype=torch.float32).to(x.device)
        for c in range(n):  
            selected_U_c = selected_U[c,:,:] # [L, tau]
            thetax[:,:,c] = torch.matmul(selected_U_c.T, x[:,:,c].T)     # [tau,b]
            removed_x[:,:,c] = torch.matmul(selected_U_c, thetax[:,:,c]).T # [L, b]
        norm_x = x - removed_x
        return norm_x, thetax

    def _get_y(self, norm_y, thetay):
        b, h, n = norm_y.shape
        restore_y = torch.zeros_like(norm_y)
        selected_V = torch.tensor(self.selected_V, dtype=torch.float32).to(norm_y.device) #[H, tau, N]
        for c in range(n):  
            selected_V_c = selected_V[c,:,:] 
            restore_y[:,:,c] = torch.matmul(selected_V_c, thetay[:,:,c]).T # [H]
        y = norm_y + np.sqrt(h/self.seq_len)*restore_y
        return y

   