import torch
import torch.nn.init as torch_init
import torch.nn as nn

from layers import *

from DR_DMU.model import WSAD


class XEncoder(nn.Module):
    def __init__(self, d_model, hid_dim, out_dim, n_heads, win_size, dropout, gamma, bias, a_nums=10, n_nums=10, norm=None):
        super(XEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = TCA(d_model, hid_dim, hid_dim, n_heads, norm)
        self.linear1 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        self.linear2 = nn.Conv1d(d_model // 2, out_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)
        self.DR_DMU = WSAD(d_model, a_nums = a_nums, n_nums = n_nums)
        
        self.linear3 = nn.Conv1d(512*2, d_model // 2, kernel_size=1)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, seq_len):
        adj = self.loc_adj(x.shape[0], x.shape[1])
        mask = self.get_mask(self.win_size, x.shape[1], seq_len)

        x = x + self.self_attn(x, mask, adj)
        
        x = self.norm(x).permute(0, 2, 1)
        x = F.gelu(self.linear1(x))
        
        x_in = x.permute(0, 2, 1)
        if self.training:
            x_k = self.DR_DMU(x_in)
            x_e = x_k["x"]
        else:
            x_k = torch.zeros(0).cuda()
            for x_split in x_in:
                x_split = x_split.unsqueeze(0)
                x_k_split = self.DR_DMU(x_split)
                x_k = torch.cat((x_k, x_k_split["x"]), 0)
            x_e = x_k
        
        # x_e = self.norm(x_e).permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        # x_e = torch.cat((x, x_e), dim = 1)
        x_e = self.dropout3(F.gelu(self.linear3(x_e)))
        x_e = self.dropout2(F.gelu(self.linear2(x_e)))
        
        # x_k = dict()
        
        if self.training:
            x_k["x"] = x

        return x_e, x_k

    def get_mask(self, window_size, temporal_scale, seq_len):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.

        m = m.repeat(self.n_heads, len(seq_len), 1, 1).cuda()

        return m
