import torch
import torch.nn.init as torch_init
import torch.nn as nn

from layers import *

from DR_DMU.model import WSAD


class XEncoder(nn.Module):
    def __init__(self, d_model, hid_dim, out_dim, n_heads, win_size, dropout, gamma, bias, norm=None):
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
        self.UR_DMU = WSAD(d_model, a_nums = 60, n_nums = 60)
        self.cat_2feat = nn.Conv1d(d_model, d_model//2, kernel_size=1)
        self.UR_DMU_linear3 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        self.UR_DMU_dropout3 = nn.Dropout(dropout)

    def forward(self, x, seq_len):
        adj = self.loc_adj(x.shape[0], x.shape[1])
        mask = self.get_mask(self.win_size, x.shape[1], seq_len)

        x_t = x
        x = x + self.self_attn(x, mask, adj)
        # self_att = x + self.self_attn(x, mask, adj)
        if self.training:
            x_k = self.UR_DMU(x_t)
            x_t = x_k["x"]
        else:
            x_k = torch.zeros(0).cuda()
            for x_split in x_t:
                x_split = x_split.unsqueeze(0)
                x_k_split = self.UR_DMU(x_split)
                x_k = torch.cat((x_k, x_k_split["x"]), 0)
            x_t = x_k
        
        x_t = self.norm(x_t).permute(0, 2, 1)
        # self_att = self.norm(self_att)
        # x = self.cat(torch.cat((x, self_att), -1))
        
        x = self.norm(x).permute(0, 2, 1)
        x = self.dropout1(F.gelu(self.linear1(x)))
        
        x_t = self.UR_DMU_dropout3(F.gelu(self.UR_DMU_linear3(x_t)))
        x_t = self.cat_2feat(torch.cat((x, x_t), -2))
        
        x_e = self.dropout2(F.gelu(self.linear2(x_t)))
        
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
