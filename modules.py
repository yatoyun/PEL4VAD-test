import torch
import torch.nn.init as torch_init
import torch.nn as nn

from layers import *

from DR_DMU.model import WSAD
from CLIP_TSA.hard_attention import HardAttention


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
        self.UR_DMU = WSAD(d_model, a_nums = a_nums, n_nums = n_nums, dropout = dropout)
        self.hard_atten = HardAttention(k=0.95, num_samples=100, input_dim=d_model//2)
        self.conv1 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        assert d_model // 2 == 512
        
        # self.concat_feat = nn.Linear(d_model * 2, d_model)

    def forward(self, x, c_x, seq_len):
        adj = self.loc_adj(x.shape[0], x.shape[1])
        mask = self.get_mask(self.win_size, x.shape[1], seq_len)
        
        x_h = self.hard_atten(c_x)

        x = x + self.self_attn(x, mask, adj)
        
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
                
        x = torch.cat((x, x_h), -1)
        
        # self_att = x + self.self_attn(x, mask, adj)
        # x = torch.cat((x, x+self.self_attn(x, mask, adj)), -1)
        
        # x = self.norm(x)
        if self.training:
            x_k = self.UR_DMU(x)
            x = x_k["x"]
        else:
            x_k = torch.zeros(0).cuda()
            for x_split in x:
                x_split = x_split.unsqueeze(0)
                x_k_split = self.UR_DMU(x_split)
                x_k = torch.cat((x_k, x_k_split["x"]), 0)
            x = x_k
        
        # x = self.norm(x)
        # self_att = self.norm(self_att)
        # x = self.cat(torch.cat((x, self_att), -1))
        # x = self.norm(x)
        # x = x + self.self_attn(x, mask, adj)
        
        x = self.norm(x).permute(0, 2, 1)
        x = self.dropout1(F.gelu(self.linear1(x)))
        x_e = self.dropout2(F.gelu(self.linear2(x)))
        
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
