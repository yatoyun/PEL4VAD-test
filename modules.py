import torch
import torch.nn.init as torch_init
import torch.nn as nn

from layers import *

from S3R.memory_module import *
from S3R.residual_attention import *

from DR_DMU.translayer import Transformer

class XEncoder(nn.Module):
    def __init__(self, d_model, hid_dim, out_dim, n_heads, win_size, dropout, gamma, bias, norm=None):
        super(XEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = TCA(d_model, hid_dim, hid_dim, n_heads, norm)
        self.linear1 = nn.Conv1d(d_model, d_model // 4, kernel_size=1)
        # self.linear2 = nn.Conv1d(d_model // 2, d_model//4, kernel_size=1)
        self.linear3 = nn.Conv1d(d_model // 4, out_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.loc_adj = DistanceAdj(gamma, bias)
        
        self.en_normal = enNormal(d_model, modality="taskaware")
        self.de_normal = deNormal(d_model, d_model // 2, reduction=16)
        self.video_embedding = Transformer(d_model, 1, 2, 128, d_model, dropout = dropout)
        self.macro_embedding = Transformer(d_model, 1, 2, 128, d_model, dropout = dropout)
        macro_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 16, 1))
        self.macro_classifier = GlobalStatistics(mlp=macro_mlp)

    def forward(self, x, macro, seq_len):
        B, N, T, C = x.shape

        x = rearrange(x, 'b n t c -> (b n) t c') # (BN)TC
        
        adj = self.loc_adj(x.shape[0], x.shape[1])
        mask = self.get_mask(self.win_size, x.shape[1], seq_len)
        x = x + self.self_attn(x, mask, adj)

        # ========
        # enNormal
        # --------
        macro, memory_attn = self.en_normal(x, macro)

        x_video = self.video_embedding(x) # (BN)TC
        x_macro = self.macro_embedding(macro) # (BN)TC

        # ========
        # deNormal
        # --------
        x, x_macro = self.de_normal(x_video, x_macro) # (BN)TC
        
        
        x = self.norm(x).permute(0, 2, 1)
        x = self.dropout1(F.gelu(self.linear1(x)))
        # x = self.dropout2(F.gelu(self.linear2(x)))
        x_e = self.dropout3(F.gelu(self.linear3(x)))
        
        # macro score
        macro_scores = self.macro_classifier(x_macro.transpose(1, 2)) # (BN)1
        macro_scores = macro_scores.contiguous().view(-1, N, 1) # BN1
        macro_scores = macro_scores.mean(dim = 1) # B1
        
        if self.training:
            x_k = {"x": x, "macro_scores": macro_scores}
        else:
            x_k = x
        
        return x_e, x_k

    def get_mask(self, window_size, temporal_scale, seq_len):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.

        m = m.repeat(self.n_heads, len(seq_len)*10, 1, 1).cuda()

        return m
