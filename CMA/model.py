import torch

from .layers import *
import torch.nn.functional as F
import torch.nn.init as torch_init

class CMA_LA(nn.Module):
    def __init__(self, modal_a, modal_b, hid_dim=128, d_ff=512, dropout_rate=0.1):
        super(CMA_LA, self).__init__()

        self.cross_attention = CrossAttention(modal_b, modal_a, hid_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(modal_a, d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(d_ff, 512, kernel_size=1),
            nn.Dropout(dropout_rate),
        )
        self.norm = nn.LayerNorm(modal_a)

    def forward(self, x, y, adj):
        new_x = x + self.cross_attention(y, x, adj)
        new_x = self.norm(new_x)
        new_x = new_x.permute(0, 2, 1)
        new_x = self.ffn(new_x)

        return new_x.permute(0, 2, 1)
