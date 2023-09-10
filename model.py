
import torch
from modules import *
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)


class XModel(nn.Module):
    def __init__(self, cfg):
        super(XModel, self).__init__()
        self.t = cfg.t_step
        self.self_attention = XEncoder(
            d_model=cfg.feat_dim,
            hid_dim=cfg.hid_dim,
            out_dim=cfg.out_dim,
            n_heads=cfg.head_num,
            win_size=cfg.win_size,
            dropout=cfg.dropout,
            gamma=cfg.gamma,
            bias=cfg.bias,
            norm=cfg.norm,
        )
        self.classifier = nn.Conv1d(cfg.out_dim, 1, self.t, padding=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
        self.apply(weight_init)

    def forward(self, x, macro, seq_len):
        x = x.permute(0, 2, 1, 3)
        B, N, T, C = x.shape
        # print("x.shape", x.shape)
        x_e, x_v = self.self_attention(x, macro, seq_len)
        logits = F.pad(x_e, (self.t - 1, 0))
        logits = self.classifier(logits)
        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)
        # print("logits.shape", logits.view(B, N, -1).shape)
        if self.training:
            x_v["logits_ex"] = logits
        
        logits = logits.view(B, N, -1, 1).mean(1)
        # print("logits.shape", logits.shape)

        return logits, x_v
