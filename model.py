
import torch
from modules import *
import torch.nn.init as torch_init
from mgfn.mgfn_model import MSNSD


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
        self.dropout = nn.Dropout(0.7)

    def forward(self, x, seq_len):
        k = 3
        bs, t, c = x.size()
        ncrops = 1
        x_e, x_v = self.self_attention(x, seq_len)
        logits = F.pad(x_e, (self.t - 1, 0))
        logits = self.classifier(logits)

        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)
        
        # if self.training:
        #     score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores = MSNSD(x_v,logits,bs,bs//2,self.dropout,ncrops,k)

        #     x_v = (x_v, score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude)

        return logits, x_v
