
import torch
from modules import *
import torch.nn.init as torch_init
from DR_DMU.model import ADCLS_head, WSAD


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)


class XModel(nn.Module):
    def __init__(self, cfg):
        super(XModel, self).__init__()
        self.t = cfg.t_step
        self.net = WSAD(cfg.feat_dim, 60, 60, cfg)
        self.classifier = ADCLS_head(1024, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
        self.apply(weight_init)
        self.norm = nn.LayerNorm(1024)
        self.linear1 = nn.Conv1d(cfg.feat_dim, cfg.feat_dim // 2, kernel_size=1)
        self.linear2 = nn.Conv1d(cfg.feat_dim // 2, cfg.out_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)


    def forward(self, x, seq_len):
        # x_e, x_v = self.self_attention(x, seq_len)
        out = self.net(x, seq_len)
        x = out["x"]
        
        x = self.norm(x).permute(0, 2, 1)
        x = self.dropout1(F.gelu(self.linear1(x)))
        x_v = x
        x_e = self.dropout2(F.gelu(self.linear2(x)))
        
        logits = F.pad(x_e, (self.t - 1, 0))
        print(logits.shape) 
        logits = self.classifier(logits)

        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)

        return logits, x_v, out
