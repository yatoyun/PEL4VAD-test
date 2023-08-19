
import torch
from modules import *
import torch.nn.init as torch_init
from BERT import BERT5
from utils import process_feat2_torch


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
        self.bertEnable = cfg.bert

        if self.bertEnable: 
            self.bert = BERT5(cfg.feat_dim,cfg.max_seqlen,hidden=cfg.feat_dim, n_layers=4, attn_heads=16)
            self.fc1_2 = nn.Linear(cfg.feat_dim, 512)
            self.fc2_2 = nn.Linear(512, 128)
            self.fc3_2 = nn.Linear(128, 1)
            self.drop_out = nn.Dropout(0.1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.max_seqlen = cfg.max_seqlen
        self.apply(weight_init)

    def forward(self, x, seq_len):
        # bert 
        out = x
        bs, ncrops, f = out.size()
        if self.bertEnable:
            if ncrops != self.max_seqlen:
                new_out = torch.zeros(0).cuda()
                for video in out:
                    new_out = torch.cat((new_out, process_feat2_torch(video, self.max_seqlen).unsqueeze(0)))
                out = new_out
                del new_out
            output, mask = self.bert(out)#.unsqueeze(1))
            cls  = output[:,0,:]
            norm = cls.norm(p=2, dim = -1, keepdim=True)
            cls  = cls.div(norm)
            cls = self.drop_out(cls) 

        # original
        x_e, x_v = self.self_attention(x, seq_len)
        logits = F.pad(x_e, (self.t - 1, 0))
        logits = self.classifier(logits)

        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)

        if self.bertEnable:       
            #add FCs on BERT  classificationOut 
            scores2 = self.relu(self.fc1_2(cls))
            scores2 = self.drop_out(scores2)
            #scores2 = self.relu(self.fc2_2(scores2))
            scores2 = self.fc2_2(scores2) #no relu activation 
            scores2 = self.drop_out(scores2)
            scores2 = self.sigmoid(self.fc3_2(scores2))
            scores2 = scores2.view(bs, -1).mean(1)
            scores2 = scores2.unsqueeze(dim=1)    

        return logits, x_v, scores2
