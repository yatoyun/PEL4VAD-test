
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


class BertModel(nn.Module):
    def __init__(self, cfg):
        super(BertModel, self).__init__()
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
     
        #add FCs on BERT  classificationOut 
        scores2 = self.relu(self.fc1_2(cls))
        scores2 = self.drop_out(scores2)
        #scores2 = self.relu(self.fc2_2(scores2))
        scores2 = self.fc2_2(scores2) #no relu activation 
        scores2 = self.drop_out(scores2)
        scores2 = self.sigmoid(self.fc3_2(scores2))
        scores2 = scores2.view(bs, -1).mean(1)
        scores2 = scores2.unsqueeze(dim=1)    

        return scores2
