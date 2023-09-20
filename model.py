
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
            a_nums=cfg.a_nums,
            n_nums=cfg.n_nums,
            norm=cfg.norm,
        )
        self.classifier = nn.Conv1d(cfg.out_dim, 1, self.t, padding=0)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / cfg.temp))
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(weight_init)

    def forward(self, x, seq_len):
        x_e, x_v = self.self_attention(x, seq_len)
        logits = F.pad(x_e, (self.t - 1, 0))
        logits = self.classifier(logits)

        logits = logits.permute(0, 2, 1)
        logits = torch.sigmoid(logits)
        
        if self.training:
            output = MSNSD(x_v["x"].permute(0,2,1), logits, x.shape[0], x.shape[0] // 2, self.dropout, 1)
            return logits, x_v, output

        return logits, x_v

def MSNSD(features,scores,bs,batch_size,drop_out,ncrops,k=20):
    #magnitude selection and score prediction
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    scores = scores.view(bs, -1)  # (B,32)
    scores = scores.unsqueeze(dim=2)  # (B,32,1)

    normal_features = features[:batch_size]
    normal_scores = scores[:batch_size]  # [b/2, 32,1]

    abnormal_features = features[batch_size:]
    abnormal_scores = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [B,32]
    feat_magnitudes = feat_magnitudes.view(bs, -1)  # [b,32]
    nfea_magnitudes = feat_magnitudes[:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes).cuda()
    select_idx = drop_out(select_idx)


    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, f])

    abnormal_features = abnormal_features.view(n_size, t, f)
    # abnormal_features = abnormal_features.permute(1, 0, 2, 3)


    total_select_abn_feature = torch.gather(abnormal_features, 1, idx_abn_feat)
        
    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                dim=1)


    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    select_idx_normal = drop_out(select_idx_normal)
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, t, f)
    # normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.gather(normal_features, 1, idx_normal_feat)
        
    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    # return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores
    return dict(
        score_abnormal=score_abnormal, 
        score_normal=score_normal, 
        abn_feamagnitude=abn_feamagnitude, 
        nor_feamagnitude=nor_feamagnitude)
