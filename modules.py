import torch
import torch.nn.init as torch_init
import torch.nn as nn

from layers import *

# from DR_DMU.model import WSAD
from Hyper.model import *
from CLIP_TSA.hard_attention import HardAttention


class XEncoder(nn.Module):
    def __init__(self, d_model, hid_dim, out_dim, n_heads, win_size, dropout, gamma, bias, cfg, norm=None):
        super(XEncoder, self).__init__()
        self.n_heads = n_heads
        self.win_size = win_size
        self.self_attn = TCA(d_model, hid_dim, hid_dim, n_heads, norm)
        self.linear1 = nn.Conv1d(d_model // 2, d_model // 2, kernel_size=1)
        # self.linear2 = nn.Conv1d(d_model // 2, out_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model // 2)
        self.loc_adj = DistanceAdj_PEL(gamma, bias)
        # self.hard_atten = HardAttention(k=0.95, num_samples=100, input_dim=d_model//4)
        # self.hard_atten2 = HardAttention(k=0.95, num_samples=100, input_dim=d_model//4)
        self.conv1 = nn.Conv1d(d_model, d_model // 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
        
        self.manifold = getattr(manifolds, cfg.manifold)()
        self.args = cfg
        self.args.feat_dim = cfg.feat_dim_hyp
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            self.args.feat_dim = self.args.feat_dim + 1
        self.disAdj = DistanceAdj()
        self.HFSGCN = FHyperGCN(cfg)
        self.HTRGCN = FHyperGCN(cfg)
        self.relu = nn.LeakyReLU()
        self.HyperCLS = HypClassifier(cfg)

        
        assert d_model // 2 == 512
        
        # self.concat_feat = nn.Linear(d_model * 2, d_model)

    def forward(self, x, c_x, seq_len):
        adj = self.loc_adj(x.shape[0], x.shape[1])
        mask = self.get_mask(self.win_size, x.shape[1], seq_len)
        
        x_h = c_x # self.hard_atten(c_x)

        x = x + self.self_attn(x, mask, adj)
        
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
                
        x = torch.cat((x, x_h), -1)
        
        ############# hyper ###########
        disadj = self.disAdj(x.shape[0], x.shape[1], self.args).to(x.device)
        proj_x = self.expm(x)
        if not self.training:
            seq_len = None
            
        adj = self.adj(proj_x, seq_len)

        x1 = self.relu(self.HFSGCN.encode(proj_x, adj))
        # x1 = self.hard_atten2(x1)
        x1 = self.dropout(x1)
        x2 = self.relu(self.HTRGCN.encode(proj_x, disadj))
        # x2 = self.hard_atten2(x2)
        x2 = self.dropout(x2)
        x = torch.cat((x1, x2), 2)
        
        x = self.norm(x).permute(0, 2, 1)
        # x = x.permute(0, 2, 1)
        x = self.dropout1(F.gelu(self.linear1(x)))
        # x_e = self.dropout2(F.gelu(self.linear2(x)))
        x_e = x.permute(0, 2, 1)
        
        x_e = self.HyperCLS(x_e)
        

        return x_e, x

    def get_mask(self, window_size, temporal_scale, seq_len):
        m = torch.zeros((temporal_scale, temporal_scale))
        w_len = window_size
        for j in range(temporal_scale):
            for k in range(w_len):
                m[j, min(max(j - w_len // 2 + k, 0), temporal_scale - 1)] = 1.

        m = m.repeat(self.n_heads, len(seq_len), 1, 1).cuda()

        return m
    
    def expm(self, x):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=-1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
            return x
        else:
            return x

    def adj(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = self.lorentz_similarity(x, x, self.manifold.k)
        x2 = torch.exp(-x2)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0).to(logits.device)  # tensor([])
        for i in range(logits.shape[0]):
            if seq_len is None:
                tmp = torch.mean(logits[i]).view(1)
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='floor') + 1),
                                largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
        '''
        d = <x, y>   lorentz metric
        '''
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        idx = np.concatenate((np.array([-1]), np.ones(x.shape[-1] - 1)))
        diag = torch.from_numpy(np.diag(idx).astype(np.float32)).to(x.device)
        temp = x @ diag
        xy_inner = -(temp @ y.transpose(-1, -2))
        xy_inner_ = F.threshold(xy_inner, 1, 1)
        sqrt_k = k**0.5
        dist = sqrt_k * self.arccosh(xy_inner_ / k)
        dist = torch.clamp(dist, min=self.eps[x.dtype], max=200)
        return dist

    def arccosh(self, x):
        """
        Element-wise arcosh operation.
        Parameters
        ---
        x : torch.Tensor[]
        Returns
        ---
        torch.Tensor[]
            arcosh result.
        """
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))

