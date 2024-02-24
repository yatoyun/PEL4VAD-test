import torch
import torch.nn as nn

def norm(data):
    l2=torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
      
        
    def forward(self, result, _label, seq_len):
        loss = {}

        _label = _label.float()

        triplet = result["triplet_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]
        cos_loss = result["cos_loss"]
        b = _label.size(0)//2
        t = att.size(1)     
        k = 20 # t//16 + 1
        # anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        # anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt,k, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())
        
        A_att = torch.topk(A_att,k, dim = -1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        # N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())    
        # A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())
        
        N_loss = 0
        for i in range(N_att.shape[0]):
            valid_N_att = N_att[i, :seq_len[i]]
            target = torch.ones_like(valid_N_att).cuda()
            N_loss += self.bce(valid_N_att, target)

        # すべてのデータポイントに対して平均を取る
        N_loss = N_loss / N_att.shape[0]
        
        A_Nloss = 0
        for i in range(A_Natt.shape[0]):
            valid_A_Natt = A_Natt[i, :seq_len[i]]
            target = torch.zeros_like(valid_A_Natt).cuda()
            A_Nloss += self.bce(valid_A_Natt, target)
        A_Nloss = A_Nloss / A_Natt.shape[0]

        cost = 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance + 0.5 * cos_loss
        # cost = 0.1 * (A_loss + N_loss + panomaly_loss) + 0.1 * triplet + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["triplet"] = triplet
        loss['kl_loss'] = kl_loss
        return cost, loss



def train(net, normal_loader, abnormal_loader, optimizer, criterion, index):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)
    ainput, alabel = next(abnormal_loader)
    _data = torch.cat((ninput, ainput), 0)
    _label = torch.cat((nlabel, alabel), 0)
    _data = _data.cuda()
    _label = _label.cuda()
    predict = net(_data)
    cost, loss = criterion(predict, _label)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()