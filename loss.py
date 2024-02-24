import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import gen_label


def CLAS2(logits, label, seq_len, criterion):
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).cuda()  # tensor([])
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        tmp = torch.mean(tmp).view(1)
        ins_logits = torch.cat((ins_logits, tmp))

    clsloss = criterion(ins_logits, label)
    return clsloss

def CLAS3(logits, label, seq_len, criterion, beta=0.1):
    logits = logits.squeeze()
    ins_logits = torch.zeros(0).cuda()  # tensor([])
    max_seq_len = torch.max(seq_len)
    labels = torch.zeros(0).cuda()
    for i in range(logits.shape[0]):
        if label[i] == 0:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=1, largest=True)
        else:
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16+1), largest=True)
        # tmp = torch.mean(tmp).view(1)
        psesudo_label = convert_gt(tmp, label[i], beta)
        psesudo_label = F.pad(psesudo_label, (0, max_seq_len-len(psesudo_label)), mode='constant', value=0)
        labels = torch.cat((labels, psesudo_label.unsqueeze(0)))
        
        tmp = F.pad(tmp, (0, max_seq_len-len(tmp)), mode='constant', value=0)
        ins_logits = torch.cat((ins_logits, tmp.unsqueeze(0)))
    

    clsloss = criterion(ins_logits, labels)
    return clsloss    

def convert_gt(ins_logits, video_label, beta=0.1):
    pesudo_label = []
    for i, logits in enumerate(ins_logits):
        if logits < beta and video_label == 1:
            pesudo_label.append(0)
        else:
            pesudo_label.append(video_label)
    pesudo_label = torch.tensor(pesudo_label).cuda()
    return pesudo_label                


def KLV_loss(preds, label, criterion):
    preds = F.log_softmax(preds, dim=1)  # log_softmaxを使用
    
    target = F.softmax(label * 10, dim=1)  # これが意図した動作であればそのまま
    loss = criterion(preds, target)
    # if torch.isnan(preds).any():
    #     loss = torch.tensor(0.0).to(preds.device)  # float型のテンソルを使用
    # else:
    #     target = F.softmax(label * 10, dim=1)  # これが意図した動作であればそのまま
    #     loss = criterion(preds, target)
    return loss


def temporal_smooth(arr):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return loss


def temporal_sparsity(arr):
    loss = torch.sum(arr)
    # loss = torch.mean(torch.norm(arr, dim=0))
    return loss


def Smooth(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]-1]
        sm_mse = temporal_smooth(tmp_logits)
        smooth_mse.append(sm_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)

    return smooth_mse * lamda


def Sparsity(logits, seq_len, lamda=8e-5):
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sp_mse = temporal_sparsity(tmp_logits)
        spar_mse.append(sp_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)

    return spar_mse * lamda


def Smooth_Sparsity(logits, seq_len, lamda=8e-5):
    smooth_mse = []
    spar_mse = []
    for i in range(logits.shape[0]):
        tmp_logits = logits[i][:seq_len[i]]
        sm_mse = temporal_smooth(tmp_logits)
        sp_mse = temporal_sparsity(tmp_logits)
        smooth_mse.append(sm_mse)
        spar_mse.append(sp_mse)
    smooth_mse = sum(smooth_mse) / len(smooth_mse)
    spar_mse = sum(spar_mse) / len(spar_mse)

    return (smooth_mse + spar_mse) * lamda
