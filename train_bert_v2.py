import torch
from loss import *
from utils import *


def train_func(dataloader, model_PEL, model, optimizer, criterion, lamda=0, beta=0.5):
    total_loss = []
    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, t_input, label, multi_label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            logits2 = model(v_input, seq_len)
            # logits = model_PEL(v_input, seq_len)
            
            # logits *= logits2.viewf(-1, 1, 1)

            # print(loss2.shape, loss1.shape)
            logits2 = logits2.squeeze()
            loss = criterion(logits2, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss.append(loss)

    return sum(total_loss) / len(total_loss)
