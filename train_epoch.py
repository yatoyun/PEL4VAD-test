import torch
from loss import *
from utils import *

# def train_func(normal_dataloader, anomaly_dataloader, model, optimizer, criterion, criterion2, criterion3, lamda=0):
def train_func(normal_iter, anomaly_iter, model, optimizer, criterion, criterion2, criterion3, lamda=0, alpha=0):
# def train_func(dataloader, model, optimizer, criterion, criterion2, lamda=0):

    v_ninput, t_ninput, nlabel, multi_nlabel = next(normal_iter)
    v_ainput, t_ainput, alabel, multi_alabel = next(anomaly_iter)
    with torch.set_grad_enabled(True):
        model.train()
        # for i, ((v_ninput, t_ninput, nlabel, multi_nlabel), (v_ainput, t_ainput, alabel, multi_alabel)) \
        #                                             in enumerate(zip(normal_dataloader, anomaly_dataloader)):
        # cat
        v_input = torch.cat((v_ninput, v_ainput), 0)
        t_input = torch.cat((t_ninput, t_ainput), 0)
        label = torch.cat((nlabel, alabel), 0)
        multi_label = torch.cat((multi_nlabel, multi_alabel), 0)
    # for i, (v_input, t_input, label, multi_label) in enumerate(dataloader):
        
        
        seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
        v_input = v_input[:, :torch.max(seq_len), :]
        v_input = v_input.float().cuda(non_blocking=True)
        t_input = t_input.float().cuda(non_blocking=True)
        label = label.float().cuda(non_blocking=True)
        multi_label = multi_label.cuda(non_blocking=True)

        logits, x_k = model(v_input, seq_len)
        
        v_feat = x_k["x"]
        x_k["frame"] = logits
        
        # Prompt-Enhanced Learning
        logit_scale = model.logit_scale.exp()
        video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
        v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
        
        ground_truth = gen_label(video_labels)
        loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

        loss1 = CLAS2(logits, label, seq_len, criterion)
        
        UR_loss = criterion3(x_k, label)
        if lamda + alpha == 0:
            # 86.9
            lamda = 0.982#0.492
            alpha = 0.432#0.489#0.127
            # {'pel_lr': 0.00030000000000000003, 'ur_lr': 0.0008, 'lamda': 0.19, 'alpha': 0.523}
        loss = loss1 + lamda * loss2 + alpha * UR_loss[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss1, loss2, UR_loss[0]
