import torch
from loss import *
from utils import *
from mgfn.loss import *


def train_func(normal_dataloader, anomaly_dataloader, model, optimizer, criterion, criterion2, lamda=0):
    t_loss = []
    s_loss = []
    with torch.set_grad_enabled(True):
        model.train()
        for i, ((v_ninput, t_ninput, nlabel, multi_nlabel), (v_ainput, t_ainput, alabel, multi_alabel)) \
                                                    in enumerate(zip(normal_dataloader, anomaly_dataloader)):
            # cat
            v_input = torch.cat((v_ninput, v_ainput), 0)
            t_input = torch.cat((t_ninput, t_ainput), 0)
            label = torch.cat((nlabel, alabel), 0)
            multi_label = torch.cat((multi_nlabel, multi_alabel), 0)
            
            
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            t_input = t_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            multi_label = multi_label.cuda(non_blocking=True)

            logits, v_feat = model(v_input, seq_len)
            
            v_feat, score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude = v_feat
            # Prompt-Enhanced Learning
            logit_scale = model.logit_scale.exp()
            video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
            v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
            ground_truth = torch.tensor(gen_label(video_labels), dtype=v_feat.dtype).cuda()
            loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)
            
            
            batch_size = v_input.shape[0] // 2
            loss_sparse = sparsity(logits[:batch_size,:,:].view(-1), batch_size, 8e-3)
            
            loss_smooth = smooth(logits,8e-4)

            logits = logits.view(batch_size * 32 * 2, -1)
            logits = logits.squeeze()

            loss_criterion = mgfn_loss(0.0001)

            cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel, nor_feamagnitude, abn_feamagnitude) + loss_smooth + loss_sparse

            loss1 = CLAS2(logits, label, seq_len, criterion)
            loss = loss1 + lamda * loss2 + cost

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)

    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss)
