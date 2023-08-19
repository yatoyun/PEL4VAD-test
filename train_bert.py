import torch
from loss import *
from utils import *


def train_func(dataloader, model, optimizer, optimizer_bert, criterion, criterion2, lamda=0, beta=0.5):
    t_loss = []
    s_loss = []
    total_loss = []
    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, t_input, label, multi_label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            t_input = t_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            multi_label = multi_label.cuda(non_blocking=True)

            logits, v_feat, logits2 = model(v_input, seq_len)
            # Prompt-Enhanced Learning
            logit_scale = model.logit_scale.exp()
            video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
            v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
            ground_truth = torch.tensor(gen_label(video_labels), dtype=v_feat.dtype).cuda()
            loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

            loss1 = CLAS2(logits, label, seq_len, criterion)

            # print(loss2.shape, loss1.shape)

            criterion_bert = torch.nn.BCELoss()
            logits2 = logits2.squeeze()
            loss_video = criterion_bert(logits2, label)

            loss = loss1 + lamda * loss2

            optimizer.zero_grad()
            optimizer_bert.zero_grad()
            loss.backward()
            loss_video.backward()
            optimizer.step()
            optimizer_bert.step()

            t_loss.append(loss1)
            s_loss.append(loss2)
            total_loss.append(loss)

    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss), sum(total_loss) / len(total_loss)
