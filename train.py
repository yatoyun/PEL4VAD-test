import torch
from loss import *
from utils import *
from model import *
from DR_DMU.model import WSAD as URModel

# def train_func(normal_dataloader, anomaly_dataloader, pel_model, pel_optimizer, criterion, criterion2, criterion3, lamda=0):
def train_func(normal_dataloader, anomaly_dataloader, pel_model, pel_optimizer, ur_model, ur_optimizer, criterion, criterion2, criterion3, pel_best, ur_best, cfg, flag=False):
# def train_func(dataloader, pel_model, pel_optimizer, criterion, criterion2, lamda=0):
    t_loss = []
    s_loss = []
    u_loss = []
    c_loss = []
    with torch.set_grad_enabled(True):
        pel_model.train()
        ur_model.train()
        for i, ((v_ninput, t_ninput, nlabel, multi_nlabel), (v_ainput, t_ainput, alabel, multi_alabel)) \
                                                    in enumerate(zip(normal_dataloader, anomaly_dataloader)):
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

            # PEL
            logits, v_feat = pel_model(v_input, seq_len)
            
            # UR
            x_k = ur_model(v_input)
            
            # pred
            bs = v_input.shape[0]
            pel_logits = logits.clone().squeeze()[bs//2:]
            ur_logits = x_k["frame"].clone().squeeze()[bs//2:]
            
            # Prompt-Enhanced Learning
            logit_scale = pel_model.logit_scale.exp()
            video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
            v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
            
            ground_truth = gen_label(video_labels)
            loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)
            loss1 = CLAS2(logits[:bs//2], label[:bs//2], seq_len[:bs//2], criterion)
            
            # UR loss
            UR_loss = criterion3(x_k, label)[0]
            loss1_ur = CLAS2(x_k["frame"][:bs//2], label[:bs//2], seq_len[:bs//2], criterion)
            # ur_loss2 = CLAS2(x_k["frame"], label, seq_len, criterion)
        
            # pel_loss = loss1 + loss2
            if flag:
                with torch.no_grad():
                    pel_best_model = XModel(cfg).cuda()
                    ur_best_model = URModel(input_size = 1024, a_nums = 60, n_nums = 60).cuda()
                    
                    pel_best_model.eval()
                    pel_best_model.load_state_dict(pel_best)
                    pel_best_logits, _ = pel_best_model(v_input, seq_len)
                    
                    ur_best_model.eval()
                    ur_best_model.load_state_dict(ur_best)
                    ur_best_logits = ur_best_model(v_input)["frame"]
                    
                    pel_best_logits = pel_best_logits.squeeze()[bs//2:]
                    ur_best_logits = ur_best_logits.squeeze()[bs//2:]
                    predict(pel_best_logits)
                    predict(ur_best_logits)
                    
                loss_co = criterion(pel_logits, ur_best_logits.detach())
                loss_co2 = criterion(ur_logits, pel_best_logits.detach())
            else:
                loss_co = 0
                loss_co2 = 0
            
            pel_loss = loss2 + loss_co + loss1
            ur_loss = UR_loss + loss_co2 + loss1_ur

            pel_optimizer.zero_grad()
            pel_loss.backward()
            pel_optimizer.step()
            ur_optimizer.zero_grad()
            ur_loss.backward()
            ur_optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)
            u_loss.append(UR_loss)
            c_loss.append(loss_co)

    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss), sum(u_loss) / len(u_loss), sum(c_loss) / len(c_loss)


def predict(pre):
    pre[pre >= 0.8] = 1
    pre[pre <= 0.2] = 0
    mask = (pre > 0.2) & (pre < 0.8)
    pre[mask] = torch.round(pre[mask] * 10) / 10
    # pre[pre >= 0.8] = 1
    # pre[pre <= 0.2] = 0
    # pre[pre <= 0.6] = 0.5
    # pre[0.4 <= pre] = 0.5
    # mask = (pre > 0.2) & (pre < 0.8) & (pre != 0.5)
    # pre[mask] = torch.round(pre[mask] * 10) / 10