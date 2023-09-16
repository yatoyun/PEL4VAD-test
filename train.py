import torch
from loss import *
from utils import *

# def train_func(normal_dataloader, anomaly_dataloader, model, optimizer, criterion, criterion2, criterion3, lamda=0):
def train_func(normal_dataloader, anomaly_dataloader, model, optimizer, criterion, criterion2, criterion3, logger_wandb, lamda=0, alpha=0):
# def train_func(dataloader, model, optimizer, criterion, criterion2, lamda=0):
    t_loss = []
    s_loss = []
    u_loss = []
    with torch.set_grad_enabled(True):
        model.train()
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

            logits, x_k, output_MSNSD = model(v_input, seq_len)
            
            # v_feat = x_k["x"]
            # x_k["frame"] = logits
            v_feat = x_k
            
            # Prompt-Enhanced Learning
            logit_scale = model.logit_scale.exp()
            video_feat, token_feat, video_labels = get_cas(v_feat, t_input, logits, multi_label)
            v2t_logits, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
            
            ground_truth = gen_label(video_labels)
            loss2 = KLV_loss(v2t_logits, ground_truth, criterion2)

            loss1 = CLAS2(logits, label, seq_len, criterion)
            
            # UR_loss = criterion3(x_k, label, seq_len)[0]
            # UR_loss = torch.tensor(0).float()
            
            loss_criterion = mgfn_loss()
            nlabel = label[:label.shape[0] // 2]
            alabel = label[label.shape[0] // 2:]
            mg_loss = loss_criterion(output_MSNSD, nlabel, alabel)
            loss1 = loss1 + mg_loss
            
            loss = lamda * loss2 + loss1
            
            logger_wandb.log({"loss": loss.item(), "loss1":loss1.item(), "loss2": loss2.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)
            u_loss.append(0)

    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss), sum(u_loss) / len(u_loss)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=100.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class mgfn_loss(torch.nn.Module):
    def __init__(self):
        super(mgfn_loss, self).__init__()
        self.criterion = torch.nn.BCELoss()
        self.contrastive = ContrastiveLoss()



    def forward(self, output, nlabel, alabel):
        score_abnormal = output["score_abnormal"]
        score_normal = output["score_normal"]
        
        nor_feamagnitude = output["nor_feamagnitude"]
        abn_feamagnitude = output["abn_feamagnitude"]
        
        # label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal
        # score = torch.cat((score_normal, score_abnormal), 0)
        # score = score.squeeze()
        # label = label.cuda()
        # seperate = len(abn_feamagnitude) / 2

        # loss_cls = self.criterion(score_abnormal.squeeze(), alabel.cuda())
        loss_con = self.contrastive(torch.norm(abn_feamagnitude, p=1, dim=2), torch.norm(nor_feamagnitude, p=1, dim=2),
                                    1)  # try tp separate normal and abnormal
        # loss_con_n = self.contrastive(torch.norm(nor_feamagnitude[int(seperate):], p=1, dim=2),
        #                               torch.norm(nor_feamagnitude[:int(seperate)], p=1, dim=2),
        #                               0)  # try to cluster the same class
        # loss_con_a = self.contrastive(torch.norm(abn_feamagnitude[int(seperate):], p=1, dim=2),
        #                               torch.norm(abn_feamagnitude[:int(seperate)], p=1, dim=2), 1)
        loss_total = 0.001 * (0.01 * loss_con) #loss_con_n )
        
        return loss_total
