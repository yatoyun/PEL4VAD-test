import torch
from loss import *
from utils import *
from einops import rearrange

def train_func(dataloader, model, optimizer, criterion, criterion2, lamda=0):
    t_loss = []
    s_loss = []
    with torch.set_grad_enabled(True):
        model.train()
        for i, (v_input, t_input, label, multi_label, macro) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(v_input[:,:,0,:]), dim=2)[0] > 0, 1)
            v_input = v_input[:, :torch.max(seq_len), :]
            v_input = v_input.float().cuda(non_blocking=True)
            t_input = t_input.float().cuda(non_blocking=True)
            label = label.float().cuda(non_blocking=True)
            multi_label = multi_label.cuda(non_blocking=True)
            macro = macro.float().cuda(non_blocking=True)

            logits, output = model(v_input, macro, seq_len)
            
            v_feat = rearrange(output["x"],'(b n) t c -> b n t c', n=10).permute(1, 0, 2, 3)
            macro_scores = output["macro_scores"]
            logits_ex = output["logits_ex"].view(v_input.shape[0], 10, -1).permute(1, 0, 2)
            
            # Prompt-Enhanced Learning
            logit_scale = model.logit_scale.exp()
            loss2 = torch.tensor(0.0).cuda()
            for logit, v_f in zip(logits_ex, v_feat):
                logit = logit.unsqueeze(-1)
                video_feat, token_feat, video_labels = get_cas(v_f, t_input, logit, multi_label)
                v2t_logit, v2v_logits = create_logits(video_feat, token_feat, logit_scale)
                ground_truth = torch.tensor(gen_label(video_labels), dtype=v_feat.dtype).cuda()
                
                # KLV loss
                loss2_tmp = KLV_loss(v2t_logit, ground_truth, criterion2)
                loss2 = loss2 + loss2_tmp
            loss2 = loss2 / 10
                
            # BCE loss
            loss1 = CLAS2(logits, label, seq_len, criterion)
            # Macro loss
            macro_criterion = MacroLoss()
            loss_macro = macro_criterion(macro_scores, torch.zeros_like(macro_scores).squeeze())
            
            # Total loss
            loss = loss1 + lamda * loss2 + loss_macro

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_loss.append(loss1)
            s_loss.append(loss2)
    return sum(t_loss) / len(t_loss), sum(s_loss) / len(s_loss)


class MacroLoss(nn.Module):
    def __init__(
            self,
        ):
        super(MacroLoss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, input, label):

        input = input.squeeze()
        # target = torch.cat((label, label), dim = 0).cuda()
        target = label.cuda()

        loss = self.loss(input, target)

        return loss
