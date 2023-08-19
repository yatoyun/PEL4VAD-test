
import time
from utils import fixed_smooth, slide_smooth
from test import *


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        pred2 = torch.zeros(0).cuda()
        pred_video = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()
        print(len(gt))

        for i, (v_input, name) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            logits, _, logits2 = model(v_input, seq_len)
            video_logit = logits2.squeeze(dim=-1)
            logits2 = logits * logits2.view(-1, 1, 1)
            logits = torch.mean(logits, 0)
            logits = logits.squeeze(dim=-1)

            seq = len(logits)
            if cfg.smooth == 'fixed':
                logits = fixed_smooth(logits, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits = slide_smooth(logits, cfg.kappa)
            else:
                pass
            logits = logits[:seq]

            # for bert
            logits2 = torch.mean(logits2, 0)
            logits2 = torch.squeeze(logits2, -1)
            if cfg.smooth == 'fixed':
                logits2 = fixed_smooth(logits2, cfg.kappa)
            elif cfg.smooth == 'slide':
                logits2 = slide_smooth(logits2, cfg.kappa)
            else:
                pass
            logits2 = logits2[:seq]

            pred = torch.cat((pred, logits2))
            pred2 = torch.cat((pred2, logits))
            pred_video = torch.cat((pred_video, video_logit))
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16:]

        gt_video = [1]*1400 + [0]*1500

        pred = list(pred.cpu().detach().numpy())
        pred2 = list(pred2.cpu().detach().numpy())
        pred_video = list(pred_video.cpu().detach().numpy())
        far = cal_false_alarm(normal_labels, normal_preds)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred2, 16))
        roc_auc2 = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

        fpr, tpr, _ = roc_curve(list(gt_video), pred_video)
        roc_auc_video = auc(fpr, tpr)
        print(roc_auc_video)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} AUC2:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, roc_auc2, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
