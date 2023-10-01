
import time
from utils import fixed_smooth, slide_smooth
from test import *


def infer_func(model, dataloader, gt, logger, cfg):
    st = time.time()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda()
        normal_preds = torch.zeros(0).cuda()
        normal_labels = torch.zeros(0).cuda()
        abnormal_labels = torch.zeros(0).cuda()
        abnormal_preds = torch.zeros(0).cuda()
        gt_tmp = torch.tensor(gt.copy()).cuda()
        
        tmp_pred = torch.zeros(0).cuda()

        for i, (v_input, clip_input, name) in enumerate(dataloader):
            v_input = v_input.float().cuda(non_blocking=True)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)
            clip_input = clip_input[:, :torch.max(seq_len), :]
            clip_input = clip_input.float().cuda(non_blocking=True)
            
            if max(seq_len) < 1200:
                logits, _ = model(v_input, clip_input, seq_len)
                tmp_pred = torch.cat((tmp_pred, logits))
            else:
                for v_in, cl_in, seq in zip(v_input, clip_input, seq_len):
                    v_in = v_in.unsqueeze(0)
                    cl_in = cl_in.unsqueeze(0)
                    seq = torch.tensor([seq]).cuda()
                    logits, _ = model(v_in, cl_in, seq)
                    tmp_pred = torch.cat((tmp_pred, logits))
            
            assert tmp_pred.shape[0] == cfg.test_bs
            logits = tmp_pred
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

            pred = torch.cat((pred, logits))
            labels = gt_tmp[: seq_len[0]*16]
            if torch.sum(labels) == 0:
                normal_labels = torch.cat((normal_labels, labels))
                normal_preds = torch.cat((normal_preds, logits))
            else:
                abnormal_labels = torch.cat((abnormal_labels, labels))
                abnormal_preds = torch.cat((abnormal_preds, logits))
            gt_tmp = gt_tmp[seq_len[0]*16:]
        
            tmp_pred = torch.zeros(0).cuda()

        pred = list(pred.cpu().detach().numpy())
        abnormal_preds = list(abnormal_preds.cpu().detach().numpy())
        
        far = cal_false_alarm(normal_labels, normal_preds)
        # all
        fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
        roc_auc = auc(fpr, tpr)
        # anomaly
        fpr, tpr, _ = roc_curve(list(gt)[:len(abnormal_preds)*16], np.repeat(abnormal_preds, 16))
        ab_roc_auc = auc(fpr, tpr)
        pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
        pr_auc = auc(rec, pre)

    time_elapsed = time.time() - st
    logger.info('offline AUC:{:.4f} Anomaly-AUC:{:.4f} AP:{:.4f} FAR:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
        roc_auc, ab_roc_auc, pr_auc, far, time_elapsed // 60, time_elapsed % 60))
