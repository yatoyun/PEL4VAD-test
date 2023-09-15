from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed, process_feat2
from log import get_logger

from model import XModel
from dataset import *

from train import train_func
from test import test_func
from infer import infer_func
import argparse
import copy

from DR_DMU.loss import AD_Loss
# from pytorch_lamb import Lamb
from timm.scheduler import CosineLRScheduler

# tune
import optuna
from tqdm import tqdm

import wandb
import os
from tensorboardX import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from torch.cuda.amp import GradScaler, autocast


def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')

def filter_and_sort(idx_dict, threshold=0.8, min_frames=10):
    # しきい値以上の値を持つリストを抽出
    filtered_values = {k: [val for val in v if val >= threshold] for k, v in idx_dict.items()}
    
    # 指定されたフレーム数以上のリストをフィルタリング
    long_enough_values = {k: v for k, v in filtered_values.items() if len(v) >= min_frames}
    
    # 平均スコアを計算
    avg_scores = {k: sum(v) / len(v) for k, v in long_enough_values.items()}
    
    # 平均スコアで降順にソートし、IDのリストを返す
    sorted_ids = sorted(avg_scores.keys(), key=lambda x: avg_scores[x], reverse=True)
    
    return sorted_ids

def make_new_label(num, model, pesudo=True):
    # load pesudo label
    output_dir = "train-pesudo"

    total_num = 0
    train_list = list(open(cfg.train_list))[:8100]
    
    idx_dict = {}
    
    # load original video feature
    for idx, video_name in enumerate(train_list):
        feat_path = os.path.join(cfg.feat_prefix, video_name.strip('\n'))  
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        # select feature according to label
        if pesudo:
            model.eval()
            with torch.no_grad():
                with autocast():
                    v_feat = torch.from_numpy(v_feat).unsqueeze(0)
                    v_feat = v_feat.float().cuda(non_blocking=True)
                    seq_len = torch.sum(torch.max(torch.abs(v_feat), dim=2)[0] > 0, 1)

                    logits, _ = model(v_feat, seq_len)
                    pred = logits.squeeze().cpu().detach().numpy()

                    # max_len = cfg.max_seqlen if cfg.max_seqlen < pred.shape[0] else int(pred.shape[0]*0.8)
                    # selected_indices = np.where(pred >= 0.5)[0]
                    # selected_indices.sort()
                    total_num += 1
                    
                    idx_dict[idx] = pred
            
            # v_feat = v_feat.squeeze().cpu().detach().numpy()
            # if len(selected_indices) >= 10:
            #     v_feat = v_feat[selected_indices]
            # print(v_feat.shape)
        
        # process feature
        else:
            v_feat = process_feat2(v_feat, cfg.max_seqlen, is_random=False)
            
            # save feature
            save_path = feat_path.replace("train", output_dir)
            np.save(save_path, v_feat)
            return 
    print(total_num)
    sorted_ids = filter_and_sort(idx_dict)
    print(len(sorted_ids))
    if len(sorted_ids) >= len(train_list)*num/10:
        sorted_ids = sorted_ids[:int(len(train_list)*num/10)]
    for idx in sorted_ids:
        video_name = train_list[idx]
        feat_path = os.path.join(cfg.feat_prefix, video_name.strip('\n'))
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        selected_indices = np.where(idx_dict[idx] >= 0.5)[0]
        v_feat = v_feat[selected_indices]
        # print(v_feat.shape)
        v_feat = process_feat2(v_feat, cfg.max_seqlen, is_random=False)
            
        # save feature
        save_path = feat_path.replace("train", output_dir)
        np.save(save_path, v_feat)
    
    
    
    print("Total convert {} videos".format(total_num))
    return sorted_ids
    

def train(model, all_train_normal_data, all_train_anomaly_data, test_loader, gt, logger):
    all_ntrain_indices = list(range(len(all_train_normal_data)))
    all_atrain_indices = list(range(len(all_train_anomaly_data)))
    random.shuffle(all_ntrain_indices)
    random.shuffle(all_atrain_indices)
    logger.info('Model:{}\n'.format(model))
    # logger.info('Optimizer:{}\n'.format(optimizer))
    ex_indices = []
    idx_list = list(range(1, 11))
    #lr=cfg.lr)
    # optimizer = Lamb(model.parameters(), lr=0.0025, weight_decay=0.01, betas=(.9, .999))
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=cfg.lr/10)
    
    for idx in idx_list:
        if idx == 1:
            make_new_label(idx, model, pesudo = False)
            ntrain_indices = all_ntrain_indices[:int(idx/10*len(all_ntrain_indices))]
            atrain_indices = all_atrain_indices[:int(idx/10*len(all_atrain_indices))]
        else:
            atrain_indices = make_new_label(idx, model)
            ntrain_indices = all_ntrain_indices[:int(idx/10*len(all_ntrain_indices))]
            # ex_indices = atrain_indices
        # make subset and size is idex percente
        train_normal_data = Subset(all_train_normal_data, ntrain_indices)
        train_anomaly_data = Subset(all_train_anomaly_data, atrain_indices)
        print(len(train_normal_data), len(train_anomaly_data))
        # convert_list = set(atrain_indices) - set(ex_indices)
            
        # model = XModel(cfg)
        # model.cuda()
        if idx == 2:
            model = XModel(cfg).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr-(idx-1)*0.00001, weight_decay=0.01)

        
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        criterion = torch.nn.BCELoss()
        criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
        criterion3 = AD_Loss()
        # PEL_params = [p for n, p in model.named_parameters() if 'DR_DMU' not in n]
        # DR_DMU_params = model.self_attention.DR_DMU.parameters()
        
        # optimizer = optim.Adam([
        # {'params': PEL_params, 'lr': args.PEL_lr},#0.0004},
        # {'params': DR_DMU_params, 'lr': args.UR_DMU_lr, 'weight_decay': 5e-5},#0.00030000000000000003, 'weight_decay': 5e-5}
        # ])
        # lamda = 0.982#0.492
        # alpha = 0.432#0.489#0.127

        # scheduler = CosineLRScheduler(optimizer, t_initial=200, lr_min=1e-4, 
        #                               warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)

        train_nloader = DataLoader(train_normal_data, batch_size=cfg.train_bs, shuffle=True,
                                num_workers=cfg.workers, pin_memory=True)
        train_aloader = DataLoader(train_anomaly_data, batch_size=cfg.train_bs, shuffle=True,
                                num_workers=cfg.workers, pin_memory=True)
        
        initial_auc, initial_ab_auc = test_func(test_loader, model, gt, cfg.dataset)
        logger.info('Random initialize AUC{}:{:.4f} Anomaly AUC:{:.5f}'.format(cfg.metrics, initial_auc, initial_ab_auc))

        best_model_wts = copy.deepcopy(model.state_dict())
        best_auc = 0.0
        auc_ab_auc = 0.0

        st = time.time()
        for epoch in range(cfg.max_epoch//10 * idx):
            loss1, loss2, cost = train_func(train_nloader, train_aloader, model, optimizer, criterion, criterion2, criterion3, logger_wandb, args.lamda, args.alpha)
            # loss1, loss2, cost = train_func(train_loader, model, optimizer, criterion, criterion2, cfg.lamda)
            # scheduler.step(epoch + 1)
            # scheduler.step()

            log_writer.add_scalar('loss', loss1, epoch)

            auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset)
            if (idx_list[-1] == idx and auc >= best_auc) or (idx_list[-1] != idx and ab_auc >= auc_ab_auc):
                best_auc = auc
                auc_ab_auc = ab_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')        
            log_writer.add_scalar('AUC', auc, epoch)

            lr = optimizer.param_groups[0]['lr']
            logger.info('[IDX:{}/10, Epoch:{}/{}]: lr:{:.3e} | loss1:{:.4f} loss2:{:.4f} loss3:{:.4f} | AUC:{:.4f} Anomaly AUC:{:.4f}'.format(
                idx, epoch + 1, cfg.max_epoch//10 * idx, lr, loss1, loss2, cost, auc, ab_auc))

            logger_wandb.log({"AUC": auc, "Anomaly AUC": ab_auc})



        time_elapsed = time.time() - st
        # model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
        logger.info('[IDX:{}/10] Training completes in {:.0f}m {:.0f}s | best AUC{}:{:.4f} Anomaly AUC:{:.4f}\n'.
                    format(idx, time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_ab_auc))
        
        # scheduler.step()
    



def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))
    
    global logger_wandb
    name = '{}_{}_{}_{}_Mem{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs, cfg.a_nums, cfg.n_nums)
    logger_wandb = wandb.init(project=args.dataset, name=name, group=args.dataset+"UR-DMU-MS")
    logger_wandb.config.update(args)
    logger_wandb.config.update(cfg.__dict__, allow_val_change=True)

    if cfg.dataset == 'ucf-crime':
        train_normal_data = UCFDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = UCFDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True, pesudo_label=True)
        # train_data = UCFDataset(cfg, test_mode=False)
        test_data = UCFDataset(cfg, test_mode=True)
        
    elif cfg.dataset == 'xd-violence':
        train_data = XDataset(cfg, test_mode=False)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == 'shanghaiTech':
        train_data = SHDataset(cfg, test_mode=False)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError("Do not support this dataset!")
    
    print(len(train_normal_data), len(train_anomaly_data), len(test_data))
    
    # train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
    #                           num_workers=cfg.workers, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)

    model = XModel(cfg)
    gt = np.load(cfg.gt)
    device = torch.device("cuda")
    model = model.to(device)

    param = sum(p.numel() for p in model.parameters())
    logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))

    if args.mode == 'train':
        logger.info('Training Mode')
        
        train(model, train_normal_data, train_anomaly_data, test_loader, gt, logger)
        # train(model, train_loader, test_loader, gt, logger)

    elif args.mode == 'infer':
        logger.info('Test Mode')
        if cfg.ckpt_path is not None:
            load_checkpoint(model, cfg.ckpt_path, logger)
        else:
            logger.info('infer from random initialization')
        infer_func(model, test_loader, gt, logger, cfg)

    else:
        raise RuntimeError('Invalid status!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    parser.add_argument('--mode', default='train', help='model status: (train or infer)')
    parser.add_argument('--version', default='original', help='change log path name')
    parser.add_argument('--PEL_lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--UR_DMU_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lamda', default=0.19, type=float, help='lamda')
    parser.add_argument('--alpha', default=0.523, type=float, help='alpha')
    
    args = parser.parse_args()
    cfg = build_config(args.dataset)

    savepath = './logs/{}_{}_{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs)
    os.makedirs(savepath,exist_ok=True)
    log_writer = SummaryWriter(savepath)
            

    main(cfg)
