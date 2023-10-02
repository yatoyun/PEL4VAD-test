from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed, process_feat2, process_feat
from log import get_logger

from model import XModel
from dataset import *

from train import train_func
from test import test_func
from infer import infer_func
import argparse
import copy

from DR_DMU.loss import AD_Loss
from pytorch_lamb import Lamb
from timm.scheduler import CosineLRScheduler

# tune
import optuna
from tqdm import tqdm

import wandb
import os
from tensorboardX import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from torch.cuda.amp import GradScaler, autocast
import gc


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

def make_new_label(train_indices, num, model, pesudo=True):
    # load pesudo label
    output_dir = "train-pesudo"
    
    clip_set = set()
    total_frames = 0
    total_ori_frames = 0
    
    # load original video feature
    for idx, video_name in enumerate(list(open(cfg.train_list))[:8100]):
        feat_path = os.path.join(cfg.feat_prefix, video_name.strip('\n'))
        if idx not in train_indices:
            continue
        
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        video_idx = video_name.strip('\n').split('/')[-1].split('_')[0]
        video_class_name = video_idx[:-3]
        clip_path_name = video_name.strip('\n').split('_x264')[0].replace('/', '/'+video_class_name+'/') + '_x264.npy'
        clip_path = os.path.join(cfg.clip_feat_prefix, clip_path_name)
        clip_feat = np.array(np.load(clip_path), dtype=np.float32)
        
        total_ori_frames += len(v_feat)
        
        # select feature according to label
        if v_feat.shape[0] > cfg.max_seqlen//2 and pesudo:
            model.eval()
            with torch.no_grad():
                v_feat = torch.from_numpy(v_feat).unsqueeze(0)
                v_feat = v_feat.float().cuda(non_blocking=True)
                clip_feat = torch.from_numpy(clip_feat).unsqueeze(0)
                seq_len = torch.sum(torch.max(torch.abs(v_feat), dim=2)[0] > 0, 1)
                clip_feat = clip_feat[:, :torch.max(seq_len), :]
                clip_feat = clip_feat.float().cuda(non_blocking=True)
                
                logits, _ = model(v_feat, clip_feat, seq_len)
                pred = logits.squeeze().cpu().detach().numpy()

                # max_len = cfg.max_seqlen if cfg.max_seqlen < pred.shape[0] else int(pred.shape[0]*0.8)
                # num -= 1
                # if num > 5:
                #     num = 6
                # selected_indices = np.where(pred >= (num-1)/10)[0]
                selected_indices = np.where(pred >= 0.2)[0]
                # selected_indices.sort()
            
            v_feat = v_feat.squeeze().cpu().detach().numpy()
            clip_feat = clip_feat.squeeze().cpu().detach().numpy()
            
            if len(selected_indices) >= 100:
                v_feat = v_feat[selected_indices]
                clip_feat = clip_feat[selected_indices]
            # print(v_feat.shape)
        total_frames += v_feat.shape[0]
        # process feature
        v_feat = process_feat(v_feat, cfg.max_seqlen, is_random=False)
        clip_feat = process_feat(clip_feat, cfg.max_seqlen, is_random=False)
        
        # save feature
        save_path = feat_path.replace("train", output_dir)
        save_clip_path = clip_path.replace("train", output_dir)
        os.makedirs(save_clip_path.replace(save_clip_path.split('/')[-1], ''), exist_ok=True)
        np.save(save_path, v_feat)
        if save_clip_path not in clip_set:
            np.save(save_clip_path, clip_feat)
            clip_set.add(save_clip_path)
            
    print("total original frames:{}".format(total_ori_frames))
    print("total frames:{}".format(total_frames))
    
def check_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9  # GB単位
    cached = torch.cuda.memory_reserved() / 1e9  # GB単位
    print(f"Memory Allocated: {allocated} GB, Memory Cached: {cached} GB")


def train(model, all_train_normal_data, all_train_anomaly_data, test_loader, gt, logger):
    all_ntrain_indices = list(range(len(all_train_normal_data)))
    all_atrain_indices = list(range(len(all_train_anomaly_data)))
    random.shuffle(all_ntrain_indices)
    random.shuffle(all_atrain_indices)
    logger.info('Model:{}\n'.format(model))
    # logger.info('Optimizer:{}\n'.format(optimizer))
    ex_indices = []
    idx_list = [2, 5, 10]
    #lr=cfg.lr)
    # optimizer = Lamb(model.parameters(), lr=0.0025, weight_decay=0.01, betas=(.9, .999))
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=cfg.lr/10)
    # optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=cfg.lr/20)
    for idx in idx_list:
        # make subset and size is idex percente
        ntrain_indices = all_ntrain_indices[:int(idx/10*len(all_ntrain_indices))]
        atrain_indices = all_atrain_indices[:int(idx/10*len(all_atrain_indices))]
        train_normal_data = Subset(all_train_normal_data, ntrain_indices)
        train_anomaly_data = Subset(all_train_anomaly_data, atrain_indices)
        print(len(train_normal_data), len(train_anomaly_data))
        
        new_indices = atrain_indices #list(set(atrain_indices) - set(ex_indices))
        if idx == idx_list[0]:
            make_new_label(atrain_indices, idx, model, pesudo = False)
        else:
            del train_nloader
            del train_aloader
            tmp_model = XModel(cfg)
            tmp_model.cuda()
            tmp_model.load_state_dict(best_model_wts)
            make_new_label(new_indices, idx, model)
        ex_indices = atrain_indices
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.005)
        if idx >= 2:
            model = XModel(cfg)
            model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.005)

        
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        criterion = torch.nn.BCELoss()
        criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
        criterion3 = AD_Loss()


        train_nloader = DataLoader(train_normal_data, batch_size=cfg.train_bs, shuffle=True,
                                num_workers=cfg.workers, pin_memory=True)
        train_aloader = DataLoader(train_anomaly_data, batch_size=cfg.train_bs, shuffle=True,
                                num_workers=cfg.workers, pin_memory=True)
        
        initial_auc, initial_ab_auc = test_func(test_loader, model, gt, cfg.dataset)
        logger.info('Random initialize AUC{}:{:.4f} Anomaly AUC:{:.5f}'.format(cfg.metrics, initial_auc, initial_ab_auc))

        best_model_wts = copy.deepcopy(model.state_dict())
        best_auc = 0.0
        auc_ab_auc = 0.0

        #################################
        # check_gpu_memory()
        
        st = time.time()
        
        if idx == idx_list[0]:
            bce_flag = 0
        else:
            bce_flag = idx / 10 * 0.3
            
        for epoch in range(cfg.max_epoch):
            gc.collect()
            torch.cuda.empty_cache()
            loss1, loss2, cost = train_func(train_nloader, train_aloader, model, optimizer, criterion, criterion2, criterion3, logger_wandb, bce_flag, args.lamda, args.alpha)
            # loss1, loss2, cost = train_func(train_loader, model, optimizer, criterion, criterion2, cfg.lamda)
            # scheduler.step(epoch + 1)
            # scheduler.step()

            log_writer.add_scalar('loss', loss1, epoch)

            auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset)
            if (idx_list[-1] == idx and auc >= best_auc) or (idx_list[-1] != idx and ab_auc >= auc_ab_auc):
                best_auc = auc
                auc_ab_auc = ab_auc
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')        
            log_writer.add_scalar('AUC', auc, epoch)

            lr = optimizer.param_groups[0]['lr']
            logger.info('[IDX:{}/10, Epoch:{}/{}]: lr:{:.3e} | loss1:{:.4f} loss2:{:.4f} loss3:{:.4f} | AUC:{:.4f} Anomaly AUC:{:.4f}'.format(
                idx, epoch + 1, cfg.max_epoch, lr, loss1, loss2, cost, auc, ab_auc))

            logger_wandb.log({"AUC": auc, "Anomaly AUC": ab_auc})



        time_elapsed = time.time() - st
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
        logger.info('[IDX:{}/10] Training completes in {:.0f}m {:.0f}s | best AUC{}:{:.4f} Anomaly AUC:{:.4f}\n'.
                    format(idx, time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_ab_auc))
        
        # scheduler.step()
    



def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))
    
    if args.mode == 'train':
        global logger_wandb
        name = '{}_{}_{}_{}_Mem{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs, cfg.a_nums, cfg.n_nums)
        logger_wandb = wandb.init(project=args.dataset+"(clip+i3d)", name=name, group="MS"+args.version+"(clip-pel-ur)")
        logger_wandb.config.update(args)
        logger_wandb.config.update(cfg.__dict__, allow_val_change=True)

    if cfg.dataset == 'ucf-crime':
        train_normal_data = UCFDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = UCFDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True, pesudo_label=True)
        # train_data = UCFDataset(cfg, test_mode=False)
        test_data = UCFDataset(cfg, test_mode=True)
        
    elif cfg.dataset == 'xd-violence':
        train_normal_data = XDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = XDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True)
        # train_data = XDataset(cfg, test_mode=False)
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
    print("len gt:{}, sum gt:{}".format(len(gt), sum(gt)))
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
    parser.add_argument('--lamda', default=1, type=float, help='lamda')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
    parser.add_argument('--t_step', default=9, type=int, help='t_step')
    parser.add_argument('--k', default=20, type=int, help='k')
    parser.add_argument('--win_size', default=9, type=int, help='win_size')
    parser.add_argument('--gamma', default=0.6, type=float, help='gamma')
    parser.add_argument('--bias', default=0.2, type=float, help='bias')
    parser.add_argument('--mem_num', default=50, type=int, help='mem_num')
    
    args = parser.parse_args()
    cfg = build_config(args.dataset)
    
    cfg.k = args.k
    cfg.t_step = args.t_step
    cfg.win_size = args.win_size
    cfg.gamma = args.gamma
    cfg.bias = args.bias
    cfg.a_nums = args.mem_num
    cfg.n_nums = args.mem_num

    savepath = './logs/{}_{}_{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs)
    os.makedirs(savepath,exist_ok=True)
    log_writer = SummaryWriter(savepath)
            

    main(cfg)

#0.8592009087263632 and parameters: {'pel_lr': 0.0007000000000000001, 'ur_lr': 0.001}.
#0.8584976052870801 and parameters: {'pel_lr': 0.0008, 'ur_lr': 0.0007000000000000001}