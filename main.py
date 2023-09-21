from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed
from log import get_logger

from model import XModel
from dataset import *

from train_epoch import train_func
from test import test_func
from infer import infer_func
import argparse
import copy

from DR_DMU.loss import AD_Loss
from pytorch_lamb import Lamb
from timm.scheduler import CosineLRScheduler

# tune
import optuna

import wandb
import os
from tensorboardX import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from torch.cuda.amp import GradScaler


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


def train(model, train_nloader, train_aloader, test_loader, gt, logger):
# def train(model, train_loader, test_loader, gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion3 = AD_Loss()
    # PEL_params = [p for n, p in model.named_parameters() if 'UR_DMU' or '2feat' not in n]
    # UR_DMU_params = [p for n, p in model.named_parameters() if 'UR_DMU' in n]
    # Cat_2feat_params = model.self_attention.cat_2feat.parameters()
        
    # # optimizer = optim.Adam([
    # # {'params': PEL_params, 'lr': args.PEL_lr},#0.0004},
    # # {'params': UR_DMU_params, 'lr': args.UR_DMU_lr, 'weight_decay': 5e-5},#0.00030000000000000003, 'weight_decay': 5e-5}
    # # ])
    # lamda = 0.982#0.492
    # alpha = 0.432#0.489#0.127
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.005)#lr=cfg.lr)
    # optimizer = Lamb(model.parameters(), lr=0.0025, weight_decay=0.01, betas=(.9, .999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    # scheduler = CosineLRScheduler(optimizer, t_initial=200, lr_min=1e-4, 
    #                               warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)

    logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))

    initial_auc, initial_ab_auc = test_func(test_loader, model, gt, cfg.dataset)
    logger.info('Random initialize AUC{}:{:.4f} Anomaly AUC:{:.5f}'.format(cfg.metrics, initial_auc, initial_ab_auc))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    auc_ab_auc = 0.0
    # cfg.max_epoch *= len(train_nloader)

    st = time.time()
    print(len(train_nloader), len(train_aloader))
    for epoch in range(cfg.max_epoch):
        for idx, (n_input, a_input) in enumerate(zip(train_nloader, train_aloader)):
        
            loss1, loss2, cost = train_func(n_input, a_input, model, optimizer, criterion, criterion2,  criterion3, logger_wandb, args.lamda, args.alpha)
            # loss1, loss2, cost = train_func(train_loader, model, optimizer, criterion, criterion2, cfg.lamda)
            # scheduler.step(epoch + 1)

            log_writer.add_scalar('loss', loss1, epoch)
            turn_point = 100
            if (epoch >= turn_point and (idx+1) % 1 == 0):
                auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset)
                if auc >= best_auc:
                    best_auc = auc
                    auc_ab_auc = ab_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')        
                log_writer.add_scalar('AUC', auc, epoch)

                lr = optimizer.param_groups[0]['lr']
                logger.info('[Epoch:{}/{}, Batch:{}/{}]: loss1:{:.4f} loss2:{:.4f} loss3:{:.4f} | AUC:{:.4f} Anomaly AUC:{:.4f}'.format(
                    epoch + 1, cfg.max_epoch, idx, len(train_nloader), loss1, loss2, cost, auc, ab_auc))

                logger_wandb.log({"AUC": auc, "Anomaly AUC": ab_auc})

        # scheduler.step()
        auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset)
        if auc >= best_auc:
            best_auc = auc
            auc_ab_auc = ab_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')        
        log_writer.add_scalar('AUC', auc, epoch)

        lr = optimizer.param_groups[0]['lr']
        logger.info('[Epoch:{}/{}]: lr:{:.5f} | loss1:{:.4f} loss2:{:.4f} loss3:{:.4f} | AUC:{:.4f} Anomaly AUC:{:.4f}'.format(
            epoch + 1, cfg.max_epoch, lr, loss1, loss2, cost, auc, ab_auc))

        logger_wandb.log({"AUC": auc, "Anomaly AUC": ab_auc})



    time_elapsed = time.time() - st
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
    logger.info('Training completes in {:.0f}m {:.0f}s | best AUC{}:{:.4f} Anomaly AUC:{:.4f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_ab_auc))


def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))
    
    if args.mode == 'train':
        global logger_wandb
        name = '{}_{}_{}_{}_Mem{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs, cfg.a_nums, cfg.n_nums)
        logger_wandb = wandb.init(project=args.dataset+"(clip+i3d)", name=name, group="epoch-"+args.version+"(clip-pel-ur)")
        logger_wandb.config.update(args)
        logger_wandb.config.update(cfg.__dict__, allow_val_change=True)

    if cfg.dataset == 'ucf-crime':
        train_normal_data = UCFDataset(cfg, test_mode=False, pre_process=True)
        train_anomaly_data = UCFDataset(cfg, test_mode=False, is_abnormal=True, pre_process=True)
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

    train_nloader = DataLoader(train_normal_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True)
    train_aloader = DataLoader(train_anomaly_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True)
    
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
        
        train(model, train_nloader, train_aloader, test_loader, gt, logger)
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
    parser.add_argument('--alpha', default=1, type=float, help='alpha')
    
    args = parser.parse_args()
    cfg = build_config(args.dataset)

    savepath = './logs/{}_{}_{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs)
    os.makedirs(savepath,exist_ok=True)
    log_writer = SummaryWriter(savepath)
            

    main(cfg)

#0.8592009087263632 and parameters: {'pel_lr': 0.0007000000000000001, 'ur_lr': 0.001}.
#0.8584976052870801 and parameters: {'pel_lr': 0.0008, 'ur_lr': 0.0007000000000000001}