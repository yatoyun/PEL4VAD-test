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

from train import train_func
from test import test_func, test_func_ur
from infer import infer_func
import argparse
import copy

from DR_DMU.loss import AD_Loss
from DR_DMU.model import WSAD as URModel
# from pytorch_lamb import Lamb
from timm.scheduler import CosineLRScheduler

# tune
import optuna

import os
from tensorboardX import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from torch.cuda.amp import GradScaler

# torch.autograd.set_detect_anomaly(True)


def load_checkpoint(pel_model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = pel_model.state_dict()
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
                logger.info('{} not found in pel_model dict.'.format(name))
    else:
        logger.info('Not found pretrained checkpoint file.')


def train(pel_model, ur_model, train_nloader, train_aloader, test_loader, gt, logger):
# def train(pel_model, train_loader, test_loader, gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    criterion3 = AD_Loss()
    
    # lamda = 0.982#0.492
    # alpha = 0.432#0.489#0.127
    logger.info('pel_model:{}\n'.format(pel_model))
    logger.info('ur_model:{}\n'.format(ur_model))


    for i in range(cfg.max_epoch // 10):
        pel_optimizer = optim.Adam(pel_model.parameters(), lr=5e-5, weight_decay=5e-6)#lr=cfg.lr)
        ur_optimizer = optim.Adam(ur_model.parameters(), lr=5e-7, weight_decay=5e-8)#lr=cfg.lr)
        # optimizer = Lamb(pel_model.parameters(), lr=0.0025, weight_decay=0.01, betas=(.9, .999))
        pel_scheduler = optim.lr_scheduler.CosineAnnealingLR(pel_optimizer, T_max=20, eta_min=0)
        # ur_scheduler = optim.lr_scheduler.CosineAnnealingLR(ur_optimizer, T_max=10, eta_min=0)
        # scheduler = CosineLRScheduler(optimizer, t_initial=200, lr_min=1e-4, 
        #                               warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)


        # logger.info('Optimizer PEL:{}\n'.format(pel_optimizer))
        # logger.info('Optimizer UR:{}\n'.format(ur_optimizer))
        
        pel_initial_auc, pel_initial_ab_auc = test_func(test_loader, pel_model, gt, cfg.dataset)
        logger.info('Random initialize AUC{}:{:.4f} Anomaly AUC:{:.5f}'.format(cfg.metrics, pel_initial_auc, pel_initial_ab_auc))
        
        ur_initial_auc, ur_initial_ab_auc = test_func_ur(test_loader, ur_model, gt, cfg.dataset)
        logger.info('Random initialize AUC{}:{:.4f} Anomaly AUC:{:.5f}'.format(cfg.metrics, ur_initial_auc, ur_initial_ab_auc))

        # pel_best_model_wts = torch.load("./ckpt/original__8636.pkl") #copy.deepcopy(pel_model.state_dict())
        # ur_best_model_wts = torch.load("./ckpt/ucf_trans_2022.pkl") #copy.deepcopy(ur_model.state_dict())
        
        pel_best_model_wts = copy.deepcopy(pel_model.state_dict())
        ur_best_model_wts = copy.deepcopy(ur_model.state_dict())    
        
        pel_best_auc = pel_initial_auc
        pel_auc_ab_auc = pel_initial_ab_auc
        
        ur_best_auc = ur_initial_auc
        ur_auc_ab_auc = ur_initial_ab_auc

        st = time.time()
        for epoch in range(10):
            loss1, loss2, cost, co_loss = train_func(train_nloader, train_aloader, pel_model, pel_optimizer, ur_model, ur_optimizer, criterion, criterion2, criterion3, pel_best_model_wts, ur_best_model_wts, cfg, True)
            # loss1, loss2, cost = train_func(train_loader, pel_model, optimizer, criterion, criterion2, cfg.lamda)
            # scheduler.step(epoch + 1)
            # scheduler.step()
            # pel_scheduler.step()
            # ur_scheduler.step()

            log_writer.add_scalar('loss', loss1, epoch)

            pel_auc, pel_ab_auc = test_func(test_loader, pel_model, gt, cfg.dataset)
            ur_auc, ur_ab_auc = test_func_ur(test_loader, ur_model, gt, cfg.dataset)
            if pel_auc >= pel_best_auc:
                pel_best_auc = pel_auc
                pel_auc_ab_auc = pel_ab_auc
                # pel_best_model_wts = copy.deepcopy(pel_model.state_dict())
                torch.save(pel_model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')    
            
            if ur_ab_auc >= ur_auc_ab_auc:
                ur_best_auc = ur_auc
                ur_auc_ab_auc = ur_ab_auc
                # ur_best_model_wts = copy.deepcopy(ur_model.state_dict())
                torch.save(ur_model.state_dict(), cfg.save_dir + cfg.model_name + '_UR_current' + '.pkl')
                
            log_writer.add_scalar('AUC', pel_auc, epoch)

            lr = pel_optimizer.param_groups[0]['lr']
            ur_lr = ur_optimizer.param_groups[0]['lr']
            logger.info('[Epoch:{}/{}]: pel-lr:{:.4e} ur-lr:{:.4e}| loss1:{:.4f} loss2:{:.4f} loss3:{:.4f} co-loss:{:.4f} | PEL.AUC:{:.4f} An-AUC:{:.4f}, UR.AUC:{:.4f} An-AUC:{:.4f}'.format(
                epoch + 1, cfg.max_epoch, lr, ur_lr, loss1, loss2, cost, co_loss, pel_auc, pel_ab_auc, ur_auc, ur_ab_auc))
            
            
        pel_best_model_wts = torch.load(cfg.save_dir + cfg.model_name + '_current' + '.pkl')
        ur_best_model_wts = torch.load(cfg.save_dir + cfg.model_name + '_UR_current' + '.pkl')
        pel_model.load_state_dict(pel_best_model_wts)
        ur_model.load_state_dict(ur_best_model_wts)



    time_elapsed = time.time() - st
    # pel_model.load_state_dict(pel_best_model_wts)
    # ur_model.load_state_dict(ur_best_model_wts)
    pel_best_model_wts = torch.load(cfg.save_dir + cfg.model_name + '_current' + '.pkl')
    ur_best_model_wts = torch.load(cfg.save_dir + cfg.model_name + '_UR_current' + '.pkl')
    pel_model.load_state_dict(pel_best_model_wts)
    ur_model.load_state_dict(ur_best_model_wts)
    
    torch.save(pel_model.state_dict(), cfg.save_dir + cfg.model_name + "co_teach" + '_' + str(round(pel_best_auc, 4)).split('.')[1] + '.pkl')
    torch.save(ur_model.state_dict(), cfg.save_dir + cfg.model_name + "UR_co_teach" + '_' + str(round(ur_best_auc, 4)).split('.')[1] + '.pkl')
    logger.info('Training completes in {:.0f}m {:.0f}s | PEL. best AUC{}:{:.4f} Anomaly AUC:{:.4f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, pel_best_auc, pel_auc_ab_auc))
    logger.info('Training completes in {:.0f}m {:.0f}s | UR. best AUC{}:{:.4f} Anomaly AUC:{:.4f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, ur_best_auc, ur_auc_ab_auc))


def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

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

    pel_model = XModel(cfg)
    gt = np.load(cfg.gt)
    device = torch.device("cuda")
    pel_model = pel_model.to(device)
    load_checkpoint(pel_model, "./ckpt/original__8636.pkl", logger)
    
    ur_model = URModel(input_size = 1024, a_nums = 60, n_nums = 60)
    ur_model = ur_model.to(device)
    load_checkpoint(ur_model, "./ckpt/ucf_trans_2022.pkl", logger)

    param = sum(p.numel() for p in pel_model.parameters())
    logger.info('total params:{:.4f}M'.format(param / (1000 ** 2)))

    if args.mode == 'train':
        logger.info('Training Mode')
        
        train(pel_model, ur_model, train_nloader, train_aloader, test_loader, gt, logger)
        # train(pel_model, train_loader, test_loader, gt, logger)

    elif args.mode == 'infer':
        logger.info('Test Mode')
        if cfg.ckpt_path is not None:
            load_checkpoint(pel_model, cfg.ckpt_path, logger)
        else:
            logger.info('infer from random initialization')
        infer_func(pel_model, test_loader, gt, logger, cfg)

    else:
        raise RuntimeError('Invalid status!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    parser.add_argument('--mode', default='train', help='pel_model status: (train or infer)')
    parser.add_argument('--version', default='original', help='change log path name')
    parser.add_argument('--PEL_lr', default=0.0003, type=float, help='learning rate')
    parser.add_argument('--UR_DMU_lr', default=0.0008, type=float, help='learning rate')
    parser.add_argument('--lamda', default=0.19, type=float, help='lamda')
    parser.add_argument('--alpha', default=0.523, type=float, help='alpha')
    
    args = parser.parse_args()
    cfg = build_config(args.dataset)

    savepath = './logs/{}_{}_{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs)
    os.makedirs(savepath,exist_ok=True)
    log_writer = SummaryWriter(savepath)
            

    main(cfg)
