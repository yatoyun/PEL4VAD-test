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
from test import test_func
from infer import infer_func
import argparse
import copy

import os
from tensorboardX import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


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
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    PEL_params = [p for n, p in model.named_parameters() if 'mgfn' not in n]
    MGFN_params = model.self_attention.mgfn.parameters()
    
    optimizer = optim.Adam([
    {'params': PEL_params, 'lr': 5e-4},
    {'params': MGFN_params, 'lr': cfg.lr}
    ])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))

    initial_auc, initial_ab_auc = test_func(test_loader, model, gt, cfg.dataset)
    logger.info('Random initialize AUC{}:{:.4f} Anomaly AUC:{:.5f}'.format(cfg.metrics, initial_auc, initial_ab_auc))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    auc_ab_auc = 0.0

    st = time.time()
    for epoch in range(cfg.max_epoch):
        loss1, loss2 = train_func(train_nloader, train_aloader, model, optimizer, criterion, criterion2, cfg.lamda)
        # scheduler.step()

        log_writer.add_scalar('loss', loss1, epoch)

        auc, ab_auc = test_func(test_loader, model, gt, cfg.dataset)
        if auc >= best_auc:
            best_auc = auc
            auc_ab_auc = ab_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')        
        log_writer.add_scalar('AUC', auc, epoch)

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} | AUC:{:.4f} Anomaly AUC:{:.4f}'.format(
            epoch + 1, cfg.max_epoch, loss1, loss2, auc, ab_auc))



    time_elapsed = time.time() - st
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
    logger.info('Training completes in {:.0f}m {:.0f}s | best AUC{}:{:.4f} Anomaly AUC:{:.4f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_ab_auc))


def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

    if cfg.dataset == 'ucf-crime':
        train_normal_data = UCFDataset(cfg, test_mode=False)
        train_anomaly_data = UCFDataset(cfg, test_mode=False, is_abnormal=True)
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
        train(model, train_nloader, train_aloader, test_loader, gt, logger)

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
    
    args = parser.parse_args()
    cfg = build_config(args.dataset)

    savepath = './logs/{}_{}_{}_{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs)
    os.makedirs(savepath,exist_ok=True)
    log_writer = SummaryWriter(savepath)
            

    main(cfg)
