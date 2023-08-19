from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from utils import setup_seed
from log import get_logger
from dataset import *

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


def train(model, train_loader, test_loader, gt, logger):
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.KLDivLoss(reduction='batchmean')
    
    # original
    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    # logger.info('Model:{}\n'.format(model))
    # logger.info('Optimizer:{}\n'.format(optimizer))
    
    ##############TEST################
    # two optimizers
    optimizer = torch.optim.Adam([
    {'params': model.self_attention.parameters()},
    {'params': model.classifier.parameters()},
    {'params': model.logit_scale}
    ], lr=cfg.lr)
    
    optimizer_bert = torch.optim.Adagrad([
    {'params': model.bert.parameters()},
    {'params': model.fc1_2.parameters()},
    {'params': model.fc2_2.parameters()},
    {'params': model.fc3_2.parameters()},
    ], lr=1e-4, weight_decay=0.0010000000474974513)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)
    scheduler_bert = optim.lr_scheduler.MultiStepLR(optimizer_bert, milestones=[50,200])

    
    logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))
    logger.info('Optimizer_bert:{}\n'.format(optimizer_bert))
    ##############################
    
    initial_auc, n_far, initial_auc2 = test_func(test_loader, model, gt, cfg.dataset)
    logger.info('Random initialize {}:{:.4f} initial_AUC2:{:.4f} FAR:{:.5f}'.format(cfg.metrics, initial_auc, initial_auc2, n_far))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    auc_far = 0.0

    st = time.time()
    for epoch in range(cfg.max_epoch):
        loss1, loss2, total_loss = train_func(train_loader, model, optimizer, optimizer_bert, criterion, criterion2, cfg.lamda, cfg.beta)
        scheduler.step()
        # added
        scheduler_bert.step()

        log_writer.add_scalar('loss', total_loss, epoch)

        auc, far, auc2 = test_func(test_loader, model, gt, cfg.dataset)
        if auc >= best_auc:
            best_auc = auc
            auc_far = far
            best_auc2 = auc2
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_current' + '.pkl')        
        log_writer.add_scalar('AUC', auc, epoch)
        log_writer.add_scalar('AUC2', auc2, epoch)

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} main_loss:{:.4f} | AUC:{:.4f} AUC2:{:.4f} FAR:{:.5f}'.format(
            epoch + 1, cfg.max_epoch, loss1, loss2, total_loss, auc, best_auc2, far))

    time_elapsed = time.time() - st
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
    logger.info('Training completes in {:.0f}m {:.0f}s | best {}:{:.4f} FAR:{:.5f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc, auc_far))
    


def main(cfg):
    logger = get_logger(cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

    if cfg.dataset == 'ucf-crime':
        train_data = UCFDataset(cfg, test_mode=False)
        test_data = UCFDataset(cfg, test_mode=True)
    elif cfg.dataset == 'xd-violence':
        train_data = XDataset(cfg, test_mode=False)
        test_data = XDataset(cfg, test_mode=True)
    elif cfg.dataset == 'shanghaiTech':
        train_data = SHDataset(cfg, test_mode=False)
        test_data = SHDataset(cfg, test_mode=True)
    else:
        raise RuntimeError("Do not support this dataset!")

    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
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
        train(model, train_loader, test_loader, gt, logger)

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

    savepath = './logs/{}_{}_{}_{}_bert{}'.format(args.dataset, args.version, cfg.lr, cfg.train_bs, cfg.bert)
    os.makedirs(savepath,exist_ok=True)
    log_writer = SummaryWriter(savepath)

    if cfg.bert:
        from model_bert import XModel
        from train_bert import train_func
        from test_bert import test_func
        from infer_bert import infer_func
    else:
        from model import XModel
        from train import train_func
        from test import test_func
        from infer import infer_func        

    main(cfg)
