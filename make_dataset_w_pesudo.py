from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
from configs import build_config
from dataset import *
import copy
import pickle
import os
from tqdm import tqdm

from utils import process_feat, process_feat2



def main(cfg, output_dir):
    # load pesudo label
    pesudo_labels_dict =  pickle.load(open(cfg.pesudo_label, 'rb'))

    
    # load original video feature
    for video_name in tqdm(list(open(cfg.train_list))):
        feat_path = os.path.join(cfg.feat_prefix, video_name.strip('\n'))
        
        v_feat = np.array(np.load(feat_path), dtype=np.float32)
        
        
        # select feature according to label
        if v_feat.shape[0] > cfg.max_seqlen//2 and "Normal" not in video_name:
            temp = v_feat
            label = pesudo_labels_dict[video_name.split('/')[-1].split("_x264")[0]]
            v_feat = v_feat[:len(label), :]
            selected_indices = np.where(label == 1)[0]
            v_feat = v_feat[selected_indices]
            if v_feat.shape[0] <= 10:
                v_feat = temp
        
        # process feature
        v_feat = process_feat2(v_feat, cfg.max_seqlen, is_random=False)
        
        # save feature
        save_path = feat_path.replace("train", output_dir)
        np.save(save_path, v_feat)
    




if __name__ == '__main__':
    cfg = build_config("ucf")

    output_dir = "train-pesudo"
    os.makedirs(os.path.join(cfg.feat_prefix, output_dir), exist_ok=True)
            

    main(cfg, output_dir)
