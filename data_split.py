import os
import tqdm
import numpy as np
from utils import process_feat
import sys

args = sys.argv

train_path = args[1]

feat_prefix = "./data/ucf-i3d"

train_list = list(open(train_path))

# make dir
# os.makedirs("./data/ucf-i3d/train-200-s", exist_ok=True)
os.makedirs("./data/ucf-i3d/test-s", exist_ok=True)

max_len = 200
print(len(train_list))
for path in tqdm.tqdm(train_list):
    # feat_path = os.path.join(feat_prefix, path.strip('\n'))
    path = path.strip('\n')
    v_feat_all = np.array(np.load(path), dtype=np.float32)
    v_feat_all = np.transpose(v_feat_all, (1, 0, 2))
    assert len(v_feat_all) == 10
    for i, v_feat in enumerate(v_feat_all):
        # v_feat = process_feat(v_feat, max_len, is_random=False)
        # output_path = path.replace("train", "train-200-s")
        output_path = path.replace("test", "test-s")
        output_path = os.path.join(feat_prefix, output_path.split("/i3d/")[-1])
        if i != 0:
            output_path = output_path.replace("_i3d", "__" + str(i))
        else:
            output_path = output_path.replace("_i3d", "")
        np.save(output_path, v_feat)
        # print(output_path)
