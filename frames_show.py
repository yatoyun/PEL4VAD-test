import os
import tqdm
import numpy as np
from utils import process_feat

train_path = "./list/ucf/train.list"

train_clip_path = "./list/ucf/train-clip.list"

feat_prefix = "./data/ucf-i3d"

train_list = list(open(train_path))

# make dir
# os.makedirs("./data/ucf-i3d/train-200", exist_ok=True)

# max_len = 200
# print(len(train_list))
# for path in tqdm.tqdm(train_list):
#     feat_path = os.path.join(feat_prefix, path.strip('\n'))
#     v_feat = np.array(np.load(feat_path), dtype=np.float32)
#     v_feat = process_feat(v_feat, max_len, is_random=False)
#     output_path = feat_path.replace("train", "train-200")
#     np.save(output_path, v_feat)

i3d_list = []

print(len(train_list))
for feat_path in train_list:
    if "__" in feat_path:
        continue
    feat_path = os.path.join(feat_prefix, feat_path.strip('\n'))
    v_feat = np.array(np.load(feat_path), dtype=np.float32)
    i3d_list.append(v_feat.shape[0])

clip_list = []
train_path = train_clip_path
train_list = list(open(train_path))

print(len(train_list))
for feat_path in train_list:
    # feat_path = os.path.join(feat_prefix, path.strip('\n'))
    feat_path = feat_path.strip('\n')
    v_feat = np.array(np.load(feat_path), dtype=np.float32)
    clip_list.append(v_feat.shape[0])

i3d_list = np.array(i3d_list)
clip_list = np.array(clip_list)

for i3d, clip in zip(i3d_list, clip_list):
    print(i3d - clip)
