from configs import build_config
import os

cfg = build_config("xd")

list_i3d = list(open(cfg.train_list))
list_clip = list(open("../CLIP-TSA/list/vit/xd-vit-train.list"))

clip_feat_prefix = cfg.clip_feat_prefix

count = 0
path_list = []
for path in list_i3d:
    if "_label_A" in path:
        label = 0.0
        video_class_name = 'normal'
    else:
        label = 1.0
        video_class_name = 'abnormal'
    video_name = path.strip('\n')[:-7]
            
    clip_path_name = video_name.replace('/', '/'+video_class_name+'/') + '.npy'
    clip_path = os.path.join(clip_feat_prefix, clip_path_name)
    path_list.append(clip_path)
    if not os.path.exists(clip_path):
        print(clip_path)
    else:
        count += 1

list_clip = [path.strip('\n') for path in list_clip]
print(set(path_list) - set(list_clip))

print(count)
