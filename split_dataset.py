import numpy as np
import os
from tqdm import tqdm

# Replace with the actual directory path
input_directory = "/home/yukaneko/dev/ShanghaiTech_features/SH_Train_ten_crop_i3d"
output_directory = "./data/sh-i3d/rtfm/train"

os.makedirs(output_directory, exist_ok=True)

# Iterate over each file in the directory
for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith("_i3d.npy"):
        # Load the numpy array from the file
        filepath = os.path.join(input_directory, filename)
        data = np.load(filepath)

        # Extract the first 1024 dimensions
        rgb_data = data[:, :, :1024]

        # Save each slice [:, i, :] as a new file
        for i in range(10):
            slice_data = rgb_data[:, i, :]
            new_filename = filename.replace("_i3d.npy", f"_{i}.npy")
            new_filepath = os.path.join(output_directory, new_filename)
            np.save(new_filepath, slice_data)