import os
from PIL import Image
import numpy as np
import shutil

input_dir = "RAW"
output_dir = "channels_distorted"
num_folders = 20

os.makedirs(output_dir, exist_ok=True)
for i in range(num_folders):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith((".tif", ".tiff"))
])

counters = {i: 0 for i in range(num_folders)}


def to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)

    min_val = arr.min()
    max_val = arr.max()

    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


for idx, filename in enumerate(files):
    folder_index = idx % num_folders
    folder_path = os.path.join(output_dir, str(folder_index))

    input_path = os.path.join(input_dir, filename)

    img = Image.open(input_path)
    arr = np.array(img)

    arr_uint8 = to_uint8(arr)

    img_uint8 = Image.fromarray(arr_uint8)

    new_name = f"{counters[folder_index]:04d}.png"
    output_path = os.path.join(folder_path, new_name)

    img_uint8.save(output_path, format="PNG")
    print(output_path)

    counters[folder_index] += 1

# Copy channel 0 images to input folder
src_dir = "channels_distorted/0"
dst_dir = "input"

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith(".png"):
        shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

print("Done.")