import numpy as np
import os
import cv2
import shutil

scene = 'penguin'

ms_imgs = np.load(scene + "/ms_imgs.npy")
print("MS:", ms_imgs.shape)

num_views = ms_imgs.shape[0]
height, width = ms_imgs.shape[1:3]

output_dir = scene + "/channels_distorted"
os.makedirs(output_dir, exist_ok=True)

num_channels = ms_imgs.shape[-1]

# Save png for each individual channel
for c in range(num_channels):
    channel_dir = os.path.join(output_dir, str(c))
    os.makedirs(channel_dir, exist_ok=True)
    
    for i in range(num_views):
        img = ms_imgs[i, :, :, c]
        
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype(np.uint8)
        
        filename = os.path.join(channel_dir, f"{i:04d}.png")
        cv2.imwrite(filename, img_uint8)

# Copy channel 0 images to input folder
src_dir = os.path.join(scene, "channels_distorted", "0")
dst_dir = os.path.join(scene, "input")

os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith(".png"):
        shutil.copy(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

print("Done.")