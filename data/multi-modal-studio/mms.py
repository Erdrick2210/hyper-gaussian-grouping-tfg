import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imsave


path_input = 'mms-data_birdhouse/mms-data_demosaicked_undistorted/scenes/birdhouse/modalities/multispectral' # per birdhouse sol
# path_input = 'mms-data/toys/modalities/multispectral' # si es descarrega tot el dataset | POSAR L'ESCENA CORRECTA
# demosaicked AND undistorted, the later by COLMAP
path_output = 'birdhouse' # 'birdhouse' 'bouquet' 'clock' 'fan' 'forestgang1' 'globe' 'toys'
num_frames = 50
fnames = ['{:04d}.npy'.format(i) for i in range(num_frames)]  # 0000.npy ... 0049.npy
num_channels = 9
max_value = 2**16
print(fnames)

# show all the channels of an image

nframe = 0
ima = np.load(os.path.join(path_input, fnames[nframe]))
print(ima.shape)
assert ima.shape[-1] == num_channels

rows = int(np.ceil(np.sqrt(num_channels)))
cols = rows
plt.figure()
i = 0
for r in range(rows):
    for c in range(cols):
        if i<num_channels:
            plt.subplot(rows, cols, i+1)
            plt.imshow(ima[:, :, i] / max_value, cmap='gray')
            # to check they are different
            # plt.imshow(ima[:, :, i] / 2 ** 16 - ima[:, :, 0] / 2 ** 16, cmap='gray')
            plt.axis('off')
            #print(i)
        i += 1

plt.show(block=False)


# show all the viewpoints for a certain channel

if not os.path.isdir(path_output):
    os.makedirs(path_output)
    print('made dir {}'.format(path_output))

rows = int(np.ceil(np.sqrt(num_frames)))
cols = rows
plt.figure()
nchannel = 0
for nframe in range(num_frames):
    ima = np.load(os.path.join(path_input, fnames[nframe]))
    plt.subplot(rows, cols, nframe + 1)
    plt.imshow(ima[:, :, nchannel] / max_value, cmap='gray')
    plt.axis('off')

plt.show(block=False)

# save channels of all frames as individual images
if False:
    for nframe in range(num_frames):
        multispectral_ima = np.load(os.path.join(path_input, fnames[nframe]))
        for nchannel in range(num_channels):
            ima = np.round((255 * (multispectral_ima[:, :, nchannel] / max_value))).astype(np.uint8)
            imsave(os.path.join(path_output, 'frames', '{:04d}_{:04d}.png'.format(nframe, nchannel)), ima)

# keep only 3 channels per time instant, this is for COLMAP except the undistorsion part
if True:
    channels = [0, 3, 6]
    os.makedirs(os.path.join(path_output, 'input'), exist_ok=True)
    for nframe in range(num_frames):
        multispectral_ima = np.load(os.path.join(path_input, fnames[nframe]))
        rgb_ima = np.round((255 * (multispectral_ima[:, :, channels] / max_value))).astype(np.uint8)
        imsave(os.path.join(path_output, 'input', '{:04d}.png'.format(nframe)), rgb_ima)

# save channels of all frames as individual images, one directory per channel, this is for undistorsion with COLMAP
if True:
    for nframe in range(num_frames):
        multispectral_ima = np.load(os.path.join(path_input, fnames[nframe]))
        for nchannel in range(num_channels):
            path_output_channel = os.path.join(path_output, 'channels_distorted', str(nchannel))
            os.makedirs(path_output_channel, exist_ok=True)
            ima = np.round((255 * (multispectral_ima[:, :, nchannel] / max_value))).astype(np.uint8)
            imsave(os.path.join(path_output_channel,'{:04d}.png'.format(nframe)), ima)