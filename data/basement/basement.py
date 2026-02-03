from PIL import Image
import numpy as np
import os

scene = "dataset_arnau"

# Llista d'imatges per canal
channel_images = [
    "0-NOFILTER.tiff",
    "1-BP850-27.tiff",
    "2-BP635-27.tiff",
    "3-BP590-27.tiff",
    "4-BP525-27.tiff",
    "5-BP505-27.tiff",
    "6-BP470-27.tiff",
    "7-BP324-27.tiff",
    "8-BP550-27.tiff",
]

channels_root = "channels_distorted"
input_dir = "input"

os.makedirs(channels_root, exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

# Guardar PNGs per canal

for c, img_name in enumerate(channel_images):
    channel_dir = os.path.join(channels_root, str(c))
    os.makedirs(channel_dir, exist_ok=True)

    for altura in range(5):
        for a in range(10):
            input_path = f"{scene}/altura{altura}/a{a}/{img_name}"
            output_path = os.path.join(channel_dir, f"00{altura}{a}.png")

            img = Image.open(input_path)

            # Convertir a uint8
            img_np = np.array(img).astype(np.float32)
            img_np = 255 * (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_uint8 = img_np.astype(np.uint8)

            Image.fromarray(img_uint8).save(output_path)

            print(f"saved: {output_path}")


# Crear RGB artificial

rgb_channels = [2, 4, 6]  # R, G, B artificial

EXPOSURE_GAIN = 1.5

ref_dir = os.path.join(channels_root, str(rgb_channels[0]))
filenames = sorted(os.listdir(ref_dir))

for fname in filenames:
    if not fname.endswith(".png"):
        continue

    imgs = []
    for c in rgb_channels:
        img_path = os.path.join(channels_root, str(c), fname)
        img = Image.open(img_path).convert("L")
        img_np = np.array(img).astype(np.float32)

        # Augmentem l'exposició
        img_np *= EXPOSURE_GAIN

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        imgs.append(img_np)

    rgb = np.stack(imgs, axis=-1)  # (H, W, 3)
    rgb_img = Image.fromarray(rgb, mode="RGB")

    output_path = os.path.join(input_dir, fname)
    rgb_img.save(output_path)

    print(f"saved RGB: {output_path}")

print("Done.")