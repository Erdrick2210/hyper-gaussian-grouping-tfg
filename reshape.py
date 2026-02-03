from PIL import Image
import numpy as np
import cv2
import os


# Returns the image shape (png or npy)
def get_shape(img_route):
    print(img_route)
    try:
        if img_route.lower().endswith(".png"):
            img = Image.open(img_route)
            w, h = img.size
            return w, h
        elif img_route.lower().endswith(".npy"):
            arr = np.load(img_route)
            c, h, w = arr.shape
            return w, h, c
        else:
            return "Error: unsupported file format (must be .png or .npy)."
    except FileNotFoundError:
        return "Error: img does not exist in specified route."
    except Exception as e:
        return f"Error: {e}"


# Rescales the image to half
def reshape(img_route, output_route):
    try:
        with Image.open(img_route) as img:
            w, h = img.size
            if w < 1600:
                return
            new_w = w // 2
            new_h = h // 2

            img_reshaped = img.resize((new_w, new_h), Image.LANCZOS)

            img_reshaped.save(output_route)
            print(f"Img saved in: {output_route} ({new_w}x{new_h})")

    except Exception as e:
        print(f"Error processing {img_route}: {e}")


# Rescales the image .png to width
def resize(img_route, output_route, width=1200):
    img = Image.open(img_route)

    orig_w, orig_h = img.size
    scale = orig_w / width
    resolution = (int(orig_w / scale), int(orig_h / scale))

    img_resized = img.resize(resolution, Image.LANCZOS)

    img_resized.save(output_route)
    print(f"Img resized and saved in: {output_route} ({resolution[0]}x{resolution[1]})")


# Rescales the image .npy to width
def resize_npy(npy_route, output_route, width=1200):
    try:
        arr = np.load(npy_route)  # (C, H, W)
        
        if arr.ndim == 3 and arr.shape[0] <= 50:
            C, H, W = arr.shape
        else:
            raise ValueError(f"Unexpected shape: {arr.shape}")
        
        scale = max(W / 1200, 1.0) # 1600
        new_W = int(W / scale)
        new_H = int(H / scale)

        resized_channels = []
        for c in range(C):
            resized_c = cv2.resize(arr[c, :, :], (new_W, new_H), interpolation=cv2.INTER_AREA)
            resized_channels.append(resized_c)

        resized = np.stack(resized_channels, axis=0)  # (C, new_H, new_W)
        np.save(output_route, resized)

        print(f"Array resized and saved in: {output_route} ({new_W}x{new_H})")

    except Exception as e:
        print(f"Error processing {npy_route}: {e}")


# Process more than 1 folder
def process_folders(base_input, base_output, divide_by_2=False, npy=False):
    for folder in os.listdir(base_input):
        folder_route = os.path.join(base_input, folder)
        if os.path.isdir(folder_route):

            output_route = os.path.join(base_output, folder)
            os.makedirs(output_route, exist_ok=True)

            for file_name in os.listdir(folder_route):
                if file_name.lower().endswith(".png") or file_name.lower().endswith(".npy"):
                    img_route = os.path.join(folder_route, file_name)
                    output_img_route = os.path.join(output_route, file_name)
                    if divide_by_2:
                        reshape(img_route, output_img_route)
                    elif npy:
                        resize_npy(img_route, output_img_route)
                    else:
                        resize(img_route, output_img_route)


# Process 1 folder
def process_folder(route):
    for file_name in os.listdir(route):
        img_route = os.path.join(route, file_name)
        if file_name.lower().endswith(".png"):
            resize(img_route, img_route)
        if file_name.lower().endswith(".npy"):
            resize_npy(img_route, img_route)


route = "data/basement15/images"

#print(get_shape("data/basement15/images/0000.npy"))

process_folder(route)

print("Done")