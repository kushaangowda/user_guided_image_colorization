import numpy as np
from PIL import Image
from skimage import color
import h5py
import os
from tqdm import tqdm

def process_images(image_folder, num_bins=64, total_images=1000):
    freq_a = {}
    freq_b = {}
    count = 0
    with h5py.File('../image_data_1k.h5', 'w') as hf:
        file_paths = os.listdir(image_folder)
        for image_filename in tqdm(file_paths):
            if count == total_images:
                break
            try:
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(image_folder, image_filename)
                    
                    img = Image.open(image_path)
                    img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img_lab = color.rgb2lab(img_resized)
                    img_gray = img_lab[:, :, 0]
                    img_a = img_lab[:, :, 1]
                    img_b = img_lab[:, :, 2]

                    if np.max(img_gray) > 100 or np.min(img_gray) < 0 or \
                       np.max(img_a) > 127 or np.min(img_a) < -128 or \
                       np.max(img_b) > 127 or np.min(img_b) < -128:
                        raise Exception("Error: Image not in LAB color space")

                    a_bins = np.arange(-128, 128, int(256/num_bins))
                    b_bins = np.arange(-128, 128, int(256/num_bins))
                    img_a_binned = np.digitize(img_a, a_bins) - 1
                    img_b_binned = np.digitize(img_b, b_bins) - 1

                    img_a_binned = np.clip(img_a_binned, 0, num_bins-1)
                    img_b_binned = np.clip(img_b_binned, 0, num_bins-1)

                    perc_pixels = 0.7
                    bin_range = [14,18]

                    if np.sum((img_a_binned >= bin_range[0]) & (img_a_binned <= bin_range[1])) >= perc_pixels*(256*256) \
                    and np.sum((img_b_binned >= bin_range[0]) & (img_b_binned <= bin_range[1])) >= perc_pixels*(256*256):
                        continue

                    unique, counts = np.unique(img_a_binned, return_counts=True)
                    for u, c in zip(unique, counts):
                        if u in freq_a:
                            freq_a[u] += c
                        else:
                            freq_a[u] = c

                    unique, counts = np.unique(img_b_binned, return_counts=True)
                    for u, c in zip(unique, counts):
                        if u in freq_b:
                            freq_b[u] += c
                        else:
                            freq_b[u] = c

                    group = hf.create_group(image_filename)
                    group.create_dataset('original', data=np.array(img_resized))
                    group.create_dataset('grayscale', data=img_gray)
                    group.create_dataset('A_channel', data=img_a_binned)
                    group.create_dataset('B_channel', data=img_b_binned)

                    count += 1
            except Exception as e:
                pass
    
    weights_a = []
    for i in range(num_bins):
        try:
            weights_a.append(1.0/freq_a[i])
        except:
            weights_a.append(1e-10)
    weights_a = weights_a / sum(weights_a)
    
    weights_b = []
    for i in range(num_bins):
        try:
            weights_b.append(1.0/freq_b[i])
        except:
            weights_b.append(1e-10)
    weights_b = weights_b / sum(weights_b)
    
    np.save('../class_weights.npy', np.array([weights_a, weights_b]))


if __name__ == '__main__':

    process_images('../data/', num_bins=32)