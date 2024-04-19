import numpy as np
from PIL import Image
from skimage import color
import h5py
import os
from tqdm import tqdm

def process_images(image_folder):
    with h5py.File('../image_data.h5', 'w') as hf:
        file_paths = os.listdir(image_folder)[:1000]
        for image_filename in tqdm(file_paths):
            try:
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(image_folder, image_filename)
                    
                    img = Image.open(image_path)
                    img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)
                    img_array = np.array(img_resized)
                    img_gray = np.array(img_resized.convert('L'))
                    img_lab = color.rgb2lab(img_resized)
                    img_a = img_lab[:, :, 1]
                    img_b = img_lab[:, :, 2]

                    a_bins = np.arange(-100, 100, 10)
                    b_bins = np.arange(-100, 100, 10)
                    img_a_binned = np.digitize(img_a, a_bins) - 1
                    img_b_binned = np.digitize(img_b, b_bins) - 1

                    group = hf.create_group(image_filename)
                    group.create_dataset('original', data=img_array)
                    group.create_dataset('grayscale', data=img_gray)
                    group.create_dataset('A_channel', data=img_a_binned)
                    group.create_dataset('B_channel', data=img_b_binned)
            except:
                pass

if __name__ == '__main__':

    process_images('../data/')