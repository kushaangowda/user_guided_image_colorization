import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.distributed import DistributedSampler

def generate_random_binary_mask(image_shape, k):
    # Unpack the image dimensions
    dim, height, width = image_shape

    # Calculate the total number of pixels in the image
    total_pixels = height * width

    # Calculate the number of pixels to be covered
    num_covered_pixels = int(total_pixels * (k / 100))

    # Generate random indices for the pixels in the image
    random_indices = torch.randperm(total_pixels)[:num_covered_pixels]

    # Convert flat indices to 2D coordinates
    x_indices, y_indices = torch.div(random_indices, width, rounding_mode='trunc'), random_indices % width

    # Create an empty mask of zeros
    mask = torch.zeros((height, width), dtype=torch.uint8)

    # Set the selected positions in the mask to 1
    mask[x_indices, y_indices] = 1

    return torch.unsqueeze(mask, 0)



class ImageDataset(Dataset):
    def __init__(self, file_path, indices):
        self.file = h5py.File(file_path, 'r')
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        group_key = list(self.file)[self.indices[idx]]
        data = self.file[group_key]
        
        input_color = torch.tensor(data['original'][:], dtype=torch.float32).permute(2,0,1) / 255.0  # Normalize
        input_gray = torch.tensor(data['grayscale'][:], dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize and add channel
        output_a = torch.unsqueeze( torch.tensor(data['A_channel'][:], dtype=torch.float32), 0)
        output_b = torch.unsqueeze(torch.tensor(data['B_channel'][:], dtype=torch.float32), 0)
        masks = generate_random_binary_mask(input_color.shape, 20)
        return torch.cat((input_gray, masks*input_color),dim=0), torch.cat((output_a, output_b),dim=0)

    def close(self):
        self.file.close()

def create_loaders(file_path, batch_size=32, test_size=0.2, random_seed=42, n_w=2):
    with h5py.File(file_path, 'r') as file:
        total_images = len(file)
        indices = list(range(total_images))
    
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_seed)
    
    train_dataset = ImageDataset(file_path, train_indices)
    test_dataset = ImageDataset(file_path, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                              sampler=DistributedSampler(train_dataset,shuffle=True), num_workers=n_w)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_test_loader(file_path, batch_size=32, test_size=0.2, random_seed=42, n_w=2):
    with h5py.File(file_path, 'r') as file:
        total_images = len(file)
        indices = list(range(total_images))
    
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_seed)
    
    test_dataset = ImageDataset(file_path, test_indices)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

if __name__ == '__main__':
    file_path = '../image_data.h5'
    train_loader, test_loader = create_loaders(file_path, batch_size=64)

    for (inputs_gray, inputs_mask, inputs_user_img), (targets_a, targets_b) in train_loader:
        print(inputs_gray.shape, inputs_mask.shape, inputs_user_img.shape, targets_a.shape, targets_b.shape)