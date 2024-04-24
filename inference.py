import torch
import torch.nn as nn
from torch.nn import functional as F
from model.UNet import UNet
from utils.data_loader import create_loaders
from tqdm import tqdm

def inference(file_path,in_channels,out_channels,n_layers,bn_layers,model_path=None):
    _, test_loader = create_loaders(file_path, batch_size=4, test_size=0.2, 
                                                random_seed=42, n_w=2)
    model = UNet(in_channels=in_channels,out_channels=out_channels,
                    blocks=n_layers,bn_blocks=bn_layers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_path is not None:
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("Error: Failed to load model")
            exit(0)
            
    test_iterator = iter(test_loader)
    model.eval()
    images,labels = next(test_iterator)
    # Move tensors to configured device
    images = images.to(device)
    labels = labels.to(device).long().view(-1, 256, 256)
    labels = torch.clamp(labels, 0, 19)
    # Calculate accuracy
    outputs = model(images).view(-1, 20, 256, 256)