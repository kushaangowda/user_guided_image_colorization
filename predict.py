import torch
import torch.nn as nn
from torch.nn import functional as F
from model.UNet import UNet
from utils.data_loader import create_test_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import os
import shutil
import numpy as np
import cv2
from PIL import Image

def dataload(file_path,batch_size,n_w):
    test_loader = create_test_loader(file_path, batch_size=batch_size, test_size=0.2, 
                                                random_seed=42, n_w=n_w)
    return test_loader

def setup(in_channels,out_channels,n_layers=5,bn_layers=2,model_path=None):
    assert len(in_channels) == n_layers and len(out_channels) == n_layers, \
    'Error: channels should be same as number of layers'

    model = UNet(in_channels=in_channels,out_channels=out_channels,
                    blocks=n_layers,bn_blocks=bn_layers)
                    
    if model_path is not None:
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("Couldn't load model weights")

    return model

def pixelwise_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).float()
    return correct.mean()

def unbin(arr):
    bs, h, w = arr.shape
    arr1 = torch.ones_like(arr)
    arr1 = -95 + arr*10
    return arr1

def save_img(images, labels, outputs, idx):
    orig_gray = images[:,0,:,:].cpu().numpy()*100
    orig_a = unbin(labels[:,0,:,:]).cpu().numpy()
    orig_b = unbin(labels[:,1,:,:]).cpu().numpy()

    _, pred_a = torch.max(outputs[:,0,:,:,:], 1)
    _, pred_b = torch.max(outputs[:,1,:,:,:], 1)
    
    pred_a = unbin(pred_a).cpu().numpy()
    pred_b = unbin(pred_b).cpu().numpy()

    for i in range(len(images)):        
        orig_lab_image = np.stack([orig_gray[i], orig_a[i], orig_b[i]], axis=-1).astype(np.float32)
        rgb_image = cv2.cvtColor(orig_lab_image, cv2.COLOR_LAB2RGB)
        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        image_path = os.path.join("predictions", f'image_orig_{idx}.png')
        Image.fromarray(rgb_image).save(image_path)
        
        pred_lab_image = np.stack([orig_gray[i], pred_a[i], pred_b[i]], axis=-1).astype(np.float32)
        rgb_image = cv2.cvtColor(pred_lab_image, cv2.COLOR_LAB2RGB)
        rgb_image = np.clip(rgb_image, 0, 1)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        image_path = os.path.join("predictions", f'image_pred_{idx}.png')
        Image.fromarray(rgb_image).save(image_path)
        print(f'Saved {image_path}')

    

def predict(test_loader,model,device,top_k=5,num_batches=None):
    if os.path.exists("predictions"):
        shutil.rmtree("predictions")
    os.mkdir("predictions")

    print(f"Using device {device}")
    
    avg_test_loss = 0
    avg_test_acc = 0
    total_test_batch = 0

    test_iterator = iter(test_loader)

    if num_batches is None:
        num_batches = len(test_loader)

    acc_vals = []
    idx = -1

    for i in  tqdm(range(num_batches)):
        model.eval()
        images,labels = next(test_iterator)
        # Move tensors to configured device
        images = images.to(device)
        labels = labels.to(device).long()
        labels = torch.clamp(labels, 0, 19)
        # Calculate accuracy
        outputs = model(images)

        test_acc = pixelwise_accuracy(outputs.view(-1, 20, 256, 256), labels.view(-1, 256, 256)).item()
        
        acc_vals.append(test_acc)

        total_test_batch += 1


    test_iterator = iter(test_loader)

    top_indices = np.argsort(np.array(acc_vals))[-top_k:]

    for i in  range(num_batches):
        images,labels = next(test_iterator)

        if i in top_indices:
            model.eval()
            # Move tensors to configured device
            images = images.to(device)
            labels = labels.to(device).long()
            labels = torch.clamp(labels, 0, 19)
            # Calculate accuracy
            outputs = model(images)

            save_img(images, labels, outputs, i)

    avg_test_acc = avg_test_acc/total_test_batch

    print(
        f'Test Accuracy: {avg_test_acc:.4f}'
    )

# if __name__ == '__main__':
#     train(data_loader,test_loader,model,epochs,device,criterion,optim,local_rank,rank)
