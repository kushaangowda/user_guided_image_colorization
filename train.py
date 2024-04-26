import torch
import torch.nn as nn
from torch.nn import functional as F
from model.UNet import UNet
from utils.data_loader import create_loaders
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import datetime
import numpy as np
import wandb

def dataload(file_path,batch_size,n_w):
    train_loader, test_loader = create_loaders(file_path, batch_size=batch_size, test_size=0.2, 
                                                random_seed=42, n_w=n_w)
    return train_loader, test_loader

def setup(lr,wd,in_channels,out_channels,n_layers=5,bn_layers=2,num_bins=40,model_path=None):
    assert len(in_channels) == n_layers and len(out_channels) == n_layers, \
    'Error: channels should be same as number of layers'

    model = UNet(in_channels=in_channels,out_channels=out_channels,
                    blocks=n_layers,bn_blocks=bn_layers,num_bins=num_bins)
                    
    if model_path is not None:
        try:
            print("Loading model")
            model.load_state_dict(torch.load(model_path))
        except:
            print("Couldn't load model weights")

    optim = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    data = np.load("class_weights.npy")
    criterion1 = nn.CrossEntropyLoss(weight=torch.Tensor(data[0]).to("cuda"))
    criterion2 = nn.CrossEntropyLoss(weight=torch.Tensor(data[1]).to("cuda"))
    return model,[criterion1,criterion2],optim

def pixelwise_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).float()
    return correct.mean()

def train(data_loader,test_loader,model,epochs,device,criteria,optim,local_rank,rank,num_bins,l_mask=10):
    # x = torch.rand(4,4,256,256)
    # in_channels = [4,64,64]
    # out_channels = [64,64,128]
    # blocks = len(in_channels)
    # model = UNet(in_channels,out_channels,blocks=blocks,bn_blocks=2)
    # out = model(x)
    # print(x.shape,out.shape)

    print(f"Proc {rank} using device {device}")
    model = DDP(model,device_ids=[local_rank])
    total_step = len(data_loader)
    best_test_acc = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(epochs):
        avg_train_loss = 0
        avg_test_loss = 0
        avg_train_acc = 0
        avg_test_acc = 0
        total_train_batch = 0
        total_test_batch = 0
        
        data_loader.sampler.set_epoch(epoch)

        print(f"Epoch {epoch + 1}:")
        data_iterator = iter(data_loader)

        for i in tqdm(range(len(data_loader)),desc='Training',disable=(rank != 0)):
            model.train()
            images,labels = next(data_iterator)
            # Move tensors to configured device
            images = images.to(device)
            labels = labels.to(device).long()
            labels = torch.clamp(labels, 0, num_bins-1)
            optim.zero_grad()

            masks = images[:,-1].type(torch.bool)
            images = images[:,:-1]

            masked_labels_0 = labels[:,0].clone()
            masked_labels_0[~masks] = -1 
            masked_labels_1 = labels[:,1].clone()
            masked_labels_1[~masks] = -1 

            # Forward pass
            outputs = model(images)
            loss = criteria[0](outputs[:,0], labels[:,0]) + criteria[1](outputs[:,1], labels[:,1])
            # loss += l_mask * (criteria[0](outputs[:,0], masked_labels_0) + criteria[1](outputs[:,1], masked_labels_1))

            # Backward and optimize
            loss.backward()
            optim.step()

            model.eval()
            # Calculate accuracy and loss
            avg_train_loss += loss.item()

            train_acc = pixelwise_accuracy(outputs.reshape(-1, num_bins, 256, 256), labels.reshape(-1, 256, 256))
            avg_train_acc += train_acc.item()

            total_train_batch += 1

        test_iterator = iter(test_loader)
        for i in  tqdm(range(len(test_loader)),desc='Testing'):
            model.eval()
            images,labels = next(test_iterator)
            # Move tensors to configured device
            images = images.to(device)
            labels = labels.to(device).long()
            labels = torch.clamp(labels, 0, num_bins-1)

            masks = images[:,-1].type(torch.bool)
            images = images[:,:-1]

            masked_labels_0 = labels[:,0].clone()
            masked_labels_0[~masks] = -1 
            masked_labels_1 = labels[:,1].clone()
            masked_labels_1[~masks] = -1 

            # Calculate accuracy
            outputs = model(images)
            loss = criteria[0](outputs[:,0], labels[:,0])+criteria[1](outputs[:,1], labels[:,1])
            # loss += l_mask * (criteria[0](outputs[:,0], masked_labels_0) + criteria[1](outputs[:,1], masked_labels_1))

            avg_test_loss += loss.item()

            test_acc = pixelwise_accuracy(outputs.reshape(-1, num_bins, 256, 256), labels.reshape(-1, 256, 256))
            avg_test_acc += test_acc.item()

            total_test_batch += 1

        avg_train_acc = avg_train_acc/total_train_batch
        avg_train_loss = avg_train_loss/total_train_batch
        avg_test_acc = avg_test_acc/total_test_batch
        avg_test_loss = avg_test_loss/total_test_batch
        if rank == 0:
            print(
                f'Proc: {rank} Epoch [{epoch+1}/{epochs}], \
                Training Accuracy: {avg_train_acc:.4f}, \
                Training Loss: {avg_train_loss:.4f}, \
                Test Accuracy: {avg_test_acc:.4f}, \
                Test Loss: {avg_test_loss:.4f}'
            )

            wandb.log({
                "Average Training Loss": avg_train_loss,
                "Average Training Accuracy": avg_train_acc,
                "Average Test Loss": avg_test_loss,
                "Average Test Accuracy": avg_test_acc,
                "epoch": epoch
            })


        if avg_test_acc > best_test_acc and rank == 0:
            best_test_acc = avg_test_acc
            torch.save(model.module.state_dict(), f'best_model_{timestamp}.pth')
            print(f'Best model saved with Test Acc: {avg_test_acc:.4f}')

# if __name__ == '__main__':
#     train(data_loader,test_loader,model,epochs,device,criterion,optim,local_rank,rank)
