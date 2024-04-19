import torch
import torch.nn as nn
from torch.nn import functional as F
from model.UNet import UNet

def train(data_loader,model,epochs,device,criterion,optim,local_rank,rank):
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
    for epoch in range(epochs):
        avg_acc = 0
        avg_loss = 0
        total_batch = 0
        
        data_loader.sampler.set_epoch(epoch)

        print(f"Epoch {epoch + 1}:")
        data_iterator = iter(data_loader)
        try:
            for i in tqdm(range(len(data_loader))):
                model.train()
                images,labels = next(cifar10_iterator)
                # Move tensors to configured device
                images = images.to(device)
                labels = labels.to(device)

                optim.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optim.step()

                model.eval()
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                total = labels.size(0)
                accuracy = 100 * correct / total

                avg_acc += (correct/total)
                avg_loss += loss.item()

                total_batch += 1

            print(f'Proc: {rank} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss/total_batch:.4f}, Accuracy: {avg_acc/total_batch:.2f}, Data loading time: {avg_data_time}, Training Time: {avg_training_time}, Total running time: {avg_total_time}')

if __name__ == '__main__':
    train(data_loader,model,epochs,device,criterion,optim,local_rank,rank)
