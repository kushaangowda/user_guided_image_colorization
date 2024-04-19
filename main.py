""" Import the dependencies"""
import torch
import argparse
import os
from torch.distributed import init_process_group, destroy_process_group
from train import train

def main(world_size,rank,local_rank,epochs,batch_size,n_workers=2):

    if not torch.cuda.is_available():
        print("Error: Distrbuted training is not supported without GPU")

    # init the process group for DDL
    init_process_group(backend='nccl',rank=rank,world_size=world_size)
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda')
    data_loader = dataload(batch_size,n_w) # load the data
    (model,criterion,optim) = setup(optim_name,device) # setup the model and the hyperparameters
    train(data_loader,model,epochs,device,criterion,optim,local_rank,rank):

    destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Image colorization training")
    parser.add_argument('--batch_size', default=16, type=int, help='per GPU')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=0, type=int)

    args = parser.parse_args()

    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args.batch_size
    epochs = args.epochs
    main(args.world_size,args.rank,args.local_rank,epochs,batch_size)