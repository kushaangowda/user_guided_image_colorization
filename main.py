""" Import the dependencies"""
import torch
import argparse
import os
from torch.distributed import init_process_group, destroy_process_group
from train import train, dataload, setup

def main(file_path,in_channels,out_channels,lr,wd,world_size,rank,local_rank,epochs,batch_size,n_workers=2):

    if not torch.cuda.is_available():
        print("Error: Distrbuted training is not supported without GPU")

    # init the process group for DDL
    init_process_group(backend='nccl',rank=rank,world_size=world_size)
    torch.cuda.set_device(local_rank)

    device = torch.device('cuda')
    data_loader, test_loader = dataload(file_path,batch_size,n_workers) # load the data
    (model,criterion,optim) = setup(
                                lr,wd,in_channels,out_channels,
                                n_layers=4,bn_layers=1,
                                model_path="best_model_proc_0.pth"
                            ) # setup the model and the hyperparameters
    model = model.to(device)
    train(data_loader,test_loader,model,epochs,device,criterion,optim,local_rank,rank)

    destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Image colorization training")
    parser.add_argument('--batch_size', default=16, type=int, help='per GPU')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=0, type=int)
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--workers',default=1,type=int)
    parser.add_argument('--lr',default=1e-4,type=float)
    parser.add_argument('--wd',default=1e-3,type=float)

    args = parser.parse_args()

    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    batch_size = args.batch_size
    epochs = args.epochs
    # in_channels = [4,32,64,128,256]
    # out_channels = [32,64,128,128,256]
    in_channels = [4,8,16,32]
    out_channels = [8,16,32,64]

    main(args.data_path,in_channels,out_channels,args.lr,args.wd,args.world_size,
            args.rank,args.local_rank,epochs,batch_size,n_workers=args.workers)