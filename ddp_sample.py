import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)

    print('| distributed init (rank {}), gpu {}'.format(args.rank, args.local_rank), flush=True)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=60*12))

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(args):
    init_distributed_mode(args)
    local_rank = args.local_rank
    global_rank= args.rank
    num_tasks=args.world_size
    dataset_train=torch.ones(4096,10)

    print(f"Running basic DDP example on rank {local_rank}.")

    # create model and move it to GPU with id rank
    model = ToyModel().cuda()
    ddp_model = DDP(model, device_ids=[local_rank])

    sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    batch_size=64
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=False,
        drop_last=True,persistent_workers=True
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    for epoch in range(100):
        data_loader_train.sampler.set_epoch(epoch)
        for i,data in enumerate(data_loader_train):
            optimizer.zero_grad()
            outputs = ddp_model(data.cuda())
            labels = torch.randn(batch_size, 5).cuda()
            loss=loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if global_rank==0:
                print(epoch,i,loss.item())
    
    print("Done!!")
    cleanup()


def parse_args():
    return argparse.Namespace()

if __name__=="__main__":
    args = parse_args()
    demo_basic(args)

