
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    print(size)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y_hat = model(X)
        y = y.to(device)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def get_dataloader(rank, world_size, length=64000, input_dim=512, output_dim=10, batch_size=64):
    # TODO: why is it SO much slower to move entire X, y to xla_device here compared to doing it in batches in the train call?
    #  Check profile.
    X = torch.randn(length, input_dim)
    y = torch.randint(output_dim, (length, ))
    dataset = TensorDataset(X, y)
    dist_sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=dist_sampler), dist_sampler


def main(rank, world_size):
    setup(rank, world_size)

    lr = 1e-3
    n_epochs = 10
    dataloader, sampler = get_dataloader(world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    model = MLP()

    fsdp_model = FullyShardedDataParallel(
        model,
        # cpu_offload=CPUOffload(offload_params=True),
        device_id=rank
    )
    print(fsdp_model.module)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=lr)
    for i in range(n_epochs):
        sampler.set_epoch(i)
        print(f"-----EPOCH: {i}------")
        train(dataloader=dataloader, model=fsdp_model, loss_fn=loss_fn, optimizer=optimizer, device=rank)
        print("\n")
    print("Fin.")

    cleanup()

# TODO: how to check how the model has been sharded? print model?


if __name__ == '__main__':
    world_size = 4
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)
