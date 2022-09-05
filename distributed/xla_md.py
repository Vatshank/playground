import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


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
    # size = len(dataloader.dataset)
    # print(size)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y_hat = model(X)
        y = y.to(device)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # xm.mark_step()
        xm.optimizer_step(optimizer)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}  [{current:>5d}/UNKNOWN]")


def get_dataloader(length=64000, input_dim=512, output_dim=10, batch_size=64):
    # TODO: why is it SO much slower to move entire X, y to xla_device here compared to doing it in batches in the train call?
    #  Check profile.
    X = torch.randn(length, input_dim)#.to(device)
    y = torch.randint(output_dim, (length, ))#.to(device)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main(index):
    device = xm.xla_device()
    lr = 1e-3
    n_epochs = 10
    dataloader = get_dataloader()
    mp_device_loader = pl.MpDeviceLoader(dataloader, device)
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for i in range(n_epochs):
        print(f"-----EPOCH: {i}------")
        train(dataloader=mp_device_loader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        print("\n")
    print("Fin.")


if __name__ == '__main__':
    xmp.spawn(main, args=())
