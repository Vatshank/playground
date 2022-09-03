import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(size)
    for batch, (X, y) in enumerate(dataloader):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def get_dataloader(length=64000, input_dim=512, output_dim=10, batch_size=64):
    X = torch.randn(length, input_dim)
    y = torch.randint(output_dim, (length, ))
    # y = nn.functional.one_hot(y_ind, output_dim)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    lr = 1e-3
    n_epochs = 10
    dataloader = get_dataloader()
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for i in range(n_epochs):
        print(f"-----EPOCH: {i}------")
        train(dataloader=dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        print("\n")
    print("Fin.")


if __name__ == '__main__':
    main()
