import torch
from torch.nn import Sequential, Linear


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.net = Sequential(
            Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)
