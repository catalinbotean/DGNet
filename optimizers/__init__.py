import torch
import torch.nn as nn


def get_optimizer(network: nn.Module):
    return torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
