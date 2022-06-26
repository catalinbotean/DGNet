import torch


def get_loss_function():
    return torch.nn.CrossEntropyLoss()
