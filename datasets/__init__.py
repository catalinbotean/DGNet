import torch

from torchvision import datasets
from transforms import get_transforms

transform = get_transforms()
training_set = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))


def get_training_loader():
    return torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)


def get_validation_loader():
    return torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)
