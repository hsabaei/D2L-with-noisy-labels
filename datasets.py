import os
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from util import other_class  # Adapt this function for PyTorch if needed

# Set random seed
np.random.seed(123)
torch.manual_seed(123)

NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100}

def get_data(dataset='mnist', noise_ratio=0, random_shuffle=False):
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)

    elif dataset == 'cifar-10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset == 'cifar-100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    else:
        return None, None, None, None

    if noise_ratio > 0:
        data_file = f"data/{dataset}_train_labels_{noise_ratio}.npy"
        if os.path.isfile(data_file):
            y_train = np.load(data_file)
        else:
            y_train = np.array(train_set.targets)
            n_samples = y_train.shape[0]
            n_noisy = int(noise_ratio * n_samples / 100)
            noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
            for i in noisy_idx:
                y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
            np.save(data_file, y_train)
            train_set.targets = y_train.tolist()

    if random_shuffle:
        idx_perm = torch.randperm(len(train_set))
        train_set.data = train_set.data[idx_perm]
        train_set.targets = [train_set.targets[i] for i in idx_perm]

    # PyTorch DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    return train_loader, test_loader

def validation_split(dataset, val_split=0.1):
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(123)
    torch.manual_seed(123)

    # Get data loaders
    train_loader, test_loader = get_data(dataset='mnist', noise_ratio=40, random_shuffle=True)

    # Splitting the dataset into training and validation datasets
    train_dataset, val_dataset = validation_split(train_loader.dataset, val_split=0.1)

    # Creating new DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Example: accessing data
    for images, labels in train_loader:
        break

