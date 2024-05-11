import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

from util import get_deep_representations
from models import get_model
from loss import cross_entropy

np.random.seed(1234)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def feature_visualization(model_name='ce', dataset='mnist',
                          num_classes=10, noise_ratio=40, n_samples=100):
    print('Dataset: %s, model_name: ce/%s, noise ratio: %s%%' % (model_name, dataset, noise_ratio))
    features_ce = [None, None]
    features_other = [None, None]

    # Load CIFAR-10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    # Sample training set
    cls_a = 0
    cls_b = 3
    cls_a_idx = torch.where(trainset.targets == cls_a)[0]
    cls_b_idx = torch.where(trainset.targets == cls_b)[0]
    cls_a_idx = np.random.choice(cls_a_idx.numpy(), n_samples, replace=False)
    cls_b_idx = np.random.choice(cls_b_idx.numpy(), n_samples, replace=False)

    X_a = trainset.data[cls_a_idx]
    X_b = trainset.data[cls_b_idx]

    model = get_model(dataset, input_tensor=None, input_shape=(3, 32, 32))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #### Get deep representations of CE model
    model_path = 'model/ce_%s_%s.pth' % (dataset, noise_ratio)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rep_a = get_deep_representations(model, X_a)
    rep_b = get_deep_representations(model, X_b)

    rep_a = TSNE(n_components=2).fit_transform(rep_a)
    rep_b = TSNE(n_components=2).fit_transform(rep_b)
    features_ce[0] = rep_a
    features_ce[1] = rep_b

    #### Get deep representations of other model
    model_path = 'model/%s_%s_%s.pth' % (model_name, dataset, noise_ratio)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    rep_a = get_deep_representations(model, X_a)
    rep_b = get_deep_representations(model, X_b)

    rep_a = TSNE(n_components=2).fit_transform(rep_a)
    rep_b = TSNE(n_components=2).fit_transform(rep_b)
    features_other[0] = rep_a
    features_other[1] = rep_b

    # Plot
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.15)

    a_clean_idx = (trainset.targets[cls_a_idx].numpy() == trainset.targets[cls_a_idx].numpy())
    a_noisy_idx = (trainset.targets[cls_a_idx].numpy() != trainset.targets[cls_a_idx].numpy())
    b_clean_idx = (trainset.targets[cls_b_idx].numpy() == trainset.targets[cls_b_idx].numpy())
    b_noisy_idx = (trainset.targets[cls_b_idx].numpy() != trainset.targets[cls_b_idx].numpy())

    ## Plot features learned by cross-entropy
    ax = fig.add_subplot(gs[0, 0])
    A = features_ce[0]
    B = features_ce[1]
    ax.scatter(A[a_clean_idx][:, 0].ravel(), A[a_clean_idx][:, 1].ravel(), c='b', marker='o', s=10, label='class A: clean')
    ax.scatter(A[a_noisy_idx][:, 0].ravel(), A[a_noisy_idx][:, 1].ravel(), c='m', marker='x', s=30, label='class A: noisy')
    ax.scatter(B[b_clean_idx][:, 0].ravel(), B[b_clean_idx][:, 1].ravel(), c='r', marker='o', s=10, label='class B: clean')
    ax.scatter(B[b_noisy_idx][:, 0].ravel(), B[b_noisy_idx][:, 1].ravel(), c='c', marker='x', s=30, label='class B: noisy')

    ax.set_title("cross-entropy", fontsize=15)
    legend = ax.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)

    ax = fig.add_subplot(gs[0, 1])
    A = features_other[0]
    B = features_other[1]
    ax.scatter(A[a_clean_idx][:, 0].ravel(), A[a_clean_idx][:, 1].ravel(), c='b', marker='o', s=10, label='class A: clean')
    ax.scatter(A[a_noisy_idx][:, 0].ravel(), A[a_noisy_idx][:, 1].ravel(), c='m', marker='x', s=30, label='class A: noisy')
    ax.scatter(B[b_clean_idx][:, 0].ravel(), B[b_clean_idx][:, 1].ravel()-5, c='r', marker='o', s=10, label='class B: clean')
    ax.scatter(B[b_noisy_idx][:, 0].ravel(), B[b_noisy_idx][:, 1].ravel(), c='c', marker='x', s=30, label='class B: noisy')

    ax.set_title("D2L", fontsize=15)
    legend = ax.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)

    plt.savefig("plots/representations_%s_%s_%s.png" % (model_name, dataset, noise_ratio), dpi=300)
    plt.show()

if __name__ == "__main__":
    feature_visualization(model_name='d2l', dataset='cifar-10', num_classes=10, noise_ratio=60, n_samples=500)
