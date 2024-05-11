import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setting a random seed for numpy for reproducibility
np.random.seed(1024)

# Defining model identifiers, labels, and plot styling
MODELS = ['ce', 'forward', 'backward', 'boot_soft', 'boot_hard', 'lid_dataset']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']

# Import your adapted modules for PyTorch
from datasets import get_data
from models import get_model
from util import mle_batch

def load_lid_and_acc_data(model_name, dataset, noise_ratio):
    lid_path = f"log/lid_{model_name}_{dataset}_{noise_ratio}.npy"
    acc_path = f"log/acc_{model_name}_{dataset}_{noise_ratio}.npy"

    if os.path.isfile(lid_path) and os.path.isfile(acc_path):
        lids = np.load(lid_path)
        acc_data = np.load(acc_path)
        acc_train = acc_data[0]
        acc_test = acc_data[1]
        return lids, acc_train, acc_test
    return None, None, None

def lid_trend_through_training(model_name='ce', dataset='mnist', noise_ratio=0., device='cuda'):
    print(f'Dataset: {dataset}, noise ratio: {noise_ratio}%')
    
    train_data, _ = get_data(dataset)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    X_train_sample, _ = next(iter(train_loader))
    lid_X = mle_batch(X_train_sample, k=20)  # Assuming mle_batch is adapted for PyTorch

    lids, acc_train, acc_test = load_lid_and_acc_data(model_name, dataset, noise_ratio)
    if lids is not None:
        plot(model_name, dataset, noise_ratio, lids, acc_train, acc_test)

def plot(model_name, dataset, noise_ratio, lids, acc_train, acc_test):
    fig, ax1 = plt.subplots()
    x_axis = np.arange(len(lids))

    ax1.plot(x_axis, lids, 'r-', label='LID score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('LID score', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(x_axis, acc_train, 'b--', label='Training Accuracy')
    ax2.plot(x_axis, acc_test, 'g-.', label='Test Accuracy')
    ax2.set_ylabel('Accuracy', color='b')
    ax2.tick_params('y', colors='b')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f'LID Trend for {model_name} on {dataset} with noise ratio {noise_ratio}')
    plt.savefig(f"plots/lid_trend_{model_name}_{dataset}_{noise_ratio}.png")
    plt.show()

def lid_trend_of_learning_models(model_list, dataset, noise_ratio):
    fig, ax = plt.subplots()

    for model_name in model_list:
        lids, acc_train, acc_test = load_lid_and_acc_data(model_name, dataset, noise_ratio)
        if lids is not None:
            x_axis = np.arange(0, len(lids), 1)
            idx = MODELS.index(model_name)
            ax.plot(x_axis, lids, color=COLORS[idx], marker=MARKERS[idx], label=MODEL_LABELS[idx])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("LID score", fontsize=15)
    ax.legend(loc='best')
    plt.savefig(f"plots/lid_trend_all_models_{dataset}_{noise_ratio}.png")
    plt.show()


if __name__ == "__main__":
    lid_trend_through_training('ce', 'cifar-100', 0.0)
    lid_trend_of_learning_models(['ce', 'forward', 'backward', 'boot_soft', 'boot_hard'], 'cifar-100', 0.0)
