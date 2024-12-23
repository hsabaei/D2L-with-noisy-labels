"""
Train test error/accuracy/loss plot.

Author: Xingjun Ma
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datasets import get_data
from models import get_model
from loss import cross_entropy
from lass_tf import lass

torch.manual_seed(1024)

MODELS = ['ce', 'forward', 'backward', 'boot_soft', 'boot_hard', 'd2l']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']

def complexity_plot(model_list, dataset='mnist', num_classes=10, noise_ratio=10, epochs=50, n_samples=500):
    """
    The complexity (Critical Sample Ratio) of the hypothesis learned throughout training.
    """
    print('Dataset: %s, epochs: %s, noise ratio: %s%%' % (dataset, epochs, noise_ratio))

    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)
    xnew = np.arange(0, epochs, 5)

    # load data
    _, _, _, X_test, Y_test = get_data(dataset)
    Y_test = torch.tensor(Y_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    shuffle = torch.randperm(X_test.size(0))
    X_test = X_test[shuffle]
    Y_test = Y_test[shuffle]
    X_test = X_test[:n_samples]
    Y_test = Y_test[:n_samples]

    # load model
    image_shape = X_test.shape[1:]
    model = get_model(dataset, input_shape=image_shape)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for model_name in model_list:
        file_name = f"log/crs_{model_name}_{dataset}_{noise_ratio}.npy"
        if os.path.isfile(file_name):
            crs = np.load(file_name)
            for i in xnew:
                crs[i] = np.mean(crs[i:i+5])
            crs = crs[xnew]
            ax.plot(xnew, crs, c=COLORS[MODELS.index(model_name)], marker=MARKERS[MODELS.index(model_name)],
                    markersize=3, linewidth=2, label=MODEL_LABELS[MODELS.index(model_name)])
            continue

        crs = np.zeros(epochs)
        for i in range(epochs):
            # the critical sample ratio of the representations learned at every epoch
            # need to save those epochs first, in this case, use separate folders for each model
            model_path = f"model/{model_name}/{dataset}_{noise_ratio}.{i:02d}.pth"
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # LASS to estimate the critical sample ratio
            scale_factor = 255. / (X_test.max() - X_test.min())
            csr_model = lass(model, X_test, Y_test, a=0.25 / scale_factor, b=0.2 / scale_factor,
                             r=0.3 / scale_factor, iter_max=100)
            X_adv, adv_ind = csr_model.find(bs=500)
            crs[i] = adv_ind.float().mean().item()

            print(f'model: {model_name}, epoch: {i}, CRS: {crs[i]}')

        # save result to avoid recomputing
        np.save(file_name, crs)

        ax.plot(xnew, crs[xnew], c=COLORS[MODELS.index(model_name)], marker=MARKERS[MODELS.index(model_name)],
                markersize=3, linewidth=2, label=MODEL_LABELS[MODELS.index(model_name)])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Hypothesis complexity (CSR score)", fontsize=15)
    legend = plt.legend(loc='upper left')
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig(f"plots/complexity_trend_all_models_{dataset}_{noise_ratio}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # mnist: epoch=50, cifar-10: epoch=120
    complexity_plot(model_list=['ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'],
                    dataset='cifar-10', num_classes=10, noise_ratio=60, epochs=120, n_samples=500)
