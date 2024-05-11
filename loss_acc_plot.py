import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import get_model  
from datasets import get_data  

np.random.seed(1024)
torch.manual_seed(1024)

MODELS = ['ce', 'forward', 'backward', 'boot_soft', 'boot_hard', 'd2l']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']

def test_acc(model_list, dataset='mnist', noise_ratio=0.):
    """
    Test accuracy throughout training.
    """
    print('Dataset: %s, noise ratio: %s%%' % (dataset, noise_ratio))

    # Plot initialization
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model_name in model_list:
        file_name = 'log/acc_%s_%s_%s.npy' % (model_name, dataset, noise_ratio)
        if os.path.isfile(file_name):
            accs = np.load(file_name)
            test_accs = accs[1]

            # Plot line
            idx = MODELS.index(model_name)
            xnew = np.arange(0, len(test_accs), 1)
            test_accs = test_accs[xnew]
            ax.plot(xnew, test_accs, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Test accuracy", fontsize=15)
    legend = plt.legend(loc='lower right', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/test_acc_trend_all_models_%s_%s.png" % (dataset, noise_ratio), dpi=300)
    plt.show()

def test_acc_last_epoch(model_list, dataset='mnist', num_classes=10, noise_ratio=10, epochs=50):
    """
    Test accuracy at the last epoch.
    """
    print('Dataset: %s, epochs: %s, noise ratio: %s%%' % (dataset, epochs, noise_ratio))

    # Load data
    X_test, Y_test = get_data(dataset)
    # Convert class vectors to binary class matrices
    Y_test = F.one_hot(Y_test, num_classes=num_classes)

    # Load model
    for model_name in model_list:
        model = get_model(dataset, num_classes=num_classes)

        # Load model weights
        model_path = 'model/%s_%s_%s.pth' % (model_name, dataset, noise_ratio)
        model.load_state_dict(torch.load(model_path))

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == Y_test).sum().item()
            test_acc = correct / len(Y_test)
            print('Model: %s, Epoch: %s, Test accuracy: %s' % (model_name, epochs - 1, test_acc))

def print_loss_acc_log(model_list, dataset='mnist', noise_ratio=0.1):
    """
    Print loss and accuracy logs.
    """
    print('Dataset: %s, noise ratio: %s' % (dataset, noise_ratio))

    for model_name in model_list:
        loss_file = 'log/loss_%s_%s_%s.npy' % (model_name, dataset, noise_ratio)
        acc_file = 'log/acc_%s_%s_%s.npy' % (model_name, dataset, noise_ratio)
        if os.path.isfile(loss_file):
            losses = np.load(loss_file)
            val_loss = losses[1, -5:]
            print('--------- Val loss ---------')
            print(val_loss)
        if os.path.isfile(acc_file):
            accs = np.load(acc_file)
            print('Epochs: ', len(accs[1]))
            val_acc = accs[1, -5:]
            print('--------- Val accuracy ---------')
            print(val_acc)

if __name__ == "__main__":
    # Example usage
    # test_acc(model_list=['ce'], dataset='cifar-10', noise_ratio=40)
    # test_acc_last_epoch(model_list=['ce'], dataset='cifar-10', num_classes=10, noise_ratio=40, epochs=120)
    print_loss_acc_log(model_list=['boot_hard'], dataset='cifar-100', noise_ratio=0)
