import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from util import get_lr_scheduler, uniform_noise_model_P
from models import get_model
from loss import cross_entropy, boot_soft, boot_hard, forward, backward, lid_paced_loss
from callback_util import D2LCallback, LoggerCallback

D2L = {'mnist': {'init_epoch': 5, 'epoch_win': 5}, 'svhn': {'init_epoch': 20, 'epoch_win': 5},
       'cifar-10': {'init_epoch': 40, 'epoch_win': 5}, 'cifar-100': {'init_epoch': 60, 'epoch_win': 5}}

# prepare folders
folders = ['data', 'model', 'log']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        self.epochs_since_last_save = 0
        self.best = 0

    def __call__(self, val_loss, model):
        self.epochs_since_last_save += 1
        if self.save_best_only:
            if val_loss < self.best or self.best == 0:
                self.best = val_loss
                self.save_model(model)
        elif self.epochs_since_last_save >= self.period:
            self.save_model(model)
            self.epochs_since_last_save = 0

    def save_model(self, model):
        if self.save_weights_only:
            torch.save(model.state_dict(), self.filepath)
        else:
            torch.save(model, self.filepath)

def extract_data_and_labels(dataloader):
    X = []
    y = []
    for data, labels in dataloader:
        X.append(data)
        y.append(labels)
    return torch.cat(X), torch.cat(y)

def train(dataset='mnist', model_name='d2l', batch_size=128, epochs=50, noise_ratio=0):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param dataset:
    :param model_name:
    :param batch_size:
    :param epochs:
    :param noise_ratio:
    :return:
    """
    print('Dataset: %s, model: %s, batch: %s, epochs: %s, noise ratio: %s%%' %
          (dataset, model_name, batch_size, epochs, noise_ratio))


    # load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset == 'mnist':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'svhn':
        trainset = datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        testset = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    elif dataset == 'cifar-10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'cifar-100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(trainset.classes)
    print("num_classes:", num_classes)

    # load model
    model = get_model(dataset, num_classes=num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # for backward, forward loss
    # suppose the model knows noise ratio
    P = uniform_noise_model_P(num_classes, noise_ratio/100.)
    # create loss
    if model_name == 'forward':
        P = uniform_noise_model_P(num_classes, noise_ratio / 100.)
        criterion = forward(P)
    elif model_name == 'backward':
        P = uniform_noise_model_P(num_classes, noise_ratio / 100.)
        criterion = backward(P)
    elif model_name == 'boot_hard':
        criterion = boot_hard
    elif model_name == 'boot_soft':
        criterion = boot_soft
    elif model_name == 'd2l':
        if dataset == 'cifar-100':
            criterion = lid_paced_loss(beta1=6.0, beta2=0.1)
        else:
            criterion = lid_paced_loss(beta1=0.1, beta2=1.0)
    else:
        criterion = nn.CrossEntropyLoss()

    ## do real-time updates using callbakcs
    callbacks = []
    if model_name == 'd2l':
        init_epoch = D2L[dataset]['init_epoch']
        epoch_win = D2L[dataset]['epoch_win']
        y_train = [label for _, label in train_loader.dataset]  # Extract labels from train loader
        d2l_learning = D2LCallback(model, trainset, y_train, dataset=dataset, noise_ratio=noise_ratio,
                                   epochs=epochs, pace_type=model_name,
                                   init_epoch=init_epoch, epoch_win=epoch_win)
        callbacks.append(d2l_learning)

        model_checkpoint = ModelCheckpoint("model/%s_%s_%s.pth" % (model_name, dataset, noise_ratio),
                                           monitor='val_loss', verbose=0, save_best_only=False,
                                           save_weights_only=True, period=1)
        callbacks.append(model_checkpoint)

    else:
        model_checkpoint = ModelCheckpoint("model/%s_%s_%s.pth" % (model_name, dataset, noise_ratio),
                                           monitor='val_loss', verbose=0, save_best_only=False,
                                           save_weights_only=True, period=epochs)
        callbacks.append(model_checkpoint)

    # learning rate scheduler if use sgd
    lr_scheduler = get_lr_scheduler(optimizer, dataset)  # Pass optimizer to get_lr_scheduler
    callbacks.append(lr_scheduler)

    # acc, loss, lid
    X_train, y_train = extract_data_and_labels(train_loader)
    X_test, y_test = extract_data_and_labels(test_loader)

    log_callback = LoggerCallback(model, X_train, y_train, X_test, y_test, dataset,
                              model_name, noise_ratio, epochs)
    callbacks.append(log_callback)

    # train model
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(targets, outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('[Epoch %d] Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
              (epoch + 1, train_loss / len(train_loader), 100. * correct / total, correct, total))

        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(targets, outputs)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print('[Epoch %d] Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' %
              (epoch + 1, test_loss / len(test_loader), 100. * correct / total, correct, total))

        #for callback in callbacks:
         #   callback.on_epoch_begin(epoch + 1)



def main(args):
    assert args.dataset in ['mnist', 'svhn', 'cifar-10', 'cifar-100'], \
        "dataset parameter must be either 'mnist', 'svhn', 'cifar-10', 'cifar-100'"
    assert args.model_name in ['ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'], \
        "dataset parameter must be either 'ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'"
    train(args.dataset, args.model_name, args.batch_size, args.epochs, args.noise_ratio)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'svhn', 'cifar-10', 'cifar-100'",
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--model_name',
        help="Model name: 'ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'.",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-r', '--noise_ratio',
        help="The percentage of noisy labels [0, 100].",
        required=False, type=int
    )
    parser.set_defaults(epochs=150)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(noise_ratio=0)

    args = parser.parse_args(['-d', 'cifar-10', '-m', 'd2l', '-e', '120', '-b', '128', '-r', '60'])
    main(args)
