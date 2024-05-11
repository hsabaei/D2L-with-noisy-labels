import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import cifar100_resnet  

class MNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SVHNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR100Model(CIFAR10Model):
    def __init__(self, num_classes=100):
        super(CIFAR100Model, self).__init__(num_classes)
        # We'll modify the last layer of the classifier to output 100 classes
        # The rest of the architecture remains the same as CIFAR-10
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, num_classes)  # Adjust for 100 classes
        )

    def forward(self, x):
        # The forward pass remains the same as CIFAR-10
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def get_model(dataset='mnist', num_classes=10):
    if dataset == 'mnist':
        return MNISTModel(num_classes=num_classes)
    elif dataset == 'svhn':
        return SVHNModel(num_classes=num_classes)
    elif dataset == 'cifar-10':
        return CIFAR10Model(num_classes=num_classes)
    elif dataset == 'cifar-100':
        return CIFAR100Model(num_classes=num_classes)
    else:
        raise ValueError("Unsupported dataset")
    


