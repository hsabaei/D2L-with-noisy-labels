import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(input, num_filters, decay, stride=1):
    out = nn.Conv2d(input.size(1), num_filters, kernel_size=3, stride=stride, padding=1, 
                    bias=False)(input)
    out = nn.BatchNorm2d(num_filters)(out)
    out = F.relu(out)
    return out

def residual_block(input, num_filters, decay, increase_filter=False, first_block=False):
    if increase_filter:
        first_stride = 2
    else:
        first_stride = 1

    # Shortcut path
    shortcut = input
    if increase_filter or first_block:
        shortcut = nn.Conv2d(input.size(1), num_filters, kernel_size=1, stride=first_stride, 
                             bias=False)(input)
        shortcut = nn.BatchNorm2d(num_filters)(shortcut)

    # Residual path
    out = conv_block(input, num_filters, decay, stride=first_stride)
    out = conv_block(out, num_filters, decay)

    # Add shortcut value to the main path
    out += shortcut
    out = F.relu(out)
    return out

def cifar100_resnet(depth, num_classes):
    num_conv = 3
    decay = 2e-3
    filters = 16

    # Input layer
    input = torch.zeros([1, 3, 32, 32])  # Example input

    # Initial conv + BN + relu
    out = conv_block(input, filters, decay)

    # First set of residual blocks
    out = residual_block(out, filters, decay, first_block=True)
    for _ in range(1, depth):
        out = residual_block(out, filters, decay)

    # Second set of residual blocks
    filters *= 2
    out = residual_block(out, filters, decay, increase_filter=True)
    for _ in range(1, depth):
        out = residual_block(out, filters, decay)

    # Third set of residual blocks
    filters *= 2
    out = residual_block(out, filters, decay, increase_filter=True)
    for _ in range(1, depth):
        out = residual_block(out, filters, decay)

    # Final layers and classification
    out = nn.BatchNorm2d(filters)(out)
    out = F.relu(out)
    out = F.avg_pool2d(out, kernel_size=8)
    out = out.view(out.size(0), -1)  # Flatten
    out = nn.Linear(out.size(1), num_classes)(out)

    return out

# Create the network
network = cifar100_resnet(5, 100)


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def residual(num_conv, filters, decay, more_filters=False, first=False):
    def f(input):
        in_channels = input.size(1)
        out_channels = filters

        stride = 2 if more_filters and not first else 1

        if not first:
            b = F.relu(nn.BatchNorm2d(in_channels)(input))
        else:
            b = input

        b = nn.Conv2d(in_channels, out_channels, kernel_size=num_conv, stride=stride, padding=1, 
                      bias=False)(b)
        b = F.relu(nn.BatchNorm2d(out_channels)(b))
        res = nn.Conv2d(out_channels, out_channels, kernel_size=num_conv, padding=1, bias=False)(b)

        # Check and match the number of filters for the shortcut
        shortcut = input
        if in_channels != out_channels or stride != 1:
            shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)(input)

        return shortcut + res

    return f