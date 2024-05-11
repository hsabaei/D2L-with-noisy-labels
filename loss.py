import torch
import torch.nn.functional as F
import numpy as np


def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        # Assuming y_true is not one-hot and y_pred is the output logits.
        y_pred = F.softmax(y_pred, dim=1)  # Convert logits to probabilities
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

        y_pred = torch.clamp(y_pred, 1e-7, 1.0)
        y_true_one_hot = torch.clamp(y_true_one_hot, 1e-4, 1.0)

        # Calculate cross-entropy from both sides
        forward = -torch.mean(torch.sum(y_true_one_hot * torch.log(y_pred), dim=-1))
        backward = -torch.mean(torch.sum(y_pred * torch.log(y_true_one_hot), dim=-1))
        
        return alpha * forward + beta * backward

    return loss


def cross_entropy(y_true, y_pred):
    return F.cross_entropy(y_pred, torch.max(y_true, 1)[1])

def boot_soft(y_true, y_pred, beta=0.95):
    """
    Bootstrapping Soft
    """
    y_pred = F.softmax(y_pred, dim=-1)
    return -torch.sum((beta * y_true + (1. - beta) * y_pred) * torch.log(y_pred), dim=-1).mean()

def boot_hard(y_true, y_pred, beta=0.8):
    """
    Bootstrapping Hard
    """
    y_pred = F.softmax(y_pred, dim=-1)
    pred_labels = F.one_hot(torch.argmax(y_pred, 1), num_classes=y_true.shape[1]).float()
    return -torch.sum((beta * y_true + (1. - beta) * pred_labels) * torch.log(y_pred), dim=-1).mean()

def forward(P):
    """
    Forward loss correction
    """
    P = torch.tensor(P, requires_grad=False)
    def loss(y_true, y_pred):
        y_pred = F.softmax(y_pred, dim=-1)
        return -torch.sum(y_true * torch.log(torch.mm(y_pred, P)), dim=-1).mean()
    return loss

def backward(P):
    """
    Backward loss correction
    """
    P_inv = torch.tensor(np.linalg.inv(P), requires_grad=False)
    def loss(y_true, y_pred):
        y_pred = F.softmax(y_pred, dim=-1)
        return -torch.sum(torch.mm(y_true, P_inv) * torch.log(y_pred), dim=-1).mean()
    return loss

def pairwise_distances(x):
    # Compute pairwise distances in a batch-wise manner
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(x, 0, 1))
    return dist

def lid(logits, k=20):
    """
    Calculate LID for each data point in the array.
    """
    batch_size = logits.size(0)
    k = min(k, batch_size - 1)  # Ensure k is smaller than the batch size

    dist = pairwise_distances(logits)
    dist = torch.sqrt(dist + 1e-15)

    _, indices = torch.topk(-dist, k=k, sorted=True)
    indices = indices[:, 1:]  # Exclude the point itself
    dist_k = torch.gather(dist, 1, indices)

    m = dist_k / dist_k[:, -1, None]
    lids = -k / torch.sum(torch.log(m + 1e-15), dim=1)
    return lids


def lid_paced_loss(alpha=1.0, beta1=0.1, beta2=1.0):
    if alpha == 1.0:
        return symmetric_cross_entropy(alpha=beta1, beta=beta2)
    else:
        def loss(y_true, y_pred):
            pred_labels = F.one_hot(torch.argmax(y_pred, 1), num_classes=y_true.shape[1]).float()
            y_new = alpha * y_true + (1. - alpha) * pred_labels
            y_pred = F.softmax(y_pred, dim=-1)
            y_pred = torch.clamp(y_pred, 1e-7, 1.0)
            return -torch.mean(torch.sum(y_new * torch.log(y_pred), dim=-1))
        return loss
