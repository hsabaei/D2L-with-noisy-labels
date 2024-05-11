import os
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# Set random seed
np.random.seed(123)
torch.manual_seed(123)

# Function Definitions

def lid(logits, k=20):
    epsilon = 1e-12
    batch_size = logits.shape[0]
    r = torch.sum(logits * logits, 1)
    r1 = r.view(-1, 1)
    D = r1 - 2 * torch.matmul(logits, logits.T) + r1.T + torch.ones(batch_size, batch_size, device=logits.device)

    D1 = -torch.sqrt(D)
    D2 = torch.topk(D1, k=k, largest=True).values
    D3 = -D2[:, 1:]

    m = D3.T * (1.0 / D3[:, -1])
    v_log = torch.sum(torch.log(m + epsilon), dim=0)
    lids = -k / v_log
    return lids

def mle_single(data, x, k):
    """
    lid of a single query point x.
    numpy implementation.

    :param data: Tensor of shape (num_samples, feature_dim) representing the dataset
    :param x: Tensor of shape (feature_dim,) representing a single query point
    :param k: Number of nearest neighbors to consider
    :return: Lid value for the single query point
    """
    data = data.float()
    x = x.float()
    if x.dim() == 1:
        x = x.view(1, -1)

    k = min(k, len(data) - 1)
    f = lambda v: - k / torch.sum(torch.log(v / v[-1] + 1e-8))
    a = torch.cdist(x, data)
    a, _ = torch.sort(a, dim=1)
    a = a[:, 1:k + 1]
    a = torch.apply_along_axis(f, axis=1, arr=a)
    return a[0].item()

def mle_batch(data, batch, k):
    """
    lid of a batch of query points X.
    PyTorch implementation.

    :param data: 
    :param batch: 
    :param k: 
    :return: 
    """
    data = torch.tensor(data, dtype=torch.float32)
    batch = torch.tensor(batch, dtype=torch.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / torch.sum(torch.log(v / v[-1] + 1e-8), dim=1)

    # Compute pairwise distances using torch.cdist
    dists = torch.cdist(batch, data)

    # Sort distances and take the first k distances
    sorted_dists, _ = torch.sort(dists, dim=1)
    sorted_dists = sorted_dists[:, 1:k + 1]

    # Apply the function f to the sorted distances
    a = f(sorted_dists)
    return a

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

def get_lids_random_batch(model, X, k=20, batch_size=128):
    """
    Get the local intrinsic dimensionality (LID) of each Xi in X using PyTorch.
    :param model: if None: lid of raw inputs, otherwise LID of deep representations.
    :param X: normal images in PyTorch tensor format.
    :param k: the number of nearest neighbours for LID estimation.
    :param batch_size: default 100.
    :return: lids: LID of normal images of shape (num_examples, lid_dim).
    """
    def mle_batch_torch(data, batch, k):
        # Define your PyTorch version of mle_batch here (as shown in a previous response)
        pass

    X = torch.tensor(X, dtype=torch.float32)  # Ensure X is a PyTorch tensor

    if model is None:
        lids = []
        n_batches = int(np.ceil(X.size(0) / float(batch_size)))
        for i_batch in range(n_batches):
            start = i_batch * batch_size
            end = min(X.size(0), (i_batch + 1) * batch_size)
            X_batch = X[start:end].reshape(end - start, -1)

            lid_batch = mle_batch_torch(X_batch, X_batch, k=k)  # PyTorch mle_batch
            lids.extend(lid_batch)

        lids = torch.stack(lids)
        return lids

    # get deep representations
    lid_dim = len(model.layers)  # Assuming you want the LID of all layers

    def estimate(i_batch):
        start = i_batch * batch_size
        end = min(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = torch.zeros((n_feed, lid_dim))
        X_batch = X[start:end]

        with torch.no_grad():
            for i, layer in enumerate(model.children()):
                X_batch = layer(X_batch)
                X_act = X_batch.view(n_feed, -1)
                lid_batch[:, i] = mle_batch_torch(X_act, X_act, k=k)  # PyTorch mle_batch

        return lid_batch

    lids = []
    n_batches = int(np.ceil(X.size(0) / float(batch_size)))
    for i_batch in range(n_batches):
        lid_batch = estimate(i_batch)
        lids.extend(lid_batch)

    lids = torch.stack(lids)

    return lids

def get_lr_scheduler(optimizer, dataset):
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1

    elif dataset == 'cifar-10':
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1

    elif dataset == 'cifar-100':
        def scheduler(epoch):
            if epoch > 160:
                return 0.0001
            elif epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1

    else:
        # Default scheduler
        def scheduler(epoch):
            return 0.1

    return StepLR(optimizer, step_size=1, gamma=scheduler)

def uniform_noise_model_P(num_classes, noise):
    """
    The noise matrix flips any class to any other with probability
    noise / (num_classes - 1).

    :param num_classes: The number of classes.
    :param noise: The noise level.
    :return: A PyTorch tensor representing the noise matrix.
    """
    assert (noise >= 0.) and (noise <= 1.)

    P = (noise / (num_classes - 1)) * torch.ones((num_classes, num_classes))
    diag = (1 - noise) * torch.ones(num_classes)
    P = P * (1 - torch.eye(num_classes)) + torch.eye(num_classes) * diag

    # Assert that the sum of each row is approximately 1
    assert torch.allclose(P.sum(dim=1), torch.ones(num_classes), atol=1e-1)

    return P



def get_deep_representations(model, X, batch_size=128):
    """
    Get the deep representations before logits.
    :param model: PyTorch model
    :param X: Tensor of shape (num_samples, input_dim) representing the input data
    :param batch_size: Batch size for processing
    :return: Tensor of shape (num_samples, output_dim) containing the deep representations
    """
    output_dim = model.layers[-3].output.shape[-1].value
    get_encoding = torch.jit.trace(model, X)

    n_batches = int(torch.ceil(X.shape[0] / float(batch_size)))
    output = torch.zeros(len(X), output_dim)
    for i in range(n_batches):
        start = i * batch_size
        end = min(len(X), (i + 1) * batch_size)
        output[start:end] = get_encoding(X[start:end])

    return output


