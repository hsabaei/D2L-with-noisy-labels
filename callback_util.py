import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from util import get_lids_random_batch
from loss import cross_entropy, lid_paced_loss
from lass_tf import lass

class D2LCallback(object):
    def __init__(self, model, X_train, y_train, dataset, noise_ratio, epochs=150,
                 pace_type='d2l', init_epoch=5, epoch_win=5, lid_subset_size=1280,
                 lid_k=20, verbose=1):
        super(D2LCallback, self).__init__()
        self.model = model
        self.turning_epoch = -1
        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.epochs = epochs
        self.pace_type = pace_type
        self.mean_lid = -1.
        self.lids = []
        self.p_lambda = 0.
        self.init_epoch = init_epoch
        self.epoch_win = epoch_win
        self.lid_subset_size = lid_subset_size
        self.lid_k = lid_k
        self.verbose = verbose
        self.alpha = 1.0

    def on_epoch_begin(self, epoch, logs={}):
        rand_idxes = torch.randperm(self.X_train.size(0))[:self.lid_subset_size]
        lid = np.mean(get_lids_random_batch(self.model, self.X_train[rand_idxes], k=self.lid_k, batch_size=128))

        self.p_lambda = epoch * 1. / self.epochs

        if lid > 0:
            self.lids.append(lid)
        else:
            self.lids.append(self.lids[-1])

        if self.found_turning_point(self.lids):
            self.update_learning_pace()

        if len(self.lids) > 5:
            print('lid = ..., ', self.lids[-5:])
        else:
            print('lid = ..., ', self.lids)

        if self.verbose > 0:
            print('--Epoch: %s, LID: %.2f, min LID: %.2f, lid window: %s, turning epoch: %s, lambda: %.2f' %
                  (epoch, lid, np.min(self.lids), self.epoch_win, self.turning_epoch, self.p_lambda))

        return

    def found_turning_point(self, lids):
        if len(lids) > self.init_epoch + self.epoch_win:
            if self.turning_epoch > -1:
                return True
            else:
                smooth_lids = lids[-self.epoch_win - 1:-1]
                if lids[-1] - np.mean(smooth_lids) > 2 * np.std(smooth_lids):
                    self.turning_epoch = len(lids) - 2
                    min_model_path = 'model/%s_%s_%s.pth' % (self.pace_type, self.dataset, self.noise_ratio)
                    self.model.load_state_dict(torch.load(min_model_path))
                    return True
        else:
            return False

    def update_learning_pace(self):
        expansion = self.lids[-1] / np.min(self.lids)
        self.alpha = np.exp(-self.p_lambda * expansion)

        print('## Turning epoch: %s, lambda: %.2f, expansion: %.2f, alpha: %.2f' %
              (self.turning_epoch, self.p_lambda, expansion, self.alpha))

        self.model = lid_paced_loss(self.alpha, self.model)

class LoggerCallback(object):
    def __init__(self, model, X_train, y_train, X_test, y_test, dataset,
                 model_name, noise_ratio, epochs):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset = dataset
        self.model_name = model_name
        self.noise_ratio = noise_ratio
        self.epochs = epochs

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        self.lid_k = 20
        self.lid_subset = 128
        self.lids = []

        self.csr_subset = 500
        self.csr_batchsize = 100
        self.csrs = []

    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)

        file_name = 'log/loss_%s_%s_%s.npy' % \
                    (self.model_name, self.dataset, self.noise_ratio)
        np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
        file_name = 'log/acc_%s_%s_%s.npy' % \
                    (self.model_name, self.dataset, self.noise_ratio)
        np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))

        if epoch % 1 == 0:
            rand_idxes = torch.randperm(self.X_train.size(0))[:self.lid_subset * 10]
            lid = np.mean(get_lids_random_batch(self.model, self.X_train[rand_idxes],
                                                k=self.lid_k, batch_size=self.lid_subset))
            self.lids.append(lid)

            file_name = 'log/lid_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.lids))

            if len(np.array(self.lids).flatten()) > 20:
                print('lid = ...', self.lids[-20:])
            else:
                print('lid = ', self.lids)

            scale_factor = 255. / (np.max(self.X_test) - np.min(self.X_test))
            csr_model = lass(self.model.layers[0].input, self.model.layers[-1].output,
                             a=0.25 / scale_factor,
                             b=0.2 / scale_factor,
                             r=0.3 / scale_factor,
                             iter_max=100)
            rand_idxes = torch.randperm(self.X_test.size(0))[:self.csr_subset]
            X_adv, adv_ind = csr_model.find(self.X_test[rand_idxes], bs=self.csr_batchsize)
            csr = np.sum(adv_ind) * 1. / self.csr_subset
            self.csrs.append(csr)

            file_name = 'log/csr_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.csrs))

            if len(self.csrs) > 20:
                print('csr = ...', self.csrs[-20:])
            else:
                print('csr = ', self.csrs)

        return

def get_lr_scheduler(dataset):
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1
        return scheduler
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1
        return scheduler
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 160:
                return 0.0001
            elif epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1
        return scheduler
