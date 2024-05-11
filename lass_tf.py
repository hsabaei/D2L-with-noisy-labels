import torch
import numpy as np

class lass(object):
    def __init__(self, model, a=0.25/255., b=0.2/255., r=0.3/255., iter_max=100, clip_min=-np.inf, clip_max=np.inf):
        """
        Initialize LASS for PyTorch.
        :param model: PyTorch model
        :param a, b, r, iter_max, clip_min, clip_max: Adversarial search parameters
        """
        self.model = model
        self.a = a
        self.b = b
        self.r = r
        self.iter_max = iter_max
        self.clip_min = clip_min
        self.clip_max = clip_max

    def find(self, X, y_target, bs=500):
        """
        Search for adversarial samples.
        :param X: Input tensor (batch of images).
        :param y_target: Target labels for adversarial generation.
        :param bs: Batch size for processing.
        :return: Adversarial examples, Indicator array for adversarial samples.
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        X_adv = X.clone().detach()
        Y_pred_vec = torch.argmax(self.model(X), dim=1).detach().cpu().numpy()
        adv_ind = np.zeros(X.size(0), dtype=bool)
        converged = False
        adv_num_old = 0

        for i in range(self.iter_max):
            if converged:
                break
            for batch_start in range(0, X.size(0), bs):
                batch_end = min(batch_start + bs, X.size(0))
                x_batch = X_adv[batch_start:batch_end].clone().detach().requires_grad_(True)

                self.model.zero_grad()
                y_pred = self.model(x_batch)
                loss = torch.nn.functional.cross_entropy(y_pred, y_target[batch_start:batch_end])
                loss.backward()

                with torch.no_grad():
                    grad_sign = x_batch.grad.sign()
                    step = self.a * grad_sign + self.b * torch.randn_like(x_batch)
                    x_batch += step
                    diff = x_batch - X[batch_start:batch_end]
                    diff = torch.clamp(diff, -self.r, self.r)
                    x_batch = torch.clamp(X[batch_start:batch_end] + diff, self.clip_min, self.clip_max)

                    X_adv[batch_start:batch_end] = x_batch.detach()

            with torch.no_grad():
                preds_adv = torch.argmax(self.model(X_adv), dim=1).detach().cpu().numpy()
                adv_ind = np.logical_or(adv_ind, Y_pred_vec != preds_adv)
                adv_num_new = np.sum(adv_ind)

                if adv_num_new - adv_num_old < 20:
                    converged = True
                adv_num_old = adv_num_new

        return X_adv, adv_ind
