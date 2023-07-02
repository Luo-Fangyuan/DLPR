# %%

from typing import Tuple
import torch
from torch import optim
import math
from utils import device


def _tuple_mul(x: Tuple):
    out = 1
    for val in x:
        out *= val
    return out


class GNCCP(optim.Optimizer):

    def __init__(self, params, zeta_step=0.1, max_iter=100, converge_eps=0.0001, val_1=0, val_2=1, convex=False):
        assert val_1 < val_2
        assert max_iter > 0
        self.eps = (val_2 - val_1) / 1e6
        self.val_1 = val_1
        self.val_2 = val_2
        self.zeta_step = zeta_step
        self.convex = convex
        self.max_iter = max_iter
        self.converge_eps = converge_eps
        if self.convex:
            self.zeta = 0
        else:
            self.zeta = 1
        self.alpha_index = 0
        self._optimized = False
        defaults = {}
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if self._optimized:
            raise Warning("Optimized")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        count_params = 0
        sum_grads = 0
        for group in self.param_groups:
            for X in group['params']:
                if X.grad is not None:
                    F_zeta_grad = (1 - abs(self.zeta)) * X.grad + 2 * self.zeta * X
                    Y = torch.zeros_like(X, device=device.get())
                    Y.fill_(self.val_1)
                    Y[F_zeta_grad < 0] = self.val_2
                    alpha = 1.0 / (self.alpha_index + 2)
                    X += alpha * (Y - X)

                    count_params += _tuple_mul(Y.shape)
                    sum_grads += torch.sum(abs(F_zeta_grad))

        mean_grads = sum_grads / count_params

        if mean_grads < self.converge_eps or self.alpha_index >= self.max_iter - 1:
            self.zeta -= self.zeta_step
            self.zeta = max(self.zeta, -1)  
            self.alpha_index = 0
        else:
            self.alpha_index += 1

        if self.zeta <= -1 + self.eps:
            for group in self.param_groups:
                for X in group['params']:
                    if X.grad is not None:
                        set_to_val2 = (X - self.val_1) > (self.val_2 - X)
                        torch.fill_(X, self.val_1)
                        X[set_to_val2] = self.val_2
            self._optimized = True
        return loss

    def optimized(self):
        return self._optimized