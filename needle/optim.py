"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p

            pid = id(p)
            if self.momentum != 0.0:
                if pid not in self.u:
                    self.u[pid] = np.zeros_like(p)
                self.u[pid] = (
                    self.momentum * self.u[pid] + (1.0 - self.momentum) * g
                )
                update = self.u[pid]
            else:
                update = g
            p.data = p.data - self.lr * update

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - (b1 ** self.t)
        bc2 = 1.0 - (b2 ** self.t)

        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p

            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p)
                self.v[pid] = np.zeros_like(p)

            self.m[pid] = b1 * self.m[pid] + (1.0 - b1) * g.data
            gg = g * g
            self.v[pid] = b2 * self.v[pid] + (1.0 - b2) * gg.data

            m_hat = self.m[pid] / bc1
            v_hat = self.v[pid] / bc2

            denom = v_hat ** 0.5 + self.eps
            step = self.lr * (m_hat / denom)
            p.data = p.data - step
