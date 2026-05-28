import numpy as np


class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1 - s)


class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(float)


class Softmax:
    @staticmethod
    def forward(z):
        # Subtract max for numerical stability
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def gradient(z):
        # When used with CrossEntropy the combined gradient simplifies to
        # (y_pred - y_true), so we return 1 here and handle it in back_prop
        # via the loss gradient directly.
        return np.ones_like(z)