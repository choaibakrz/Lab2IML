import numpy as np


class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]


class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def gradient(y_true, y_pred):
        # Combined Softmax + CrossEntropy gradient = (y_pred - y_true) / N
        # This is mathematically correct when the output layer uses Softmax
        return (y_pred - y_true) / y_true.shape[0]