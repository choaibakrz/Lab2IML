import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        return ...

class ReLU:
    @staticmethod
    def forward(z):
        return ...

    @staticmethod
    def gradient(z):
        return ...
