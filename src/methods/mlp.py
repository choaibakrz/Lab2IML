import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        self.dimensions = dimensions
        self.activations = activations
        self.n_layers = len(dimensions) - 1
        self.learning_rate = 1e-3

        self.weights = {}
        self.biases = {}
        for i in range(self.n_layers):
            fan_in  = dimensions[i]
            fan_out = dimensions[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights[i + 1] = np.random.uniform(-limit, limit, (fan_in, fan_out))
            self.biases[i + 1]  = np.zeros(fan_out)

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        a = {0: x}
        z = {}
        for i in range(1, self.n_layers + 1):
            z[i] = a[i - 1] @ self.weights[i] + self.biases[i]
            a[i] = self.activations[i - 1].forward(z[i])
        return z, a


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        _, a = self.feed_forward(x)
        return a[self.n_layers]


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        dw = {}
        db = {}
        N = y_true.shape[0]

        # Delta at output layer
        delta = loss.gradient(y_true, a[self.n_layers]) * \
                self.activations[self.n_layers - 1].gradient(z[self.n_layers])

        for i in range(self.n_layers, 0, -1):
            dw[i] = a[i - 1].T @ delta / N
            db[i] = delta.mean(axis=0)
            if i > 1:
                delta = (delta @ self.weights[i].T) * \
                        self.activations[i - 2].gradient(z[i - 1])

        return dw, db


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.weights[index] -= self.learning_rate * dw
        self.biases[index]  -= self.learning_rate * delta

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        self.learning_rate = learning_rate
        N = x.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            x_shuffled      = x[perm]
            y_true_shuffled = y_true[perm]

            for start in range(0, N, batch_size):
                xb = x_shuffled[start:start + batch_size]
                yb = y_true_shuffled[start:start + batch_size]

                z, a = self.feed_forward(xb)
                dw, db = self.back_prop(z, a, yb, loss)

                for i in range(1, self.n_layers + 1):
                    self.update_w_b(i, dw[i], db[i])

            if (epoch + 1) % 10 == 0:
                _, a_full = self.feed_forward(x)
                l = loss.loss(y_true, a_full[self.n_layers])
                print(f"Epoch {epoch + 1}/{epochs}  loss={l:.4f}")