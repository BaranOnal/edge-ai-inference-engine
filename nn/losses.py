import numpy as np

class MSE:
    def compute(self, y, y_hat):
        m = y.shape[1]

        cost = (1/m) * np.sum((y_hat - y) ** 2)
        return cost

    def derivative(self, y, y_hat):
        m = y.shape[1]

        return (2/m) * (y_hat -y)


class BinaryCrossEntropy:
    def compute(self, y, y_hat):
        m = y.shape[1]
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        cost = -(1/m) * np.sum( y * np.log(y_hat) + (1-y) * np.log(1 - y_hat) )
        return cost

    def derivative(self, y, y_hat):
        # d(Sigmoid + BinaryCE)  = y_hat - y
        return y_hat - y


class CategoricalCrossEntropy:
    def compute(self, y, y_hat):
        m = y.shape[1]
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        # Formula= - (1/m) * Σ (y * log(y_hat))
        cost = - (1 / m) * np.sum(y * np.log(y_hat))
        return cost

    def derivative(self, y, y_hat):
        # d(Softmax + Categorical)  = y_hat - y
        return y_hat - y


losses = {
    "squared_loss": MSE,
    "binary_crossentropy": BinaryCrossEntropy,
   "categorical_crossentropy": CategoricalCrossEntropy
}