import numpy as np

class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)

        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dA):
        return dA * self.output * (1 - self.output)

class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.input <= 0] = 0
        return dZ

class Softmax:
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=0, keepdims=True)) #floating point overflow
        self.output = exps / np.sum(exps, axis=0, keepdims=True)
        return self.output

    def backward(self, dA):
        #  derivation of Softmax + Cross-Entropy = y_hat - y
        return dA

activations = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "softmax": Softmax
}