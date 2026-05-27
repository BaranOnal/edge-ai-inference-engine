import numpy as np


class SGD:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, layer):
        layer.weights -= self.lr * layer.dw
        layer.bias -= self.lr * layer.db


class Momentum:
    def __init__(self, learning_rate, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.v_W = {}
        self.v_B = {}

    def update(self, layer):
        if layer not in self.v_W:
            self.v_W[layer] = np.zeros_like(layer.weights)
            self.v_B[layer] = np.zeros_like(layer.bias)

        self.v_W[layer] = self.beta * self.v_W[layer] + (1 - self.beta) * layer.dw
        self.v_B[layer] = self.beta * self.v_B[layer] + (1 - self.beta) * layer.db

        layer.weights -= self.lr * self.v_W[layer]
        layer.bias -= self.lr * self.v_B[layer]



class RMSprop:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-7):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s_W = {}
        self.s_B = {}

    def update(self, layer):
        if layer not in self.s_W:
            self.s_W[layer] = np.zeros_like(layer.weights)
            self.s_B[layer] = np.zeros_like(layer.bias)

        self.s_W[layer] = self.beta * self.s_W[layer] + (1 - self.beta) * (layer.dw ** 2)
        self.s_B[layer] = self.beta * self.s_B[layer] + (1 - self.beta) * (layer.db ** 2)

        layer.weights -= (self.lr / (np.sqrt(self.s_W[layer] + self.epsilon))) * layer.dw
        layer.bias -= (self.lr / (np.sqrt(self.s_B[layer] + self.epsilon))) * layer.db

class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1 #for bias correction
        self.v_W = {}
        self.v_B = {}
        self.s_W = {}
        self.s_B = {}


    def update(self, layer):
        if layer not in self.v_W:
            self.v_W[layer] = np.zeros_like(layer.weights)
            self.v_B[layer] = np.zeros_like(layer.bias)
            self.s_W[layer] = np.zeros_like(layer.weights)
            self.s_B[layer] = np.zeros_like(layer.bias)

        self.v_W[layer] = self.beta1 * self.v_W[layer] + (1 - self.beta1) * layer.dw
        self.v_B[layer] = self.beta1 * self.v_B[layer] + (1 - self.beta1) * layer.db
        self.s_W[layer] = self.beta2 * self.s_W[layer] + (1 - self.beta2) * (layer.dw ** 2)
        self.s_B[layer] = self.beta2 * self.s_B[layer] + (1 - self.beta2) * (layer.db ** 2)

        v_W_corr = self.v_W[layer] / (1 - (self.beta1 ** self.t))
        v_B_corr = self.v_B[layer] / (1 - (self.beta1 ** self.t))
        s_W_corr = self.s_W[layer] / (1 - (self.beta2 ** self.t))
        s_B_corr = self.s_B[layer] / (1 - (self.beta2 ** self.t))

        self.t += 1

        layer.weights -= (self.lr * v_W_corr) / np.sqrt(s_W_corr + self.epsilon)
        layer.bias -= (self.lr * v_B_corr) / np.sqrt(s_B_corr + self.epsilon)



optimizers = {
    "SGD": SGD,
    "Momentum": Momentum,
    "RMSprop": RMSprop,
    "Adam": Adam
}