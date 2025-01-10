import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        
        return input_gradient
    
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, input):
        self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)
        return input * self.mask

    def backward(self, output_gradient):
        return output_gradient * self.mask


class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)

class Sigmoid:
    def forward(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, output_gradient):
        sig = self.forward(self.input)
        return output_gradient * sig * (1 - sig)

class Tanh:
    def forward(self, input):
        self.input = input
        return np.tanh(input)

    def backward(self, output_gradient):
        return output_gradient * (1 - np.tanh(self.input) ** 2)
