import  numpy as np

class MeanSquaredError:
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    def backward(self):
        return 2 * (self.predicted - self.target) / self.target.size
