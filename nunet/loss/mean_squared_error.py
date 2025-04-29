import numpy as np

from .loss_module import LossModule


class MSELoss(LossModule):

    def __init__(self):
        self.predictions = None
        self.target = None
        self.input_size = None

    def forward(self, predictions: np.ndarray, target: np.ndarray):

        self.predictions = predictions
        self.target = target
        self.input_size = predictions.size

        squared_errors = (predictions - target) ** 2
        loss = np.mean(squared_errors)
        return loss

    def backward(self):
        # Gradient of MSE = 2 * (predictions - target) / N
        gradient = 2 * (self.predictions - self.target) / self.input_size

        return gradient
