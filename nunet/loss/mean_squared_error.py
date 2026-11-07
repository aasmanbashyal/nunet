import numpy as np

from .loss_module import LossModule


class MSELoss(LossModule):
    """Mean squared error over all elements, without implicit broadcasting."""

    def __init__(self):
        self.predictions = None
        self.target = None
        self.input_size = None

    def forward(self, predictions: np.ndarray, target: np.ndarray):
        predictions = np.asarray(predictions, dtype=float)
        target = np.asarray(target, dtype=float)
        if predictions.shape != target.shape:
            raise ValueError("predictions and target must have the same shape")
        if predictions.size == 0:
            raise ValueError("predictions and target must not be empty")
        if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(target)):
            raise ValueError("predictions and target must contain only finite values")
        self.predictions = predictions
        self.target = target
        self.input_size = predictions.size
        return np.mean((predictions - target) ** 2)

    def backward(self):
        if self.predictions is None:
            raise RuntimeError("forward must be called before backward")
        return 2 * (self.predictions - self.target) / self.input_size
