import numpy as np

from nunet.layer import Module


class ReLU(Module):
    def __init__(self):
        self._last_input = None

    def forward(self, input_data):
        self._last_input = np.asarray(input_data)
        return np.maximum(0, self._last_input)

    def backward(self, gradwrtoutput):
        if self._last_input is None:
            raise RuntimeError("forward must be called before backward")
        grad = np.asarray(gradwrtoutput)
        if grad.shape != self._last_input.shape:
            raise ValueError("gradient must have the same shape as the forward output")
        return grad * (self._last_input > 0)
