import numpy as np

from nunet.layer import Module


class Sigmoid(Module):
    def __init__(self):
        self._last_output = None

    def forward(self, input_data):
        input_array = np.asarray(input_data, dtype=float)
        exp_values = np.exp(-np.abs(input_array))
        self._last_output = np.where(
            input_array >= 0,
            1 / (1 + exp_values),
            exp_values / (1 + exp_values),
        )
        return self._last_output

    def backward(self, gradwrtoutput):
        if self._last_output is None:
            raise RuntimeError("forward must be called before backward")
        grad = np.asarray(gradwrtoutput)
        if grad.shape != self._last_output.shape:
            raise ValueError("gradient must have the same shape as the forward output")
        return grad * self._last_output * (1 - self._last_output)
