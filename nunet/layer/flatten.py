"""Reshape a batch of features while retaining the backward shape."""

import numpy as np

from .module import Module


class Flatten(Module):
    """Flatten dimensions from start_dim onward, preserving the batch by default."""

    def __init__(self, start_dim=1):
        if isinstance(start_dim, bool) or not isinstance(start_dim, (int, np.integer)) or start_dim < 0:
            raise ValueError("start_dim must be a nonnegative integer")
        self.start_dim = int(start_dim)
        self._input_shape = None
        self._output_shape = None

    def forward(self, input):
        input = np.asarray(input)
        if self.start_dim >= input.ndim:
            raise ValueError("start_dim must be smaller than the input rank")
        self._input_shape = input.shape
        flattened_size = int(np.prod(input.shape[self.start_dim:]))
        self._output_shape = (*input.shape[:self.start_dim], flattened_size)
        return input.reshape(self._output_shape)

    def backward(self, gradwrtoutput):
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        gradient = np.asarray(gradwrtoutput)
        if gradient.shape != self._output_shape:
            raise ValueError("gradient shape must match the last forward output")
        return gradient.reshape(self._input_shape)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"
