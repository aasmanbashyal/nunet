"""Maximum pooling for NCHW arrays."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .conv2d import _pair
from .module import Module


class MaxPool2d(Module):
    """Pool spatial windows; ties send the gradient to the first maximum."""

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = self.kernel_size if stride is None else _pair(stride, "stride")
        self._input_shape = None
        self._indices = None

    def forward(self, input):
        input = np.asarray(input, dtype=float)
        if input.ndim != 4 or any(size == 0 for size in input.shape):
            raise ValueError("expected a nonempty NCHW input")
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if input.shape[2] < kh or input.shape[3] < kw:
            raise ValueError("kernel_size exceeds the input dimensions")
        windows = sliding_window_view(input, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
        windows = windows.reshape(*windows.shape[:4], kh * kw)
        self._input_shape = input.shape
        self._indices = np.argmax(windows, axis=-1)
        return np.take_along_axis(windows, self._indices[..., None], axis=-1)[..., 0]

    def backward(self, gradwrtoutput):
        if self._indices is None:
            raise RuntimeError("forward must be called before backward")
        gradient = np.asarray(gradwrtoutput, dtype=float)
        if gradient.shape != self._indices.shape:
            raise ValueError("gradient shape must match the last forward output")
        result = np.zeros(self._input_shape, dtype=float)
        oh, ow = gradient.shape[2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        for row in range(kh):
            for col in range(kw):
                result[:, :, row:row + sh * oh:sh, col:col + sw * ow:sw] += (
                    gradient * (self._indices == row * kw + col)
                )
        return result

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"
