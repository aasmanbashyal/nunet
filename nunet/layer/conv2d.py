"""Two-dimensional cross-correlation with an explicit backward pass."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from nunet.dtype import Tensor
from .module import Module


def _pair(value, name, minimum=1):
    values = (value, value) if isinstance(value, (int, np.integer)) else value
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must be an integer or a pair of integers")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or v < minimum for v in values):
        raise ValueError(f"{name} values must be integers >= {minimum}")
    return tuple(int(v) for v in values)


class Conv2d(Module):
    """Apply a convolution to arrays shaped (batch, channels, height, width).

    As in PyTorch, the operation is cross-correlation, without reversing the
    kernel. Only the most recent forward pass is retained for backward.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, init_method="xavier"):
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) or value <= 0
               for value in (in_channels, out_channels)):
            raise ValueError("channel counts must be positive integers")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding", minimum=0)
        self.bias = bool(bias)
        shape = (self.out_channels, self.in_channels, *self.kernel_size)
        self.W = Tensor(shape, init_method="zeros" if init_method == "xavier" else init_method)
        if init_method == "xavier":
            kernel_area = np.prod(self.kernel_size)
            limit = np.sqrt(6 / ((self.in_channels + self.out_channels) * kernel_area))
            self.W[:] = np.random.uniform(-limit, limit, shape)
        self.b = Tensor(self.out_channels, init_method="zeros") if self.bias else None
        self._input_shape = None
        self._windows = None
        self._padded_shape = None

    def forward(self, input):
        input = np.asarray(input, dtype=float)
        if input.ndim != 4 or input.shape[1] != self.in_channels:
            raise ValueError(f"expected NCHW input with {self.in_channels} channels")
        if any(size == 0 for size in input.shape):
            raise ValueError("input dimensions must be nonzero")
        ph, pw = self.padding
        kh, kw = self.kernel_size
        sh, sw = self.stride
        padded = np.pad(input, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        if padded.shape[2] < kh or padded.shape[3] < kw:
            raise ValueError("kernel_size exceeds the padded input dimensions")
        windows = sliding_window_view(padded, (kh, kw), axis=(2, 3))
        self._windows = windows[:, :, ::sh, ::sw]
        self._input_shape = input.shape
        self._padded_shape = padded.shape
        output = np.einsum("nchwkl,ockl->nohw", self._windows, np.asarray(self.W))
        if self.bias:
            output += np.asarray(self.b)[None, :, None, None]
        return output

    def backward(self, gradwrtoutput):
        if self._windows is None:
            raise RuntimeError("forward must be called before backward")
        gradient = np.asarray(gradwrtoutput, dtype=float)
        n, _, oh, ow, kh, kw = self._windows.shape
        if gradient.shape != (n, self.out_channels, oh, ow):
            raise ValueError("gradient shape must match the last forward output")
        self.W.grad = np.einsum("nohw,nchwkl->ockl", gradient, self._windows)
        if self.bias:
            self.b.grad = gradient.sum(axis=(0, 2, 3))
        input_gradient = np.zeros(self._padded_shape, dtype=float)
        sh, sw = self.stride
        weights = np.asarray(self.W)
        for row in range(kh):
            for col in range(kw):
                input_gradient[:, :, row:row + sh * oh:sh, col:col + sw * ow:sw] += (
                    np.einsum("nohw,oc->nchw", gradient, weights[:, :, row, col])
                )
        ph, pw = self.padding
        height, width = self._input_shape[2:]
        return input_gradient[:, :, ph:ph + height, pw:pw + width]

    @property
    def param(self):
        return [self.W, self.b] if self.bias else [self.W]

    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, bias={self.bias})")
