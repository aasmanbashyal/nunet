import numpy as np

from nunet.dtype import Tensor
from .module import Module


class Linear(Module):
    """Apply an affine transform to the final dimension of an input array."""

    def __init__(self, in_features, out_features, bias=True, init_method='xavier'):
        super().__init__()
        for name, value in [('in_features', in_features), ('out_features', out_features)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.init_method = init_method
        self.input = None
        self.W = Tensor((in_features, out_features), init_method='zeros')
        self.b = Tensor(out_features, init_method='zeros') if bias else None
        self.init_weights()

    def init_weights(self):
        if self.init_method == 'xavier':
            limit = np.sqrt(6 / (self.in_features + self.out_features))
            self.W[...] = np.random.uniform(-limit, limit, self.W.shape)
        else:
            self.W[...] = Tensor.init_weights(self.W.shape, self.init_method)
        if self.b is not None:
            self.b.fill(0)

    def forward(self, input):
        input = np.asarray(input, dtype=float)
        if input.ndim == 0 or input.shape[-1] != self.in_features:
            raise ValueError(f"input must have final dimension {self.in_features}, got {input.shape}")
        self.input = input
        output = input @ np.asarray(self.W)
        if self.b is not None:
            output += np.asarray(self.b)
        return output

    def backward(self, gradwrtoutput):
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        gradwrtoutput = np.asarray(gradwrtoutput, dtype=float)
        expected_shape = self.input.shape[:-1] + (self.out_features,)
        if gradwrtoutput.shape != expected_shape:
            raise ValueError(f"output gradient shape must be {expected_shape}, got {gradwrtoutput.shape}")
        inputs = self.input.reshape(-1, self.in_features)
        output_grads = gradwrtoutput.reshape(-1, self.out_features)
        self.W.grad = inputs.T @ output_grads
        if self.b is not None:
            self.b.grad = output_grads.sum(axis=0)
        return gradwrtoutput @ np.asarray(self.W).T

    @property
    def param(self):
        return [self.W, self.b] if self.b is not None else [self.W]

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias}, "
                f"init_method={self.init_method!r})")
