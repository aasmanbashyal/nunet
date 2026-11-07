from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):

    @property
    def param(self):
        return []

    @property
    def grad(self):
        return [param.grad for param in self.param]

    @abstractmethod
    def forward(self, input):
        pass

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    @abstractmethod
    def backward(self, gradwrtoutput):
        pass

    def load_param(self, *params):
        """Copy parameter values without replacing arrays held by optimizers."""
        current_params = self.param
        if len(params) != len(current_params):
            raise ValueError(f"expected {len(current_params)} parameters, got {len(params)}")
        values = [np.asarray(value, dtype=param.dtype)
                  for param, value in zip(current_params, params)]
        for param, value in zip(current_params, values):
            if value.shape != param.shape:
                raise ValueError(f"parameter shape {value.shape} must match {param.shape}")
        for param, value in zip(current_params, values):
            np.copyto(param, value)

    def __repr__(self):
        return f"{self.__class__.__name__}()"
