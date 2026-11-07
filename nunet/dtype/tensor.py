import numpy as np


class Tensor(np.ndarray):
    """A NumPy parameter array with a manually populated gradient buffer."""

    def __new__(cls, shape, init_method="rand"):
        base_array = cls.init_weights(shape=shape, init_method=init_method)
        tensor = base_array.view(cls)
        tensor._gradient = np.zeros(tensor.shape, dtype=float)
        return tensor

    def __array_finalize__(self, source):
        # Views and arithmetic results do not participate in automatic differentiation.
        self._gradient = None

    @property
    def grad(self):
        if self._gradient is None:
            self._gradient = np.zeros(self.shape, dtype=float)
        return self._gradient

    @grad.setter
    def grad(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != self.shape:
            raise ValueError(f"gradient shape {value.shape} must match tensor shape {self.shape}")
        self.grad[...] = value

    # Keep the original public name as an alias, using the same buffer and setter.
    gradient = grad

    @staticmethod
    def init_weights(shape, init_method):
        if init_method == 'ones':
            return np.ones(shape)
        elif init_method == 'rand':
            return np.random.random(shape)
        elif init_method == 'normal':
            return np.random.normal(loc=0.0, scale=0.1, size=shape)
        elif init_method == 'zeros':
            return np.zeros(shape)
        raise ValueError(f"unknown initialization method: {init_method!r}")
