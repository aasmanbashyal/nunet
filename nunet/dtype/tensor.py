import numpy as np


class Tensor(np.ndarray):

    def __new__(cls, shape, init_method="rand"):
        base_array = cls.init_weights(shape=shape, init_method=init_method)
        tensor = base_array.view(cls)
        tensor._gradient = np.zeros(shape)
        return tensor

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, value):
        self._gradient = value

    @staticmethod
    def init_weights(shape, init_method):
        if init_method == 'ones':
            return np.ones(shape)

        elif init_method == 'rand':
            return np.random.rand(shape)

        elif init_method == 'normal':
            return np.random.normal(size=shape)
        else:
            return np.zeros(shape)
