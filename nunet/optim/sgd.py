import numpy as np

from nunet.dtype import Tensor
from .optim_module import BaseOptimizer


class SGD(BaseOptimizer):
    """Stochastic gradient descent for flat or nested parameter lists."""

    def __init__(self, params, lr=0.01, clip_value=5.0):
        if not np.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if clip_value is not None and (not np.isfinite(clip_value) or clip_value <= 0):
            raise ValueError("clip_value must be positive and finite, or None")
        self.params = []
        seen = set()
        for param in self._iter_params(params):
            if id(param) not in seen:
                seen.add(id(param))
                self.params.append(param)
        self.lr = lr
        self.clip_value = clip_value

    @classmethod
    def _iter_params(cls, params):
        if params is None:
            return
        if isinstance(params, Tensor):
            yield params
        elif isinstance(params, (list, tuple)):
            for param in params:
                yield from cls._iter_params(param)
        else:
            raise TypeError("parameters must be Tensor instances in a flat or nested list")

    def zero_grad(self):
        for param in self.params:
            param.grad.fill(0)

    def step(self):
        # Validate every gradient before updating any parameters.
        for param in self.params:
            if not np.all(np.isfinite(param.grad)):
                raise ValueError("parameter gradients must be finite")
        for param in self.params:
            grad = param.grad
            if self.clip_value is not None:
                grad = np.clip(grad, -self.clip_value, self.clip_value)
            param[...] = np.asarray(param) - self.lr * grad

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr}, clip_value={self.clip_value})"
