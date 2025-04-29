import numpy as np
from .optim_module import BaseOptimizer


class SGD(BaseOptimizer):

    def __init__(self, params, lr=0.01, clip_value=5.0):
        self.params = params
        self.lr = lr
        self.clip_value = clip_value  # Maximum allowed gradient value for clipping

    def zero_grad(self):
        for param_pair in self.params:
            for param in param_pair:
                param.grad = np.zeros_like(param)

    def step(self):
        for param_pair in self.params:
            for param in param_pair:
                # Check for NaN values in gradients
                if np.any(np.isnan(param.grad)):
                    print("Warning: NaN values detected in gradients, skipping update for this parameter")
                    continue
                
                # Clip gradients to prevent exploding gradients
                param.grad = np.clip(param.grad, -self.clip_value, self.clip_value)
                
                # Update parameters
                param -= self.lr * param.grad

    def __repr__(self):
        return f"{self.__class__.__name__}(lr={self.lr}, clip_value={self.clip_value})"
