import numpy as np
from .loss_module import LossModule


class CategoricalCrossEntropyLoss(LossModule):

    def __init__(self, reduction='mean', from_logits=False):
        self.reduction = reduction
        self.from_logits = from_logits
        self.input = None
        self.target = None
        super().__init__()

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(
                f"input and target should have the same shape, but got {input.shape} and {target.shape}")

        self.input = input
        self.target = target

        if self.from_logits:
            shifted = input - np.max(input, axis=1, keepdims=True)
            exp_vals = np.exp(np.clip(shifted, -709, 709))
            probs = exp_vals / (np.sum(exp_vals, axis=1, keepdims=True) + 1e-15)
            input_to_use = probs
        else:
            input_to_use = input
        
        eps = 1e-15
        input_clipped = np.clip(input_to_use, eps, 1.0 - eps)
        
        log_probs = np.log(input_clipped)
        loss = -np.sum(target * log_probs, axis=1)
        
        if np.any(np.isnan(loss)):
            loss = np.nan_to_num(loss, nan=1e3)
        
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

    def backward(self):
        if self.from_logits:
            shifted = self.input - np.max(self.input, axis=1, keepdims=True)
            exp_vals = np.exp(np.clip(shifted, -709, 709))
            probs = exp_vals / (np.sum(exp_vals, axis=1, keepdims=True) + 1e-15)
            grad = probs - self.target
        else:
            grad = self.input - self.target
        
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.reduction == 'mean':
            grad = grad / self.input.shape[0]
        
        return grad

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\", from_logits={self.from_logits})"


class BinaryCrossEntropyLoss(LossModule):

    def __init__(self, reduction='mean', from_logits=False):
        self.reduction = reduction
        self.from_logits = from_logits
        self.input = None
        self.target = None
        super().__init__()

    def forward(self, input, target):
        if input.shape != target.shape:
            raise ValueError(
                f"input and target should have the same shape, but got {input.shape} and {target.shape}")

        self.input = input
        self.target = target
        
        if self.from_logits:
            input_to_use = 1.0 / (1.0 + np.exp(np.clip(-input, -709, 709)))
        else:
            input_to_use = input

        eps = 1e-15
        input_clipped = np.clip(input_to_use, eps, 1 - eps)
        
        loss = -(target * np.log(input_clipped) + (1 - target) * np.log(1 - input_clipped))
        
        if np.any(np.isnan(loss)):
            loss = np.nan_to_num(loss, nan=1e3)
        
        if self.reduction == 'mean':
            return np.mean(loss)
        elif self.reduction == 'sum':
            return np.sum(loss)
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

    def backward(self):
        if self.from_logits:
            sigmoid = 1.0 / (1.0 + np.exp(np.clip(-self.input, -709, 709)))
            grad = sigmoid - self.target
        else:
            eps = 1e-15
            input_clipped = np.clip(self.input, eps, 1 - eps)
            
            grad = (input_clipped - self.target) / (input_clipped * (1 - input_clipped))
        
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.reduction == 'mean':
            grad = grad / self.input.shape[0]
        
        return grad

    def __repr__(self):
        return f"{self.__class__.__name__}(reduction=\"{self.reduction}\", from_logits={self.from_logits})"
