import numpy as np

from .loss_module import LossModule


def _validate_input(input, target, from_logits):
    input = np.asarray(input, dtype=float)
    target = np.asarray(target, dtype=float)
    if input.shape != target.shape:
        raise ValueError(
            f"input and target must have the same shape, got {input.shape} and {target.shape}"
        )
    if input.size == 0:
        raise ValueError("input and target must not be empty")
    if not np.all(np.isfinite(input)) or not np.all(np.isfinite(target)):
        raise ValueError("input and target must contain only finite values")
    if np.any((target < 0) | (target > 1)):
        raise ValueError("target values must be between 0 and 1")
    if not from_logits and np.any((input < 0) | (input > 1)):
        raise ValueError("probabilities must be between 0 and 1")
    return input, target


class CategoricalCrossEntropyLoss(LossModule):
    """Cross entropy for (batch, classes) arrays and one-hot or soft targets.

    Mean reduction averages over the batch. Probability inputs are clipped
    below 1e-15; their gradient is zero below that clipping threshold.
    Use ``from_logits=True`` to avoid clipping and retain extreme-logit gradients.
    """

    def __init__(self, reduction="mean", from_logits=False):
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction
        self.from_logits = from_logits
        self.input = None
        self.target = None
        self._probs = None

    def forward(self, input, target):
        input, target = _validate_input(input, target, self.from_logits)
        if input.ndim != 2:
            raise ValueError("categorical cross entropy expects (batch, classes) arrays")
        if not np.allclose(np.sum(target, axis=1), 1.0):
            raise ValueError("each target row must sum to 1")

        if self.from_logits:
            shifted = input - np.max(input, axis=1, keepdims=True)
            exp_values = np.exp(shifted)
            normalizer = np.sum(exp_values, axis=1, keepdims=True)
            self._probs = exp_values / normalizer
            log_probs = shifted - np.log(normalizer)
        else:
            log_probs = np.log(np.maximum(input, 1e-15))

        self.input = input
        self.target = target
        loss = -np.sum(target * log_probs, axis=1)
        return np.mean(loss) if self.reduction == "mean" else np.sum(loss)

    def backward(self):
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        if self.from_logits:
            grad = self._probs * np.sum(self.target, axis=1, keepdims=True) - self.target
        else:
            grad = -self.target / np.maximum(self.input, 1e-15)
            grad = np.where(self.input >= 1e-15, grad, 0.0)
        if self.reduction == "mean":
            grad = grad / self.input.shape[0]
        return grad

    def __repr__(self):
        return f'{self.__class__.__name__}(reduction="{self.reduction}", from_logits={self.from_logits})'


class BinaryCrossEntropyLoss(LossModule):
    """Binary cross entropy, averaged over all elements for mean reduction.

    Probability inputs are clipped to [1e-15, 1 - 1e-15], with zero gradient
    outside that interval. Logits use the stable softplus formulation.
    """

    def __init__(self, reduction="mean", from_logits=False):
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction
        self.from_logits = from_logits
        self.input = None
        self.target = None

    def forward(self, input, target):
        input, target = _validate_input(input, target, self.from_logits)
        if self.from_logits:
            loss = np.maximum(input, 0) - input * target + np.log1p(np.exp(-np.abs(input)))
        else:
            probs = np.clip(input, 1e-15, 1 - 1e-15)
            loss = -(target * np.log(probs) + (1 - target) * np.log1p(-probs))
        self.input = input
        self.target = target
        return np.mean(loss) if self.reduction == "mean" else np.sum(loss)

    def backward(self):
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        if self.from_logits:
            exp_values = np.exp(-np.abs(self.input))
            probs = np.where(self.input >= 0, 1 / (1 + exp_values), exp_values / (1 + exp_values))
            grad = probs - self.target
        else:
            probs = np.clip(self.input, 1e-15, 1 - 1e-15)
            grad = (probs - self.target) / (probs * (1 - probs))
            grad = np.where((self.input >= 1e-15) & (self.input <= 1 - 1e-15), grad, 0.0)
        if self.reduction == "mean":
            grad = grad / self.input.size
        return grad

    def __repr__(self):
        return f'{self.__class__.__name__}(reduction="{self.reduction}", from_logits={self.from_logits})'
