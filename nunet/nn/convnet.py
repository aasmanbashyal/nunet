"""A small image classifier composed of NuNet layers."""

import numpy as np

from nunet.layer.activation import ReLU
from nunet.layer.conv2d import Conv2d
from nunet.layer.flatten import Flatten
from nunet.layer.linear import Linear
from nunet.layer.pooling import MaxPool2d
from nunet.layer.sequential import Sequential


class ConvNet(Sequential):
    """Conv2d, ReLU, MaxPool2d and a linear classifier returning class logits.

    input_shape excludes the batch dimension and is (channels, height, width).
    Train with CategoricalCrossEntropyLoss(from_logits=True).
    """

    def __init__(self, input_shape, num_classes, hidden_channels=8):
        if (not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3
                or any(isinstance(size, bool) or not isinstance(size, (int, np.integer))
                       or size <= 0 for size in input_shape)):
            raise ValueError("input_shape must contain three positive integers")
        channels, height, width = input_shape
        if height < 2 or width < 2:
            raise ValueError("input height and width must be at least 2 for pooling")
        if isinstance(num_classes, bool) or not isinstance(num_classes, (int, np.integer)) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        convolution = Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.input_shape = tuple(input_shape)
        super().__init__(
            convolution,
            ReLU(),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(hidden_channels * (height // 2) * (width // 2), num_classes),
        )

    def forward(self, input):
        input = np.asarray(input)
        if input.ndim != 4 or input.shape[1:] != self.input_shape:
            raise ValueError(f"expected input shape (batch, {self.input_shape})")
        return super().forward(input)
