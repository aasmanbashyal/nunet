# NuNet

NuNet is a small neural network library built with NumPy, inspired by PyTorch
and [micrograd](https://github.com/karpathy/micrograd). It implements each
layer's backward pass explicitly. It does not build an automatic differentiation
graph.

## Setup

Python 3.11 or newer is required. From the repository root:

```sh
uv sync
uv run --with "pytest>=8.3.5,<9" python -m pytest -q
uv run python -m examples.train_convnet
```

The existing Poetry build configuration and lockfile are retained. `uv sync`
resolves `pyproject.toml` independently of `poetry.lock`. To run only the core
tests without installing the notebook dependencies:

```sh
uv run --no-project --with "numpy>=2.2.5,<3" --with "pytest>=8.3.5,<9" python -m pytest -q
```

## Train an MLP

```python
import numpy as np
from nunet.nn import MLP
from nunet.loss import MSELoss
from nunet.optim import SGD

np.random.seed(7)
X = np.linspace(-1, 1, 64).reshape(-1, 1)
y = 2 * X + 0.5
model = MLP(1, [16], 1, init_method="xavier", verbose=False)
optimizer = SGD(model.param, lr=0.05)
train_loss, val_loss = model.fit(
    X, y, MSELoss(), optimizer, batch_size=16, epochs=100
)
predictions = model.predict(X)
```

`fit` accepts `X_val`, `y_val`, and `patience`. With validation data, it restores
a copy of the best validation weights after training. Loading weights updates
existing parameter objects, so an optimizer remains connected to the model.
Epoch losses account for a smaller final batch. Sum-reduced losses are reported
as the sum divided by the number of samples.

## Train a convolutional network

Images use NCHW layout: batch, channels, height, width. Class targets are one-hot
arrays with shape `(batch, num_classes)`. The model returns raw logits.

```python
import numpy as np
from nunet.nn import ConvNet
from nunet.loss import CategoricalCrossEntropyLoss
from nunet.optim import SGD

model = ConvNet(input_shape=(1, 8, 8), num_classes=2, hidden_channels=4)
loss_fn = CategoricalCrossEntropyLoss(from_logits=True)
optimizer = SGD(model.param, lr=0.05)
X = np.random.randn(8, 1, 8, 8)
y = np.eye(2)[np.arange(8) % 2]

for _ in range(20):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    model.backward(loss_fn.backward())
    optimizer.step()

predictions = model(X).argmax(axis=1)
```

`ConvNet` combines a padded 3x3 convolution, ReLU, 2x2 max pooling, flattening,
and a linear classifier. It inherits `Sequential` and uses the explicit training
loop above. For a deterministic example with learnable image patterns, run
`examples.train_convnet` as shown in Setup.

## API

| Import | Purpose |
| --- | --- |
| `nunet.layer.Linear(in_features, out_features, bias=True, init_method="xavier")` | Transform the last input dimension. |
| `nunet.layer.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)` | NCHW cross-correlation with trainable kernels. Kernel size, stride, and padding accept integers or pairs. |
| `nunet.layer.MaxPool2d(kernel_size=2, stride=None)` | Pool spatial windows. Stride defaults to kernel size; ties select the first maximum. |
| `nunet.layer.Flatten(start_dim=1)` | Flatten feature dimensions while preserving the batch by default. |
| `nunet.layer.Sequential(*layers)` | Apply layers in order and backpropagate in reverse. |
| `nunet.layer.activation.ReLU`, `Sigmoid` | Elementwise activations. |
| `nunet.loss.MSELoss()` | Mean squared error across all elements. |
| `nunet.loss.CategoricalCrossEntropyLoss(reduction="mean", from_logits=False)` | Class distributions shaped `(batch, classes)` with matching targets. |
| `nunet.loss.BinaryCrossEntropyLoss(reduction="mean", from_logits=False)` | Binary targets matching the prediction shape. Mean reduction averages all elements. |
| `nunet.optim.SGD(params, lr=0.01, clip_value=5.0)` | SGD with elementwise gradient clipping. Use `clip_value=None` to disable clipping. |

Cross-entropy losses support `mean` and `sum` reductions. Pass
`from_logits=True` for raw model outputs; this uses stable log-space formulas.
Probability mode expects inputs in `[0, 1]`. Categorical loss clips probabilities
below `1e-15`; binary loss also clips above `1 - 1e-15`. Gradients beyond those
clipping thresholds are zero.
MSE and cross-entropy reject mismatched input and target shapes.

Call the loss before `loss.backward()`, then pass that gradient to
`model.backward(...)`. Each layer caches only its most recent forward pass, and
backward replaces parameter gradients. Reusing the same layer instance more
than once in a model or accumulating gradients across separate backward calls
is not supported. Run the optimizer step after backward.

Trainable layers expose `.param` and `.grad` lists. `Sequential.param` preserves
the original nested list layout, with an entry for each child layer. SGD accepts
either nested model parameters or a flat layer parameter list. Tensor `.grad`
and `.gradient` refer to the same gradient buffer. Tensor arithmetic itself
does not track derivatives.

Use `layer.load_param(*values)` for an individual layer and
`model.load_param(parameter_tree)` for a sequential model. Values must match
the existing parameter shapes.

## Development status

- [x] Refactor parameter handling, gradients, and training checkpoints.
- [x] Implement and test convolution, pooling, and a small convolutional network.
- [x] Document the API and provide runnable training examples.

Tests cover numerical gradients, convolution against a loop reference, stable
losses, optimizer updates, early stopping, and learning a small synthetic image
dataset. The implementation runs on CPU with NumPy; it has no GPU backend,
dilation, grouped convolution, or automatic differentiation.

Useful next changes are to run this suite in GitHub Actions, move pandas,
Matplotlib, and Jupyter into optional example dependencies, and add a versioned
checkpoint format for saving models between sessions.
