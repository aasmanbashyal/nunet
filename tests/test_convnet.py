import numpy as np
import pytest

from nunet.layer.conv2d import Conv2d
from nunet.layer.flatten import Flatten
from nunet.layer.pooling import MaxPool2d
from nunet.loss import CategoricalCrossEntropyLoss
from nunet.nn.convnet import ConvNet
from nunet.optim import SGD


def numerical_gradient(array, objective, epsilon=1e-6):
    result = np.empty_like(array, dtype=float)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + epsilon
        plus = objective()
        array[index] = original - epsilon
        minus = objective()
        array[index] = original
        result[index] = (plus - minus) / (2 * epsilon)
    return result


def reference_convolution(input, weights, bias, stride, padding):
    ph, pw = padding
    sh, sw = stride
    padded = np.pad(input, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    kh, kw = weights.shape[2:]
    oh = (padded.shape[2] - kh) // sh + 1
    ow = (padded.shape[3] - kw) // sw + 1
    result = np.zeros((input.shape[0], weights.shape[0], oh, ow))
    for batch, channel, row, col in np.ndindex(result.shape):
        window = padded[batch, :, row * sh:row * sh + kh, col * sw:col * sw + kw]
        result[batch, channel, row, col] = np.sum(window * weights[channel]) + bias[channel]
    return result


def test_convolution_matches_reference_with_rectangular_geometry():
    rng = np.random.default_rng(14)
    layer = Conv2d(2, 3, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0))
    layer.load_param(rng.normal(size=layer.W.shape), rng.normal(size=layer.b.shape))
    input = rng.normal(size=(2, 2, 4, 5))
    expected = reference_convolution(input, layer.W, layer.b, layer.stride, layer.padding)
    np.testing.assert_allclose(layer(input), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bias", [True, False])
def test_convolution_gradients_match_finite_differences(bias):
    rng = np.random.default_rng(21)
    layer = Conv2d(2, 2, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0), bias=bias)
    layer.W[:] = rng.normal(size=layer.W.shape)
    input = rng.normal(size=(1, 2, 3, 4))
    upstream = rng.normal(size=layer(input).shape)
    input_gradient = layer.backward(upstream)
    parameter_gradients = [parameter.grad.copy() for parameter in layer.param]
    objective = lambda: np.sum(layer(input) * upstream)
    np.testing.assert_allclose(input_gradient, numerical_gradient(input, objective), atol=1e-8)
    for parameter, expected in zip(layer.param, parameter_gradients):
        np.testing.assert_allclose(expected, numerical_gradient(parameter, objective), atol=1e-8)


def test_convolution_load_keeps_optimizer_parameter_references():
    layer = Conv2d(1, 2, 2, bias=False)
    parameter = layer.W
    layer.load_param(np.ones_like(layer.W))
    assert layer.W is parameter
    assert len(layer.param) == 1
    np.testing.assert_array_equal(layer.W, 1)
    with pytest.raises(ValueError):
        layer.load_param(np.zeros((1,)))
    np.testing.assert_array_equal(layer.W, 1)


def test_pooling_accumulates_overlapping_window_gradients():
    layer = MaxPool2d(2, stride=1)
    input = np.array([[[[1, 2, 3], [4, 10, 5], [6, 7, 8]]]])
    np.testing.assert_array_equal(layer(input), np.full((1, 1, 2, 2), 10))
    expected = np.zeros_like(input, dtype=float)
    expected[0, 0, 1, 1] = 4
    np.testing.assert_array_equal(layer.backward(np.ones((1, 1, 2, 2))), expected)


def test_pooling_gradient_and_rectangular_windows():
    rng = np.random.default_rng(3)
    input = rng.normal(size=(1, 2, 3, 5))
    layer = MaxPool2d((2, 3), stride=(1, 2))
    output = layer(input)
    assert output.shape == (1, 2, 2, 2)
    upstream = rng.normal(size=output.shape)
    gradient = layer.backward(upstream)
    objective = lambda: np.sum(layer(input) * upstream)
    np.testing.assert_allclose(gradient, numerical_gradient(input, objective), atol=1e-8)


def test_pooling_ties_choose_first_maximum():
    layer = MaxPool2d(2)
    layer(np.ones((1, 1, 2, 2)))
    np.testing.assert_array_equal(layer.backward(np.ones((1, 1, 1, 1))), [[[[1, 0], [0, 0]]]])


def test_flatten_restores_original_shape_and_checks_gradient():
    layer = Flatten()
    input = np.arange(48).reshape(2, 3, 2, 4)
    output = layer(input)
    assert output.shape == (2, 24)
    np.testing.assert_array_equal(layer.backward(output), input)
    with pytest.raises(ValueError):
        layer.backward(np.ones((48,)))


@pytest.mark.parametrize("layer", [Conv2d(1, 1, 2), MaxPool2d(2), Flatten()])
def test_backward_requires_a_forward_pass(layer):
    with pytest.raises(RuntimeError, match="forward"):
        layer.backward(np.ones((1, 1, 2, 2)))


@pytest.mark.parametrize("kwargs", [
    {"kernel_size": 0}, {"kernel_size": (2, -1)}, {"stride": 0},
    {"padding": -1}, {"in_channels": 0}, {"out_channels": 1.5},
])
def test_convolution_rejects_invalid_configuration(kwargs):
    options = {"in_channels": 1, "out_channels": 2, "kernel_size": 3}
    options.update(kwargs)
    with pytest.raises(ValueError):
        Conv2d(**options)


def test_convolution_rejects_invalid_input_and_upstream_shapes():
    layer = Conv2d(2, 3, 3)
    with pytest.raises(ValueError, match="NCHW"):
        layer(np.ones((1, 3, 5, 5)))
    with pytest.raises(ValueError, match="kernel"):
        layer(np.ones((1, 2, 1, 1)))
    layer(np.ones((1, 2, 4, 4)))
    with pytest.raises(ValueError, match="gradient"):
        layer.backward(np.ones((1, 3, 1, 1)))


def test_convnet_learns_small_image_classification_problem():
    np.random.seed(7)
    rng = np.random.default_rng(7)
    input = rng.normal(scale=0.05, size=(12, 1, 6, 6))
    input[:6, :, :, :3] += 1
    input[6:, :, :, 3:] += 1
    labels = np.array([0] * 6 + [1] * 6)
    targets = np.eye(2)[labels]
    model = ConvNet((1, 6, 6), num_classes=2, hidden_channels=3)
    loss = CategoricalCrossEntropyLoss(from_logits=True)
    optimizer = SGD(model.param, lr=0.1)
    initial = loss(model(input), targets)
    for _ in range(60):
        optimizer.zero_grad()
        loss(model(input), targets)
        model.backward(loss.backward())
        optimizer.step()
    predictions = model(input)
    assert loss(predictions, targets) < initial * 0.2
    np.testing.assert_array_equal(np.argmax(predictions, axis=1), labels)


def test_convnet_validates_declared_image_shape():
    with pytest.raises(ValueError):
        ConvNet((1, 1, 4), 2)
    model = ConvNet((1, 4, 6), 2)
    with pytest.raises(ValueError, match="shape"):
        model(np.ones((2, 1, 6, 4)))
