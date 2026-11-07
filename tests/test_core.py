from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nunet.dtype import Tensor
from nunet.layer import Linear, Sequential
from nunet.optim import SGD


def numerical_gradient(values, objective, epsilon=1e-6):
    gradient = np.zeros(values.shape)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + epsilon
        positive = objective()
        values[index] = original - epsilon
        negative = objective()
        values[index] = original
        gradient[index] = (positive - negative) / (2 * epsilon)
    return gradient


def test_tensor_gradient_aliases_share_one_shape_checked_buffer():
    tensor = Tensor((2, 3), init_method='zeros')
    buffer = tensor.grad
    assert tensor.gradient is buffer
    tensor.gradient = np.ones((2, 3))
    assert tensor.grad is buffer
    assert_array_equal(tensor.grad, 1)
    tensor.grad = np.full((2, 3), 2)
    assert_array_equal(tensor.gradient, 2)
    with pytest.raises(ValueError, match='gradient shape'):
        tensor.grad = np.ones(3)
    assert tensor[:, 0].grad.shape == (2,)
    assert tensor.T.grad.shape == (3, 2)
    with pytest.raises(ValueError, match='initialization'):
        Tensor(2, init_method='typo')


@pytest.mark.parametrize('shape', [(3,), (4, 3), (2, 2, 3)])
@pytest.mark.parametrize('bias', [False, True])
def test_linear_gradients_match_finite_differences(shape, bias):
    rng = np.random.default_rng(8)
    layer = Linear(3, 2, bias=bias)
    layer.W[...] = rng.normal(size=layer.W.shape)
    inputs = rng.normal(size=shape)
    upstream = rng.normal(size=shape[:-1] + (2,))
    layer(inputs)
    input_gradient = layer.backward(upstream)
    weight_gradient = layer.W.grad.copy()
    bias_gradient = layer.b.grad.copy() if bias else None

    def objective():
        return np.sum(layer(inputs) * upstream)

    assert_allclose(input_gradient, numerical_gradient(inputs, objective), rtol=1e-6, atol=1e-7)
    assert_allclose(weight_gradient, numerical_gradient(layer.W, objective), rtol=1e-6, atol=1e-7)
    if bias:
        assert_allclose(bias_gradient, numerical_gradient(layer.b, objective), rtol=1e-6, atol=1e-7)
    else:
        assert len(layer.param) == len(layer.grad) == 1


def test_linear_rejects_wrong_shapes_and_does_not_mask_nonfinite_values():
    layer = Linear(2, 3, init_method='ones')
    with pytest.raises(RuntimeError, match='forward'):
        layer.backward(np.ones((1, 3)))
    with pytest.raises(ValueError, match='final dimension'):
        layer(np.ones((4, 1)))
    layer(np.ones((4, 2)))
    with pytest.raises(ValueError, match='output gradient shape'):
        layer.backward(np.ones((3,)))
    assert np.isnan(layer([[np.nan, 1]])).all()


@pytest.mark.parametrize('nested', [False, True])
def test_sgd_updates_flat_and_nested_parameters_and_zeroes_in_place(nested):
    layer = Linear(2, 1, bias=False, init_method='ones')
    params = [[layer.param, None], [], [layer.W]] if nested else layer.param
    optimizer = SGD(params, lr=0.1)
    gradient_buffer = layer.W.grad
    layer.W.gradient = [[10], [-2]]
    optimizer.step()
    assert_allclose(layer.W, [[0.5], [1.2]])
    assert_array_equal(layer.W.gradient, [[10], [-2]])
    optimizer.zero_grad()
    assert layer.W.grad is gradient_buffer
    assert_array_equal(layer.W.gradient, 0)


def test_sgd_can_disable_clipping_and_rejects_nonfinite_gradients_before_updates():
    layer = Linear(1, 1, init_method='ones')
    optimizer = SGD(layer.param, lr=0.1, clip_value=None)
    layer.W.grad = [[10]]
    layer.b.grad = [np.inf]
    with pytest.raises(ValueError, match='finite'):
        optimizer.step()
    assert_array_equal(layer.W, [[1]])
    layer.b.grad = [0]
    optimizer.step()
    assert_array_equal(layer.W, [[0]])


@pytest.mark.parametrize('kwargs', [{'lr': -1}, {'lr': np.nan}, {'clip_value': 0}, {'clip_value': np.inf}])
def test_sgd_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        SGD([], **kwargs)


def test_loading_linear_parameters_preserves_optimizer_references():
    layer = Linear(2, 1, init_method='zeros')
    optimizer = SGD(layer.param, lr=0.1)
    weights, bias = layer.param
    layer.load_param(np.ones((2, 1)), np.ones(1))
    assert layer.W is weights
    assert layer.b is bias
    layer.W.grad = np.ones((2, 1))
    layer.b.grad = np.ones(1)
    optimizer.step()
    assert_allclose(layer.W, 0.9)
    assert_allclose(layer.b, 0.9)
    with pytest.raises(ValueError, match='expected 2'):
        layer.load_param(np.ones((2, 1)))
    with pytest.raises(ValueError, match='parameter shape'):
        layer.load_param(np.zeros((2, 1)), np.zeros(2))
    assert_allclose(layer.W, 0.9)


@pytest.mark.parametrize('unpack', [False, True])
@pytest.mark.parametrize('single_nested', [False, True])
def test_sequential_loads_nested_checkpoints_without_replacing_parameters(unpack, single_nested):
    if single_nested:
        model = Sequential(Sequential(Linear(2, 1, bias=False)))
    else:
        model = Sequential(Linear(2, 3, bias=False), Sequential(Linear(3, 2), Linear(2, 1)))
    optimizer = SGD(model.param, lr=0.1)
    expected = model(np.ones((3, 2))).copy()
    checkpoint = deepcopy(model.param)
    references = list(optimizer.params)
    for param in references:
        param.fill(0)
    if unpack:
        model.load_param(*checkpoint)
    else:
        model.load_param(checkpoint)
    assert_allclose(model(np.ones((3, 2))), expected)
    assert all(actual is original for actual, original in zip(SGD(model.param).params, references))
    model.backward(np.ones((3, 1)))
    assert len(model.grad) == len(references)
    optimizer.step()
    assert not np.allclose(model(np.ones((3, 2))), expected)
    with pytest.raises(ValueError, match='per-layer structure'):
        model.load_param([None])
