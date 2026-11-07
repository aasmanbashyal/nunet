import numpy as np
import pytest

from nunet.layer.activation import ReLU, Sigmoid
from nunet.loss import BinaryCrossEntropyLoss, CategoricalCrossEntropyLoss, MSELoss


def numerical_gradient(fn, values, epsilon=1e-6):
    gradient = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (fn(plus) - fn(minus)) / (2 * epsilon)
    return gradient


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("from_logits", [False, True])
def test_categorical_gradient_matches_finite_differences(reduction, from_logits):
    values = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    if from_logits:
        values = np.log(values)
    targets = np.array([[0.0, 0.0, 1.0], [0.2, 0.3, 0.5]])
    loss = CategoricalCrossEntropyLoss(reduction, from_logits)
    loss(values, targets)
    actual = loss.backward()
    expected = numerical_gradient(lambda x: loss(x, targets), values)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("from_logits", [False, True])
def test_binary_gradient_matches_finite_differences_for_multiple_outputs(reduction, from_logits):
    values = np.array([[0.2, 0.3, 0.5], [0.6, 0.8, 0.1]])
    if from_logits:
        values = np.log(values / (1 - values))
    targets = np.array([[0.0, 1.0, 1.0], [0.2, 0.0, 0.5]])
    loss = BinaryCrossEntropyLoss(reduction, from_logits)
    loss(values, targets)
    actual = loss.backward()
    expected = numerical_gradient(lambda x: loss(x, targets), values)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_categorical_extreme_logits_keep_loss_and_gradient():
    loss = CategoricalCrossEntropyLoss(from_logits=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert loss(np.array([[1000, -1000]]), np.array([[0, 1]])) == 2000
        np.testing.assert_allclose(loss.backward(), [[1, -1]])


def test_binary_extreme_logits_keep_loss_and_gradient():
    loss = BinaryCrossEntropyLoss(from_logits=True)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert loss(np.array([[1000, -1000]]), np.array([[0, 1]])) == 1000
        np.testing.assert_allclose(loss.backward(), [[0.5, -0.5]])


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss])
def test_clipped_probability_endpoints_are_finite(loss_cls):
    loss = loss_cls()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        assert np.isfinite(loss(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])))
        assert np.all(np.isfinite(loss.backward()))
        assert loss.backward()[0, 0] == 0


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss])
def test_gradient_below_probability_clip_matches_flat_objective(loss_cls):
    loss = loss_cls()
    target = np.array([[1.0, 0.0]])
    values = np.array([[0.5e-15, 0.75]])
    loss(values, target)
    assert loss.backward()[0, 0] == 0
    lower = values.copy()
    upper = values.copy()
    lower[0, 0] = 0.4e-15
    upper[0, 0] = 0.6e-15
    assert loss(lower, target) == loss(upper, target)


def test_binary_gradient_above_probability_clip_matches_flat_objective():
    loss = BinaryCrossEntropyLoss()
    target = np.zeros((1, 1))
    loss(np.array([[1 - 0.5e-15]]), target)
    np.testing.assert_array_equal(loss.backward(), [[0]])
    assert loss(np.array([[1.0]]), target) == loss(np.array([[1 - 0.5e-15]]), target)


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss, MSELoss])
def test_loss_rejects_implicit_broadcasting(loss_cls):
    with pytest.raises(ValueError, match="same shape"):
        loss_cls()(np.zeros((2, 1)), np.zeros(2))


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss, MSELoss])
def test_loss_rejects_empty_arrays(loss_cls):
    with pytest.raises(ValueError, match="empty"):
        loss_cls()(np.empty((0, 2)), np.empty((0, 2)))


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss, MSELoss])
def test_loss_rejects_nonfinite_values(loss_cls):
    with pytest.raises(ValueError, match="finite"):
        loss_cls()(np.array([[np.nan, 0.3]]), np.array([[0.0, 1.0]]))


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss])
def test_cross_entropy_rejects_unknown_reduction(loss_cls):
    with pytest.raises(ValueError, match="reduction"):
        loss_cls(reduction="unknown")


@pytest.mark.parametrize("loss_cls", [CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss])
def test_cross_entropy_rejects_invalid_probabilities(loss_cls):
    with pytest.raises(ValueError, match="probabilities"):
        loss_cls()(np.array([[-0.1, 1.1]]), np.array([[0.0, 1.0]]))


def test_categorical_rejects_missing_batch_dimension():
    with pytest.raises(ValueError, match="batch, classes"):
        CategoricalCrossEntropyLoss()(np.array([0.2, 0.8]), np.array([0, 1]))


def test_categorical_rejects_invalid_target_distribution():
    with pytest.raises(ValueError, match="sum to 1"):
        CategoricalCrossEntropyLoss()(np.array([[0.2, 0.8]]), np.array([[0, 0]]))


def test_mse_gradient_matches_finite_differences():
    values = np.array([[0.2, 0.3], [0.5, -0.6]])
    targets = np.array([[1.0, 0.5], [0.2, -0.1]])
    loss = MSELoss()
    loss(values, targets)
    actual = loss.backward()
    expected = numerical_gradient(lambda x: loss(x, targets), values)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("module", [ReLU, Sigmoid, MSELoss, CategoricalCrossEntropyLoss, BinaryCrossEntropyLoss])
def test_backward_requires_forward(module):
    with pytest.raises(RuntimeError, match="forward"):
        if module in (ReLU, Sigmoid):
            module().backward(np.ones(2))
        else:
            module().backward()


def test_sigmoid_handles_integer_inputs_and_extreme_values():
    sigmoid = Sigmoid()
    with np.errstate(over="raise", invalid="raise"):
        actual = sigmoid(np.array([-1000, -1, 0, 1, 1000]))
    expected = np.array([0, 1 / (1 + np.e), 0.5, 1 / (1 + np.exp(-1)), 1])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(sigmoid.backward(np.ones(5)), expected * (1 - expected))


@pytest.mark.parametrize("module", [ReLU, Sigmoid])
def test_activation_rejects_broadcast_gradient(module):
    layer = module()
    layer(np.ones((2, 3)))
    with pytest.raises(ValueError, match="same shape"):
        layer.backward(np.ones(3))


def test_relu_derivative_and_nonfinite_propagation():
    relu = ReLU()
    np.testing.assert_array_equal(relu([-2, 0, 3]), [0, 0, 3])
    np.testing.assert_array_equal(relu.backward([2, 2, 2]), [0, 0, 2])
    assert np.isnan(relu([np.nan])[0])
