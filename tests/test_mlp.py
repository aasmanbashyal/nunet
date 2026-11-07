import numpy as np
import pytest

from nunet.loss import CategoricalCrossEntropyLoss, MSELoss
from nunet.nn import MLP
from nunet.optim import SGD


def test_epoch_loss_weights_the_last_partial_batch():
    model = MLP(1, [], 1, verbose=False)
    model.load_param([[np.zeros((1, 1)), np.zeros(1)]])
    X = np.zeros((3, 1))
    y = np.array([[1.0], [1.0], [3.0]])
    expected = 11.0 / 3.0
    assert model.validate(X, y, MSELoss(), batch_size=2) == pytest.approx(expected)
    assert model.train(X, y, MSELoss(), SGD(model.param, lr=0), batch_size=2) == pytest.approx(expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_classification_epoch_loss_is_independent_of_batch_partition(reduction):
    model = MLP(1, [], 2, verbose=False)
    model.load_param([[np.zeros((1, 2)), np.zeros(2)]])
    X = np.zeros((3, 1))
    y = np.eye(2)[[0, 1, 0]]
    loss = CategoricalCrossEntropyLoss(reduction=reduction, from_logits=True)
    assert model.validate(X, y, loss, batch_size=2) == pytest.approx(np.log(2))
    assert model.train(X, y, loss, SGD(model.param, lr=0), batch_size=2) == pytest.approx(np.log(2))


def test_early_stopping_restores_a_snapshot_without_replacing_parameters(monkeypatch):
    model = MLP(1, [], 1, verbose=False)
    weight = model.layers[0].W
    optimizer = SGD(model.param)
    training_epochs = iter([1.0, 2.0, 3.0])
    validation_losses = iter([0.5, 0.7, 0.9])

    def train(*args):
        weight[:] = next(training_epochs)
        return 1.0

    monkeypatch.setattr(model, 'train', train)
    monkeypatch.setattr(model, 'validate', lambda *args: next(validation_losses))
    X = np.ones((2, 1))
    train_losses, val_losses = model.fit(
        X, X, MSELoss(), optimizer, X_val=X, y_val=X, epochs=3, patience=1
    )
    assert len(train_losses) == len(val_losses) == 2
    assert model.layers[0].W is weight
    np.testing.assert_array_equal(weight, [[1.0]])
    optimizer.zero_grad()
    weight.grad = np.ones_like(weight)
    optimizer.step()
    np.testing.assert_allclose(weight, [[0.99]])


def test_fit_without_validation_keeps_final_weights(monkeypatch):
    model = MLP(1, [], 1, verbose=False)
    weight = model.layers[0].W
    epochs = iter([1.0, 2.0, 3.0])

    def train(*args):
        weight[:] = next(epochs)
        return 1.0

    monkeypatch.setattr(model, 'train', train)
    X = np.ones((2, 1))
    model.fit(X, X, MSELoss(), SGD(model.param), epochs=3)
    np.testing.assert_array_equal(weight, [[3.0]])


def test_mlp_rejects_empty_data_and_incomplete_validation():
    model = MLP(1, (), 1, verbose=False)
    with pytest.raises(ValueError, match='nonzero'):
        model.validate(np.empty((0, 1)), np.empty((0, 1)), MSELoss())
    X = np.ones((2, 1))
    with pytest.raises(ValueError, match='together'):
        model.fit(X, X, MSELoss(), SGD(model.param), X_val=X)
