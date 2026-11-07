import json

import numpy as np
import pytest

from nunet.layer import Conv2d, Flatten, Linear, MaxPool2d, Sequential
from nunet.layer.activation import ReLU, Sigmoid
from nunet.nn import ConvNet, MLP
from nunet.optim import SGD


def rewrite_archive(path, update):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    update(arrays)
    with path.open('wb') as stream:
        np.savez(stream, **arrays)


@pytest.mark.parametrize('factory,shape', [
    (lambda: MLP(3, [4], 2, verbose=False), (5, 3)),
    (lambda: ConvNet((1, 4, 6), 2, hidden_channels=3), (5, 1, 4, 6)),
])
def test_roundtrip_predictions_and_optimizer_references(tmp_path, factory, shape):
    original = factory()
    restored = factory()
    optimizer = SGD(restored.param, lr=0.1)
    references = list(optimizer.params)
    input = np.random.default_rng(8).normal(size=shape)
    expected = original(input)
    path = tmp_path / 'weights.custom-extension'
    assert original.save(path) is original
    assert path.exists()
    assert not path.with_name(path.name + '.npz').exists()
    assert restored.load(path) is restored
    np.testing.assert_array_equal(restored(input), expected)
    assert all(a is b for a, b in zip(optimizer.params, SGD(restored.param).params))
    for parameter in references:
        parameter.grad = np.ones(parameter.shape)
    before = [parameter.copy() for parameter in references]
    optimizer.step()
    for parameter, previous in zip(references, before):
        np.testing.assert_allclose(parameter, previous - 0.1)


def test_nested_bias_free_model_and_initialization_compatibility(tmp_path):
    source = Sequential(Sequential(Linear(2, 3, bias=False, init_method='ones')), Sigmoid())
    target = Sequential(Sequential(Linear(2, 3, bias=False, init_method='zeros')), Sigmoid())
    path = tmp_path / 'nested.npz'
    source.save(path)
    target.load(path)
    np.testing.assert_array_equal(source([[1, 2]]), target([[1, 2]]))
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {'__metadata__', 'layers.0.layers.0.parameters.0'}
        assert json.loads(archive['__metadata__'].item())['version'] == 1


@pytest.mark.parametrize('source,target', [
    (lambda: Sequential(Linear(2, 3), ReLU()), lambda: Sequential(Linear(2, 3), Sigmoid())),
    (lambda: Conv2d(1, 2, 3, padding=1), lambda: Conv2d(1, 2, 3, padding=0)),
    (lambda: MaxPool2d(2, stride=1), lambda: MaxPool2d(2, stride=2)),
    (lambda: Flatten(1), lambda: Flatten(0)),
    (lambda: ConvNet((1, 4, 6), 2), lambda: ConvNet((1, 6, 4), 2)),
    (lambda: Linear(2, 3), lambda: Linear(3, 3)),
])
def test_incompatible_architecture_rejected_before_mutation(tmp_path, source, target):
    model = target()
    before = [parameter.copy() for parameter in SGD(model.param).params]
    path = tmp_path / 'incompatible.npz'
    source().save(path)
    with pytest.raises(ValueError, match='architecture'):
        model.load(path)
    for previous, parameter in zip(before, SGD(model.param).params):
        np.testing.assert_array_equal(parameter, previous)


@pytest.mark.parametrize('damage', ['shape', 'dtype', 'object', 'missing', 'extra'])
def test_damaged_parameter_rejected_before_any_parameter_changes(tmp_path, damage):
    source = Linear(2, 2, init_method='ones')
    target = Linear(2, 2, init_method='zeros')
    path = tmp_path / 'broken.npz'
    source.save(path)

    def update(arrays):
        if damage == 'shape':
            arrays['parameters.1'] = np.ones(3)
        elif damage == 'dtype':
            arrays['parameters.1'] = np.ones(2, dtype=np.float32)
        elif damage == 'object':
            arrays['parameters.1'] = np.array([object(), object()])
        elif damage == 'missing':
            del arrays['parameters.1']
        else:
            arrays['unexpected'] = np.ones(1)

    rewrite_archive(path, update)
    with pytest.raises(ValueError, match='checkpoint'):
        target.load(path)
    np.testing.assert_array_equal(target.W, 0)
    np.testing.assert_array_equal(target.b, 0)


@pytest.mark.parametrize('metadata', [
    {'format': 'nunet.weights', 'version': 99},
    {'format': 'nunet.weights', 'version': True},
    {'format': 'unknown', 'version': 1},
    [], 'not JSON',
])
def test_invalid_metadata_is_rejected(tmp_path, metadata):
    model = Linear(2, 1)
    path = tmp_path / 'metadata.npz'
    model.save(path)
    encoded = metadata if isinstance(metadata, str) else json.dumps(metadata)
    rewrite_archive(path, lambda arrays: arrays.update(__metadata__=np.array(encoded)))
    with pytest.raises(ValueError, match='checkpoint'):
        model.load(path)


def test_corrupt_file_and_non_archive_are_rejected(tmp_path):
    model = Linear(2, 1)
    path = tmp_path / 'corrupt.npz'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        model.load(path)
    with path.open('wb') as stream:
        np.save(stream, np.zeros(1))
    with pytest.raises(ValueError, match='checkpoint'):
        model.load(path)


def test_failed_save_keeps_existing_checkpoint(tmp_path, monkeypatch):
    source = Linear(2, 1, init_method='ones')
    path = tmp_path / 'weights.npz'
    source.save(path)
    previous = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError('simulated disk failure')

    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='disk failure'):
        source.save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_readonly_parameter_does_not_allow_partial_loading(tmp_path):
    source = Linear(2, 1, init_method='ones')
    target = Linear(2, 1, init_method='zeros')
    path = tmp_path / 'weights.npz'
    source.save(path)
    target.b.flags.writeable = False
    with pytest.raises(ValueError, match='writeable'):
        target.load(path)
    np.testing.assert_array_equal(target.W, 0)


def test_custom_and_shared_layers_are_rejected(tmp_path):
    class CustomLinear(Linear):
        pass

    with pytest.raises(TypeError, match='custom module'):
        CustomLinear(2, 2).save(tmp_path / 'custom.npz')
    layer = Linear(2, 2)
    with pytest.raises(ValueError, match='shared layer'):
        Sequential(layer, layer).save(tmp_path / 'shared.npz')
