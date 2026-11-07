"""Versioned, weights-only checkpoints for NuNet's built-in modules."""

import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np


_FORMAT = "nunet.weights"
_VERSION = 1
_METADATA = "__metadata__"


def _describe(model):
    # Explicit fields exclude transient forward caches and initialization choices.
    # No classes are imported or constructed based on checkpoint contents.
    from nunet.layer import Conv2d, Flatten, Linear, MaxPool2d, Sequential
    from nunet.layer.activation import ReLU, Sigmoid
    from nunet.nn import ConvNet, MLP

    fields = {
        Linear: ("in_features", "out_features", "bias"),
        Conv2d: ("in_channels", "out_channels", "kernel_size", "stride", "padding", "bias"),
        MaxPool2d: ("kernel_size", "stride"),
        Flatten: ("start_dim",),
        ReLU: (), Sigmoid: (), Sequential: (), MLP: (),
        ConvNet: ("input_shape",),
    }
    parameters = {}
    seen = set()

    def plain(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    def visit(module, path):
        cls = type(module)
        if cls not in fields:
            raise TypeError(f"checkpoints do not support custom module {cls.__name__}")
        if id(module) in seen:
            raise ValueError("checkpoints do not support shared layer instances")
        seen.add(id(module))
        description = {
            "type": f"{cls.__module__}.{cls.__qualname__}",
            "config": {name: plain(getattr(module, name)) for name in fields[cls]},
        }
        if isinstance(module, Sequential):
            description["layers"] = [
                visit(child, f"{path}layers.{index}.")
                for index, child in enumerate(module.layers)
            ]
        else:
            description["parameters"] = []
            for index, parameter in enumerate(module.param):
                if parameter.dtype.kind not in "fiu":
                    raise ValueError("checkpoint parameters must have real numeric dtypes")
                name = f"{path}parameters.{index}"
                parameters[name] = parameter
                description["parameters"].append({
                    "name": name, "shape": list(parameter.shape), "dtype": parameter.dtype.str,
                })
        return description

    return visit(model, ""), parameters


def save(model, path):
    """Atomically save weights and compatibility metadata to exactly ``path``.

    Only built-in layers and models are supported. Optimizer settings, gradients,
    random state, and forward caches are not stored.
    """
    description, parameters = _describe(model)
    metadata = {"format": _FORMAT, "version": _VERSION, "model": description}
    arrays = {name: np.asarray(value) for name, value in parameters.items()}
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **{_METADATA: np.array(json.dumps(metadata)), **arrays})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load(model, path):
    """Load weights into an already constructed matching model, without pickle.

    Validate the complete archive before copying any weights. The existing
    parameter arrays are retained so optimizers still reference the same model.
    """
    description, parameters = _describe(model)
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("expected a NuNet NPZ checkpoint")
        with archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("checkpoint contains duplicate entries")
            if _METADATA not in archive.files:
                raise ValueError("checkpoint metadata is missing")
            encoded = archive[_METADATA]
            if encoded.shape != () or encoded.dtype.kind != "U":
                raise ValueError("checkpoint metadata must be a JSON string")
            metadata = json.loads(encoded.item())
            if not isinstance(metadata, dict) or metadata.get("format") != _FORMAT:
                raise ValueError("unrecognized checkpoint format")
            if type(metadata.get("version")) is not int or metadata["version"] != _VERSION:
                raise ValueError(f"unsupported checkpoint version: {metadata.get('version')!r}")
            if metadata != {"format": _FORMAT, "version": _VERSION, "model": description}:
                raise ValueError("checkpoint architecture or parameter specifications do not match the model")
            if set(archive.files) != {_METADATA, *parameters}:
                raise ValueError("checkpoint parameter entries do not match the model")
            values = {}
            for name, parameter in parameters.items():
                value = archive[name]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(f"checkpoint parameter {name} has an incompatible shape or dtype")
                if not parameter.flags.writeable:
                    raise ValueError(f"model parameter {name} is not writeable")
                values[name] = value
    except (ValueError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(f"Invalid NuNet checkpoint: {error}") from error

    for name, parameter in parameters.items():
        np.copyto(parameter, values[name], casting="no")
