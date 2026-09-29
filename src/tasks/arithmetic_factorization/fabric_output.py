"""Expose immutable shared outputs as Fabric-compatible named tuples."""

from collections import namedtuple
from dataclasses import fields, is_dataclass
from functools import lru_cache


@lru_cache(maxsize=None)
def _record_type(dataclass_type):
    return namedtuple(
        f"Fabric{dataclass_type.__name__}",
        [field.name for field in fields(dataclass_type)],
    )


def _as_fabric_output(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _record_type(type(value))(*(
            _as_fabric_output(getattr(value, field.name)) for field in fields(value)
        ))
    if isinstance(value, dict):
        return {key: _as_fabric_output(item) for key, item in value.items()}
    if isinstance(value, tuple):
        items = tuple(_as_fabric_output(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, list):
        return [_as_fabric_output(item) for item in value]
    return value


def fabric_output_hook(module, inputs, output):
    """Change containers only; preserve tensors, model state and gradients.

    Both Fabric AMP and its backward-hook visitor reject frozen dataclasses.
    Named tuples retain field access while supporting both standard visitors.
    Register only on the model handed to Fabric, after observation transfer.
    """
    return _as_fabric_output(output)
