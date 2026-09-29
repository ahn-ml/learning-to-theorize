"""Eight ground-truth image-editing primitives.

Each primitive is a deterministic, magnitude-fixed edit on one 32x32 RGB CIFAR-10
image held as ``uint8`` in ``[0, 255]``.  The paper-producing generator applied
every primitive at its default magnitude, so no primitive samples randomness
here; :mod:`tasks.image_editing.data.generator` owns all sampling.

Unlike GridWorld, this data module is not torch-free: the released magnitudes
are defined by ``torchvision.transforms.functional`` dispatching to Pillow, and
the paper HDF5 artifacts only reproduce byte-for-byte through those exact
operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image


IMAGE_SIZE = 32
NUM_CHANNELS = 3


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8))


def _to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image)


def _brightness(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        return _to_numpy(TF.adjust_brightness(_to_pil(image), factor))

    return apply


def _hue(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        return _to_numpy(TF.adjust_hue(_to_pil(image), factor))

    return apply


def _horizontal_flip(image: np.ndarray) -> np.ndarray:
    return _to_numpy(TF.hflip(_to_pil(image)))


def _vertical_flip(image: np.ndarray) -> np.ndarray:
    return _to_numpy(TF.vflip(_to_pil(image)))


def _rotation(image: np.ndarray) -> np.ndarray:
    return _to_numpy(TF.rotate(_to_pil(image), 90))


def _masking(image: np.ndarray) -> np.ndarray:
    """Paint a gray square over the top-left ``ratio`` fraction of the image.

    The patch uses the fixed ``max_ratio`` and starts at the top-left corner.
    """

    height, width = image.shape[:2]
    mask_size = int(MASKING_RATIO * min(height, width))
    masked = image.copy()
    masked[0:mask_size, 0:mask_size, :] = MASKING_VALUE
    return masked


MASKING_RATIO = 0.5
MASKING_VALUE = 128

BRIGHTNESS_PLUS_FACTOR = 1.5
BRIGHTNESS_MINUS_FACTOR = 0.5
HUE_PLUS_FACTOR = 0.3
HUE_MINUS_FACTOR = -0.3
ROTATION_DEGREES = 90


@dataclass(frozen=True, slots=True)
class Primitive:
    """One named, deterministic edit applied to a single image."""

    name: str
    apply: Callable[[np.ndarray], np.ndarray]
    repeatable: bool

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape != (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS):
            raise ValueError(
                f"primitive input must be shaped ({IMAGE_SIZE}, {IMAGE_SIZE}, "
                f"{NUM_CHANNELS}), got {tuple(image.shape)}"
            )
        return self.apply(image)


# Registry order is irrelevant to behaviour; every consumer sorts by name so the
# canonical program enumeration stays stable.
_PRIMITIVES: tuple[Primitive, ...] = (
    Primitive("brightness_plus", _brightness(BRIGHTNESS_PLUS_FACTOR), repeatable=True),
    Primitive("brightness_minus", _brightness(BRIGHTNESS_MINUS_FACTOR), repeatable=True),
    Primitive("hue_plus", _hue(HUE_PLUS_FACTOR), repeatable=True),
    Primitive("hue_minus", _hue(HUE_MINUS_FACTOR), repeatable=True),
    Primitive("rotation", _rotation, repeatable=True),
    Primitive("horizontal_flip", _horizontal_flip, repeatable=False),
    Primitive("vertical_flip", _vertical_flip, repeatable=False),
    Primitive("masking", _masking, repeatable=False),
)

PRIMITIVES: Mapping[str, Primitive] = {
    primitive.name: primitive for primitive in _PRIMITIVES
}

#: Primitive pairs that partially cancel, so the generator never composes them.
OPPOSING_PRIMITIVES: tuple[tuple[str, str], ...] = (
    ("brightness_plus", "brightness_minus"),
    ("hue_plus", "hue_minus"),
)


def primitive_names() -> tuple[str, ...]:
    """Return every primitive name in canonical (sorted) order."""

    return tuple(sorted(PRIMITIVES))


def repeatable_primitives() -> tuple[str, ...]:
    """Return primitives the generator may apply twice, in canonical order."""

    return tuple(
        sorted(name for name, prim in PRIMITIVES.items() if prim.repeatable)
    )


def non_repeatable_primitives() -> tuple[str, ...]:
    """Return primitives that are idempotent or self-cancelling when repeated."""

    return tuple(
        sorted(name for name, prim in PRIMITIVES.items() if not prim.repeatable)
    )


def apply_program(image: np.ndarray, program: tuple[str, ...]) -> np.ndarray:
    """Apply a program left to right and return the edited image."""

    result = image.copy()
    for name in program:
        try:
            primitive = PRIMITIVES[name]
        except KeyError as error:
            raise ValueError(f"unknown image-editing primitive {name!r}") from error
        result = primitive(result)
    return result


__all__ = [
    "BRIGHTNESS_MINUS_FACTOR",
    "BRIGHTNESS_PLUS_FACTOR",
    "HUE_MINUS_FACTOR",
    "HUE_PLUS_FACTOR",
    "IMAGE_SIZE",
    "MASKING_RATIO",
    "MASKING_VALUE",
    "NUM_CHANNELS",
    "OPPOSING_PRIMITIVES",
    "PRIMITIVES",
    "Primitive",
    "ROTATION_DEGREES",
    "apply_program",
    "non_repeatable_primitives",
    "primitive_names",
    "repeatable_primitives",
]
