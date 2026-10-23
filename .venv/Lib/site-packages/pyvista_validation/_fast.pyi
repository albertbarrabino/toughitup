"""The C extension; see ``_accelerate`` for how it is used."""

from collections.abc import Callable
from typing import TypeVar

_F = TypeVar('_F', bound=Callable[..., object])

def wrap(function: _F, name: str, text_signature: str, module: str, /) -> _F: ...
