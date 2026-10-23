"""Bind the public functions to their C fast paths when the extension is available.

Set ``PYVISTA_VALIDATION_ACCELERATE`` to ``0``, ``false``, ``off`` or ``no``, in any
case, to stay on the pure Python implementations.
"""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

_wrap: Callable[[Any, str, str, str], Any] | None = None


def disabled(value: str | None, /) -> bool:
    """Return whether an environment value asks for the pure Python implementations."""
    return value is not None and value.strip().lower() in {'0', 'false', 'off', 'no'}


if not disabled(os.environ.get('PYVISTA_VALIDATION_ACCELERATE')):
    try:
        from pyvista_validation._fast import wrap as _wrap
    except ImportError:  # pragma: no cover - the extension is optional
        _wrap = None

# Whether the C fast paths are in use.
enabled = _wrap is not None
# The Python implementations, by name.
reference: dict[str, Callable[..., Any]] = {}


def accelerate(namespace: dict[str, Any], names: Iterable[str]) -> None:
    """Rebind each named function in ``namespace`` to its C builtin, if it has one."""
    for name in names:
        function = namespace[name]
        reference[name] = function
        if _wrap is not None:
            namespace[name] = _wrap(function, name, text_signature(function), function.__module__)


def text_signature(function: Callable[..., object]) -> str:
    """Write the signature the way CPython reads ``__text_signature__`` for builtins."""
    parts = ['$module']
    positional_only = True
    starred = False
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind is parameter.POSITIONAL_ONLY:
            positional_only = True
        elif positional_only:
            parts.append('/')
            positional_only = False
        if parameter.kind is parameter.VAR_POSITIONAL:
            parts.append(f'*{parameter.name}')
            starred = True
            continue
        if parameter.kind is parameter.VAR_KEYWORD:
            parts.append(f'**{parameter.name}')
            continue
        if parameter.kind is parameter.KEYWORD_ONLY and not starred:
            parts.append('*')
            starred = True
        default = '' if parameter.default is parameter.empty else f'={parameter.default!r}'
        parts.append(f'{parameter.name}{default}')
    if positional_only:
        parts.append('/')
    return '(' + ', '.join(parts) + ')'
