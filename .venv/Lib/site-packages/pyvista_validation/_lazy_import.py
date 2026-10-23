"""Lazily imported names from this package's optional dependencies.

VTK and SciPy are both optional and both slow to import, so their names are
resolved on first access and then cached in this module's globals.

VTK names follow the same backend selection as PyVista: ``PYVISTA_VTK_BACKEND``
if set (``vtk`` meaning stock ``vtkmodules``), otherwise ``cvista`` when it is
installed, otherwise ``vtkmodules``. A flat backend such as cvista exposes its
classes directly off the root package; stock VTK keeps them in submodules, and
importing only the needed submodule keeps the cost down.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation
    from vtkmodules.vtkCommonMath import vtkMatrix3x3
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform

__all__ = ['Rotation', 'vtkMatrix3x3', 'vtkMatrix4x4', 'vtkTransform']

_SCIPY_MODULES = {'Rotation': 'scipy.spatial.transform'}
_VTK_SUBMODULES = {
    'vtkMatrix3x3': 'vtkCommonMath',
    'vtkMatrix4x4': 'vtkCommonMath',
    'vtkTransform': 'vtkCommonTransforms',
}


def _placeholder(name: str) -> type[object]:
    """Return a class nothing is an instance of, standing in for a missing package.

    Unlike ``None`` or an empty tuple it still composes into the ``Union`` type aliases.
    """
    return type(name, (), {})


def _vtk_root() -> str:
    """Return the package VTK names resolve against, matching PyVista's selection."""
    backend = os.environ.get('PYVISTA_VTK_BACKEND')
    if backend:
        return 'vtkmodules' if backend == 'vtk' else backend
    if importlib.util.find_spec('cvista') is not None:
        return 'cvista'
    return 'vtkmodules'


def _import_vtk(name: str) -> type[object]:
    """Import a VTK class from the active backend, or a placeholder without one."""
    root = _vtk_root()
    try:
        package = importlib.import_module(root)
    except ModuleNotFoundError:
        return _placeholder(name)
    # Flat backends resolve classes by name off the root; stock VTK needs the submodule.
    if hasattr(package, name):
        return cast('type[object]', getattr(package, name))
    return cast(
        'type[object]', getattr(importlib.import_module(f'{root}.{_VTK_SUBMODULES[name]}'), name)
    )


def _import_scipy(name: str) -> type[object]:
    """Import a SciPy class, or a placeholder when SciPy is not installed."""
    try:
        return cast('type[object]', getattr(importlib.import_module(_SCIPY_MODULES[name]), name))
    except ModuleNotFoundError:
        return _placeholder(name)


def __getattr__(name: str) -> type[object]:
    """Resolve a name from its optional dependency, caching it for next time."""
    if name in _VTK_SUBMODULES:
        value = _import_vtk(name)
    elif name in _SCIPY_MODULES:
        value = _import_scipy(name)
    else:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg)

    setattr(sys.modules[__name__], name, value)  # __getattr__ only runs on a miss, so this caches
    return value
