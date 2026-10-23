"""Core type aliases."""

from __future__ import annotations

from typing import Union

from pyvista_validation import _lazy_import

from ._array_like import NumberType
from ._array_like import _ArrayLike
from ._array_like import _ArrayLike1D
from ._array_like import _ArrayLike2D
from ._array_like import _Scalar

# Generic over the Python scalar type of sequence items, which defaults to float.
VectorLike = _ArrayLike1D[NumberType]
MatrixLike = _ArrayLike2D[NumberType]
ArrayLike = _ArrayLike[NumberType]

RotationLike = Union[MatrixLike, _lazy_import.vtkMatrix3x3, _lazy_import.Rotation]
TransformLike = Union[RotationLike, _lazy_import.vtkMatrix4x4, _lazy_import.vtkTransform]

# A scalar or any array-like.
_ArrayLikeOrScalar = Union[NumberType, _Scalar, ArrayLike[NumberType]]
