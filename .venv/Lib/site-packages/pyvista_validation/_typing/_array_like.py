"""Array-like type definitions.

The array-like aliases are generic over ``NumberType``, the Python scalar type of a
sequence's items, which defaults to ``float`` so that ``ArrayLike`` and
``ArrayLike[float]`` are the same type. NumPy arrays of any numeric dtype are
accepted whatever the parameter, since a dtype is not a Python scalar type.
``np.float64`` items are accepted through their ``float`` subclass, and ``int``
and ``bool`` items through numeric promotion.
"""

from __future__ import annotations

from collections.abc import Sequence
import sys
from typing import TYPE_CHECKING
from typing import TypeAlias
from typing import Union

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    # Type variable defaults (PEP 696) reached the standard library in 3.13.
    from typing_extensions import TypeVar

# Every NumPy scalar type this package produces or preserves.
_Scalar = (
    np.float64
    | np.float32
    | np.float16
    | np.int64
    | np.int32
    | np.int16
    | np.int8
    | np.uint64
    | np.uint32
    | np.uint16
    | np.uint8
    | np.bool_
)
_Floating = np.float64 | np.float32 | np.float16
_Integer = np.int64 | np.int32 | np.int16 | np.int8 | np.uint64 | np.uint32 | np.uint16 | np.uint8
# What ``must_be_real`` admits: to NumPy a boolean is not a number.
_Real = _Floating | _Integer
# Text, which NumPy holds as fixed-width unicode or bytes; accepted with ``must_be_real=False``.
_Text = np.str_ | np.bytes_
_AnyScalar = _Scalar | _Text

# Anything np.dtype() accepts for numeric data: a scalar type, a dtype, or a dtype name.
if TYPE_CHECKING:
    from typing_extensions import Never

    # Empty list literals, which NumPy turns into float64 arrays; only a type checker
    # can tell them apart from other lists, so the runtime value is a placeholder.
    _EmptyList: TypeAlias = (
        list[Never] | list[list[Never]] | list[list[list[Never]]] | list[list[list[list[Never]]]]
    )
    _DTypeLike: TypeAlias = (
        type[np.generic[object] | float | int | bool] | np.dtype[np.generic[object]] | str
    )
else:
    _DTypeLike = npt.DTypeLike
    _EmptyList = list

# For overload signatures that return the same dtype they are given; used bare, it is any
# of them.
_ScalarT = TypeVar('_ScalarT', bound=_Scalar, default=_Scalar)
NumpyArray = npt.NDArray[_ScalarT]

# The Python scalar type of a sequence's items. Its default makes a bare ``ArrayLike`` mean
# ``ArrayLike[float]``, which accepts ints and bools as well through numeric promotion.
NumberType = TypeVar('NumberType', bound=float, default=float)
Number = float

_NestedBool = (
    Sequence[bool]
    | Sequence[Sequence[bool]]
    | Sequence[Sequence[Sequence[bool]]]
    | Sequence[Sequence[Sequence[Sequence[bool]]]]
)
_NestedInt = (
    Sequence[int]
    | Sequence[Sequence[int]]
    | Sequence[Sequence[Sequence[int]]]
    | Sequence[Sequence[Sequence[Sequence[int]]]]
)
_NestedFloat = (
    Sequence[float]
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]]
    | Sequence[Sequence[Sequence[Sequence[float]]]]
)
_NestedStr = (
    Sequence[str]
    | Sequence[Sequence[str]]
    | Sequence[Sequence[Sequence[str]]]
    | Sequence[Sequence[Sequence[Sequence[str]]]]
)

# What ndarray.tolist() returns for each dtype family, up to four dimensions.
_NestedListBool = (
    list[bool] | list[list[bool]] | list[list[list[bool]]] | list[list[list[list[bool]]]]
)
_NestedListInt = list[int] | list[list[int]] | list[list[list[int]]] | list[list[list[list[int]]]]
_NestedListFloat = (
    list[float] | list[list[float]] | list[list[list[float]]] | list[list[list[list[float]]]]
)
_NestedTupleBool = (
    tuple[bool, ...]
    | tuple[tuple[bool, ...], ...]
    | tuple[tuple[tuple[bool, ...], ...], ...]
    | tuple[tuple[tuple[tuple[bool, ...], ...], ...], ...]
)
_NestedTupleInt = (
    tuple[int, ...]
    | tuple[tuple[int, ...], ...]
    | tuple[tuple[tuple[int, ...], ...], ...]
    | tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]
)
_NestedTupleFloat = (
    tuple[float, ...]
    | tuple[tuple[float, ...], ...]
    | tuple[tuple[tuple[float, ...], ...], ...]
    | tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
)
_NestedListStr = list[str] | list[list[str]] | list[list[list[str]]] | list[list[list[list[str]]]]
_NestedTupleStr = (
    tuple[str, ...]
    | tuple[tuple[str, ...], ...]
    | tuple[tuple[tuple[str, ...], ...], ...]
    | tuple[tuple[tuple[tuple[str, ...], ...], ...], ...]
)
_FiniteNestedList = _NestedListFloat
_FiniteNestedTuple = _NestedTupleFloat

# What converting an array of each dtype family to lists or tuples produces; 0-D gives a scalar.
_ToListBool = bool | _NestedListBool
_ToListInt = int | _NestedListInt
_ToListFloat = float | _NestedListFloat
_ToList = _ToListBool | _ToListInt | _ToListFloat
_ToTupleBool = bool | _NestedTupleBool
_ToTupleInt = int | _NestedTupleInt
_ToTupleFloat = float | _NestedTupleFloat
_ToTuple = _ToTupleBool | _ToTupleInt | _ToTupleFloat
_ToListStr = str | _NestedListStr
_ToTupleStr = str | _NestedTupleStr
# The list and tuple results once text is admitted too.
_ToAnyList = _ToList | _ToListStr
_ToAnyTuple = _ToTuple | _ToTupleStr

# Sequences may mix Python and NumPy scalars, or hold arrays as their innermost items.
# Spelled with ``Union`` because a ``types.UnionType`` cannot be subscripted on Python 3.10.
_ArrayLike1D = Union[
    npt.NDArray[_Scalar],
    Sequence[Union[NumberType, _Scalar]],
    Sequence[npt.NDArray[_Scalar]],
]
_ArrayLike2D = Union[
    npt.NDArray[_Scalar],
    Sequence[Sequence[Union[NumberType, _Scalar]]],
    Sequence[Sequence[npt.NDArray[_Scalar]]],
]
_ArrayLike3D = Union[
    npt.NDArray[_Scalar],
    Sequence[Sequence[Sequence[Union[NumberType, _Scalar]]]],
    Sequence[Sequence[Sequence[npt.NDArray[_Scalar]]]],
]
_ArrayLike4D = Union[
    npt.NDArray[_Scalar],
    Sequence[Sequence[Sequence[Sequence[Union[NumberType, _Scalar]]]]],
    Sequence[Sequence[Sequence[Sequence[npt.NDArray[_Scalar]]]]],
]
_ArrayLike = Union[
    _ArrayLike1D[NumberType],
    _ArrayLike2D[NumberType],
    _ArrayLike3D[NumberType],
    _ArrayLike4D[NumberType],
]

# The same shapes once text is admitted: any array the package handles, scalars included.
_AnyItem = Union[float, str, bytes, _AnyScalar]
_AnyArrayLike1D = Union[
    npt.NDArray[_AnyScalar], Sequence[_AnyItem], Sequence[npt.NDArray[_AnyScalar]]
]
_AnyArrayLike2D = Union[
    npt.NDArray[_AnyScalar],
    Sequence[Sequence[_AnyItem]],
    Sequence[Sequence[npt.NDArray[_AnyScalar]]],
]
_AnyArrayLike3D = Union[
    npt.NDArray[_AnyScalar],
    Sequence[Sequence[Sequence[_AnyItem]]],
    Sequence[Sequence[Sequence[npt.NDArray[_AnyScalar]]]],
]
_AnyArrayLike4D = Union[
    npt.NDArray[_AnyScalar],
    Sequence[Sequence[Sequence[Sequence[_AnyItem]]]],
    Sequence[Sequence[Sequence[Sequence[npt.NDArray[_AnyScalar]]]]],
]
_AnyArrayLike = Union[_AnyArrayLike1D, _AnyArrayLike2D, _AnyArrayLike3D, _AnyArrayLike4D]
_AnyArrayLikeOrScalar = Union[float, str, bytes, _AnyScalar, _AnyArrayLike]
