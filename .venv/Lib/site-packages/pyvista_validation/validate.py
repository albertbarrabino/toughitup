"""Functions that validate input and return a standard representation.

A ``validate`` function typically:

* Uses :py:mod:`~pyvista_validation.check` functions to
  check the type and/or value of input arguments.
* Applies (optional) constraints -- for example input or output must have a
  specific length, shape, type, data-type, etc.
* Accepts many different input types or values and standardizes the
  output as a single representation with known properties.

"""

from __future__ import annotations

import inspect
import itertools
import reprlib
import sys
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypedDict
from typing import cast
from typing import overload

import numpy as np

from pyvista_validation import _lazy_import
from pyvista_validation._accelerate import accelerate
from pyvista_validation._cast_array import _cast_to_numpy
from pyvista_validation._cast_array import _cast_to_tuple
from pyvista_validation._cast_array import _tolist
from pyvista_validation.check import check_contains
from pyvista_validation.check import check_finite
from pyvista_validation.check import check_integer
from pyvista_validation.check import check_length
from pyvista_validation.check import check_ndim
from pyvista_validation.check import check_nonnegative
from pyvista_validation.check import check_range
from pyvista_validation.check import check_real
from pyvista_validation.check import check_shape
from pyvista_validation.check import check_sorted
from pyvista_validation.check import check_string
from pyvista_validation.check import check_subdtype

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    # Type variable defaults (PEP 696) reached the standard library in 3.13.
    from typing_extensions import TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any
    from typing import TypeAlias

    import numpy.typing as npt
    from typing_extensions import Never
    from typing_extensions import Unpack

    from pyvista_validation._typing import MatrixLike
    from pyvista_validation._typing import RotationLike
    from pyvista_validation._typing import TransformLike
    from pyvista_validation._typing import VectorLike
    from pyvista_validation._typing import _AnyArrayLikeOrScalar
    from pyvista_validation._typing import _AnyScalar
    from pyvista_validation._typing import _ArrayLikeOrScalar
    from pyvista_validation._typing import _DTypeLike
    from pyvista_validation._typing import _EmptyList
    from pyvista_validation._typing import _Floating
    from pyvista_validation._typing import _Integer
    from pyvista_validation._typing import _NestedBool
    from pyvista_validation._typing import _NestedFloat
    from pyvista_validation._typing import _NestedInt
    from pyvista_validation._typing import _NestedStr
    from pyvista_validation._typing import _Real
    from pyvista_validation._typing import _Scalar
    from pyvista_validation._typing import _ScalarT
    from pyvista_validation._typing import _ToAnyList
    from pyvista_validation._typing import _ToAnyTuple
    from pyvista_validation._typing import _ToList
    from pyvista_validation._typing import _ToListBool
    from pyvista_validation._typing import _ToListFloat
    from pyvista_validation._typing import _ToListInt
    from pyvista_validation._typing import _ToListStr
    from pyvista_validation._typing import _ToTuple
    from pyvista_validation._typing import _ToTupleBool
    from pyvista_validation._typing import _ToTupleFloat
    from pyvista_validation._typing import _ToTupleInt
    from pyvista_validation._typing import _ToTupleStr

    from .check import _ShapeLike

    # Everything a validate function can return, by function, for calls whose output mode
    # is not known statically.
    _ArrayOut: TypeAlias = npt.NDArray[_Scalar] | _ToList | _ToTuple
    _NumberOut: TypeAlias = bool | int | float | npt.NDArray[_Scalar]
    _DataRangeOut: TypeAlias = (
        npt.NDArray[_Scalar]
        | list[bool]
        | list[int]
        | list[float]
        | tuple[bool, bool]
        | tuple[int, int]
        | tuple[float, float]
    )
    _ArrayNx3Out: TypeAlias = (
        npt.NDArray[_Scalar]
        | list[list[bool]]
        | list[list[int]]
        | list[list[float]]
        | tuple[tuple[bool, bool, bool], ...]
        | tuple[tuple[int, int, int], ...]
        | tuple[tuple[float, float, float], ...]
    )
    _ArrayNOut: TypeAlias = (
        npt.NDArray[_Scalar]
        | list[bool]
        | list[int]
        | list[float]
        | tuple[bool, ...]
        | tuple[int, ...]
        | tuple[float, ...]
    )
    _ArrayNUnsignedOut: TypeAlias = npt.NDArray[_Integer] | list[int] | tuple[int, ...]
    _Array3Out: TypeAlias = (
        npt.NDArray[_Scalar]
        | list[bool]
        | list[int]
        | list[float]
        | tuple[bool, bool, bool]
        | tuple[int, int, int]
        | tuple[float, float, float]
    )
    _DimensionalityOut: TypeAlias = bool | int | float | npt.NDArray[_Scalar]
    # The same, once text is admitted with must_be_real=False.
    _AnyArrayOut: TypeAlias = npt.NDArray[_AnyScalar] | _ToAnyList | _ToAnyTuple
    _DataRangeAnyOut: TypeAlias = (
        _DataRangeOut | npt.NDArray[_AnyScalar] | list[str] | tuple[str, str]
    )
    _ArrayNx3AnyOut: TypeAlias = (
        _ArrayNx3Out | npt.NDArray[_AnyScalar] | list[list[str]] | tuple[tuple[str, str, str], ...]
    )
    _ArrayNAnyOut: TypeAlias = _ArrayNOut | npt.NDArray[_AnyScalar] | list[str] | tuple[str, ...]
    _Array3AnyOut: TypeAlias = (
        _Array3Out | npt.NDArray[_AnyScalar] | list[str] | tuple[str, str, str]
    )
    # A 2-D float array, which is what NumPy's typed linear algebra operates on.
    _Matrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# For validate_arrayN_unsigned, whose dtype_out must be an integer type.
_IntegerT = TypeVar('_IntegerT', bound='_Integer', default='_Integer')
# For the overloads that keep a real dtype, which must_be_real=True guarantees.
_RealT = TypeVar('_RealT', bound='_Real', default='_Real')


class _SortedKwargs(TypedDict, total=False):
    """Keyword arguments for ``check_sorted``, passed through ``must_be_sorted``."""

    ascending: bool
    strict: bool
    axis: int | None


# The keyword arguments of validate_array, grouped so each validate function can declare the
# subset it accepts; the ones it sets itself are left out and rejected statically.
class _CheckKwargs(TypedDict, total=False):
    """Checks every validate function accepts."""

    must_have_dtype: _DTypeLike | None
    must_have_length: int | VectorLike | None
    must_have_min_length: int | None
    must_have_max_length: int | None
    must_be_finite: bool
    must_be_in_range: VectorLike | None
    strict_lower_bound: bool
    strict_upper_bound: bool
    as_any: bool
    copy: bool
    name: str


class _NonnegativeKwargs(TypedDict, total=False):
    """The ``must_be_nonnegative`` check."""

    must_be_nonnegative: bool


class _IntegerKwargs(TypedDict, total=False):
    """The ``must_be_integer`` check."""

    must_be_integer: bool


class _SortedCheckKwargs(TypedDict, total=False):
    """The ``must_be_sorted`` check."""

    must_be_sorted: bool | _SortedKwargs


class _NdimKwargs(TypedDict, total=False):
    """The ``must_have_ndim`` check."""

    must_have_ndim: int | VectorLike | None


class _ShapeKwargs(TypedDict, total=False):
    """The ``must_have_shape`` check."""

    must_have_shape: _ShapeLike | list[_ShapeLike] | None


class _ReshapeKwargs(TypedDict, total=False):
    """The ``reshape_to`` option."""

    reshape_to: int | tuple[int, ...] | None


class _BroadcastKwargs(TypedDict, total=False):
    """The ``broadcast_to`` option."""

    broadcast_to: int | tuple[int, ...] | None


class _OutputKwargs(TypedDict, total=False):
    """The options that select the output type."""

    dtype_out: _DTypeLike | None
    to_list: bool
    to_tuple: bool


class _RealKwargs(TypedDict, total=False):
    """The option that decides whether booleans and text are accepted."""

    must_be_real: bool


class _ArrayKwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _SortedCheckKwargs,
    _NdimKwargs,
    _ShapeKwargs,
    _ReshapeKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Keyword arguments of ``validate_array`` other than the output options."""


class _FlatKwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _SortedCheckKwargs,
    _ShapeKwargs,
    _ReshapeKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Every check except ``must_have_ndim``, which the one-dimensional overloads name."""


class _AllKwargs(_ArrayKwargs, _OutputKwargs, _RealKwargs, total=False):
    """Every keyword argument of ``validate_array``."""


class _NumberKwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _SortedCheckKwargs,
    _NdimKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Keyword arguments ``validate_number`` forwards."""


class _DataRangeKwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _NdimKwargs,
    _ReshapeKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Keyword arguments ``validate_data_range`` forwards."""


class _ArrayNKwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _SortedCheckKwargs,
    _NdimKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Keyword arguments ``validate_arrayN`` and ``validate_arrayNx3`` forward."""


class _ArrayNKwargsAll(_ArrayNKwargs, _OutputKwargs, _RealKwargs, total=False):
    """Every keyword argument of ``validate_arrayN``."""


class _ArrayNUnsignedKwargs(
    _CheckKwargs,
    _SortedCheckKwargs,
    _NdimKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Keyword arguments ``validate_arrayN_unsigned`` forwards."""


class _Array3Kwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _SortedCheckKwargs,
    _NdimKwargs,
    total=False,
):
    """Keyword arguments ``validate_array3`` forwards."""


class _DimensionalityKwargs(
    _CheckKwargs,
    _NonnegativeKwargs,
    _IntegerKwargs,
    _SortedCheckKwargs,
    _NdimKwargs,
    _BroadcastKwargs,
    total=False,
):
    """Keyword arguments ``validate_dimensionality`` forwards."""


# fmt: off
@overload
def validate_array(arr: npt.NDArray[_RealT] | _RealT, /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[_RealT]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _EmptyList, /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[_ScalarT] | _ScalarT, /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: bool | _NestedBool, /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: int | _NestedInt, /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_array(arr: float | _NestedFloat, /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_array(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr, /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.str_]: ...
@overload
def validate_array(arr: npt.NDArray[_RealT] | Sequence[_RealT] | Sequence[Sequence[_RealT]] | Sequence[npt.NDArray[_RealT]] | Sequence[Sequence[npt.NDArray[_RealT]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[bool], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[int], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[float], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_array(arr: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> npt.NDArray[_AnyScalar]: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _EmptyList, /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListFloat: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[np.bool_] | np.bool_ | bool | _NestedBool | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[bool]: ...
@overload
def validate_array(arr: npt.NDArray[np.bool_] | np.bool_ | bool | _NestedBool | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListBool: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[_Integer] | _Integer | int | _NestedInt | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[int]: ...
@overload
def validate_array(arr: npt.NDArray[_Integer] | _Integer | int | _NestedInt | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListInt: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[_Floating] | _Floating | float | _NestedFloat | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[float]: ...
@overload
def validate_array(arr: npt.NDArray[_Floating] | _Floating | float | _NestedFloat | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListFloat: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[str]: ...
@overload
def validate_array(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListStr: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[bool]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListBool: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[int]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListInt: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[float]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToListFloat: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> list[bool] | list[int] | list[float]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToList: ...
@overload
def validate_array(arr: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayKwargs]) -> _ToAnyList: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _EmptyList, /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleFloat: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[np.bool_] | np.bool_ | bool | _NestedBool | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[bool, ...]: ...
@overload
def validate_array(arr: npt.NDArray[np.bool_] | np.bool_ | bool | _NestedBool | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleBool: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[_Integer] | _Integer | int | _NestedInt | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[int, ...]: ...
@overload
def validate_array(arr: npt.NDArray[_Integer] | _Integer | int | _NestedInt | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleInt: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[_Floating] | _Floating | float | _NestedFloat | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[float, ...]: ...
@overload
def validate_array(arr: npt.NDArray[_Floating] | _Floating | float | _NestedFloat | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleFloat: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[str, ...]: ...
@overload
def validate_array(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleStr: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[bool, ...]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleBool: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[int, ...]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleInt: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[float, ...]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTupleFloat: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], must_have_ndim: Literal[1], **kwargs: Unpack[_FlatKwargs]) -> tuple[bool, ...] | tuple[int, ...] | tuple[float, ...]: ...
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToTuple: ...
@overload
def validate_array(arr: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayKwargs]) -> _ToAnyTuple: ...  # type: ignore[overload-overlap]
@overload
def validate_array(arr: _ArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayKwargs]) -> _ArrayOut: ...
@overload
def validate_array(arr: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayKwargs]) -> _AnyArrayOut: ...
# fmt: on
def validate_array(
    arr: _AnyArrayLikeOrScalar,
    /,
    *,
    must_have_shape: _ShapeLike | list[_ShapeLike] | None = None,
    must_have_ndim: int | VectorLike | None = None,
    must_have_dtype: _DTypeLike | None = None,
    must_have_length: int | VectorLike | None = None,
    must_have_min_length: int | None = None,
    must_have_max_length: int | None = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: bool | _SortedKwargs = False,
    must_be_in_range: VectorLike | None = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    reshape_to: int | tuple[int, ...] | None = None,
    broadcast_to: int | tuple[int, ...] | None = None,
    dtype_out: _DTypeLike | None = None,
    as_any: bool = True,
    copy: bool = False,
    to_list: bool = False,
    to_tuple: bool = False,
    name: str = 'Array',
) -> _AnyArrayOut:
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape,
    data-type, and/or has values that meet specific
    requirements such as being sorted, integer-like, or finite.

    The array's output can also be reshaped or broadcast, cast as a
    nested tuple or list array, or cast to a specific data type.

    See Also
    --------
    validate_number
        Specialized function for single numbers.

    validate_array3
        Specialized function for 3-element arrays.

    validate_arrayN
        Specialized function for one-dimensional arrays.

    validate_arrayNx3
        Specialized function for Nx3 dimensional arrays.

    validate_data_range
        Specialized function for data ranges.

    Parameters
    ----------
    arr : array_like
        Array to be validated, in any form that can be converted to
        a :class:`np.ndarray`. This includes lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ``ndarrays``.

    must_have_shape : int | tuple[int, ...] | list[int, tuple[int, ...]], optional
        :func:`Check <pyvista_validation.check.check_shape>`
        if the array has a specific shape. Specify a single shape
        or a ``list`` of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of ``-1`` for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (that is, 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    must_have_ndim : int | VectorLike, optional
        :func:`Check <pyvista_validation.check.check_ndim>` if
        the array has the specified number of dimensions. Specify a
        single dimension or a sequence of allowable dimensions. If a
        sequence, the array must have at least one of the specified
        number of dimensions.

    must_have_dtype : DTypeLike | list[DTypeLike, ...], optional
        :func:`Check <pyvista_validation.check.check_subdtype>`
        if the array's data-type has the given ``dtype``. Specify a
        :class:`np.dtype` object or dtype-like base class which the
        array's data must be a subtype of. If a ``list``, the array's data
        must be a subtype of at least one of the specified ``dtypes``.

    must_have_length : int | VectorLike, optional
        :func:`Check <pyvista_validation.check.check_length>`
        if the array has the given length. If multiple values are given,
        the array's length must match one of the values.

        .. note ::

            The array's length is determined after reshaping the array
            (if ``reshape`` is not ``None``) and after broadcasting (if
            ``broadcast_to`` is not ``None``). Therefore, the values of
            `length`` should take the array's new shape into
            consideration if applicable.

    must_have_min_length : int, optional
        :func:`Check <pyvista_validation.check.check_length>`
        if the array's length is this value or greater.

    must_have_max_length : int, optional
        :func:`Check <pyvista_validation.check.check_length>`
        if the array' length is this value or less.

    must_be_nonnegative : bool, default: False
        :func:`Check <pyvista_validation.check.check_nonnegative>`
        if all elements of the array are nonnegative.

    must_be_finite : bool, default: False
        :func:`Check <pyvista_validation.check.check_finite>`
        if all elements of the array are finite, that is, not ``infinity``
        and not Not a Number (``NaN``).

    must_be_real : bool, default: True
        :func:`Check <pyvista_validation.check.check_real>`
        if the array has real numbers, that is, its data type is integer or
        floating. Boolean, text and complex arrays are rejected; pass ``False``
        to accept them. To reject only booleans, use ``must_have_dtype=np.number``.

    must_be_integer : bool, default: False
        :func:`Check <pyvista_validation.check.check_integer>`
        if the array's values are integer-like (that is, that
        ``np.all(arr, np.floor(arr))``).

    must_be_sorted : bool | dict, default: False
        :func:`Check <pyvista_validation.check.check_sorted>`
        if the array's values are sorted. If ``True``, the check is
        performed with default parameters:

        * ``ascending=True``: the array must be sorted in ascending order
        * ``strict=False``: sequential elements with the same value are allowed
        * ``axis=-1``: the sorting is checked along the array's last axis

        To check for descending order, enforce strict ordering, or to check
        along a different axis, use a ``dict`` with keyword arguments that
        will be passed to ``check_sorted``.

    must_be_in_range : VectorLike, optional
        :func:`Check <pyvista_validation.check.check_range>`
        if the array's values are all within a specific range. Range
        must be array-like with two elements specifying the minimum and
        maximum data values allowed, respectively. By default, the range
        endpoints are inclusive, that is, values must be >= minimum and <=
        maximum. Use ``strict_lower_bound`` and/or ``strict_upper_bound``
        to further restrict the allowable range.

        ..note ::

            Use ``np.inf`` to check for open intervals, e.g.:

            * ``[-np.inf, upper_bound]`` to check if values are less
              than (or equal to)  ``upper_bound``
            * ``[lower_bound, np.inf]`` to check if values are greater
              than (or equal to) ``lower_bound``

    strict_lower_bound : bool, default: False
        Enforce a strict lower bound for the range specified by
        ``must_be_in_range``, that is, array values must be strictly greater
        than the specified minimum.

    strict_upper_bound : bool, default: False
        Enforce a strict upper bound for the range specified by
        ``must_be_in_range``, that is, array values must be strictly less
        than the specified maximum.

    reshape_to : int | tuple[int, ...], optional
        Reshape the output array to a new shape with :func:`np.reshape`.
        The shape should be compatible with the original shape. If an
        integer, then the result will be a 1-D array of that length. One
        shape dimension can be -1.

    broadcast_to : int | tuple[int, ...], optional
        Broadcast the array with :func:`np.broadcast_to` to a
        read-only view with the specified shape. Broadcasting is done
        after reshaping (if ``reshape_to`` is not ``None``).

    dtype_out : DTypeLike, optional
        Set the data-type of the returned array. By default, the
        ``dtype`` is inferred from the input data.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through. Otherwise, a
        subclass is returned as a base ``np.ndarray`` view of the same data.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A new array is always
        made from a sequence, and one may also be made to satisfy ``dtype_out``.

    to_list : bool, default: False
        Return the validated array as a ``list`` or nested ``list``. Scalar
        values are always returned as a ``Number``  (that is, ``int`` or ``float``).
        Has no effect if ``to_tuple=True``.

    to_tuple : bool, default: False
        Return the validated array as a ``tuple`` or nested ``tuple``. Scalar
        values are always returned as a ``Number``  (that is, ``int`` or ``float``).

    name : str, default: "Array"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    array_like
        Validated array. Returned object is:

        * an instance of ``np.ndarray`` (default), or
        * a nested ``list`` (if ``to_list=True``), or
        * a nested ``tuple`` (if ``to_tuple=True``), or
        * a ``Number`` (that is, ``int`` or ``float``) if the input is a scalar.

    Examples
    --------
    Validate a one-dimensional array has at least length two, is
    monotonically increasing (that is, has strict ascending order), and
    is within some range.

    >>> from pyvista_validation import validate_array
    >>> array_in = (1, 2, 3, 5, 8, 13)
    >>> rng = (0, 20)
    >>> validate_array(
    ...     array_in,
    ...     must_have_shape=(-1),
    ...     must_have_min_length=2,
    ...     must_be_sorted=dict(strict=True),
    ...     must_be_in_range=rng,
    ... )
    array([ 1,  2,  3,  5,  8, 13])

    """
    arr_out = _cast_to_numpy(arr, as_any=as_any, copy=copy)
    if must_be_real:
        check_real(arr_out, name=name)

    # Check dtype
    if must_have_dtype is not None:
        check_subdtype(arr_out, must_have_dtype, name=name)

    # Check shape
    if must_have_shape is not None:
        check_shape(arr_out, must_have_shape, name=name)
    if must_have_ndim is not None:
        check_ndim(arr_out, ndim=must_have_ndim, name=name)

    # Do reshape _after_ checking shape to prevent unexpected reshaping
    if reshape_to is not None and arr_out.shape != reshape_to:
        arr_out = arr_out.reshape(reshape_to)
    if broadcast_to is not None and arr_out.shape != broadcast_to:
        arr_out = np.broadcast_to(arr_out, broadcast_to, subok=True)

    # Check length _after_ reshaping otherwise length may be wrong
    if (
        must_have_length is not None
        or must_have_min_length is not None
        or must_have_max_length is not None
    ):
        check_length(
            arr_out,
            exact_length=must_have_length,
            min_length=must_have_min_length,
            max_length=must_have_max_length,
            allow_scalar=True,
            name=name,
        )

    # Check data values. These checks are for numbers, which the cast states; NumPy raises for
    # anything else.
    numeric = cast('npt.NDArray[_Scalar]', arr_out)
    if must_be_nonnegative:
        check_nonnegative(numeric, name=name)
    # Check finite before setting dtype since dtype change can fail with inf
    if must_be_finite:
        check_finite(numeric, name=name)
    if must_be_integer:
        check_integer(numeric, strict=False, name=name)
    if must_be_in_range is not None:
        check_range(
            numeric,
            must_be_in_range,
            strict_lower=strict_lower_bound,
            strict_upper=strict_upper_bound,
            name=name,
        )
    if must_be_sorted:
        if isinstance(must_be_sorted, bool):
            check_sorted(arr_out, name=name)
        else:
            check_sorted(arr_out, **must_be_sorted, name=name)

    # Process output
    if dtype_out is not None:
        # Copy was done earlier, so don't do it again here
        arr_out = _astype(arr_out, dtype_out)
    if to_tuple:
        return _cast_to_tuple(arr_out)
    if to_list:
        return _tolist(arr_out)
    return arr_out


# fmt: off
@overload
def validate_axes(*axes: VectorLike | MatrixLike, normalize: Literal[True] = ..., must_be_orthogonal: bool = ..., must_have_orientation: Literal['right', 'left'] | None = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_axes(*axes: VectorLike | MatrixLike, normalize: Literal[False], must_be_orthogonal: bool = ..., must_have_orientation: Literal['right', 'left'] | None = ..., name: str = ...) -> npt.NDArray[_Scalar]: ...
@overload
def validate_axes(*axes: VectorLike | MatrixLike, normalize: bool = ..., must_be_orthogonal: bool = ..., must_have_orientation: Literal['right', 'left'] | None = ..., name: str = ...) -> npt.NDArray[_Scalar]: ...
# fmt: on
def validate_axes(
    *axes: VectorLike | MatrixLike,
    normalize: bool = True,
    must_be_orthogonal: bool = True,
    must_have_orientation: Literal['right', 'left'] | None = 'right',
    name: str = 'Axes',
) -> npt.NDArray[_Scalar]:
    """Validate 3D axes vectors.

    By default, the axes are normalized and checked to ensure they are orthogonal and
    have a right-handed orientation.

    Parameters
    ----------
    *axes : VectorLike | MatrixLike
        Axes to be validated. Axes may be specified as a single argument of a 3x3
        array of row vectors or as separate arguments for each 3-element axis vector.
        If only two vectors are given and ``must_have_orientation`` is not ``None``,
        the third vector is automatically calculated as the cross-product of the
        two vectors such that the axes have the correct orientation.

    normalize : bool, default: True
        If ``True``, the axes vectors are individually normalized to each have a norm
        of 1.

    must_be_orthogonal : bool, default: True
        Check if the axes are orthogonal. If ``True``, the cross product between any
        two axes vectors must be parallel to the third.

    must_have_orientation : str, default: 'right'
        Check if the axes have a specific orientation. If ``right``, the
        cross-product of the first axis vector with the second must have a positive
        direction. If ``left``, the direction must be negative. If ``None``, the
        orientation is not checked.

    name : str, default: "Axes"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 axes array of row vectors.

    Examples
    --------
    Validate an axes array.

    >>> import numpy as np
    >>> from pyvista_validation import validate_axes
    >>> validate_axes(np.eye(3))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Validate individual axes vectors as a 3x3 array.

    >>> validate_axes([1, 0, 0], [0, 1, 0], [0, 0, 1])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Create a validated left-handed axes array from two vectors.

    >>> validate_axes([1, 0, 0], [0, 1, 0], must_have_orientation='left')
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])

    """
    check_length(axes, exact_length=[1, 2, 3], name=f'{name} arguments')
    if must_have_orientation is not None:
        check_contains(
            ['right', 'left'],
            must_contain=must_have_orientation,
            name=f'{name} orientation',
        )
    elif len(axes) == 2:
        msg = f'{name} orientation must be specified when only two vectors are given.'
        raise ValueError(msg)

    # Validate axes array
    axes_array: npt.NDArray[_Scalar]
    if len(axes) == 1:
        axes_array = validate_array(axes[0], must_have_shape=(3, 3), name=name)
    else:
        vectors = np.zeros((3, 3))
        vectors[0] = validate_array3(axes[0], name=f'{name} Vector[0]')
        vectors[1] = validate_array3(axes[1], name=f'{name} Vector[1]')
        if len(axes) == 3:
            vectors[2] = validate_array3(axes[2], name=f'{name} Vector[2]')
        else:
            first, second, _ = vectors
            third = np.cross(first, second)
            # Two parallel vectors have no third axis; zero vectors are reported below
            if np.isclose(third, 0).all() and not np.isclose(vectors[:2], 0).all(axis=1).any():
                msg = f'{name} cannot be parallel.'
                raise ValueError(msg)
            if must_have_orientation == 'right':
                vectors[2] = third
            else:
                vectors[2] = np.cross(second, first)
        axes_array = vectors
    check_finite(axes_array, name=name)

    # The checks below are done in floating point; the input dtype is kept for the output
    matrix = axes_array.astype(np.float64, copy=False).reshape((3, 3))
    if np.isclose(matrix, 0).all(axis=1).any():
        msg = f'{name} cannot be zeros.'
        raise ValueError(msg)

    # Normalize axes for dot and cross product calcs
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    axes_norm: _Matrix = matrix / norms
    norm0, norm1, norm2 = axes_norm
    # Parallel and antiparallel unit vectors have a dot product of magnitude one
    pairs = ((norm0, norm1), (norm0, norm2), (norm1, norm2))
    if any(np.isclose(abs(first @ second), 1) for first, second in pairs):
        msg = f'{name} cannot be parallel.'
        raise ValueError(msg)
    cross_0_1 = np.cross(norm0, norm1)
    cross_1_2 = np.cross(norm1, norm2)

    if must_be_orthogonal and not (
        (np.allclose(cross_0_1, norm2) or np.allclose(cross_0_1, -norm2))
        and (np.allclose(cross_1_2, norm0) or np.allclose(cross_1_2, -norm0))
    ):
        msg = f'{name} are not orthogonal.'
        raise ValueError(msg)

    # Check orientation
    # Note: this check is skipped for two vectors since the third axis is
    # computed from the first two, and this check is only relevant for the
    # non-orthogonal case
    if must_have_orientation:
        dot = cross_0_1 @ norm2
        if must_have_orientation == 'right' and dot < 0:
            msg = f'{name} do not have a right-handed orientation.'
            raise ValueError(msg)
        if must_have_orientation == 'left' and dot > 0:
            msg = f'{name} do not have a left-handed orientation.'
            raise ValueError(msg)

    if normalize:
        return axes_norm
    return axes_array


# fmt: off
@overload
def validate_rotation(rotation: npt.NDArray[_ScalarT], must_have_handedness: Literal['right', 'left'] | None = ..., *, tolerance: float = ..., name: str = ...) -> npt.NDArray[_ScalarT]: ...
@overload
def validate_rotation(rotation: Sequence[Sequence[int]], must_have_handedness: Literal['right', 'left'] | None = ..., *, tolerance: float = ..., name: str = ...) -> npt.NDArray[np.int64]: ...
@overload
def validate_rotation(rotation: Sequence[Sequence[float]], must_have_handedness: Literal['right', 'left'] | None = ..., *, tolerance: float = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_rotation(rotation: _lazy_import.vtkMatrix3x3 | _lazy_import.Rotation, must_have_handedness: Literal['right', 'left'] | None = ..., *, tolerance: float = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_rotation(rotation: RotationLike, must_have_handedness: Literal['right', 'left'] | None = ..., *, tolerance: float = ..., name: str = ...) -> npt.NDArray[_Scalar]: ...
# fmt: on
def validate_rotation(
    rotation: RotationLike,
    must_have_handedness: Literal['right', 'left'] | None = None,
    *,
    tolerance: float = 1e-6,
    name: str = 'Rotation',
) -> npt.NDArray[_Scalar]:
    """Validate a rotation as a 3x3 matrix.

    The rotation is valid if it is orthogonal and has a determinant
    of ``1`` (right-handed or "proper" rotation) or ``-1`` (left-handed or "improper"
    rotation). By default, right- and left-handed rotations are allowed.
    Use ``must_have_handedness`` to restrict the handedness.

    Parameters
    ----------
    rotation : RotationLike
        3x3 rotation matrix or a SciPy ``Rotation`` object.

    must_have_handedness : 'right' | 'left' | None, default: None
        Check if the rotation has a specific handedness. If ``right``, the
        determinant must be ``1``. If ``left``, the determinant must be ``-1``.
        By default, either handedness is allowed.

    tolerance : float, default: 1e-6
        Tolerance used for checking orthogonality.

    name : str, default: "Rotation"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 rotation matrix.

    Examples
    --------
    Validate a rotation matrix. The identity matrix is used as a toy example.

    >>> import numpy as np
    >>> from pyvista_validation import validate_rotation
    >>> rotation = np.eye(3)
    >>> validate_rotation(rotation)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    By default, left-handed rotations (which include reflections) are allowed.

    >>> rotation *= -1  # Add reflections
    >>> validate_rotation(rotation)
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])

    """
    check_contains(
        ['right', 'left', None], must_contain=must_have_handedness, name='must_have_handedness'
    )
    rotation_matrix = validate_transform3x3(rotation, name=name)
    # The checks below are done in floating point; the input dtype is kept for the output
    matrix = rotation_matrix.astype(np.float64, copy=False).reshape((3, 3))
    norm_diff = np.linalg.norm(matrix @ matrix.T - np.eye(3), ord='fro')
    if not norm_diff < tolerance:
        msg = f'{name} is not valid. Rotation must be orthogonal.'
        raise ValueError(msg)

    if must_have_handedness is not None:
        det = np.linalg.det(matrix)
        if must_have_handedness == 'right' and not det > 0:
            msg = (
                f'{name} has incorrect handedness. Expected a right-handed rotation, but got a '
                f'left-handed rotation instead.'
            )
            raise ValueError(msg)
        if must_have_handedness == 'left' and not det < 0:
            msg = (
                f'{name} has incorrect handedness. Expected a left-handed rotation, but got a '
                f'right-handed rotation instead.'
            )
            raise ValueError(msg)
    return rotation_matrix


# fmt: off
@overload
def validate_transform4x4(transform: npt.NDArray[_ScalarT], /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[_ScalarT | np.float64]: ...
@overload
def validate_transform4x4(transform: Sequence[Sequence[int]], /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[np.int64 | np.float64]: ...
@overload
def validate_transform4x4(transform: Sequence[Sequence[float]], /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_transform4x4(transform: _lazy_import.vtkMatrix3x3 | _lazy_import.vtkMatrix4x4 | _lazy_import.vtkTransform | _lazy_import.Rotation, /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_transform4x4(transform: TransformLike, /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[_Scalar]: ...
# fmt: on
def validate_transform4x4(
    transform: TransformLike, /, *, must_be_finite: bool = True, name: str = 'Transform'
) -> npt.NDArray[_Scalar]:
    """Validate transform-like input as a 4x4 ``ndarray``.

    Parameters
    ----------
    transform : TransformLike
        Transformation matrix as a 3x3 or 4x4 array, ``vtkMatrix3x3`` or
        ``vtkMatrix4x4``, ``vtkTransform``, or a SciPy ``Rotation`` instance.
        If the input is 3x3, the array is padded using a 4x4 identity matrix.

    must_be_finite : bool, default: True
        :func:`Check <pyvista_validation.check.check_finite>`
        if all elements of the array are finite, that is, not ``infinity``
        and not Not a Number (``NaN``).

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 4x4 transformation matrix.

    See Also
    --------
    validate_transform3x3
        Similar function for 3x3 transforms.

    validate_array
        Generic array validation function.

    """
    check_string(name, name='Name')
    try:
        # VTK and SciPy objects raise TypeError here and are handled below
        arr = validate_array(
            cast('MatrixLike', transform),
            must_have_shape=[(3, 3), (4, 4)],
            must_be_finite=must_be_finite,
            name=name,
        )
    except TypeError:
        # Not array-like; only now touch the lazy VTK and SciPy imports
        if isinstance(transform, _lazy_import.vtkMatrix4x4):
            return _array_from_vtkmatrix(transform, shape=(4, 4))
        if isinstance(transform, _lazy_import.vtkTransform):
            return _array_from_vtkmatrix(transform.GetMatrix(), shape=(4, 4))
        try:
            arr = validate_transform3x3(transform, must_be_finite=must_be_finite, name=name)
        except TypeError:
            msg = (
                'Input transform must be one of:\n'
                '\tvtkMatrix4x4\n'
                '\tvtkMatrix3x3\n'
                '\tvtkTransform\n'
                '\t4x4 np.ndarray\n'
                '\t3x3 np.ndarray\n'
                '\tscipy.spatial.transform.Rotation\n'
                f'Got {reprlib.repr(transform)} with type {type(transform)} instead.'
            )
            raise TypeError(msg) from None

    if arr.shape == (3, 3):
        arr4 = np.eye(4)
        arr4[:3, :3] = arr
        return arr4
    return arr


# fmt: off
@overload
def validate_transform3x3(transform: npt.NDArray[_ScalarT], /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[_ScalarT]: ...
@overload
def validate_transform3x3(transform: Sequence[Sequence[int]], /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[np.int64]: ...
@overload
def validate_transform3x3(transform: Sequence[Sequence[float]], /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_transform3x3(transform: _lazy_import.vtkMatrix3x3 | _lazy_import.Rotation, /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[np.float64]: ...
@overload
def validate_transform3x3(transform: TransformLike, /, *, must_be_finite: bool = ..., name: str = ...) -> npt.NDArray[_Scalar]: ...
# fmt: on
def validate_transform3x3(
    transform: TransformLike, /, *, must_be_finite: bool = True, name: str = 'Transform'
) -> npt.NDArray[_Scalar]:
    """Validate transform-like input as a 3x3 ``ndarray``.

    Parameters
    ----------
    transform : RotationLike
        Transformation matrix as a 3x3 array, vtk matrix, or a SciPy ``Rotation``
        instance.

        .. note::

           Although ``RotationLike`` inputs are accepted, no checks are done
           to verify that the transformation is actually a rotation.
           Therefore, any 3x3 transformation is acceptable.

    must_be_finite : bool, default: True
        :func:`Check <pyvista_validation.check.check_finite>`
        if all elements of the array are finite, that is, not ``infinity``
        and not Not a Number (``NaN``).

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 transformation matrix.

    See Also
    --------
    validate_transform4x4
        Similar function for 4x4 transforms.

    validate_array
        Generic array validation function.

    """
    check_string(name, name='Name')
    try:
        # VTK and SciPy objects raise TypeError here and are handled below
        return validate_array(
            cast('MatrixLike', transform),
            must_have_shape=(3, 3),
            must_be_finite=must_be_finite,
            name=name,
        )
    except TypeError:
        # Not array-like; only now touch the lazy VTK and SciPy imports
        if isinstance(transform, _lazy_import.vtkMatrix3x3):
            return _array_from_vtkmatrix(transform, shape=(3, 3))
        if isinstance(transform, _lazy_import.Rotation):
            return validate_transform3x3(
                transform.as_matrix(), must_be_finite=must_be_finite, name=name
            )

    error_message = (
        f'Input transform must be one of:\n'
        '\tvtkMatrix3x3\n'
        '\t3x3 np.ndarray\n'
        '\tscipy.spatial.transform.Rotation\n'
        f'Got {reprlib.repr(transform)} with type {type(transform)} instead.'
    )
    raise TypeError(error_message)


def _array_from_vtkmatrix(
    matrix: _lazy_import.vtkMatrix3x3 | _lazy_import.vtkMatrix4x4,
    shape: tuple[Literal[3], Literal[3]] | tuple[Literal[4], Literal[4]],
) -> npt.NDArray[np.float64]:
    """Convert a vtk matrix to an array."""
    array = np.zeros(shape)
    for i, j in itertools.product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


# fmt: off
@overload
def validate_number(num: np.bool_ | bool, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> bool: ...
@overload
def validate_number(num: _Integer | int, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> int: ...
@overload
def validate_number(num: _Floating | float, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> float: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> bool: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> int: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> float: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> bool | int | float: ...
@overload
def validate_number(num: _RealT, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[_RealT]: ...  # type: ignore[overload-overlap]
@overload
def validate_number(num: _ScalarT, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_number(num: bool, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_number(num: int, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_number(num: float, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[bool], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[int], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: type[float], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_NumberKwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_number(num: float | _Scalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_NumberKwargs]) -> _NumberOut: ...
# fmt: on
def validate_number(
    num: float | _Scalar, /, *, reshape: bool = True, **kwargs: Unpack[_AllKwargs]
) -> _NumberOut:
    """Validate a real, finite number.

    By default, the number is checked to ensure it:

    * is scalar or is an array which can be reshaped as a scalar
    * is a real number
    * is finite

    Parameters
    ----------
    num : float
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    output : int | float
        Validated number.

    See Also
    --------
    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a number.

    >>> from pyvista_validation import validate_number
    >>> validate_number(1)
    1

    1D arrays are automatically reshaped.

    >>> validate_number([42.0])
    42.0

    Additional checks can be added as needed.

    >>> validate_number(10, must_be_in_range=[0, 10], must_be_integer=True)
    10

    """
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('to_list', True)
    kwargs.setdefault('must_be_finite', True)

    shape: _ShapeLike | list[_ShapeLike]
    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'reshape_to', ())
    else:
        shape = ()
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_have_shape', shape)

    return cast('_NumberOut', validate_array(num, **kwargs))


# fmt: off
@overload
def validate_data_range(rng: npt.NDArray[np.bool_] | Sequence[bool] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[bool, bool]: ...
@overload
def validate_data_range(rng: npt.NDArray[_Integer] | Sequence[int] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[int, int]: ...
@overload
def validate_data_range(rng: npt.NDArray[_Floating] | Sequence[float] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[float, float]: ...
@overload
def validate_data_range(rng: npt.NDArray[np.str_] | Sequence[str] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[str, str]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[bool, bool]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[int | _Integer], must_be_real: bool = ..., to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[int, int]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[float | _Floating], must_be_real: bool = ..., to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[float, float]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[bool, bool] | tuple[int, int] | tuple[float, float]: ...
@overload
def validate_data_range(rng: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_tuple: Literal[True] = True, **kwargs: Unpack[_DataRangeKwargs]) -> tuple[bool, bool] | tuple[int, int] | tuple[float, float] | tuple[str, str]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: npt.NDArray[_RealT], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_data_range(rng: npt.NDArray[_ScalarT], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: Sequence[bool], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: Sequence[int], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_data_range(rng: Sequence[float], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_data_range(rng: npt.NDArray[np.str_] | Sequence[str], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.str_]: ...
@overload
def validate_data_range(rng: npt.NDArray[_RealT] | Sequence[_RealT] | Sequence[Sequence[_RealT]] | Sequence[npt.NDArray[_RealT]] | Sequence[Sequence[npt.NDArray[_RealT]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[bool], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[int], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[float], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_data_range(rng: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[False], to_tuple: Literal[False], **kwargs: Unpack[_DataRangeKwargs]) -> npt.NDArray[_AnyScalar]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: npt.NDArray[np.bool_] | Sequence[bool] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[bool]: ...
@overload
def validate_data_range(rng: npt.NDArray[_Integer] | Sequence[int] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[int]: ...
@overload
def validate_data_range(rng: npt.NDArray[_Floating] | Sequence[float] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[float]: ...
@overload
def validate_data_range(rng: npt.NDArray[np.str_] | Sequence[str] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[str]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[bool]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[int]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[float]: ...
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[bool] | list[int] | list[float]: ...
@overload
def validate_data_range(rng: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_DataRangeKwargs]) -> list[bool] | list[int] | list[float] | list[str]: ...  # type: ignore[overload-overlap]
@overload
def validate_data_range(rng: VectorLike, /, *, dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_DataRangeKwargs]) -> _DataRangeOut: ...
@overload
def validate_data_range(rng: _AnyArrayLikeOrScalar, /, *, dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_DataRangeKwargs]) -> _DataRangeAnyOut: ...
# fmt: on
def validate_data_range(
    rng: _AnyArrayLikeOrScalar, /, **kwargs: Unpack[_AllKwargs]
) -> _DataRangeAnyOut:
    """Validate a data range.

    By default, the data range is checked to ensure:

    * it has two values
    * it has real numbers
    * the lower bound is not more than the upper bound

    Parameters
    ----------
    rng : VectorLike
        Range to validate in the form ``(lower_bound, upper_bound)``.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    tuple
        Validated range as ``(lower_bound, upper_bound)``.

    See Also
    --------
    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a data range.

    >>> from pyvista_validation import validate_data_range
    >>> validate_data_range([-5, 5])
    (-5, 5)

    Add additional constraints if needed.

    >>> validate_data_range([0, 1.0], must_be_nonnegative=True)
    (0.0, 1.0)

    """
    kwargs.setdefault('name', 'Data Range')
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_have_shape', 2)
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_be_sorted', True)
    if not kwargs.get('to_list'):
        kwargs.setdefault('to_tuple', True)
    return cast('_DataRangeAnyOut', _validate_any(rng, **kwargs))


# fmt: off
@overload
def validate_arrayNx3(arr: npt.NDArray[_RealT], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[_ScalarT], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: Sequence[bool] | Sequence[Sequence[bool]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: Sequence[int] | Sequence[Sequence[int]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_arrayNx3(arr: Sequence[float] | Sequence[Sequence[float]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[np.str_] | Sequence[str] | Sequence[Sequence[str]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.str_]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[_RealT] | Sequence[_RealT] | Sequence[Sequence[_RealT]] | Sequence[npt.NDArray[_RealT]] | Sequence[Sequence[npt.NDArray[_RealT]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[bool], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[int], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[float], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_arrayNx3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_AnyScalar]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: npt.NDArray[np.bool_] | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[bool]]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[_Integer] | Sequence[int] | Sequence[Sequence[int]] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[int]]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[_Floating] | Sequence[float] | Sequence[Sequence[float]] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[float]]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[np.str_] | Sequence[str] | Sequence[Sequence[str]] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[str]]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[bool]]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[int]]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[float]]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[bool]] | list[list[int]] | list[list[float]]: ...
@overload
def validate_arrayNx3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[list[bool]] | list[list[int]] | list[list[float]] | list[list[str]]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: npt.NDArray[np.bool_] | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[bool, bool, bool], ...]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[_Integer] | Sequence[int] | Sequence[Sequence[int]] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[int, int, int], ...]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[_Floating] | Sequence[float] | Sequence[Sequence[float]] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[float, float, float], ...]: ...
@overload
def validate_arrayNx3(arr: npt.NDArray[np.str_] | Sequence[str] | Sequence[Sequence[str]] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[str, str, str], ...]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[bool, bool, bool], ...]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[int, int, int], ...]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[float, float, float], ...]: ...
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[bool, bool, bool], ...] | tuple[tuple[int, int, int], ...] | tuple[tuple[float, float, float], ...]: ...
@overload
def validate_arrayNx3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[tuple[bool, bool, bool], ...] | tuple[tuple[int, int, int], ...] | tuple[tuple[float, float, float], ...] | tuple[tuple[str, str, str], ...]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayNx3(arr: VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayNKwargs]) -> _ArrayNx3Out: ...
@overload
def validate_arrayNx3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayNKwargs]) -> _ArrayNx3AnyOut: ...
# fmt: on
def validate_arrayNx3(  # noqa: N802
    arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = True, **kwargs: Unpack[_AllKwargs]
) -> _ArrayNx3AnyOut:
    """Validate an array is numeric and has shape Nx3.

    The array is checked to ensure its input values:

    * have shape ``(N, 3)`` or can be reshaped to ``(N, 3)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N, 3)``.

    Parameters
    ----------
    arr : VectorLike | MatrixLike
        Array to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 3 elements are considered valid
        input and are reshaped to ``(1, 3)`` to ensure the output is
        two-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated array with shape ``(N, 3)``.

    See Also
    --------
    validate_arrayN
        Similar function for one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate an Nx3 array.

    >>> from pyvista_validation import validate_arrayNx3
    >>> validate_arrayNx3(((1, 2, 3), (4, 5, 6)))
    array([[1, 2, 3],
           [4, 5, 6]])

    One-dimensional 3-element arrays are automatically reshaped to 2D.

    >>> validate_arrayNx3([1, 2, 3])
    array([[1, 2, 3]])

    Add additional constraints.

    >>> validate_arrayNx3(((1, 2, 3), (4, 5, 6)), must_be_in_range=[0, 10])
    array([[1, 2, 3],
           [4, 5, 6]])

    """
    shape: _ShapeLike | list[_ShapeLike]
    if reshape:
        shape = [3, (-1, 3)]
        _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'reshape_to', (-1, 3))
    else:
        shape = (-1, 3)
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_have_shape', shape)

    return cast('_ArrayNx3AnyOut', _validate_any(arr, **kwargs))


# fmt: off
@overload
def validate_arrayN(arr: npt.NDArray[_RealT] | _RealT, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_RealT]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: list[Never] | list[list[Never]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: npt.NDArray[_ScalarT] | _ScalarT, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: bool | Sequence[bool] | Sequence[Sequence[bool]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: int | Sequence[int] | Sequence[Sequence[int]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_arrayN(arr: float | Sequence[float] | Sequence[Sequence[float]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_arrayN(arr: npt.NDArray[np.str_] | np.str_ | str | Sequence[str] | Sequence[Sequence[str]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.str_]: ...
@overload
def validate_arrayN(arr: npt.NDArray[_RealT] | Sequence[_RealT] | Sequence[Sequence[_RealT]] | Sequence[npt.NDArray[_RealT]] | Sequence[Sequence[npt.NDArray[_RealT]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[bool], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[int], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[float], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_arrayN(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> npt.NDArray[_AnyScalar]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: list[Never] | list[list[Never]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[float]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: npt.NDArray[np.bool_] | np.bool_ | bool | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[bool]: ...
@overload
def validate_arrayN(arr: npt.NDArray[_Integer] | _Integer | int | Sequence[int] | Sequence[Sequence[int]] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[int]: ...
@overload
def validate_arrayN(arr: npt.NDArray[_Floating] | _Floating | float | Sequence[float] | Sequence[Sequence[float]] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[float]: ...
@overload
def validate_arrayN(arr: npt.NDArray[np.str_] | np.str_ | str | Sequence[str] | Sequence[Sequence[str]] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[str]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[bool]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[int]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[float]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[bool] | list[int] | list[float]: ...
@overload
def validate_arrayN(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNKwargs]) -> list[bool] | list[int] | list[float] | list[str]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: list[Never] | list[list[Never]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[float, ...]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: npt.NDArray[np.bool_] | np.bool_ | bool | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[bool, ...]: ...
@overload
def validate_arrayN(arr: npt.NDArray[_Integer] | _Integer | int | Sequence[int] | Sequence[Sequence[int]] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[int, ...]: ...
@overload
def validate_arrayN(arr: npt.NDArray[_Floating] | _Floating | float | Sequence[float] | Sequence[Sequence[float]] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[float, ...]: ...
@overload
def validate_arrayN(arr: npt.NDArray[np.str_] | np.str_ | str | Sequence[str] | Sequence[Sequence[str]] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[str, ...]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[bool, ...]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[int, ...]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[float, ...]: ...
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[bool, ...] | tuple[int, ...] | tuple[float, ...]: ...
@overload
def validate_arrayN(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNKwargs]) -> tuple[bool, ...] | tuple[int, ...] | tuple[float, ...] | tuple[str, ...]: ...  # type: ignore[overload-overlap]
@overload
def validate_arrayN(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayNKwargs]) -> _ArrayNOut: ...
@overload
def validate_arrayN(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayNKwargs]) -> _ArrayNAnyOut: ...
# fmt: on
def validate_arrayN(  # noqa: N802
    arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = True, **kwargs: Unpack[_AllKwargs]
) -> _ArrayNAnyOut:
    """Validate a numeric 1D array.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N,)``

    Parameters
    ----------
    arr : VectorLike
        Array to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array.

    See Also
    --------
    validate_arrayN_unsigned
        Similar function for non-negative integer arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four elements.

    >>> from pyvista_validation import validate_arrayN
    >>> validate_arrayN((1, 2, 3, 4))
    array([1, 2, 3, 4])

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> validate_arrayN(42.0)
    array([42.])

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> validate_arrayN([[1, 2]])
    array([1, 2])

    Add additional constraints if needed.

    >>> validate_arrayN((1, 2, 3), must_have_length=3)
    array([1, 2, 3])

    """
    shape: _ShapeLike | list[_ShapeLike]
    if reshape:
        shape = [(), (-1), (1, -1)]
        _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'reshape_to', (-1))
    else:
        shape = -1
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_have_shape', shape)
    return cast('_ArrayNAnyOut', _validate_any(arr, **kwargs))


# fmt: off
@overload
def validate_arrayN_unsigned(arr: VectorLike, /, *, reshape: bool = ..., dtype_out: type[int] = ..., must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNUnsignedKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_arrayN_unsigned(arr: VectorLike, /, *, reshape: bool = ..., dtype_out: type[_IntegerT], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNUnsignedKwargs]) -> npt.NDArray[_IntegerT]: ...
@overload
def validate_arrayN_unsigned(arr: VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike = ..., must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNUnsignedKwargs]) -> npt.NDArray[_Integer]: ...
@overload
def validate_arrayN_unsigned(arr: VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike = ..., must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_ArrayNUnsignedKwargs]) -> list[int]: ...
@overload
def validate_arrayN_unsigned(arr: VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike = ..., must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_ArrayNUnsignedKwargs]) -> tuple[int, ...]: ...
@overload
def validate_arrayN_unsigned(arr: VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike = ..., must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_ArrayNUnsignedKwargs]) -> _ArrayNUnsignedOut: ...
# fmt: on
def validate_arrayN_unsigned(  # noqa: N802
    arr: VectorLike, /, *, reshape: bool = True, **kwargs: Unpack[_ArrayNKwargsAll]
) -> _ArrayNUnsignedOut:
    """Validate a numeric 1D array of non-negative (unsigned) integers.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are integer-like
    * are non-negative
    * can be represented by the output data type

    The returned array is formatted so that its values:

    * have shape ``(N,)``
    * have an integer data type

    Parameters
    ----------
    arr : VectorLike
        Array to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array with non-negative integers.

    See Also
    --------
    validate_arrayN
        Similar function for numeric one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four non-negative integer-like elements.

    >>> import numpy as np
    >>> from pyvista_validation import validate_arrayN_unsigned
    >>> arr = validate_arrayN_unsigned((1.0, 2.0, 3.0, 4.0))
    >>> arr
    array([1, 2, 3, 4])

    Verify that the output data type is integral.

    >>> np.issubdtype(arr.dtype, int)
    True

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> validate_arrayN_unsigned(42)
    array([42])

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> validate_arrayN_unsigned([[1, 2]])
    array([1, 2])

    Add additional constraints if needed.

    >>> validate_arrayN_unsigned((1, 2, 3), must_be_in_range=[1, 3])
    array([1, 2, 3])

    """
    # Set default dtype out but allow overriding as long as the dtype
    # is also integral
    dtype_out = kwargs.setdefault('dtype_out', int)
    if dtype_out is not int:
        check_subdtype(cast('_DTypeLike', dtype_out), np.integer)

    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_be_integer', True)
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_be_nonnegative', True)

    # A value above what the output dtype can hold would otherwise wrap around
    limit = _integer_max(cast('_DTypeLike', dtype_out))
    user_range = kwargs.get('must_be_in_range')
    rng = _cast_to_numpy([0, limit] if user_range is None else user_range)
    kwargs['must_be_in_range'] = [rng.item(0), min(rng.item(1), limit)]

    return cast('_ArrayNUnsignedOut', validate_arrayN(arr, reshape=reshape, **kwargs))


# fmt: off
@overload
def validate_array3(arr: npt.NDArray[_RealT] | _RealT, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[_RealT]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: npt.NDArray[_ScalarT] | _ScalarT, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: bool | Sequence[bool] | Sequence[Sequence[bool]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: int | Sequence[int] | Sequence[Sequence[int]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_array3(arr: float | Sequence[float] | Sequence[Sequence[float]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_array3(arr: npt.NDArray[np.str_] | np.str_ | str | Sequence[str] | Sequence[Sequence[str]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.str_]: ...
@overload
def validate_array3(arr: npt.NDArray[_RealT] | Sequence[_RealT] | Sequence[Sequence[_RealT]] | Sequence[npt.NDArray[_RealT]] | Sequence[Sequence[npt.NDArray[_RealT]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[_RealT]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[bool], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[int], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[float], must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[np.float64]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_array3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[False] = False, to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> npt.NDArray[_AnyScalar]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: npt.NDArray[np.bool_] | np.bool_ | bool | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[bool]: ...
@overload
def validate_array3(arr: npt.NDArray[_Integer] | _Integer | int | Sequence[int] | Sequence[Sequence[int]] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[int]: ...
@overload
def validate_array3(arr: npt.NDArray[_Floating] | _Floating | float | Sequence[float] | Sequence[Sequence[float]] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[float]: ...
@overload
def validate_array3(arr: npt.NDArray[np.str_] | np.str_ | str | Sequence[str] | Sequence[Sequence[str]] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[str]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[bool]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[int]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[float]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[bool] | list[int] | list[float]: ...
@overload
def validate_array3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: Literal[True], to_tuple: Literal[False] = False, **kwargs: Unpack[_Array3Kwargs]) -> list[bool] | list[int] | list[float] | list[str]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: npt.NDArray[np.bool_] | np.bool_ | bool | Sequence[bool] | Sequence[Sequence[bool]] | Sequence[npt.NDArray[np.bool_]] | Sequence[Sequence[npt.NDArray[np.bool_]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[bool, bool, bool]: ...
@overload
def validate_array3(arr: npt.NDArray[_Integer] | _Integer | int | Sequence[int] | Sequence[Sequence[int]] | Sequence[npt.NDArray[_Integer]] | Sequence[Sequence[npt.NDArray[_Integer]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[int, int, int]: ...
@overload
def validate_array3(arr: npt.NDArray[_Floating] | _Floating | float | Sequence[float] | Sequence[Sequence[float]] | Sequence[npt.NDArray[_Floating]] | Sequence[Sequence[npt.NDArray[_Floating]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[float, float, float]: ...
@overload
def validate_array3(arr: npt.NDArray[np.str_] | np.str_ | str | Sequence[str] | Sequence[Sequence[str]] | Sequence[npt.NDArray[np.str_]] | Sequence[Sequence[npt.NDArray[np.str_]]], /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[str, str, str]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[bool, bool, bool]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[int, int, int]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[float, float, float]: ...
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[bool, bool, bool] | tuple[int, int, int] | tuple[float, float, float]: ...
@overload
def validate_array3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: Literal[True], **kwargs: Unpack[_Array3Kwargs]) -> tuple[bool, bool, bool] | tuple[int, int, int] | tuple[float, float, float] | tuple[str, str, str]: ...  # type: ignore[overload-overlap]
@overload
def validate_array3(arr: float | _Scalar | VectorLike | MatrixLike, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_Array3Kwargs]) -> _Array3Out: ...
@overload
def validate_array3(arr: _AnyArrayLikeOrScalar, /, *, reshape: bool = ..., broadcast: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: Literal[False], to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_Array3Kwargs]) -> _Array3AnyOut: ...
# fmt: on
def validate_array3(
    arr: _AnyArrayLikeOrScalar,
    /,
    *,
    reshape: bool = True,
    broadcast: bool = False,
    **kwargs: Unpack[_AllKwargs],
) -> _Array3AnyOut:
    """Validate a numeric 1D array with 3 elements.

    The array is checked to ensure its input values:

    * have shape ``(3,)`` or can be reshaped to ``(3,)``
    * are numeric and real

    The returned array is formatted so that it has shape ``(3,)``.

    Parameters
    ----------
    arr : float | VectorLike | MatrixLike
        Array to validate.

    reshape : bool, default: True
        If ``True``, 2D vectors with shape ``(1, 3)`` are considered valid
        input, and are reshaped to ``(3,)`` to ensure the output is
        consistently one-dimensional.

    broadcast : bool, default: False
        If ``True``, scalar values or 1D arrays with a single element
        are considered valid input and the single value is broadcast to
        a length 3 array.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array with 3 elements.

    See Also
    --------
    validate_number
        Similar function for a single number.

    validate_arrayN
        Similar function for one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with three elements.

    >>> from pyvista_validation import validate_array3
    >>> validate_array3((1, 2, 3))
    array([1, 2, 3])

    2D 3-element arrays are automatically reshaped to be 1D.

    >>> validate_array3([[1, 2, 3]])
    array([1, 2, 3])

    Scalar 0-dimensional values can be automatically broadcast as
    a 3-element 1D array.

    >>> validate_array3(42.0, broadcast=True)
    array([42., 42., 42.])

    Add additional constraints if needed.

    >>> validate_array3((1, 2, 3), must_be_nonnegative=True)
    array([1, 2, 3])

    """
    shape: list[_ShapeLike] = [(3,)]
    if reshape:
        shape.append((1, 3))
        shape.append((3, 1))
        _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'reshape_to', (-1))
    if broadcast:
        shape.append(())  # allow 0D scalars
        shape.append((1,))  # 1D 1-element vectors
        _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'broadcast_to', (3,))
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_have_shape', shape)

    return cast('_Array3AnyOut', _validate_any(arr, **kwargs))


def _set_default_kwarg_mandatory(kwargs: dict[str, object], key: str, default: object) -> None:
    """Set a kwarg and raise ValueError if not set to its default value."""
    val = kwargs.pop(key, default)
    if val != default:
        calling_fname = inspect.stack()[1].function
        msg = (
            f"Parameter '{key}' cannot be set for function `{calling_fname}`.\n"
            f'Its value is automatically set to `{default}`.'
        )
        raise ValueError(msg)
    kwargs[key] = default


# fmt: off
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_DimensionalityKwargs]) -> int: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: type[bool | np.bool_], must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_DimensionalityKwargs]) -> bool: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: type[int | _Integer], must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_DimensionalityKwargs]) -> int: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: type[float | _Floating], must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_DimensionalityKwargs]) -> float: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[True] = True, to_tuple: bool = False, **kwargs: Unpack[_DimensionalityKwargs]) -> bool | int | float: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_DimensionalityKwargs]) -> npt.NDArray[np.int64]: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: type[_ScalarT], must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_DimensionalityKwargs]) -> npt.NDArray[_ScalarT]: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: Literal[False], to_tuple: Literal[False] = False, **kwargs: Unpack[_DimensionalityKwargs]) -> npt.NDArray[_Scalar]: ...
@overload
def validate_dimensionality(dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike, /, *, reshape: bool = ..., dtype_out: _DTypeLike | None = None, must_be_real: bool = ..., to_list: bool = False, to_tuple: bool = False, **kwargs: Unpack[_DimensionalityKwargs]) -> _DimensionalityOut: ...
# fmt: on
def validate_dimensionality(
    dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike,
    /,
    *,
    reshape: bool = True,
    **kwargs: Unpack[_AllKwargs],
) -> _DimensionalityOut:
    """Validate a dimensionality.

    By default, the dimensionality is checked to ensure it:

    * is scalar or is an array which can be reshaped as a scalar
    * is an integer in the inclusive range ``[0, 3]``
    * or is a valid alias among ``'0D'``, ``'1D'``, ``'2D'``, or ``'3D'``

    Parameters
    ----------
    dimensionality : Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | ArrayLike
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    int
        Validated dimensionality.

    Examples
    --------
    Validate a dimensionality.

    >>> from pyvista_validation import validate_dimensionality
    >>> validate_dimensionality('1D')
    1

    1D arrays are automatically reshaped.

    >>> validate_dimensionality([3])
    3

    """
    kwargs.setdefault('name', 'Dimensionality')
    kwargs.setdefault('to_list', True)
    kwargs.setdefault('must_be_finite', True)
    kwargs.setdefault('must_be_in_range', [0, 3])
    kwargs.setdefault('dtype_out', int)
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_be_integer', True)

    as_array = _asarray_any(dimensionality)
    if as_array.dtype.kind == 'U':
        # An alias such as '2D' is its number without the suffix
        try:
            as_array = np.char.replace(as_array.astype(np.str_), 'D', '').astype(np.int64)
        except ValueError:
            msg = (
                f'`{dimensionality}` is not a valid dimensionality.'
                ' Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D"].'
            )
            raise ValueError(msg) from None

    shape: _ShapeLike | list[_ShapeLike]
    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'reshape_to', ())
    else:
        shape = ()
    _set_default_kwarg_mandatory(cast('dict[str, object]', kwargs), 'must_have_shape', shape)

    return cast(
        '_DimensionalityOut', validate_array(cast('npt.NDArray[_Scalar]', as_array), **kwargs)
    )


def _integer_max(dtype: _DTypeLike, /) -> int:
    """Return the largest value an integer dtype holds, typing what NumPy leaves as ``Any``."""
    return int(np.iinfo(cast('type[np.integer[Any]]', dtype)).max)


def _asarray_any(obj: object, /) -> npt.NDArray[np.generic[object]]:
    """Convert any object to an array, typing what NumPy leaves as ``Any``."""
    return cast('npt.NDArray[np.generic[object]]', np.asarray(obj))


def _astype(array: npt.NDArray[_AnyScalar], dtype: _DTypeLike, /) -> npt.NDArray[_AnyScalar]:
    """Cast an array to a dtype without copying, typing what NumPy leaves as ``Any``."""
    return cast('npt.NDArray[_AnyScalar]', array.astype(dtype, copy=False))


def _validate_any(arr: _AnyArrayLikeOrScalar, /, **kwargs: Unpack[_AllKwargs]) -> _AnyArrayOut:
    """Forward to ``validate_array``; a ``must_be_real`` typed ``bool`` selects no overload."""
    return validate_array(cast('_ArrayLikeOrScalar', arr), **kwargs)


if not TYPE_CHECKING:
    accelerate(
        globals(),
        [
            'validate_array',
            'validate_axes',
            'validate_rotation',
            'validate_transform4x4',
            'validate_transform3x3',
            'validate_number',
            'validate_data_range',
            'validate_arrayNx3',
            'validate_arrayN',
            'validate_arrayN_unsigned',
            'validate_array3',
            'validate_dimensionality',
        ],
    )
