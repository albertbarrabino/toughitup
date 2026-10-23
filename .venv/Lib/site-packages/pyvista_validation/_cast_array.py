"""Array casting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
from typing import overload

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

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


# Overloads follow NumPy's dtype inference: NumPy inputs keep their dtype, Python bools, ints
# and floats become bool, int64 and float64. Empty lists come first so that they are typed
# as float64 rather than matching the bool sequence overload. Since bool subclasses int and
# np.float64 subclasses float, mypy flags those pairs as overlapping; NumPy tells them apart
# at runtime, so the overlap is ignored where it is reported.


# fmt: off
@overload
def _cast_to_list(arr: _EmptyList, /) -> _ToListFloat: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_list(arr: npt.NDArray[np.bool_] | np.bool_ | bool | _NestedBool, /) -> _ToListBool: ...
@overload
def _cast_to_list(arr: npt.NDArray[_Integer] | _Integer | int | _NestedInt, /) -> _ToListInt: ...
@overload
def _cast_to_list(arr: npt.NDArray[_Floating] | _Floating | float | _NestedFloat, /) -> _ToListFloat: ...
@overload
def _cast_to_list(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr, /) -> _ToListStr: ...
@overload
def _cast_to_list(arr: _ArrayLikeOrScalar, /) -> _ToList: ...
@overload
def _cast_to_list(arr: _AnyArrayLikeOrScalar, /) -> _ToAnyList: ...
# fmt: on
def _cast_to_list(arr: _AnyArrayLikeOrScalar, /) -> _ToAnyList:
    """Cast an array to a nested list.

    Parameters
    ----------
    arr : float | ArrayLike
        Array to cast.

    Returns
    -------
    list
        List or nested list array.

    """
    return _tolist(_cast_to_numpy(arr))


# fmt: off
@overload
def _cast_to_tuple(arr: _EmptyList, /) -> _ToTupleFloat: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_tuple(arr: npt.NDArray[np.bool_] | np.bool_ | bool | _NestedBool, /) -> _ToTupleBool: ...
@overload
def _cast_to_tuple(arr: npt.NDArray[_Integer] | _Integer | int | _NestedInt, /) -> _ToTupleInt: ...
@overload
def _cast_to_tuple(arr: npt.NDArray[_Floating] | _Floating | float | _NestedFloat, /) -> _ToTupleFloat: ...
@overload
def _cast_to_tuple(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr, /) -> _ToTupleStr: ...
@overload
def _cast_to_tuple(arr: _ArrayLikeOrScalar, /) -> _ToTuple: ...
@overload
def _cast_to_tuple(arr: _AnyArrayLikeOrScalar, /) -> _ToAnyTuple: ...
# fmt: on
def _cast_to_tuple(arr: _AnyArrayLikeOrScalar, /) -> _ToAnyTuple:
    """Cast an array to a nested tuple.

    Parameters
    ----------
    arr : float | ArrayLike
        Array to cast.

    Returns
    -------
    tuple
        Tuple or nested tuple array.

    """
    return cast('_ToAnyTuple', _to_tuple(_cast_to_list(arr)))


# fmt: off
@overload
def _cast_to_numpy(arr: npt.NDArray[_ScalarT] | _ScalarT, /, *, as_any: bool = ..., dtype: None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_numpy(arr: _EmptyList, /, *, as_any: bool = ..., dtype: None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.float64]: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_numpy(arr: bool | _NestedBool, /, *, as_any: bool = ..., dtype: None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_numpy(arr: int | _NestedInt, /, *, as_any: bool = ..., dtype: None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.int64]: ...
@overload
def _cast_to_numpy(arr: float | _NestedFloat, /, *, as_any: bool = ..., dtype: None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.float64]: ...
@overload
def _cast_to_numpy(arr: npt.NDArray[np.str_] | np.str_ | str | _NestedStr, /, *, as_any: bool = ..., dtype: None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.str_]: ...
@overload
def _cast_to_numpy(arr: _ArrayLikeOrScalar, /, *, as_any: bool = ..., dtype: type[_ScalarT], copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[_ScalarT]: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_numpy(arr: _ArrayLikeOrScalar, /, *, as_any: bool = ..., dtype: type[bool], copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.bool_]: ...  # type: ignore[overload-overlap]
@overload
def _cast_to_numpy(arr: _ArrayLikeOrScalar, /, *, as_any: bool = ..., dtype: type[int], copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.int64]: ...
@overload
def _cast_to_numpy(arr: _ArrayLikeOrScalar, /, *, as_any: bool = ..., dtype: type[float], copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[np.float64]: ...
@overload
def _cast_to_numpy(arr: _ArrayLikeOrScalar, /, *, as_any: bool = ..., dtype: _DTypeLike | None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[_Scalar]: ...
@overload
def _cast_to_numpy(arr: _AnyArrayLikeOrScalar, /, *, as_any: bool = ..., dtype: _DTypeLike | None = None, copy: bool = ..., must_be_real: bool = ...) -> npt.NDArray[_AnyScalar]: ...
# fmt: on
def _cast_to_numpy(
    arr: _AnyArrayLikeOrScalar,
    /,
    *,
    as_any: bool = True,
    dtype: _DTypeLike | None = None,
    copy: bool = False,
    must_be_real: bool = False,
) -> npt.NDArray[_AnyScalar]:
    """Cast array to a NumPy ``ndarray``.

    Object arrays are not allowed but the ``dtype`` is otherwise unchecked by default.
    String arrays and complex numbers are therefore allowed.

    .. warning::

        Arrays intended for use with vtk should set ``must_be_real=True``
        since ``numpy_to_vtk`` uses the array values directly without
        checking for complex arrays.

    Parameters
    ----------
    arr : float | ArrayLike
        Array to cast.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through. Otherwise, a
        subclass is returned as a base ``np.ndarray`` view of the same data.

    dtype : DTypeLike, optional
        The data-type of the returned array.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A new array is always
        made from a sequence, and one may also be made to satisfy ``dtype``.

    must_be_real : bool, default: True
        Raise a ``TypeError`` if the array does not have real numbers, that is
        its data type is not integer or floating. Boolean and text arrays are
        not real.

    Raises
    ------
    ValueError
        If input cannot be cast as a NumPy ``ndarray``.
    TypeError
        If an object array is created or if the data is not real numbers
        and ``must_be_real`` is ``True``.

    Returns
    -------
    np.ndarray
        NumPy ``ndarray``.

    """
    try:
        out = _asarray(arr, dtype=dtype, as_any=as_any)
        # A base-class view of a subclass still shares its memory
        if copy and isinstance(arr, np.ndarray) and np.may_share_memory(out, arr):
            out = out.copy()
    except ValueError as e:
        msg = f'Input cannot be cast as {np.ndarray}.'
        raise ValueError(msg) from e
    if must_be_real and not issubclass(out.dtype.type, (np.floating, np.integer)):
        msg = f'Array must have real numbers. Got dtype {out.dtype.type}'
        raise TypeError(msg)
    if out.dtype.kind == 'O':
        shown = arr.decode(errors='replace') if isinstance(arr, bytes) else arr
        msg = f'Object arrays are not supported. Got {shown} when casting to a NumPy array.'
        raise TypeError(msg)
    return out


def _asarray(
    arr: _AnyArrayLikeOrScalar, /, *, dtype: _DTypeLike | None, as_any: bool
) -> npt.NDArray[_AnyScalar]:
    """Call NumPy's array constructors, whose dynamic return types stop here."""
    return cast(
        'npt.NDArray[_AnyScalar]',
        np.asanyarray(arr, dtype=dtype) if as_any else np.asarray(arr, dtype=dtype),
    )


def _tolist(array: npt.NDArray[_AnyScalar], /) -> _ToAnyList:
    """Convert an array to nested lists, typing what NumPy leaves as ``Any``."""
    return cast('_ToAnyList', array.tolist())


def _to_tuple(obj: object, /) -> object:
    """Recursively convert nested lists to nested tuples."""
    return tuple(_to_tuple(item) for item in obj) if isinstance(obj, list) else obj
