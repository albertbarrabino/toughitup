"""Functions that check the type and/or value of inputs.

A ``check`` function typically:

* Performs a simple validation on a single input variable.
* Raises an error if the check fails due to invalid input.
* Does not modify its input, and returns it unchanged, typed as what the check
  established, so a check can be used inline.

"""

from __future__ import annotations

from collections.abc import Container
from collections.abc import Iterable
from collections.abc import Sequence
from collections.abc import Sized
import numbers
import reprlib
import sys
from types import UnionType
from typing import TYPE_CHECKING
from typing import Union
from typing import cast
from typing import get_args
from typing import get_origin
from typing import overload

import numpy as np

from pyvista_validation._accelerate import accelerate
from pyvista_validation._cast_array import _cast_to_numpy

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    # Type variable defaults (PEP 696) reached the standard library in 3.13.
    from typing_extensions import TypeVar

if TYPE_CHECKING:
    import numpy.typing as npt

    from pyvista_validation._typing import VectorLike
    from pyvista_validation._typing import _AnyArrayLikeOrScalar
    from pyvista_validation._typing import _ArrayLikeOrScalar
    from pyvista_validation._typing import _DTypeLike
    from pyvista_validation._typing import _Scalar


_Shape = tuple[()] | tuple[int, ...]
_ShapeLike = int | _Shape
# What isinstance() accepts, plus the union objects the type-based checks unpack.
_ClassInfo = type[object] | tuple[type[object], ...] | UnionType

# Every check returns its input unchanged; these carry the input's static type through.
_ArrayT = TypeVar('_ArrayT', bound='_ArrayLikeOrScalar', default='_ArrayLikeOrScalar')
# For the checks that do not depend on the values being numbers.
_AnyArrayT = TypeVar('_AnyArrayT', bound='_AnyArrayLikeOrScalar', default='_AnyArrayLikeOrScalar')
_DTypeOrArrayT = TypeVar(
    '_DTypeOrArrayT',
    bound='_DTypeLike | _AnyArrayLikeOrScalar',
    default='_DTypeLike | _AnyArrayLikeOrScalar',
)
_SizedT = TypeVar('_SizedT', bound='float | _Scalar | Sized', default='float | _Scalar | Sized')
_NumberT = TypeVar('_NumberT', bound='float | _Scalar', default='float | _Scalar')
_StrT = TypeVar('_StrT', bound=str, default=str)
_SequenceT = TypeVar('_SequenceT', bound=Sequence[object], default=Sequence[object])
_IterableT = TypeVar('_IterableT', bound=Iterable[object], default=Iterable[object])
_ItemT = TypeVar('_ItemT', default=object)
_ObjectT = TypeVar('_ObjectT', default=object)
# What a class-based check narrows an input of unknown type to.
_T = TypeVar('_T', default=object)
_T1 = TypeVar('_T1', default=object)
_T2 = TypeVar('_T2', default=object)
_T3 = TypeVar('_T3', default=object)


def check_subdtype(
    input_obj: _DTypeOrArrayT,
    /,
    base_dtype: _DTypeLike | tuple[_DTypeLike, ...] | list[_DTypeLike],
    *,
    name: str = 'Input',
) -> _DTypeOrArrayT:
    """Check if an input's data-type is a subtype of another data-type or data-types.

    Parameters
    ----------
    input_obj : float | ArrayLike | DTypeLike
        ``dtype`` object (or object coercible to one) or an array-like object.
        If array-like, the ``dtype`` of the array is used.

    base_dtype : DTypeLike | Sequence[DTypeLike]
        ``dtype``-like object or a sequence of ``dtype``-like objects. The ``input_obj``
        must be a subtype of this value. If a sequence, ``input_obj`` must be a
        subtype of at least one of the specified ``dtypes``.

    name : str, default: "Input"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    object
        The input, unchanged.

    Raises
    ------
    TypeError
        If ``input_obj`` is not a subtype of ``base_dtype``.

    See Also
    --------
    check_real
    check_number

    Examples
    --------
    Check if ``float`` is a subtype of ``np.floating``.

    >>> import numpy as np
    >>> from pyvista_validation import check_subdtype
    >>> check_subdtype(float, np.floating)
    <class 'float'>

    Check from multiple allowable ``dtypes``.

    >>> check_subdtype(int, [np.integer, np.floating])
    <class 'int'>

    Check an array's ``dtype``.

    >>> array = np.array([1, 2, 3], dtype='uint8')
    >>> check_subdtype(array, np.integer)
    array([1, 2, 3], dtype=uint8)

    """
    input_dtype = _dtype_of(input_obj)
    if not isinstance(base_dtype, (tuple, list)):
        base_dtype = [base_dtype]
    if not any(_issubdtype(input_dtype, base) for base in base_dtype):
        # Not a subdtype, so raise error
        msg = f"{name} has incorrect dtype of '{input_dtype.name}'. "
        if len(base_dtype) == 1:
            msg += f'The dtype must be a subtype of {base_dtype[0]}.'
        else:
            msg += f'The dtype must be a subtype of at least one of \n{base_dtype}.'
        raise TypeError(msg)
    return input_obj


def check_real(array: _AnyArrayT, /, *, name: str = 'Array') -> _AnyArrayT:
    """Check if an array has real numbers (float or integer type).

    Notes
    -----
    -   Boolean and text data types are not considered real and will raise an error.
    -   Arrays with ``infinity`` or ``NaN`` values are considered real and
        will not raise an error. Use :func:`check_finite` to check for
        finite values.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    TypeError
        If the array does not have real numbers.

    See Also
    --------
    check_integer
        Similar function for integer arrays.
    check_number
        Similar function for scalar values.
    check_finite
        Check for finite values.

    Examples
    --------
    Check if an array has real numbers.

    >>> from pyvista_validation import check_real
    >>> check_real([1, 2, 3])
    [1, 2, 3]

    """
    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    if array_.dtype.type in (np.int32, np.int64, np.longlong, np.float32, np.float64):
        return array
    try:
        check_subdtype(array_, (np.floating, np.integer), name=name)
    except TypeError as e:
        msg = f'{name} must have real numbers.'
        raise TypeError(msg) from e
    return array


def check_sorted(
    array: _AnyArrayT,
    /,
    *,
    ascending: bool = True,
    strict: bool = False,
    axis: int | None = -1,
    name: str = 'Array',
) -> _AnyArrayT:
    """Check if an array's values are sorted.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    ascending : bool, default: True
        If ``True``, check if the array's elements are in ascending order.
        If ``False``, check if the array's elements are in descending order.

    strict : bool, default: False
        If ``True``, the array's elements must be strictly increasing (if
        ``ascending=True``) or strictly decreasing (if ``ascending=False``).
        Effectively, this means the array must be sorted *and* its values
        must be unique.

    axis : int | None, default: -1
        Axis along which to check sorting. If ``None``, the array is flattened
        before checking. The default is ``-1``, which checks sorting along the
        last axis.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If the array is not sorted in ascending order.

    See Also
    --------
    check_range

    Examples
    --------
    Check if an array's values are sorted,

    >>> from pyvista_validation import check_sorted
    >>> check_sorted([1, 2, 3])
    [1, 2, 3]

    """
    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    ndim = array_.ndim
    if ndim == 0:
        # Scalars are always sorted
        return array

    if axis is None:
        # Emulate np.sort(), which flattens array when axis is None
        array_ = array_.ravel(order='A')
        ndim = 1
        axis = 0
    elif axis != -1:
        check_number(axis, name='Axis')
        check_integer(axis, name='Axis')
        axis = int(axis)
        try:
            check_range(axis, rng=[-ndim, ndim - 1], name='Axis')
        except ValueError:
            msg = f'Axis {axis} is out of bounds for ndim {ndim}.'
            raise ValueError(msg) from None

    # Compare a view of the array along the axis with a view offset by one element. NumPy
    # orders text arrays too, but its stubs type the comparison per dtype family.
    moved = cast('npt.NDArray[_Scalar]', np.moveaxis(array_, axis, -1))
    first_item = moved[..., :-1]
    second_item = moved[..., 1:]

    if ascending and not strict:
        is_sorted = np.all(first_item <= second_item)
    elif ascending and strict:
        is_sorted = np.all(first_item < second_item)
    elif not ascending and not strict:
        is_sorted = np.all(first_item >= second_item)
    else:  # not ascending and strict
        is_sorted = np.all(first_item > second_item)

    if not is_sorted:
        # Show the array's elements in error msg if array is small
        msg_body = f'with {array_.size} elements'
        order = 'ascending' if ascending else 'descending'
        strict_ = 'strict ' if strict else ''
        msg = (
            f'{name} {msg_body} must be sorted in {strict_}{order} order. '
            f'Got:\n    {reprlib.repr(array_)}'
        )
        raise ValueError(msg)
    return array


def check_finite(array: _ArrayT, /, *, name: str = 'Array') -> _ArrayT:
    """Check if an array has finite values, that is, no NaN or Inf values.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If the array has any ``Inf`` or ``NaN`` values.

    See Also
    --------
    check_real

    Examples
    --------
    Check if an array's values are finite.

    >>> from pyvista_validation import check_finite
    >>> check_finite([1, 2, 3])
    [1, 2, 3]

    """
    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    if not np.all(np.isfinite(array_)):
        msg = f'{name} must have finite values.'
        raise ValueError(msg)
    return array


def check_integer(
    array: _ArrayT,
    /,
    *,
    strict: bool = False,
    name: str = 'Array',
) -> _ArrayT:
    """Check if an array has integer or integer-like float values.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    strict : bool, default: False
        If ``True``, the array's data must be a subtype of ``int`` or
        ``np.integer``. Otherwise, floats are allowed but must be
        whole numbers.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If any element's value differs from its floor.

    TypeError
        If ``strict=True`` and the array's ``dtype`` is not integral.

    See Also
    --------
    check_nonnegative
    check_subdtype

    Examples
    --------
    Check if an array has integer-like values.

    >>> from pyvista_validation import check_integer
    >>> check_integer([1.0, 2.0])
    [1.0, 2.0]

    """
    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    if strict:
        check_subdtype(array_, np.integer, name=name)
    elif not np.array_equal(array_, np.floor(array_)):
        msg = f'{name} must have integer-like values.'
        raise ValueError(msg)
    return array


def check_nonnegative(array: _ArrayT, /, *, name: str = 'Array') -> _ArrayT:
    """Check if an array's elements are all nonnegative.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If the array has any negative values.

    See Also
    --------
    check_greater_than
    check_less_than

    Examples
    --------
    Check if an array's values are non-negative.

    >>> from pyvista_validation import check_nonnegative
    >>> check_nonnegative([1, 2, 3])
    [1, 2, 3]

    """
    return check_greater_than(array, 0, strict=False, name=name)


def _validate_real_value(scalar: float | _Scalar, name: str = 'Value') -> npt.NDArray[_Scalar]:
    """Cast a real scalar to a 0-D array, raising if it is not a real number."""
    valid_scalar: npt.NDArray[_Scalar] = _cast_to_numpy(scalar)
    check_shape(valid_scalar, (), name=name)
    check_real(valid_scalar, name=name)
    return valid_scalar


def check_greater_than(
    array: _ArrayT,
    /,
    value: float | _Scalar,
    *,
    strict: bool = True,
    name: str = 'Array',
) -> _ArrayT:
    """Check if an array's elements are all greater than some value.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    value : float
        Value which the array's elements must be greater than.

    strict : bool, default: True
        If ``True``, the array's value must be strictly greater than ``value``.
        Otherwise, values must be greater than or equal to ``value``.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If not all array elements are greater than (or equal to if
        ``strict=True``) the specified value.

    See Also
    --------
    check_less_than
    check_range
    check_nonnegative

    Examples
    --------
    Check if an array's values are greater than 0.

    >>> from pyvista_validation import check_greater_than
    >>> check_greater_than([1, 2, 3], value=0)
    [1, 2, 3]

    """
    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    valid_value = _validate_real_value(value)
    if strict and not np.all(array_ > valid_value):
        msg = f'{name} values must all be greater than {value}.'
        raise ValueError(msg)
    if not np.all(array_ >= valid_value):
        msg = f'{name} values must all be greater than or equal to {value}.'
        raise ValueError(msg)
    return array


def check_less_than(
    array: _ArrayT,
    /,
    value: float | _Scalar,
    *,
    strict: bool = True,
    name: str = 'Array',
) -> _ArrayT:
    """Check if an array's elements are all less than some value.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    value : float
        Value which the array's elements must be less than.

    strict : bool, default: True
        If ``True``, the array's value must be strictly less than
        ``value``. Otherwise, values must be less than or equal to
        ``value``.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If not all array elements are less than (or equal to if
        ``strict=True``) the specified value.

    See Also
    --------
    check_greater_than
    check_range
    check_nonnegative

    Examples
    --------
    Check if an array's values are less than 0.

    >>> from pyvista_validation import check_less_than
    >>> check_less_than([-1, -2, -3], value=0)
    [-1, -2, -3]

    """
    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    valid_value = _validate_real_value(value)
    if strict and not np.all(array_ < valid_value):
        msg = f'{name} values must all be less than {value}.'
        raise ValueError(msg)
    if not np.all(array_ <= valid_value):
        msg = f'{name} values must all be less than or equal to {value}.'
        raise ValueError(msg)
    return array


def check_range(
    array: _ArrayT,
    /,
    rng: VectorLike,
    *,
    strict_lower: bool = False,
    strict_upper: bool = False,
    name: str = 'Array',
) -> _ArrayT:
    """Check if an array's values are all within a specific range.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    rng : VectorLike, optional
        Vector with two elements ``[min, max]`` specifying the minimum
        and maximum data values allowed, respectively. By default, the
        range endpoints are inclusive, that is, values must be >= min
        and <= max. Use ``strict_lower`` and/or ``strict_upper``
        to further restrict the allowable range. Use ``np.inf`` or
        ``-np.inf`` to specify open intervals, e.g. ``[0, np.inf]``.

    strict_lower : bool, default: False
        Enforce a strict lower bound for the range, that is, array values
        must be strictly greater than the minimum.

    strict_upper : bool, default: False
        Enforce a strict upper bound for the range, that is, array values
        must be strictly less than the maximum.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If any array value is outside the specified range.

    See Also
    --------
    check_less_than
    check_greater_than

    Examples
    --------
    Check if `an array's values are in the range ``[0, 1]``.

    >>> from pyvista_validation import check_range
    >>> check_range([0, 0.5, 1], rng=[0, 1])
    [0, 0.5, 1]

    """
    rng_ = rng if isinstance(rng, np.ndarray) else _cast_to_numpy(rng)
    check_shape(rng_, 2, name='Range')
    check_sorted(rng_, name='Range')

    array_ = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)
    check_greater_than(array_, rng_.item(0), strict=strict_lower, name=name)
    check_less_than(array_, rng_.item(1), strict=strict_upper, name=name)
    return array


def check_shape(
    array: _AnyArrayT,
    /,
    shape: _ShapeLike | list[_ShapeLike],
    *,
    name: str = 'Array',
) -> _AnyArrayT:
    """Check if an array has the specified shape.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    shape : ShapeLike | list[ShapeLike]
        A single shape or a list of any allowable shapes. If an integer,
        ``i``, the shape is interpreted as ``(i,)``. Use a value of
        -1 for any dimension where its size is allowed to vary, e.g.
        ``(-1,3)`` if any Nx3 array is allowed. Use ``()`` for the
        shape of scalar values (that is, 0-dimensional). If a list, the
        array must have at least one of the specified shapes.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If the array does not have any of the specified shapes.

    See Also
    --------
    check_length
    check_ndim

    Examples
    --------
    Check if an array is one-dimensional.

    >>> import numpy as np
    >>> from pyvista_validation import check_shape
    >>> check_shape([1, 2, 3], shape=(-1))
    [1, 2, 3]

    Check if an array is one-dimensional or a scalar.

    >>> check_shape(1, shape=[(), (-1)])
    1

    Check if an array is 3x3 or 4x4.

    >>> check_shape(np.eye(3), shape=[(3, 3), (4, 4)])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    """

    def _shape_is_allowed(a: _Shape, b: _Shape) -> bool:
        # a: array's actual shape
        # b: allowed shape (may have -1)
        return len(a) == len(b) and all(map(lambda x, y: True if x == y else y == -1, a, b))

    if not isinstance(shape, list):
        shape = [shape]

    array_shape = _shape_of(array)
    for input_shape in shape:
        valid_shape = _validate_shape_value(input_shape)
        if _shape_is_allowed(array_shape, valid_shape):
            return array

    msg = f'{name} has shape {array_shape} which is not allowed. '
    if len(shape) == 1:
        msg += f'Shape must be {shape[0]}.'
    else:
        msg += f'Shape must be one of {shape}.'
    raise ValueError(msg)


def check_ndim(
    array: _AnyArrayT,
    /,
    ndim: int | VectorLike,
    *,
    name: str = 'Array',
) -> _AnyArrayT:
    """Check if an array has the specified number of dimensions.

    .. note::
        Scalar values have a dimension of ``0``.

    Parameters
    ----------
    array : float | ArrayLike
        Number or array to check.

    ndim : int | Sequence[int], optional
        A single dimension or a sequence of allowable dimensions. If an
        integer, the array must have this number of dimensions. If a
        sequence, the array must have at least one of the specified number
        of dimensions.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    array_like
        The input, unchanged.

    Raises
    ------
    ValueError
        If the array does not have the required number of dimensions.

    See Also
    --------
    check_length
    check_shape

    Examples
    --------
    Check if an array is one-dimensional

    >>> import numpy as np
    >>> from pyvista_validation import check_ndim
    >>> check_ndim([1, 2, 3], ndim=1)
    [1, 2, 3]

    Check if an array is two-dimensional or a scalar.

    >>> check_ndim(1, ndim=(0, 2))
    1

    """
    ndim_ = np.atleast_1d(_cast_to_numpy(ndim))
    array_ndim = _cast_to_numpy(array).ndim
    if array_ndim not in ndim_:
        check_ndim(ndim, [0, 1], name='ndim')
        check_integer(ndim_, strict=True, name='ndim')
        expected = f'{ndim}' if len(ndim_) == 1 else f'one of {ndim}'
        msg = (
            f'{name} has the incorrect number of dimensions. '
            f'Got {array_ndim}, expected {expected}.'
        )
        raise ValueError(msg)
    return array


# fmt: off
@overload
def check_number(num: _NumberT, /, *, name: str = ...) -> _NumberT: ...  # type: ignore[overload-overlap]
@overload
def check_number(num: object, /, *, name: str = ...) -> numbers.Number: ...
# fmt: on
def check_number(num: object, /, *, name: str = 'Object') -> object:
    """Check if an object is an instance of ``Number``.

    A number is any instance of ``numbers.Number``, for example,  ``int``,
    ``float``, and ``complex``.

    Notes
    -----
    A NumPy ``ndarray`` is not an instance of ``Number``.

    Parameters
    ----------
    num : numbers.Number
        Number to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    Number
        The input, unchanged. Type checkers see it as a number.

    Raises
    ------
    TypeError
        If input is not an instance of ``Number``.

    Examples
    --------
    Check if a complex number is an instance of ``Number``.

    >>> from pyvista_validation import check_number
    >>> check_number(1 + 2j)
    (1+2j)

    """
    check_instance(num, numbers.Number, allow_subclass=True, name=name)
    return num


# fmt: off
@overload
def check_string(obj: _StrT, /, *, allow_subclass: bool = ..., name: str = ...) -> _StrT: ...
@overload
def check_string(obj: object, /, *, allow_subclass: bool = ..., name: str = ...) -> str: ...
# fmt: on
def check_string(obj: object, /, *, allow_subclass: bool = True, name: str = 'Object') -> str:
    """Check if an object is an instance of ``str``.

    Parameters
    ----------
    obj : str
        Object to check.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be ``str`` or a subclass of
        ``str``. Otherwise, subclasses are not allowed.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    str
        The input, unchanged. Type checkers see it as a string.

    Raises
    ------
    TypeError
        If input is not an instance of ``str``.

    See Also
    --------
    check_contains
    check_iterable_items
    check_sequence
    check_instance

    Examples
    --------
    Check if an object is a string.

    >>> from pyvista_validation import check_string
    >>> check_string('eggs')
    'eggs'

    """
    return check_instance(obj, str, allow_subclass=allow_subclass, name=name)


# fmt: off
@overload
def check_sequence(obj: _SequenceT, /, *, name: str = ...) -> _SequenceT: ...
@overload
def check_sequence(obj: object, /, *, name: str = ...) -> Sequence[object]: ...
# fmt: on
def check_sequence(obj: object, /, *, name: str = 'Object') -> Sequence[object]:
    """Check if an object is an instance of ``Sequence``.

    Parameters
    ----------
    obj : Sequence
        Object to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    Sequence
        The input, unchanged. Type checkers see it as a sequence.

    Raises
    ------
    TypeError
        If input is not an instance of ``Sequence``.

    See Also
    --------
    check_iterable
    check_instance

    Examples
    --------
    Check if an object is a sequence.

    >>> import numpy as np
    >>> from pyvista_validation import check_sequence
    >>> check_sequence([1, 2, 3])
    [1, 2, 3]
    >>> check_sequence('A')
    'A'

    """
    check_instance(obj, Sequence, allow_subclass=True, name=name)
    return cast('Sequence[object]', obj)


# fmt: off
@overload
def check_iterable(obj: _IterableT, /, *, name: str = ...) -> _IterableT: ...
@overload
def check_iterable(obj: object, /, *, name: str = ...) -> Iterable[object]: ...
# fmt: on
def check_iterable(obj: object, /, *, name: str = 'Object') -> Iterable[object]:
    """Check if an object is an instance of ``Iterable``.

    Parameters
    ----------
    obj : Iterable
        Iterable object to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    Iterable
        The input, unchanged. Type checkers see it as an iterable.

    Raises
    ------
    TypeError
        If input is not an instance of ``Iterable``.

    See Also
    --------
    check_sequence
    check_instance
    check_iterable_items

    Examples
    --------
    Check if an object is iterable.

    >>> import numpy as np
    >>> from pyvista_validation import check_iterable
    >>> check_iterable([1, 2, 3])
    [1, 2, 3]
    >>> check_iterable(np.array((4, 5, 6)))
    array([4, 5, 6])

    """
    check_instance(obj, Iterable, allow_subclass=True, name=name)
    return cast('Iterable[object]', obj)


# fmt: off
@overload
def check_instance(obj: object, /, classinfo: type[_T], *, allow_subclass: bool = ..., name: str = ...) -> _T: ...
@overload
def check_instance(obj: object, /, classinfo: tuple[type[_T1], type[_T2]], *, allow_subclass: bool = ..., name: str = ...) -> _T1 | _T2: ...
@overload
def check_instance(obj: object, /, classinfo: tuple[type[_T1], type[_T2], type[_T3]], *, allow_subclass: bool = ..., name: str = ...) -> _T1 | _T2 | _T3: ...
@overload
def check_instance(obj: _ObjectT, /, classinfo: _ClassInfo, *, allow_subclass: bool = ..., name: str = ...) -> _ObjectT: ...
# fmt: on
def check_instance(
    obj: object,
    /,
    classinfo: _ClassInfo,
    *,
    allow_subclass: bool = True,
    name: str = 'Object',
) -> object:
    """Check if an object is an instance of the given type or types.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | tuple[type, ...]
        ``type`` or tuple of types. Object must be an instance of one of
        the types.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be specified by ``classinfo``
         or any of its subclasses. Otherwise, subclasses are not allowed.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    object
        The input, unchanged. Type checkers see it as an instance of ``classinfo``.

    Raises
    ------
    TypeError
        If object is not an instance of any of the given types.

    See Also
    --------
    check_type
    check_number
    check_string
    check_iterable
    check_sequence

    Examples
    --------
    Check if an object is an instance of ``complex``.

    >>> from pyvista_validation import check_instance
    >>> check_instance(1 + 2j, complex)
    (1+2j)

    Check if an object is an instance of one of several types.

    >>> check_instance('eggs', (int, str))
    'eggs'

    """
    if not isinstance(name, str):
        msg = f'Name must be a string, got {type(name)} instead.'  # type: ignore[unreachable]
        raise TypeError(msg)
    if isinstance(classinfo, UnionType) or get_origin(classinfo) is Union:
        classinfo = _union_members(classinfo)
    num_classes = len(classinfo) if isinstance(classinfo, tuple) else 1
    is_instance = isinstance(obj, classinfo)
    is_error = False
    if allow_subclass and not is_instance:
        is_error = True
        if num_classes == 1:
            msg_body = 'must be an instance of'
        else:
            msg_body = 'must be an instance of any type'

    # Set flag to raise error if not type
    elif not allow_subclass:
        if isinstance(classinfo, tuple):
            if type(obj) not in classinfo:
                is_error = True
                msg_body = 'must have one of the following types'
        elif type(obj) is not classinfo:
            is_error = True
            msg_body = 'must have type'
    if is_error:
        msg = f'{name} {msg_body} {classinfo}. Got {type(obj)} instead.'
        raise TypeError(msg)
    return obj


# fmt: off
@overload
def check_type(obj: object, /, classinfo: type[_T], *, name: str = ...) -> _T: ...
@overload
def check_type(obj: object, /, classinfo: tuple[type[_T1], type[_T2]], *, name: str = ...) -> _T1 | _T2: ...
@overload
def check_type(obj: object, /, classinfo: tuple[type[_T1], type[_T2], type[_T3]], *, name: str = ...) -> _T1 | _T2 | _T3: ...
@overload
def check_type(obj: _ObjectT, /, classinfo: _ClassInfo, *, name: str = ...) -> _ObjectT: ...
# fmt: on
def check_type(obj: object, /, classinfo: _ClassInfo, *, name: str = 'Object') -> object:
    """Check if an object is one of the given type or types.

    Notes
    -----
    The use of :func:`check_instance` is generally preferred as it
    allows subclasses. Use :func:`check_type` only for cases where
    exact types are necessary.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | tuple[type, ...]
        ``type`` or tuple of types. Object must be one of the types.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    object
        The input, unchanged. Type checkers see it as an instance of ``classinfo``.

    Raises
    ------
    TypeError
        If object is not any of the given types.

    See Also
    --------
    check_instance

    Examples
    --------
    Check if an object is type ``dict`` or ``set``.

    >>> from pyvista_validation import check_type
    >>> check_type({'spam': 'eggs'}, (dict, set))
    {'spam': 'eggs'}

    """
    return check_instance(obj, classinfo, allow_subclass=False, name=name)


# fmt: off
@overload
def check_iterable_items(iterable_obj: _IterableT, /, item_type: _ClassInfo, *, allow_subclass: bool = ..., name: str = ...) -> _IterableT: ...
@overload
def check_iterable_items(iterable_obj: object, /, item_type: type[_T], *, allow_subclass: bool = ..., name: str = ...) -> Iterable[_T]: ...
@overload
def check_iterable_items(iterable_obj: object, /, item_type: _ClassInfo, *, allow_subclass: bool = ..., name: str = ...) -> Iterable[object]: ...
# fmt: on
def check_iterable_items(
    iterable_obj: object,
    /,
    item_type: _ClassInfo,
    *,
    allow_subclass: bool = True,
    name: str = 'Iterable',
) -> Iterable[object]:
    """Check if an iterable's items all have a specified type.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable to check.

    item_type : type | tuple[type, ...]
        Class types to check for. Each element of the sequence must
        have the type or one of the types specified.

    allow_subclass : bool, default: True
        If ``True``, the type of the iterable items must be any of the
        given types or a subclass thereof. Otherwise, subclasses are not
        allowed.

    name : str, default: "Iterable"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    Iterable
        The input, unchanged. Type checkers see its items as instances of ``item_type``.

    Raises
    ------
    TypeError
        If any of the items in the iterable have an incorrect type.

    See Also
    --------
    check_instance
    check_iterable
    check_iterable_items

    Examples
    --------
    Check if a ``tuple`` only has ``int`` or ``float`` elements.

    >>> from pyvista_validation import check_iterable_items
    >>> check_iterable_items((1, 2, 3.0), (int, float))
    (1, 2, 3.0)

    Check if a ``list`` only has ``list`` elements.

    >>> import pyvista_validation
    >>> check_iterable_items([[1], [2], [3]], list)
    [[1], [2], [3]]

    """
    items = check_iterable(iterable_obj, name=name)
    for item in items:
        check_instance(item, item_type, allow_subclass=allow_subclass, name=f'All items of {name}')
    return items


def check_contains(
    container: Container[object], /, must_contain: _ItemT, *, name: str = 'Input'
) -> _ItemT:
    """Check if an item is in a container.

    Parameters
    ----------
    container : Any
        Container to check.

    must_contain : Any
        Item which must be in the container.

    name : str, default: "Input"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    object
        ``must_contain``, unchanged.

    Raises
    ------
    ValueError
        If the item is not in the container.

    See Also
    --------
    check_iterable
    check_iterable_items

    Examples
    --------
    Check if ``"A"`` is in a list of strings.

    >>> from pyvista_validation import check_contains
    >>> check_contains(['A', 'B', 'C'], must_contain='A')
    'A'

    """
    if must_contain not in container:
        qualifier = 'one of' if isinstance(container, (list, tuple)) else 'in'
        msg = f"{name} '{must_contain}' is not valid. {name} must be {qualifier}: \n\t{container}"
        raise ValueError(msg)
    return must_contain


def check_length(
    sized_input: _SizedT,
    /,
    exact_length: int | VectorLike | None = None,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    must_be_1d: bool = False,
    allow_scalar: bool = False,
    name: str = 'Array',
) -> _SizedT:
    """Check if the length of an array meets specific requirements.

    Notes
    -----
    By default, this function operates on multidimensional arrays,
    where ``len(arr)`` may differ from the number of elements in the
    array. For one-dimensional cases (where ``len(arr) == arr.size``),
    set ``must_be_1d=True``.

    Parameters
    ----------
    sized_input : float | Sized
        Number or array to check.

    exact_length : int | VectorLike, optional
        Check if the array has the given length. If multiple
        numbers are given, the array's length must match one of the
        numbers.

    min_length : int, optional
        Check if the array has this length or greater.

    max_length : int, optional
        Check if the array has this length or less.

    must_be_1d : bool, default: False
        If ``True``, check if the shape of the array is one-dimensional,
        that is, that the array's shape is ``(N,)``.

    allow_scalar : bool, default: False
        If ``True``, a scalar input will be reshaped to have a length
        of 1. Otherwise, the check will fail since a scalar does not
        have a length.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Returns
    -------
    Sized
        The input, unchanged.

    Raises
    ------
    ValueError
        If the array's length is outside the specified range.

    See Also
    --------
    check_shape
    check_ndim

    Examples
    --------
    Check if an array has a length of 2 or 3.

    >>> from pyvista_validation import check_length
    >>> check_length([1, 2], exact_length=[2, 3])
    [1, 2]

    Check if an array has a minimum length of 3.

    >>> check_length((1, 2, 3), min_length=3)
    (1, 2, 3)

    Check if a multidimensional array has a maximum length of 2.

    >>> check_length([[1, 2, 3], [4, 5, 6]], max_length=2)
    [[1, 2, 3], [4, 5, 6]]

    """
    sized: float | _Scalar | Sized = sized_input
    if allow_scalar:
        # Reshape to 1D
        if isinstance(sized, (float, int, np.number, np.bool_)):
            sized = [sized]
        elif isinstance(sized, np.ndarray):
            array = cast('npt.NDArray[_Scalar]', sized)
            if array.ndim == 0:
                sized = array.reshape(1)
    if must_be_1d:
        check_shape(cast('_ArrayLikeOrScalar', sized), shape=(-1))
    array_len = len(cast('Sized', sized))
    if exact_length is not None:
        check_integer(exact_length, name="'exact_length'")
        if array_len not in np.atleast_1d(_cast_to_numpy(exact_length)):
            msg = (
                f'{name} must have a length equal to any of: {exact_length}. '
                f'Got length {array_len} instead.'
            )
            raise ValueError(msg)
    if min_length is not None:
        check_finite(min_length, name='Min length')
    if max_length is not None:
        check_finite(max_length, name='Max length')
    if min_length is not None and max_length is not None:
        check_sorted((min_length, max_length), name='Range')
    if min_length is not None and array_len < min_length:
        msg = f'{name} must have a minimum length of {min_length}. Got length {array_len} instead.'
        raise ValueError(msg)
    if max_length is not None and array_len > max_length:
        msg = f'{name} must have a maximum length of {max_length}. Got length {array_len} instead.'
        raise ValueError(msg)
    return sized_input


def _validate_shape_value(shape: _ShapeLike | None) -> _Shape:
    """Validate shape-like input and return its tuple representation."""
    if shape is None:
        # `None` is used to mean `any shape is allowed` by the array
        #  validation methods, so raise an error here.
        #  Also, setting `None` as a shape is deprecated by NumPy.
        msg = '`None` is not a valid shape. Use `()` instead.'
        raise TypeError(msg)
    if shape in [(), (-1,), (1,), (3,), (2,), (1, 3), (-1, 3)]:
        return shape

    def _is_valid_dim(d: object) -> bool:
        return isinstance(d, int) and d >= -1

    if _is_valid_dim(shape):
        return (cast('int', shape),)
    if isinstance(shape, tuple) and all(map(_is_valid_dim, shape)):
        return shape

    # Input is not valid at this point. Use checks to raise an
    # appropriate error
    check_instance(shape, (int, tuple), name='Shape')
    if isinstance(shape, int):
        shape = (shape,)
    else:
        check_iterable_items(shape, int, name='Shape')
    check_greater_than(shape, -1, name='Shape', strict=False)
    msg = 'This line should not be reachable.'  # pragma: no cover
    raise RuntimeError(msg)  # pragma: no cover


def _union_members(classinfo: _ClassInfo, /) -> tuple[type[object], ...]:
    """Return the classes of a union type, which ``typing`` reports as ``Any``."""
    return cast('tuple[type[object], ...]', get_args(classinfo))


def _dtype_of(obj: _DTypeLike | _AnyArrayLikeOrScalar, /) -> np.dtype[np.generic[object]]:
    """Return the dtype of a dtype-like object, or of the array an array-like casts to."""
    try:
        return np.dtype(cast('_DTypeLike', obj))
    except TypeError:
        return np.asanyarray(obj).dtype


def _issubdtype(dtype: np.dtype[np.generic[object]], base: _DTypeLike, /) -> bool:
    """Wrap ``np.issubdtype``, whose parameters are typed as ``Any``."""
    return np.issubdtype(dtype, base)


def _shape_of(array: _AnyArrayLikeOrScalar, /) -> tuple[int, ...]:
    """Return the shape the input would have as an array, without copying it."""
    return np.shape(cast('npt.ArrayLike', array))


if not TYPE_CHECKING:
    accelerate(
        globals(),
        [
            'check_subdtype',
            'check_real',
            'check_sorted',
            'check_finite',
            'check_integer',
            'check_nonnegative',
            'check_greater_than',
            'check_less_than',
            'check_range',
            'check_shape',
            'check_ndim',
            'check_number',
            'check_string',
            'check_sequence',
            'check_iterable',
            'check_instance',
            'check_type',
            'check_iterable_items',
            'check_contains',
            'check_length',
        ],
    )
