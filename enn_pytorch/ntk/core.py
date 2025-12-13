import functools
import numpy as np
import torch
from typing import Callable, List, Optional, Sequence, Tuple, Union


def zip_axes(
    x: torch.Tensor,
    start_axis: int = 0,
    end_axis: Optional[int] = None,
    unzip: bool = False,
) -> torch.Tensor:

    if end_axis is None:
        end_axis = len(x.shape)

    half_ndim, ragged = divmod(end_axis - start_axis, 2)
    if ragged:
        raise ValueError(
            f"Need even number of axes to zip, got {end_axis - start_axis}."
        )

    odd_axes = range(start_axis + 1, end_axis, 2)
    last_axes = range(end_axis - half_ndim, end_axis)

    if unzip:
        x = torch.moveaxis(x, list(odd_axes), list(last_axes))
    else:
        x = torch.moveaxis(x, list(last_axes), list(odd_axes))
    return x


def unzip_axes(
    x: torch.Tensor, start_axis: int = 0, end_axis: Optional[int] = None
) -> torch.Tensor:
    return zip_axes(x, start_axis, end_axis, unzip=True)


def size_at(
    x: Union[torch.Tensor, Sequence[int]], axes: Optional[Sequence[int]] = None
) -> int:
    if hasattr(x, "shape"):
        x = x.shape

    if axes is None:
        axes = range(len(x))

    return functools.reduce(lambda a, b: a * b, [x[a] for a in axes], 1)


def canonicalize_axis(axis, x) -> List[int]:
    axis = [axis] if isinstance(axis, int) else list(axis)
    n = len(x.shape)
    return list(set(np.arange(n)[axis]))


def get_diagonal(
    cov: Optional[np.ndarray], diagonal_batch: bool, diagonal_spatial: bool
) -> Optional[np.ndarray]:

    if cov is None:
        return cov

    batch_ndim = 1 if diagonal_batch else 2
    start_axis = 2 - batch_ndim
    end_axis = batch_ndim if diagonal_spatial else cov.ndim
    cov = unzip_axes(cov, start_axis, end_axis)
    return diagonal_between(cov, start_axis, end_axis)


def get_diagonal_outer_prods(
    cov1: torch.Tensor,
    cov2: Optional[torch.Tensor],
    diagonal_batch: bool,
    diagonal_spatial: bool,
    operation: Callable[[float, float], float],
    axis: Sequence[int] = (),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    axis = canonicalize_axis(axis, cov1)

    cov1 = get_diagonal(cov1, diagonal_batch, diagonal_spatial)
    cov2 = get_diagonal(cov2, diagonal_batch, diagonal_spatial)

    cov1, _ = mean_and_var(cov1, axis=axis, keepdims=True)
    cov2, _ = mean_and_var(cov2, axis=axis, keepdims=True)

    end_axis = 1 if diagonal_spatial else cov1.ndim
    prod12 = outer_prod(cov1, cov2, 0, end_axis, operation)

    start_axis = 1 if diagonal_batch else 0
    prod11 = outer_prod(cov1, cov1, start_axis, end_axis, operation)
    prod22 = (
        outer_prod(cov2, cov2, start_axis, end_axis, operation)
        if cov2 is not None
        else prod11
    )

    return prod11, prod12, prod22


def diagonal_between(
    x: torch.Tensor, start_axis: int = 0, end_axis: Optional[int] = None
) -> torch.Tensor:

    if end_axis is None:
        end_axis = x.ndim

    half_ndim, ragged = divmod(end_axis - start_axis, 2)
    if ragged:
        raise ValueError(
            f"Need even number of axes to flatten, got {end_axis - start_axis}."
        )
    if half_ndim == 0:
        return x

    side_shape = x.shape[start_axis : start_axis + half_ndim]
    side_size = size_at(side_shape)

    shape_2d = x.shape[:start_axis] + (side_size, side_size) + x.shape[end_axis:]
    shape_result = x.shape[:start_axis] + side_shape + x.shape[end_axis:]

    x = torch.diagonal(x.reshape(shape_2d), dim1=start_axis, dim2=start_axis + 1)
    x = torch.moveaxis(x, -1, start_axis)
    return x.reshape(shape_result)


def mean_and_var(
    x: Optional[torch.Tensor],
    axis=None,
    dtype: Optional[torch.dtype] = None,
    out: Optional[None] = None,
    ddof: int = 0,
    keepdims: bool = False,
    get_var: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    var = None
    if x is None:
        return x, var
    
    if len(axis) <= 0:
        return x, var

    mean = torch.mean(x, dim=axis, dtype=dtype, keepdim=keepdims)
    if get_var:
        var = torch.var(
            x, dim=axis, dtype=dtype, unbiased=(ddof == 1), keepdim=keepdims
        )
    return mean, var


def zip_flat(x, y):
    return tuple(c for xy in zip(x, y) for c in xy)


def interleave_ones(x, start_axis, end_axis, x_first):
    x_axes = x.shape[start_axis:end_axis]
    ones = (1,) * (end_axis - start_axis)
    shape = x.shape[:start_axis]
    if x_first:
        shape += zip_flat(x_axes, ones)
    else:
        shape += zip_flat(ones, x_axes)
    shape += x.shape[end_axis:]
    return x.reshape(shape)


def outer_prod(x, y, start_axis, end_axis, prod_op):
    if y is None:
        y = x
    x = interleave_ones(x, start_axis, end_axis, True)
    y = interleave_ones(y, start_axis, end_axis, False)
    return prod_op(x, y)


def elementwise(kernel_function):

    def new_kernel_function(k, **kwargs):
        if kernel_function is None:
            raise NotImplementedError(kernel_function)

        if not k.is_gaussian:
            raise ValueError(
                "The input to the activation function must be Gaussian, "
                "i.e. a random affine transform is required before the "
                "activation function."
            )
        k = kernel_function(k)  # pytype:disable=not-callable
        return k.replace(is_gaussian=False)

    return new_kernel_function


def get_axes(
    x: torch.Tensor,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    n = x.ndim
    return (
        tuple(range(0, n, 2)),
        tuple(range(1, n, 2)),
        tuple(range(0, n // 2)),
        tuple(range(n // 2, n)),
    )


def get_first(k) -> torch.Tensor:
    if isinstance(k, torch.Tensor):
        return k

    for g in ("nngp", "ntk"):
        if hasattr(k, g):
            v = getattr(k, g)
            if v is not None:
                return v

    raise ValueError(k)


def get_attr(k, g: str) -> torch.Tensor:
    if isinstance(k, torch.Tensor):
        return k
    return getattr(k, g)


def add_diagonal_regularizer(
    A: torch.Tensor, diag_reg: float, diag_reg_absolute_scale: bool
) -> torch.Tensor:
    dimension = A.shape[0]
    if not diag_reg_absolute_scale:
        diag_reg *= torch.trace(A) / dimension
    return A + diag_reg * torch.eye(dimension, device=A.device, dtype=A.dtype)


def get_cho_solve(
    A: torch.Tensor, diag_reg: float, diag_reg_absolute_scale: bool, lower: bool = False
) -> Callable[[torch.Tensor, Sequence[int]], torch.Tensor]:
    x_non_channel_shape = A.shape[1::2]
    A = A.reshape(-1, A.shape[-1])
    A = add_diagonal_regularizer(A, diag_reg, diag_reg_absolute_scale)
    C = torch.linalg.cholesky(A, upper=not lower)

    def cho_solve(b: torch.Tensor, b_axes: Sequence[int]) -> torch.Tensor:
        b_axes = canonicalize_axis(b_axes, b)
        last_b_axes = range(-len(b_axes), 0)
        x_shape = x_non_channel_shape + tuple(b.shape[a] for a in b_axes)

        b = torch.moveaxis(b, b_axes, list(last_b_axes))
        b = b.reshape(A.shape[-1], -1)

        x = torch.cholesky_solve(b, C, upper=not lower)
        x = x.reshape(x_shape)
        return x

    return cho_solve


def make_expm1_fn(normalization: float):

    def expm1_fn(evals: torch.Tensor, t: torch.Tensor):
        # Since our matrix really should be positive semidefinite,
        # we can threshold the eigenvalues to squash ones that are negative
        # for numerical reasons.
        return torch.expm1(-torch.clamp(evals, min=0.0) * t / normalization)

    return expm1_fn


def make_inv_expm1_fn(normalization: float):
    expm1_fn = make_expm1_fn(normalization)

    def inv_expm1_fn(evals: torch.Tensor, t: torch.Tensor):
        return expm1_fn(evals, t) / torch.abs(evals)

    return inv_expm1_fn


def make_2d(
    x: Optional[torch.Tensor], start_axis: int = 0, end_axis: Optional[int] = None
) -> Optional[torch.Tensor]:
    """Makes `x` 2D from `start_axis` to `end_axis`, preserving other axes.

    `x` is assumed to follow the (`X, X, Y, Y, Z, Z`) axes layout.

    Example:
      >>> x = np.ones((1, 2, 3, 3, 4, 4))
      >>> make_2d(x).shape == (12, 24)
      >>>
      >>> make_2d(x, 2).shape == (1, 2, 12, 12)
      >>>
      >>> make_2d(x, 2, 4).shape == (1, 2, 3, 3, 4, 4)
    """
    if x is None:
        return x

    if end_axis is None:
        end_axis = x.ndim

    x = unzip_axes(x, start_axis, end_axis)

    half_ndim = (end_axis - start_axis) // 2
    x = x.reshape(
        x.shape[:start_axis]
        + (
            size_at(x.shape[start_axis : start_axis + half_ndim]),
            size_at(x.shape[start_axis + half_ndim : end_axis]),
        )
        + x.shape[end_axis:]
    )
    return x


def canonicalize_get(get):
    if get is None:
        return True, get

    if not get:
        raise ValueError('"get" must be non-empty.')

    get_is_not_tuple = isinstance(get, str)
    if get_is_not_tuple:
        get = (get,)

    get = tuple(s.lower() for s in get)
    if len(set(get)) < len(get):
        raise ValueError('All entries in "get" must be unique. Got {}'.format(get))
    return get_is_not_tuple, get


def get_dependency(get, compute_cov: bool) -> Tuple[str, ...]:
    """Figure out dependency for get."""
    _, get = canonicalize_get(get)
    for g in get:
        if g not in ["nngp", "ntk"]:
            raise NotImplementedError(
                'Can only get either "nngp" or "ntk" predictions, got %s.' % g
            )
    get_dependency = ()
    if "nngp" in get or ("ntk" in get and compute_cov):
        get_dependency += ("nngp",)
    if "ntk" in get:
        get_dependency += ("ntk",)
    return get_dependency
