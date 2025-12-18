import torch
from typing import NamedTuple, Optional, Dict, Union, Sequence
from functools import lru_cache

from src.ntk.core import (
    add_diagonal_regularizer,
    canonicalize_axis,
    canonicalize_get,
    get_attr,
    get_axes,
    get_cho_solve,
    get_dependency,
    get_first,
    make_2d,
    make_expm1_fn,
    make_inv_expm1_fn,
    zip_axes,
)
from src.ntk.kernel import KernelLike


class Gaussian(NamedTuple):
    """A `(mean, covariance)` convenience namedtuple."""

    mean: torch.Tensor
    covariance: torch.Tensor


def gp_inference(
    k_train_train,
    y_train: torch.Tensor,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False,
    trace_axes: tuple = (-1,),
):

    even, odd, first, last = get_axes(get_first(k_train_train))
    trace_axes = canonicalize_axis(trace_axes, y_train)

    @lru_cache(2)
    def solve(g: str):
        k_dd = get_attr(k_train_train, g)
        return get_cho_solve(k_dd, diag_reg, diag_reg_absolute_scale)

    @lru_cache(2)
    def k_inv_y(g: str):
        return solve(g)(y_train, trace_axes)

    def predict_fn(
        get: Optional[str] = None, k_test_train=None, k_test_test=None
    ) -> Dict[str, Union[torch.Tensor, Gaussian]]:

        if get is None:
            get = ("nngp", "ntk")

        out = {}

        for g in get:
            k = g if g != "ntkgp" else "ntk"
            k_dd = get_attr(k_train_train, k)
            k_td = None if k_test_train is None else get_attr(k_test_train, k)

            if k_td is None:
                y = y_train.to(k_dd.dtype)
            else:
                y = torch.tensordot(k_td, k_inv_y(k), dims=(odd, first))
                y = torch.moveaxis(y, list(range(-len(trace_axes), 0)), trace_axes)

            if k_test_test is not None:
                if k_td is None:
                    out[g] = Gaussian(y, torch.zeros_like(k_dd, dtype=k_dd.dtype))
                else:
                    if g == "ntk" and (
                        not hasattr(k_train_train, "nngp")
                        or not hasattr(k_test_train, "nngp")
                    ):
                        raise ValueError(
                            'If `"ntk" in get`, and `k_test_test is not None`, '
                            "and `k_test_train is not None`, you need both NTK and NNGP."
                        )

                    g_init = "nngp" if g != "ntkgp" else "ntk"

                    k_td_g_inv_y = solve(k)(get_attr(k_test_train, g_init), even)
                    k_tt = get_attr(k_test_test, g_init)

                    if g == "nngp" or g == "ntkgp":
                        cov = torch.tensordot(k_td, k_td_g_inv_y, dims=(odd, first))
                        cov = k_tt - zip_axes(cov)
                        out[g] = Gaussian(y, cov)

                    elif g == "ntk":
                        term_1 = solve(g)(k_td, even)
                        cov = torch.tensordot(
                            get_attr(k_train_train, "nngp"),
                            term_1,
                            dims=([odd], [first]),
                        )
                        cov = torch.tensordot(term_1, cov, dims=([first], [first]))

                        term_2 = torch.tensordot(
                            k_td, k_td_g_inv_y, dims=([odd], [first])
                        )
                        term_2 += torch.moveaxis(term_2, first, last)
                        cov = zip_axes(cov - term_2) + k_tt
                        out[g] = Gaussian(y, cov)

                    else:
                        raise ValueError(g)
            else:
                out[g] = y

        out = out[get[0]]
        result = out.mean, out.covariance

        return result

    return predict_fn


def gradient_descent_mse_ensemble(
    kernel_fn,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    learning_rate: float = 1.0,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False,
    trace_axes: tuple = (-1,),
    device: str = "cpu",
):
    r"""Predicts the gaussian embedding induced by gradient descent on MSE loss."""

    expm1 = make_expm1_fn(y_train.numel())
    inv_expm1 = make_inv_expm1_fn(y_train.numel())

    trace_axes = canonicalize_axis(trace_axes, y_train)
    trace_axes = tuple(-y_train.ndim + a for a in trace_axes)
    n_trace_axes = len(trace_axes)
    last_t_axes = range(-n_trace_axes, 0)
    trace_shape = tuple(y_train.shape[a] for a in trace_axes)

    y_train_flat = torch.moveaxis(y_train, trace_axes, [*last_t_axes]).reshape(
        (-1,) + trace_shape
    )

    k_dd_cache = {}

    def get_k_train_train(get: Sequence[str]):
        if len(get) == 1:
            get = get[0]
            if get not in k_dd_cache:
                k_dd_cache[get] = kernel_fn(x_train, None, get)

        elif len(get) == 2:
            if not any(g in k_dd_cache for g in get):
                k_dd_cache.update(kernel_fn(x_train, None, get)._asdict())
            else:
                for g in get:
                    if g not in k_dd_cache:
                        k_dd_cache[g] = kernel_fn(x_train, None, g)

        else:
            raise ValueError(get)
        return KernelLike(**k_dd_cache)

    @lru_cache(2)
    def eigenspace(get: str):
        k_dd = getattr(get_k_train_train((get,)), get)
        k_dd = add_diagonal_regularizer(
            make_2d(k_dd), diag_reg, diag_reg_absolute_scale
        )
        evals, evecs = torch.linalg.eigh(k_dd)
        evals = evals.unsqueeze(0)
        return evals, evecs

    @lru_cache(4)
    def predict_inf(get):
        _, get = canonicalize_get(get)
        k_dd = get_k_train_train(get)
        return gp_inference(
            k_dd, y_train, diag_reg, diag_reg_absolute_scale, trace_axes
        )

    def get_kernels(get, x_test: Optional[torch.Tensor], compute_cov: bool):
        get = get_dependency(get, compute_cov)
        k_dd = get_k_train_train(get)
        if x_test is None:
            k_td = None
            nngp_tt = compute_cov or None
        else:

            k_td = kernel_fn(x_test, x_train, get)

            if compute_cov:
                nngp_tt = kernel_fn(x_test, None, "nngp")
            else:
                nngp_tt = None
        return k_dd, k_td, nngp_tt

    def predict_fn(
        t: Optional[Union[float, torch.Tensor]] = None,
        x_test: Optional[torch.Tensor] = None,
        get: Optional[str] = None,
        compute_cov: bool = False,
    ) -> Dict[str, Gaussian]:
        """Return output mean and covariance on the test set at time[s] `t`."""
        if get is None:
            get = ("nngp", "ntk")

        get_is_not_tuple, get = canonicalize_get(get)

        # train-train, test-train, test-test.
        k_dd, k_td, nngp_tt = get_kernels(get, x_test, compute_cov)

        # Infinite time.
        if t is None:
            return predict_inf(get)(get=get, k_test_train=k_td, k_test_test=nngp_tt)

        # Finite time.
        t = torch.tensor(t, device=device) * learning_rate
        t_shape = t.shape
        t = t.reshape((-1, 1))

        def reshape_mean(mean):
            k = get_first(k_dd if k_td is None else k_td)
            mean = mean.reshape(t_shape + k.shape[::2] + trace_shape)
            mean = torch.moveaxis(mean, last_t_axes, trace_axes)
            return mean

        def reshape_cov(cov):
            k = get_first(k_dd if k_td is None else k_td)
            cov_shape_t = t_shape + k.shape[::2] * 2
            return zip_axes(cov.reshape(cov_shape_t), len(t_shape))

        out = {}

        for g in get:
            evals, evecs = eigenspace(g)

            # Training set.
            if k_td is None:
                mean = torch.einsum(
                    "ji,ti,ki,k...->tj...",
                    evecs,
                    -expm1(evals, t),
                    evecs,
                    y_train_flat,
                )

            # Test set.
            else:
                neg_inv_expm1 = -inv_expm1(evals, t)
                ktd_g = make_2d(getattr(k_td, g))
                mean = torch.einsum(
                    "lj,ji,ti,ki,k...->tl...",
                    ktd_g,
                    evecs,
                    neg_inv_expm1,
                    evecs,
                    y_train_flat,
                )

            mean = reshape_mean(mean)

            if nngp_tt is not None:
                nngp_dd = make_2d(k_dd.nngp)

                # Training set.
                if k_td is None:
                    if g == "nngp":
                        cov = torch.einsum(
                            "ji,ti,ki->tjk",
                            evecs,
                            (
                                torch.clamp(evals, min=0.0)
                                * torch.exp(
                                    -2
                                    * torch.clamp(evals, min=0.0)
                                    * t
                                    / y_train.numel()
                                )
                            ),
                            evecs,
                        )

                    elif g == "ntk":
                        exp = torch.einsum(
                            "mi,ti,ki->tmk",
                            evecs,
                            torch.exp(
                                -torch.clamp(evals, min=0.0) * t / y_train.numel()
                            ),
                            evecs,
                        )
                        cov = torch.einsum("tmk,kl,tnl->tmn", exp, nngp_dd, exp)

                    else:
                        raise ValueError(g)

                # Test set.
                else:
                    _nngp_tt = make_2d(nngp_tt).unsqueeze(0)

                    if g == "nngp":
                        cov = _nngp_tt - torch.einsum(
                            "mj,ji,ti,ki,lk->tml",
                            ktd_g,
                            evecs,
                            -inv_expm1(evals, 2 * t),
                            evecs,
                            ktd_g,
                        )

                    elif g == "ntk":
                        term_1 = torch.einsum(
                            "mi,ti,ki,lk->tml",
                            evecs,
                            neg_inv_expm1,
                            evecs,
                            ktd_g,
                        )
                        term_2 = torch.einsum(
                            "mj,ji,ti,ki,lk->tml",
                            ktd_g,
                            evecs,
                            neg_inv_expm1,
                            evecs,
                            make_2d(k_td.nngp),
                        )
                        term_2 += torch.moveaxis(term_2, 1, 2)
                        cov = torch.einsum(
                            "tji,jk,tkl->til",
                            term_1,
                            nngp_dd,
                            term_1,
                        )
                        cov += -term_2 + _nngp_tt

                    else:
                        raise ValueError(g)

                out[g] = Gaussian(mean, reshape_cov(cov))

            else:
                out[g] = mean

        return out

    return predict_fn
