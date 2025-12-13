import torch
import numpy as np
from enn_pytorch.ntk.core import elementwise, get_diagonal_outer_prods
from enn_pytorch.ntk.kernel import Kernel


def sqrt(x, tol=torch.tensor(0.0)):
    return torch.sqrt(torch.maximum(x, tol))

class ABRelu:
    def __init__(
        self,
        a: float,
        b: float,
        do_stabilize: bool = False
    ):
        self.a = a
        self.b = b
        self.do_stabilize = do_stabilize

        def kernel_function(k: Kernel) -> Kernel:
            cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
            factor = torch.tensor(1.0)

            if self.do_stabilize:
                factor = torch.maximum(torch.max(torch.abs(nngp)), torch.tensor(1e-12))

            nngp /= factor
            cov1 /= factor
            if cov2 is not None:
                cov2 /= factor

            prod11, prod12, prod22 = get_diagonal_outer_prods(
                cov1,
                cov2,
                k.diagonal_batch,
                k.diagonal_spatial,
                torch.mul
            )

            def nngp_ntk_fn(nngp, prod, ntk=None):
                square_root = sqrt(prod - nngp ** 2)
                angles = torch.atan2(square_root, nngp)

                factor_coef = (self.a - self.b)**2 / (2 * np.pi)
                dot_sigma = (self.a**2 + self.b**2) / 2 - factor_coef * angles
                nngp = factor_coef * square_root + dot_sigma * nngp

                if ntk is not None:
                    ntk *= dot_sigma

                return nngp, ntk

            def nngp_fn_diag(nngp):
                return (self.a**2 + self.b**2) / 2 * nngp

            nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk=ntk)

            if k.diagonal_batch and k.diagonal_spatial:
                cov1 = nngp_fn_diag(cov1)
                if cov2 is not None:
                    cov2 = nngp_fn_diag(cov2)
            else:
                cov1, _ = nngp_ntk_fn(cov1, prod11)
                if cov2 is not None:
                    cov2, _ = nngp_ntk_fn(cov2, prod22)

            if self.do_stabilize:
                nngp *= factor
                cov1 *= factor
                if cov2 is not None:
                    cov2 *= factor

            return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    
        self.kernel_function = elementwise(kernel_function)


class Relu:
    def __init__(self, do_stabilize: bool = False):
        self.ab_relu = ABRelu(0, 1, do_stabilize)
        self.kernel_function = self.ab_relu.kernel_function
