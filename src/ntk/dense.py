import torch
from src.ntk.kernel import Kernel


def affine(mat: torch.Tensor, W_std: float, b_std: float) -> torch.Tensor:

    if mat is None:
        return None

    return W_std**2 * mat + b_std**2


class Dense:

    def __init__(
        self,
        out_dim: int,
        W_std: float = 1.0,
        b_std: float = 0.0,
        parameterization: str = "ntk",
        batch_axis: int = 0,
        channel_axis: int = -1,
    ):
        self.out_dim = out_dim
        self.W_std = W_std
        self.b_std = b_std
        self.parameterization = parameterization
        self.batch_axis = batch_axis
        self.channel_axis = channel_axis

    def kernel_function(self, k: Kernel):

        cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

        def fc(x):
            return affine(x, self.W_std, self.b_std)

        if self.parameterization == "ntk":
            cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))
            if ntk is not None:
                ntk = nngp + self.W_std**2 * ntk
        elif self.parameterization == "standard":
            input_width = k.shape1[self.channel_axis]
            if ntk is not None:
                ntk = input_width * nngp + 1.0 + self.W_std**2 * ntk
            cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))

        k = k.replace(
            cov1=cov1,
            nngp=nngp,
            cov2=cov2,
            ntk=ntk,
            is_gaussian=True,
        )

        return k
