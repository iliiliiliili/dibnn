import torch
from typing import Optional
from src.ntk.kernel import Kernel


class Serial:

    def __init__(
        self,
        layers,
    ):
        self.layers = layers

    def kernel_function(self, k: Kernel) -> Kernel:

        for layer in self.layers:
            k = layer.kernel_function(k)
        return k

    def __call__(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        get: str = "nngp",
        diagonal_batch: bool = True,
        diagonal_spatial: bool = False,
        batch_axis: Optional[int] = 0,
        channel_axis: Optional[int] = -1,
    ) -> Kernel:

        k = Kernel(
            x1=x1,
            x2=x2,
            diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial,
            batch_axis=batch_axis,
            channel_axis=channel_axis,
        )

        for layer in self.layers:
            k = layer.kernel_function(k)

        result = k[get]

        return result
