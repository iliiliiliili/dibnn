import dataclasses
import torch
from typing import Optional

from enn_pytorch.ntk.core import zip_axes

@dataclasses.dataclass
class KernelLike:

    nngp: torch.Tensor
    ntk: Optional[torch.Tensor] = None

    cov1: torch.Tensor = None
    cov2: Optional[torch.Tensor] = None
    
    is_gaussian: bool = False

    def __init__(
        self,
        nngp: torch.Tensor,
        ntk: Optional[torch.Tensor] = None,
        cov1: Optional[torch.Tensor] = None,
        cov2: Optional[torch.Tensor] = None,
        is_gaussian: bool = False,
    ):
        self.nngp = nngp
        self.ntk = ntk
        self.cov1 = cov1
        self.cov2 = cov2
        self.is_gaussian = is_gaussian


@dataclasses.dataclass
class Kernel:

    nngp: torch.Tensor
    ntk: Optional[torch.Tensor] = None

    cov1: torch.Tensor = None
    cov2: Optional[torch.Tensor] = None
    
    is_gaussian: bool = False

    def __init__(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        get: str = 'nngp',
        diagonal_batch: bool = True,
        diagonal_spatial: bool = False,
        batch_axis: Optional[int] = 0,
        channel_axis: Optional[int] = -1,
    ):
        if get != "nngp":
            raise ValueError(f"Unsupported kernel type: {get}")

        self.batch_axis = batch_axis
        self.channel_axis = channel_axis
        self.diagonal_batch = diagonal_batch
        self.diagonal_spatial = diagonal_spatial
    
        self.cov1 = self.__compute_covariance(x1, x1)
        self.cov2 = None if x2 is None else self.__compute_covariance(x2, x2)
        self.nngp = self.__compute_covariance(x1, x1 if x2 is None else x2, full_batch=True)
        self.ntk = None

        self.shape1 = x1.shape
        self.shape2 = None if x2 is None else x2.shape

    def __getitem__(self, key: str):
        return getattr(self, key if isinstance(key, str) else key[0])

    def replace(self, **kwargs) -> 'Kernel':

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        return self

    def __compute_covariance(self, x1, x2=None, full_batch=False):

        if x2 is None:
            x2 = x1

        if self.diagonal_batch and not full_batch:
            if self.diagonal_spatial:
                return self.__compute_covariance_diagonal_batch_diagonal_spatial(x1)
            else:
                return self.__compute_covariance_diagonal_batch_full_spatial(x1)
        else:
            if self.diagonal_spatial:
                return self.__compute_covariance_full_batch_diagonal_spatial(x1, x2)
            else:
                return self.__compute_covariance_full_batch_full_spatial(x1, x2)

    def __compute_covariance_diagonal_batch_diagonal_spatial(self, x):
        
        ret = torch.sum(x ** 2, dim=self.channel_axis)
        new_batch_axis = self.batch_axis - (1 if self.batch_axis > self.channel_axis else 0)
        ret = torch.movedim(ret, new_batch_axis, 0)
        return ret / x.shape[self.channel_axis]

    def __compute_covariance_diagonal_batch_full_spatial(self, x):
        
        ndim = x.ndim
        batch_axis = self.batch_axis % ndim
        channel_axis = self.channel_axis % ndim
        
        # Move batch and channel axes to standard positions
        # batch_axis -> 0, channel_axis -> -1
        axes_order = [i for i in range(ndim) if i not in (batch_axis, channel_axis)]
        x_reordered = x.moveaxis([batch_axis, channel_axis], [0, -1])
        
        # Reshape to (batch_size, -1, n_channels)
        batch_size = x_reordered.shape[0]
        n_channels = x_reordered.shape[-1]
        x_flat = x_reordered.reshape(batch_size, -1, n_channels)
        
        # Compute covariance: (batch_size, spatial_size, spatial_size)
        ret = torch.einsum('bic,bjc->bij', x_flat, x_flat)
        
        # Reshape back to original spatial dimensions
        spatial_shape = [x_reordered.shape[i] for i in range(1, len(x_reordered.shape) - 1)]
        ret = ret.reshape(batch_size, *spatial_shape, *spatial_shape)
        
        return ret / n_channels


    def __compute_covariance_full_batch_full_spatial(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch.Tensor:
        ret = torch.tensordot(x1, x2, dims=([self.channel_axis], [self.channel_axis]))
        new_batch_axis = self.batch_axis - (1 if self.batch_axis > self.channel_axis and self.channel_axis >= 0 else 0)
        ret = torch.moveaxis(ret, (new_batch_axis, x1.ndim - 1 + new_batch_axis), (0, 1))
        ret = zip_axes(ret, 2)
        return ret / x1.shape[self.channel_axis]

    def __compute_covariance_full_batch_diagonal_spatial(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError("Full batch, diagonal spatial covariance not implemented yet.")
