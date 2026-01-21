# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Utilities for perturbing data with Gaussian noise."""

from typing import Callable

import dataclasses
from src import base
from src.networks.indexers import EnsembleIndexer, LayerEnsembleIndexer, ScaledGaussianIndexer, GaussianWithUnitIndexer
from src.data_noise import base as data_noise_base
import torch


@dataclasses.dataclass
class GaussianTargetNoise(data_noise_base.DataNoise):
    """Apply Gaussian noise to the target y."""

    enn: base.EpistemicNetwork
    noise_std: float
    seed: int = 0

    def __call__(self, data: base.Batch, index: base.Index, device: str) -> base.Batch:
        """Apply Gaussian noise to the target y."""
        assert data.y.shape == (data.y.shape[0], 1)
        noise_fn = make_noise_fn(self.enn, self.noise_std, self.seed)
        y_noise = noise_fn(data.data_index, index, device)
        return data._replace(y=data.y + y_noise)


NoiseFn = Callable[[base.DataIndex, base.Index, str], torch.Tensor]

def make_noise_fn(
    enn: base.EpistemicNetwork, noise_std: float, seed: int = 0
) -> NoiseFn:
    """Factory method to create noise_fn for given ENN."""
    indexer = data_noise_base.get_indexer(enn.indexer)

    if isinstance(indexer, EnsembleIndexer):
        return _make_ensemble_gaussian_noise(noise_std, seed)

    elif isinstance(indexer, LayerEnsembleIndexer):
        return _make_layer_ensemble_gaussian_noise(noise_std, seed)

    elif isinstance(indexer, ScaledGaussianIndexer):
        return _make_gaussian_index_noise(indexer.index_dim, noise_std, seed)

    elif isinstance(indexer, GaussianWithUnitIndexer):
        index_dim = indexer.index_dim
        raw_noise = _make_gaussian_index_noise(index_dim - 1, noise_std, seed)
        noise_fn = lambda d, z: raw_noise(d, z[1:])
        return noise_fn

    else:
        raise ValueError(f"Unsupported ENN={enn}.")


def _make_ensemble_gaussian_noise(noise_std: float, seed: int) -> NoiseFn:
    """Factory method to add Gaussian noise for ensemble index."""


    def noise_fn(data_index: base.DataIndex, index: base.Index, device: str) -> torch.Tensor:
        """Assumes integer index for ensemble."""
        batch_size = data_index.shape[0]
        
        generator = torch.Generator(device=device)
        
        def indexed_randn(index: int, device: str):
            """Generates indexed random normal samples."""
            generator.manual_seed(index)
            sample = torch.randn(1, generator=generator, device=device)[0]
            return sample
        
        # if not index.shape:
        #     index = index.repeat(batch_size)
        
        index = index.unsqueeze(0).repeat(batch_size, 1)
        index += seed
        index += data_index

        samples = torch.stack([indexed_randn(idx.item(), device) for idx in index.reshape(-1)]).reshape(index.shape) * noise_std
        return samples

    return noise_fn


def _make_layer_ensemble_gaussian_noise(
    noise_std: float, seed: int, factor=50
) -> NoiseFn:
    """Factory method to add Gaussian noise for layer ensemble index."""

    def noise_fn(data_index: base.DataIndex, input_index: base.Index) -> base.Tensor:
        """Assumes integer index for ensemble."""
        batch_size = data_index.shape[0]

        if len(input_index.shape) > 1:
            total_index = 0
            f = 1
            for i in input_index[0, :]:
                total_index += i.item() * f
                f *= factor
            index = torch.full((batch_size,), total_index, dtype=torch.long)
        else:
            index = torch.full((batch_size,), input_index.item(), dtype=torch.long)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        samples = torch.randn(batch_size, 1, generator=generator) * noise_std
        return samples

    return noise_fn


def _make_gaussian_index_noise(index_dim: int, noise_std: float, seed: int) -> NoiseFn:
    """Factory method to add Gaussian noise for index MLP."""

    def noise_fn(data_index: base.DataIndex, index: base.Index) -> base.Tensor:
        """Assumes scaled Gaussian index with reserved first component."""
        batch_size = data_index.shape[0]
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        b = torch.randn(batch_size, index_dim, generator=generator)
        z = index.unsqueeze(0).repeat(batch_size, 1)
        
        noise = torch.sum(b * z, dim=1, keepdim=True) * noise_std
        return noise

    return noise_fn
