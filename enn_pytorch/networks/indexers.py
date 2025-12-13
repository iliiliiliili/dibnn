# python3
# pylint: disable=g-bad-file-header
# Copyright Illia Oleksiienko
# This file is a modified version for pytorch of the original JAX implementation
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

"""Epistemic indexers for ENNs - PyTorch version."""
import dataclasses
from typing import List, Sequence
from enn_pytorch import base
import torch
import numpy as np


class PrngIndexer(base.EpistemicIndexer):
    """Index by random generator sequence."""

    def __call__(self, key: base.RngKey) -> base.Index:
        if isinstance(key, int):
            return key
        return key


@dataclasses.dataclass
class EnsembleIndexer(base.EpistemicIndexer):
    """Index into an ensemble by integer."""

    num_ensemble: int

    def __call__(self, key: base.RngKey) -> base.Index:
        if isinstance(key, int):
            torch.manual_seed(key)
            return torch.randint(0, self.num_ensemble, []).item()
        else:
            return torch.randint(0, self.num_ensemble, [], generator=key).item()

    def batched(self, key: base.RngKey, num_samples: int) -> base.Index:
        def create_all_samples(num_ensemble):
            result = []
            for q in range(num_ensemble):
                result.append(q)
            return result

        all_samples = create_all_samples(self.num_ensemble)
        
        if isinstance(key, int):
            torch.manual_seed(key)
            generator = None
        else:
            generator = key
            
        replace = num_samples > self.num_ensemble
        if replace:
            results = torch.tensor([all_samples[i] for i in torch.randint(0, len(all_samples), [num_samples], generator=generator)])
        else:
            indices = torch.randperm(len(all_samples), generator=generator)[:num_samples]
            results = torch.tensor([all_samples[i] for i in indices])

        return results


@dataclasses.dataclass
class LayerEnsembleIndexer(base.EpistemicIndexer):
    """Index into a layer ensemble by integer."""

    num_ensembles: Sequence[int]
    correlated: bool = False

    def __call__(self, key: base.RngKey) -> base.Index:
        if isinstance(key, int):
            keys = [key + i for i in range(len(self.num_ensembles))]
        else:
            # Split key for each layer
            keys = [torch.Generator().manual_seed(key + i) for i in range(len(self.num_ensembles))]

        if self.correlated:
            if isinstance(keys[0], int):
                torch.manual_seed(keys[0])
                index = torch.randint(0, self.num_ensembles[0], []).item()
            else:
                index = torch.randint(0, self.num_ensembles[0], [], generator=keys[0]).item()
            return torch.tensor([index for _ in self.num_ensembles])

        indices = []
        for i, (k, num_ensemble) in enumerate(zip(keys, self.num_ensembles)):
            if isinstance(k, int):
                torch.manual_seed(k)
                idx = torch.randint(0, num_ensemble, []).item()
            else:
                idx = torch.randint(0, num_ensemble, [], generator=k).item()
            indices.append(idx)
        
        return torch.tensor(indices)

    def batched(self, key: base.RngKey, num_samples: int) -> base.Index:
        if self.correlated:
            raise NotImplementedError()

        def create_all_samples(i, num_ensembles, prefix):
            result = []
            for q in range(num_ensembles[i]):
                value = (*prefix, q)

                if i + 1 < len(num_ensembles):
                    result += create_all_samples(i + 1, num_ensembles, value)
                else:
                    result.append(value)

            return result

        all_samples = create_all_samples(0, self.num_ensembles, [])
        all_samples = torch.tensor(all_samples)

        if isinstance(key, int):
            torch.manual_seed(key)
            generator = None
        else:
            generator = key

        choices = torch.randperm(len(all_samples), generator=generator)[:num_samples]
        results = all_samples[choices]

        return results


@dataclasses.dataclass
class ScaledGaussianIndexer(base.EpistemicIndexer):
    """Samples index as Gaussian(0, scale)."""

    index_dim: int
    scale: float = 1.0

    def __call__(self, key: base.RngKey) -> base.Index:
        if isinstance(key, int):
            torch.manual_seed(key)
            return torch.randn(self.index_dim) * self.scale
        else:
            return torch.randn(self.index_dim, generator=key) * self.scale


@dataclasses.dataclass
class GaussianWithUnitIndexer(base.EpistemicIndexer):
    """Samples index as Gaussian(0, scale) concatenated with unit vector."""

    index_dim: int
    scale: float = 1.0

    def __call__(self, key: base.RngKey) -> base.Index:
        if isinstance(key, int):
            torch.manual_seed(key)
            gaussian = torch.randn(self.index_dim) * self.scale
            unit = torch.ones(1)
        else:
            gaussian = torch.randn(self.index_dim, generator=key) * self.scale
            unit = torch.ones(1)
        return torch.cat([gaussian, unit])


@dataclasses.dataclass
class DirichletIndexer(base.EpistemicIndexer):
    """Samples index as Dirichlet distribution."""

    num_classes: int
    alpha: float = 1.0

    def __call__(self, key: base.RngKey) -> base.Index:
        if isinstance(key, int):
            torch.manual_seed(key)
        alpha_vec = torch.ones(self.num_classes) * self.alpha
        return torch.distributions.Dirichlet(alpha_vec).sample()
