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

"""Epistemic indexers for ENNs."""
import dataclasses
from typing import List, Sequence
from enn import base
import jax
import jax.numpy as jnp
import numpy as np

class PrngIndexer(base.EpistemicIndexer):
    """Index by JAX PRNG sequence."""

    def __call__(self, key: base.RngKey) -> base.Index:
        return key


@dataclasses.dataclass
class EnsembleIndexer(base.EpistemicIndexer):
    """Index into an ensemble by integer."""

    num_ensemble: int

    def __call__(self, key: base.RngKey) -> base.Index:
        return jax.random.randint(key, [], 0, self.num_ensemble)

    def batched(self, key: base.RngKey, num_samples: int) -> base.Index:

        def create_all_samples(num_ensemble):
            result = []
            for q in range(num_ensemble):
                result.append(q)

            return result

        all_samples = create_all_samples(self.num_ensemble)

        results = jax.random.choice(key, jnp.array(all_samples), [num_samples], replace=num_samples > self.num_ensemble)

        return results

@dataclasses.dataclass
class LayerEnsembleIndexer(base.EpistemicIndexer):
    """Index into an ensemble by integer."""

    num_ensembles: Sequence[int]
    correlated: bool = False

    def __call__(self, key: base.RngKey) -> base.Index:

        keys = jax.random.split(key, len(self.num_ensembles))

        if self.correlated:
            index = jax.random.randint(keys[0], [], 0, self.num_ensembles[0])
            return jnp.array(
                [index for key, num_ensemble in zip(keys, self.num_ensembles)]
            )

        return jnp.array(
            [jax.random.randint(key, [], 0, num_ensemble) for key, num_ensemble in zip(keys, self.num_ensembles)]
        )

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
        all_samples = jnp.array(all_samples)

        choices = jax.random.choice(key, len(all_samples), [num_samples], replace=False)
        results = all_samples[choices]

        return results


@dataclasses.dataclass
class ScaledGaussianIndexer(base.EpistemicIndexer):
    """A scaled Gaussian indexer."""

    index_dim: int
    # When index_scale is 1.0 the returned random variable has expected norm = 1.
    index_scale: float = 1.0

    def __call__(self, key: base.RngKey) -> base.Index:
        return (
            self.index_scale
            / jnp.sqrt(self.index_dim)
            * jax.random.normal(key, shape=[self.index_dim])
        )

@dataclasses.dataclass
class GaussianWithUnitIndexer(base.EpistemicIndexer):
    """Produces index (1, z) for z dimension=index_dim-1 unit ball."""

    index_dim: int

    @property
    def mean_index(self) -> base.Array:
        return jnp.append(1, jnp.zeros(self.index_dim - 1))

    def __call__(self, key: base.RngKey) -> base.Index:
        return jnp.append(
            1,
            jax.random.normal(key, shape=[self.index_dim - 1])
            / jnp.sqrt(self.index_dim - 1),
        )


@dataclasses.dataclass
class DirichletIndexer(base.EpistemicIndexer):
    """Samples a Dirichlet index with parameter alpha."""

    alpha: base.Array

    def __call__(self, key: base.RngKey) -> base.Index:
        return jax.random.dirichlet(key, self.alpha)
