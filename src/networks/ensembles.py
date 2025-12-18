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

"""Implementing some types of ENN ensembles in PyTorch."""
from typing import Callable, Optional, Sequence

from src import base
from src import utils
from src.networks import indexers
from src.networks import priors
import torch
import torch.nn as nn


class Ensemble(base.EpistemicModule):
    """Naive discrete ensemble of PyTorch modules.

    This implementation assumes *one* integer index per batch, and forwards the
    entire batch with that index.
    """

    def __init__(self, ensemble: Sequence[nn.Module]):
        super().__init__()
        self.ensemble = nn.ModuleList(list(ensemble))
        self.num_ensemble = len(ensemble)

    def forward(self, inputs: torch.Tensor, index: base.Index) -> torch.Tensor:
        """Index must be a single integer indicating which net to forward."""
        if isinstance(index, torch.Tensor):
            index = index.item()
        return self.ensemble[index](inputs)


class MLPEnsembleEnn(base.EpistemicNetwork):
    """An ensemble of MLP (with flatten) and without any prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensemble: int,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
    ):
        """An ensemble of MLP (with flatten) and without any prior."""

        def create_mlp():
            layers = [nn.Flatten()]
            sizes = list(output_sizes)
            # Need to know input size - will be set on first forward
            for i, size in enumerate(sizes):
                if i == 0:
                    # First layer will use LazyLinear
                    layers.append(nn.LazyLinear(size))
                else:
                    layers.append(nn.Linear(sizes[i - 1], size))
                if i < len(sizes) - 1:
                    layers.append(nn.ReLU())

            mlp = nn.Sequential(*layers)

            # Apply custom initialization if provided
            if w_init is not None or b_init is not None:
                for module in mlp.modules():
                    if isinstance(module, (nn.Linear, nn.LazyLinear)):
                        if w_init is not None:
                            w_init(module.weight)
                        if b_init is not None and module.bias is not None:
                            b_init(module.bias)

            return mlp

        model = Ensemble([create_mlp() for _ in range(num_ensemble)])
        indexer_fn = indexers.EnsembleIndexer(num_ensemble)

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            return model(inputs, index)

        def init_fn() -> nn.Module:
            return model

        super().__init__(apply_fn, init_fn, indexer_fn)


def make_mlp_ensemble_prior_fns(
    output_sizes: Sequence[int],
    dummy_input: torch.Tensor,
    num_ensemble: int,
    seed: int = 0,
    w_init: Optional[Callable] = None,
    b_init: Optional[Callable] = None,
) -> Sequence[Callable[[torch.Tensor], torch.Tensor]]:
    """Factory method for creating ensemble of prior functions."""

    def create_and_init_mlp(seed_val):
        torch.manual_seed(seed_val)
        layers = [nn.Flatten()]
        sizes = list(output_sizes)

        # Infer input size from dummy input
        if isinstance(dummy_input, torch.Tensor):
            flat_input = dummy_input.flatten(1)
            input_size = flat_input.shape[-1]
        else:
            flat_input = torch.tensor(dummy_input).flatten(1)
            input_size = flat_input.shape[-1]

        for i, size in enumerate(sizes):
            if i == 0:
                layers.append(nn.Linear(input_size, size))
            else:
                layers.append(nn.Linear(sizes[i - 1], size))
            if i < len(sizes) - 1:
                layers.append(nn.ReLU())

        mlp = nn.Sequential(*layers)

        # Apply custom initialization if provided
        if w_init is not None or b_init is not None:
            for module in mlp.modules():
                if isinstance(module, nn.Linear):
                    if w_init is not None:
                        w_init(module.weight)
                    if b_init is not None and module.bias is not None:
                        b_init(module.bias)

        mlp.eval()  # Set to eval mode for prior
        return mlp

    prior_fns = []
    for i in range(num_ensemble):
        mlp = create_and_init_mlp(seed + i)
        # Capture mlp in closure
        prior_fns.append(lambda x, m=mlp: m(x))

    return prior_fns


def wrap_sequence_as_prior(
    prior_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor, base.Index], torch.Tensor]:
    def prior_fn(x: torch.Tensor, z: base.Index) -> torch.Tensor:
        if isinstance(z, torch.Tensor):
            z = z.item()
        return prior_fns[z](x)

    return prior_fn


def make_random_gp_ensemble_prior_fns(
    input_dim: int,
    output_dim: int,
    num_feat: int,
    gamma: priors.GpGamma,
    num_ensemble: int,
    seed: int = 0,
) -> Sequence[Callable[[torch.Tensor], torch.Tensor]]:
    """Factory method for creating an ensemble of random GPs."""
    prior_fns = []
    for i in range(num_ensemble):
        prior_fns.append(
            priors.make_random_feat_gp(
                input_dim, output_dim, num_feat, seed + i, gamma, scale=1.0
            )
        )
    return prior_fns


class MLPEnsembleMatchedPrior(base.EpistemicNetwork):
    """Ensemble of MLPs with matched prior functions."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        dummy_input: torch.Tensor,
        num_ensemble: int,
        prior_scale: float = 1.0,
        seed: int = 0,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
    ):
        """Ensemble of MLPs with matched prior functions."""
        mlp_priors = make_mlp_ensemble_prior_fns(
            output_sizes, dummy_input, num_ensemble, seed, w_init, b_init
        )
        enn = priors.EnnWithAdditivePrior(
            enn=MLPEnsembleEnn(
                output_sizes, num_ensemble, w_init=w_init, b_init=b_init
            ),
            prior_fn=wrap_sequence_as_prior(mlp_priors),
            prior_scale=prior_scale,
        )
        super().__init__(enn.apply, enn.init, enn.indexer)


class MLPEnsembleGpPrior(base.EpistemicNetwork):
    """Ensemble of MLPs with GP random kitchen sink prior functions."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        input_dim: int,
        num_ensemble: int,
        num_feat: int,
        gamma: priors.GpGamma = 1.0,
        prior_scale: float = 1,
        seed: int = 0,
    ):
        """Ensemble of MLPs with GP random kitchen sink prior functions."""
        gp_priors = make_random_gp_ensemble_prior_fns(
            input_dim, output_sizes[-1], num_feat, gamma, num_ensemble, seed
        )
        enn = priors.EnnWithAdditivePrior(
            enn=MLPEnsembleEnn(output_sizes, num_ensemble),
            prior_fn=wrap_sequence_as_prior(gp_priors),
            prior_scale=prior_scale,
        )
        super().__init__(enn.apply, enn.init, enn.indexer)


class MLPEnsembleArbitraryPrior(base.EpistemicNetwork):
    """Ensemble of MLPs with arbitrary prior functions."""

    def __init__(
        self,
        prior_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        output_sizes: Sequence[int],
        num_ensemble: int,
        prior_scale: float = 1.0,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
    ):
        """Ensemble of MLPs with arbitrary prior functions."""
        enn = priors.EnnWithAdditivePrior(
            enn=MLPEnsembleEnn(
                output_sizes, num_ensemble, w_init=w_init, b_init=b_init
            ),
            prior_fn=wrap_sequence_as_prior(prior_fns),
            prior_scale=prior_scale,
        )
        super().__init__(enn.apply, enn.init, enn.indexer)
