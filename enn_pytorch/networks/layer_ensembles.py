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

"""Implementing layer ensemble ENNs in PyTorch."""
from typing import Callable, Optional, Sequence

from enn_pytorch import base
from enn_pytorch import utils
from enn_pytorch.networks import indexers
from enn_pytorch.networks import priors
import torch
import torch.nn as nn
import numpy as np


class EnsembleLayer(nn.Module):
    """A layer that contains an ensemble of sub-layers."""
    
    def __init__(
        self,
        ensemble: Sequence[nn.Module],
        priors: Sequence[nn.Module],
        prior_scale: float = 1.0,
        activation=torch.nn.functional.relu,
    ):
        super().__init__()
        self.ensemble = nn.ModuleList(list(ensemble))
        self.priors = nn.ModuleList(list(priors))
        self.prior_scale = prior_scale
        self.activation = activation

    def forward(self, inputs: torch.Tensor, index: base.Index) -> torch.Tensor:
        """Index must be a single integer indicating which layer to forward."""
        if isinstance(index, torch.Tensor):
            index = index.item()
        
        model_output = self.activation(self.ensemble[index](inputs))

        if len(self.priors) > 0:
            with torch.no_grad():
                prior_output = self.activation(self.priors[index](inputs))
            output = model_output + self.prior_scale * prior_output
        else:
            output = model_output

        return output


class LayerEnsembleNetwork(base.EpistemicNetwork):
    """A layer-ensemble MLP (with flatten) without any prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        module_factory: Callable = None,
        correlated: bool = False,
    ):
        """Initialize a layer ensemble network.
        
        Args:
            output_sizes: Sequence of layer output sizes
            num_ensembles: Sequence of ensemble sizes per layer
            module_factory: Factory function to create layer modules (default: nn.LazyLinear)
            correlated: Whether to use correlated indices across layers
        """
        
        if module_factory is None:
            def module_factory(output_size):
                return nn.LazyLinear(output_size)
        
        class LayerEnsembleMLP(nn.Module):
            def __init__(self, output_sizes, num_ensembles):
                super().__init__()
                self.flatten = nn.Flatten()
                
                self.layers = nn.ModuleList([
                    EnsembleLayer(
                        [module_factory(output_size) for _ in range(num_ensemble)],
                        [],
                        0.0,
                        torch.nn.functional.relu if i < len(num_ensembles) - 1 else lambda x: x,
                    )
                    for i, (num_ensemble, output_size) in enumerate(
                        zip(num_ensembles, output_sizes)
                    )
                ])
            
            def forward(self, inputs: torch.Tensor, full_index: base.Index) -> base.Output:
                x = self.flatten(inputs)
                
                for layer, index in zip(self.layers, full_index):
                    x = layer(x, index)
                
                return x
        
        model = LayerEnsembleMLP(output_sizes, num_ensembles)
        indexer_fn = indexers.LayerEnsembleIndexer(num_ensembles, correlated)

        def apply_fn(model: nn.Module, x: torch.Tensor, z: base.Index) -> base.Output:
            return model(x, z)

        def init_fn() -> nn.Module:
            return model

        super().__init__(apply_fn, init_fn, indexer_fn)


class LayerEnsembleNetworkWithPriors(base.EpistemicNetwork):
    """A layer-ensemble MLP (with flatten) with priors."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        prior_scale: float = 1.0,
        module_factory: Callable = None,
        seed: int = 0,
        correlated: bool = False,
    ):
        """Initialize a layer ensemble network with priors.
        
        Args:
            output_sizes: Sequence of layer output sizes
            num_ensembles: Sequence of ensemble sizes per layer
            prior_scale: Scaling factor for prior outputs
            module_factory: Factory function to create layer modules (default: nn.LazyLinear)
            seed: Random seed for prior initialization
            correlated: Whether to use correlated indices across layers
        """
        
        if module_factory is None:
            def module_factory(output_size):
                return nn.LazyLinear(output_size)
        
        torch.manual_seed(seed)
        
        class LayerEnsembleMLP(nn.Module):
            def __init__(self, output_sizes, num_ensembles, prior_scale):
                super().__init__()
                self.flatten = nn.Flatten()
                
                self.layers = nn.ModuleList([
                    EnsembleLayer(
                        [module_factory(output_size) for _ in range(num_ensemble)],
                        [module_factory(output_size) for _ in range(num_ensemble)],
                        prior_scale,
                        torch.nn.functional.relu if i < len(num_ensembles) - 1 else lambda x: x,
                    )
                    for i, (num_ensemble, output_size) in enumerate(
                        zip(num_ensembles, output_sizes)
                    )
                ])
            
            def forward(self, inputs: torch.Tensor, full_index: base.Index) -> base.Output:
                x = self.flatten(inputs)
                
                for layer, index in zip(self.layers, full_index):
                    x = layer(x, index)
                
                return x
        
        model = LayerEnsembleMLP(output_sizes, num_ensembles, prior_scale)
        indexer_fn = indexers.LayerEnsembleIndexer(num_ensembles, correlated)

        def apply_fn(model: nn.Module, x: torch.Tensor, z: base.Index) -> base.Output:
            return model(x, z)

        def init_fn() -> nn.Module:
            return model

        super().__init__(apply_fn, init_fn, indexer_fn)


def make_einsum_layer_ensemble_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: torch.Tensor,
    num_ensembles: Sequence[int],
    prior_scale: float = 1.0,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> base.EpistemicNetwork:
    """Factory method to create layer ensemble MLP with matched prior.

    Args:
        output_sizes: Sequence of integer sizes for the MLPs.
        dummy_input: Example x input for prior initialization.
        num_ensembles: Sequence of ensemble sizes per layer.
        prior_scale: Float rescaling of the prior MLP.
        nonzero_bias: Whether to make the initial layer bias nonzero.
        seed: integer seed for prior init.

    Returns:
        EpistemicNetwork ENN of the ensemble of MLP with matched prior.
    """
    return LayerEnsembleNetworkWithPriors(
        output_sizes=output_sizes,
        num_ensembles=num_ensembles,
        prior_scale=prior_scale,
        seed=seed,
        correlated=False,
    )


def make_layer_ensemble_cor_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: torch.Tensor,
    num_ensembles: Sequence[int],
    prior_scale: float = 1.0,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> base.EpistemicNetwork:
    """Factory method to create correlated layer ensemble MLP with matched prior.

    Args:
        output_sizes: Sequence of integer sizes for the MLPs.
        dummy_input: Example x input for prior initialization.
        num_ensembles: Sequence of ensemble sizes per layer (should be uniform).
        prior_scale: Float rescaling of the prior MLP.
        nonzero_bias: Whether to make the initial layer bias nonzero.
        seed: integer seed for prior init.

    Returns:
        EpistemicNetwork ENN of the correlated ensemble of MLP with matched prior.
    """
    return LayerEnsembleNetworkWithPriors(
        output_sizes=output_sizes,
        num_ensembles=num_ensembles,
        prior_scale=prior_scale,
        seed=seed,
        correlated=True,
    )


# Alias for compatibility
def make_true_einsum_layer_ensemble_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: torch.Tensor,
    num_ensembles: Sequence[int],
    prior_scale: float = 1.0,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> base.EpistemicNetwork:
    """Alias for make_einsum_layer_ensemble_mlp_with_prior_enn."""
    return make_einsum_layer_ensemble_mlp_with_prior_enn(
        output_sizes, dummy_input, num_ensembles, prior_scale, nonzero_bias, seed
    )
