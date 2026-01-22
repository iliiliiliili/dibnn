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
import math
from typing import Callable, List, Optional, Sequence

from src import base
from src import utils
from src.networks import indexers
from src.networks import priors
import torch
import torch.nn as nn

from src.networks.functional import BatchedFunctionalLinear
from src.networks.priors import ModelWithPrior


class MlpEnsembleEnn(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensemble: int,
        nonzero_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        use_double_precision: bool = True,
    ):

        class EnsembleMlp(nn.Module):
            def __init__(self, num_ensemble, output_sizes, nonzero_bias, w_init, b_init, use_double_precision):
                super().__init__()

                self.functional_linear = BatchedFunctionalLinear()

                self.ensembled_weights = nn.ParameterList() # for each layer, size is (num_ensemble, out_size, in_size)
                self.ensembled_biases = nn.ParameterList() # for each layer, size is (num_ensemble, out_size)

                for i in range(1, len(output_sizes)):
                    layer_ensembled_weights = nn.Parameter(
                        torch.Tensor(num_ensemble, output_sizes[i], output_sizes[i - 1]).to(torch.float64 if use_double_precision else torch.float32)
                    )
                    layer_ensembled_biases = nn.Parameter(
                        torch.Tensor(num_ensemble, output_sizes[i]).to(torch.float64 if use_double_precision else torch.float32)
                    )

                    if w_init is not None:
                        w_init(layer_ensembled_weights)
                    else:
                        stddev = 1.0 / math.sqrt(output_sizes[i - 1])
                        torch.nn.init.trunc_normal_(layer_ensembled_weights, std=stddev)

                    if b_init is not None:
                        b_init(layer_ensembled_biases)
                    elif nonzero_bias and i == 1:
                        torch.nn.init.trunc_normal_(layer_ensembled_biases, std=1)
                    else:
                        torch.nn.init.zeros_(layer_ensembled_biases)

                    self.ensembled_weights.append(layer_ensembled_weights)
                    self.ensembled_biases.append(layer_ensembled_biases)

            def forward(
                self, x: torch.Tensor, indices: List[base.DataIndex]
            ) -> base.Output:

                for i, (layer_weights, layer_biases) in enumerate(
                    zip(
                        self.ensembled_weights,
                        self.ensembled_biases,
                    )
                ):

                    weight = layer_weights[indices]  # Select the ensemble weights for this index
                    bias = layer_biases[indices]      # Select the ensemble biases for this index

                    x = self.functional_linear(x, weight, bias)

                    if i < len(self.ensembled_weights) - 1:
                        x = torch.relu(x)

                return x

        indexer_fn = indexers.EnsembleIndexer(num_ensemble)

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            return model(inputs, index)

        def init_fn(seed: int) -> nn.Module:

            torch.manual_seed(seed)
            model = EnsembleMlp(num_ensemble, output_sizes, nonzero_bias, w_init, b_init, use_double_precision)

            return model

        super().__init__(apply_fn, init_fn, indexer_fn)


class MlpEnsembleEnnWithAdditivePrior(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensemble: int,
        prior_scale: float,
        nonzero_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        use_double_precision: bool = True,
    ):

        enn = MlpEnsembleEnn(
            output_sizes, num_ensemble, nonzero_bias, w_init, b_init, use_double_precision
        )

        prior_enn = MlpEnsembleEnn(
            output_sizes, num_ensemble, nonzero_bias, w_init, b_init, use_double_precision
        )

        def apply_fn(
            model: ModelWithPrior, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            output = model(inputs, index)
            return output

        def init_fn(seed: int) -> nn.Module:

            seed_train, seed_prior = utils.split_seed(seed, 2)
            model = enn.init(seed_train)
            prior_model = prior_enn.init(seed_prior)

            result = ModelWithPrior(model, prior_model, prior_scale)

            return result

        super().__init__(apply_fn, init_fn, enn.indexer)


# def make_mlp_ensemble_prior_fns(
#     output_sizes: Sequence[int],
#     dummy_input: torch.Tensor,
#     num_ensemble: int,
#     seed: int = 0,
#     w_init: Optional[Callable] = None,
#     b_init: Optional[Callable] = None,
# ) -> Sequence[Callable[[torch.Tensor], torch.Tensor]]:
#     """Factory method for creating ensemble of prior functions."""

#     def create_and_init_mlp(seed_val):
#         torch.manual_seed(seed_val)
#         layers = [nn.Flatten()]
#         sizes = list(output_sizes)

#         # Infer input size from dummy input
#         if isinstance(dummy_input, torch.Tensor):
#             flat_input = dummy_input.flatten(1)
#             input_size = flat_input.shape[-1]
#         else:
#             flat_input = torch.tensor(dummy_input).flatten(1)
#             input_size = flat_input.shape[-1]

#         for i, size in enumerate(sizes):
#             if i == 0:
#                 layers.append(nn.Linear(input_size, size))
#             else:
#                 layers.append(nn.Linear(sizes[i - 1], size))
#             if i < len(sizes) - 1:
#                 layers.append(nn.ReLU())

#         mlp = nn.Sequential(*layers)

#         # Apply custom initialization if provided
#         if w_init is not None or b_init is not None:
#             for module in mlp.modules():
#                 if isinstance(module, nn.Linear):
#                     if w_init is not None:
#                         w_init(module.weight)
#                     if b_init is not None and module.bias is not None:
#                         b_init(module.bias)

#         mlp.eval()  # Set to eval mode for prior
#         return mlp

#     prior_fns = []
#     for i in range(num_ensemble):
#         mlp = create_and_init_mlp(seed + i)
#         # Capture mlp in closure
#         prior_fns.append(lambda x, m=mlp: m(x))

#     return prior_fns


# def wrap_sequence_as_prior(
#     prior_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
# ) -> Callable[[torch.Tensor, base.Index], torch.Tensor]:
#     def prior_fn(x: torch.Tensor, z: base.Index) -> torch.Tensor:
#         if isinstance(z, torch.Tensor):
#             z = z.item()
#         return prior_fns[z](x)

#     return prior_fn


# def make_random_gp_ensemble_prior_fns(
#     input_dim: int,
#     output_dim: int,
#     num_feat: int,
#     gamma: priors.GpGamma,
#     num_ensemble: int,
#     seed: int = 0,
# ) -> Sequence[Callable[[torch.Tensor], torch.Tensor]]:
#     """Factory method for creating an ensemble of random GPs."""
#     prior_fns = []
#     for i in range(num_ensemble):
#         prior_fns.append(
#             priors.make_random_feat_gp(
#                 input_dim, output_dim, num_feat, seed + i, gamma, scale=1.0
#             )
#         )
#     return prior_fns


# class MLPEnsembleMatchedPrior(base.EpistemicNetwork):
#     """Ensemble of MLPs with matched prior functions."""

#     def __init__(
#         self,
#         output_sizes: Sequence[int],
#         dummy_input: torch.Tensor,
#         num_ensemble: int,
#         prior_scale: float = 1.0,
#         seed: int = 0,
#         w_init: Optional[Callable] = None,
#         b_init: Optional[Callable] = None,
#     ):
#         """Ensemble of MLPs with matched prior functions."""
#         mlp_priors = make_mlp_ensemble_prior_fns(
#             output_sizes, dummy_input, num_ensemble, seed, w_init, b_init
#         )
#         enn = priors.EnnWithAdditivePrior(
#             enn=MLPEnsembleEnn(
#                 output_sizes, num_ensemble, w_init=w_init, b_init=b_init
#             ),
#             prior_fn=wrap_sequence_as_prior(mlp_priors),
#             prior_scale=prior_scale,
#         )
#         super().__init__(enn.apply, enn.init, enn.indexer)


# class MLPEnsembleGpPrior(base.EpistemicNetwork):
#     """Ensemble of MLPs with GP random kitchen sink prior functions."""

#     def __init__(
#         self,
#         output_sizes: Sequence[int],
#         input_dim: int,
#         num_ensemble: int,
#         num_feat: int,
#         gamma: priors.GpGamma = 1.0,
#         prior_scale: float = 1,
#         seed: int = 0,
#     ):
#         """Ensemble of MLPs with GP random kitchen sink prior functions."""
#         gp_priors = make_random_gp_ensemble_prior_fns(
#             input_dim, output_sizes[-1], num_feat, gamma, num_ensemble, seed
#         )
#         enn = priors.EnnWithAdditivePrior(
#             enn=MLPEnsembleEnn(output_sizes, num_ensemble),
#             prior_fn=wrap_sequence_as_prior(gp_priors),
#             prior_scale=prior_scale,
#         )
#         super().__init__(enn.apply, enn.init, enn.indexer)


# class MLPEnsembleArbitraryPrior(base.EpistemicNetwork):
#     """Ensemble of MLPs with arbitrary prior functions."""

#     def __init__(
#         self,
#         prior_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
#         output_sizes: Sequence[int],
#         num_ensemble: int,
#         prior_scale: float = 1.0,
#         w_init: Optional[Callable] = None,
#         b_init: Optional[Callable] = None,
#     ):
#         """Ensemble of MLPs with arbitrary prior functions."""
#         enn = priors.EnnWithAdditivePrior(
#             enn=MLPEnsembleEnn(
#                 output_sizes, num_ensemble, w_init=w_init, b_init=b_init
#             ),
#             prior_fn=wrap_sequence_as_prior(prior_fns),
#             prior_scale=prior_scale,
#         )
#         super().__init__(enn.apply, enn.init, enn.indexer)
