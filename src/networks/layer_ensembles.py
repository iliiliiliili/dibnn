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


class MlpLayerEnsembleEnn(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: List[int],
        nonzero_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        use_double_precision: bool = False,
    ):

        class LayerEnsembleMlp(nn.Module):
            def __init__(self, num_ensembles, output_sizes, nonzero_bias, w_init, b_init, use_double_precision):
                super().__init__()

                self.functional_linear = BatchedFunctionalLinear()

                self.ensembled_weights = nn.ParameterList() # for each layer, size is (num_ensemble, out_size, in_size)
                self.ensembled_biases = nn.ParameterList() # for each layer, size is (num_ensemble, out_size)

                for i in range(1, len(output_sizes)):
                    layer_ensembled_weights = nn.Parameter(
                        torch.Tensor(num_ensembles[i - 1], output_sizes[i], output_sizes[i - 1]).to(torch.float64 if use_double_precision else torch.float32)
                    )
                    layer_ensembled_biases = nn.Parameter(
                        torch.Tensor(num_ensembles[i - 1], output_sizes[i]).to(torch.float64 if use_double_precision else torch.float32)
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
                self, x: torch.Tensor, indices: List[List[base.DataIndex]]
            ) -> base.Output:

                for i, (layer_weights, layer_biases, layer_indices) in enumerate(
                    zip(
                        self.ensembled_weights,
                        self.ensembled_biases,
                        indices.T
                    )
                ):

                    weight = layer_weights[layer_indices]  # Select the ensemble weights for this index
                    bias = layer_biases[layer_indices]      # Select the ensemble biases for this index

                    x = self.functional_linear(x, weight, bias)

                    if i < len(self.ensembled_weights) - 1:
                        x = torch.relu(x)

                return x

        indexer_fn = indexers.LayerEnsembleIndexer(num_ensembles)

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            return model(inputs, index)

        def init_fn(seed: int) -> nn.Module:

            torch.manual_seed(seed)
            model = LayerEnsembleMlp(num_ensembles, output_sizes, nonzero_bias, w_init, b_init, use_double_precision)

            return model

        super().__init__(apply_fn, init_fn, indexer_fn)


class MlpLayerEnsembleEnnWithAdditivePrior(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: List[int],
        prior_scale: float,
        nonzero_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        use_double_precision: bool = False,
    ):

        enn = MlpLayerEnsembleEnn(
            output_sizes, num_ensembles, nonzero_bias, w_init, b_init, use_double_precision
        )

        prior_enn = MlpLayerEnsembleEnn(
            output_sizes, num_ensembles, nonzero_bias, w_init, b_init, use_double_precision
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
