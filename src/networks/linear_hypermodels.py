# python3
# pylint: disable=g-bad-file-header
# Copyright Illia Oleksiienko
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
"""Implementing Dropout as an ENN in PyTorch."""
import math
from typing import Callable, List, Optional, Sequence

from src import base, utils
from src.networks import indexers
from src.networks.functional import BatchedFunctionalLinear
import torch
import torch.nn as nn

from src.networks.priors import ModelWithPrior


class MlpLinearHypermodelEnn(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        index_dim: int = 1,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        scale_down_weights=False,
        use_double_precision: bool = True,
    ):

        b_init = b_init if b_init is not None else torch.nn.init.zeros_

        class LinearHypermodelMlp(nn.Module):
            def __init__(self, output_sizes, index_dim, w_init, b_init, use_double_precision):
                super().__init__()

                self.functional_linear = BatchedFunctionalLinear()

                self.weight_hyper_layers = nn.ModuleList()
                self.bias_hyper_layers = nn.ModuleList()

                for i in range(1, len(output_sizes)):

                    weight_hyper_layer = nn.Linear(
                        index_dim, output_sizes[i] * output_sizes[i - 1], dtype=torch.float64 if use_double_precision else torch.float32
                    )
                    bias_hyper_layer = nn.Linear(
                        index_dim, output_sizes[i], dtype=torch.float64 if use_double_precision else torch.float32
                    )
                    self.weight_hyper_layers.append(weight_hyper_layer)
                    self.bias_hyper_layers.append(bias_hyper_layer)

                    if w_init is not None:
                        w_init(weight_hyper_layer.weight)
                        w_init(bias_hyper_layer.weight)
                    else:
                        stddev = 1.0 / math.sqrt(index_dim)
                        torch.nn.init.trunc_normal_(weight_hyper_layer.weight, std=stddev)
                        torch.nn.init.trunc_normal_(bias_hyper_layer.weight, std=stddev)
                    
                    if b_init is not None:
                        b_init(weight_hyper_layer.bias)
                        b_init(bias_hyper_layer.bias)
                    else:
                        torch.nn.init.zeros_(weight_hyper_layer.bias)
                        torch.nn.init.zeros_(bias_hyper_layer.bias)

            def forward(
                self, x: torch.Tensor, indices: base.DataIndex
            ) -> base.Output:

                for i, (weight_hyper_layer, bias_hyper_layer) in enumerate(
                    zip(
                        self.weight_hyper_layers,
                        self.bias_hyper_layers,
                    )
                ):

                    weight = weight_hyper_layer(indices).reshape(
                        -1, output_sizes[i], output_sizes[i - 1]
                    )
                    bias = bias_hyper_layer(indices).reshape(
                        -1, output_sizes[i]
                    )

                    if scale_down_weights:
                        weight = weight / math.sqrt(weight.shape[-1])

                    x = self.functional_linear(x, weight, bias)

                    if i < len(self.weight_hyper_layers) - 1:
                        x = torch.relu(x)

                return x

            def get_params_tuple(self):
                return (
                    self.weight_sigmas,
                    self.bias_sigmas,
                    self.weight_mus,
                    self.bias_mus,
                )

        indexer = indexers.ScaledGaussianIndexer(
            index_dims=[index_dim],
            scale=1.0,
        )

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            return model(inputs, index)

        def init_fn(seed: int) -> nn.Module:

            torch.manual_seed(seed)
            model = LinearHypermodelMlp(output_sizes, index_dim, w_init, b_init, use_double_precision)

            return model

        super().__init__(apply_fn, init_fn, indexer)


class MlpLinearHypermodelEnnWithAdditivePrior(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        prior_scale: float,
        index_dim: int = 1,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        scale_down_weights=False,
        use_double_precision: bool = True,
    ):

        enn = MlpLinearHypermodelEnn(
            output_sizes, index_dim, w_init, b_init, scale_down_weights, use_double_precision
        )

        prior_enn = MlpLinearHypermodelEnn(
            output_sizes, index_dim, w_init, b_init, scale_down_weights, use_double_precision
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