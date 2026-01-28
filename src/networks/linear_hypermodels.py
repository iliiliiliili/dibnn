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

import numpy as np

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

                    # weight = weight_hyper_layer(indices).reshape(
                    #     -1, output_sizes[i+1], output_sizes[i]
                    # )
                    # bias = bias_hyper_layer(indices).reshape(
                    #     -1, output_sizes[i+1]
                    # )

                    weight = weight_hyper_layer(indices).reshape(
                        indices.shape[0], output_sizes[i], output_sizes[i+1]
                    )
                    bias = bias_hyper_layer(indices).reshape(
                        indices.shape[0], output_sizes[i+1]
                    )

                    if scale_down_weights:
                        weight = weight / math.sqrt(weight.shape[-1])

                    out = x @ weight + bias.unsqueeze(1)
                    
                    if i < len(self.weight_hyper_layers) - 1:
                        out = torch.relu(out)
                    
                    x = out

                return x

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


class MlpLinearHypermodelEnnWithAdditivePriorIndependentLayers(base.EpistemicNetwork):

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

        def apply_fn(
            model: ModelWithPrior, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            output = model(inputs, index)
            return output

        def init_fn(seed: int) -> nn.Module:

            seed_train, seed_prior = utils.split_seed(seed, 2)
            
            model = enn.init(seed_train)
            torch.manual_seed(seed_prior)
            prior_model = PriorMLPIndependentLayers(
                output_sizes, index_dim
            )

            result = ModelWithPrior(model, prior_model, prior_scale)

            return result

        super().__init__(apply_fn, init_fn, enn.indexer)


class PriorHyperLinear(nn.Module):
    """Linear hypermodel."""

    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        index_dim_per_layer: int,
        weight_scaling: float = 1.0,
        bias_scaling: float = 1.0,
        fixed_bias_val: float = 0.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.index_dim_per_layer = index_dim_per_layer
        self.weight_scaling = weight_scaling
        self.bias_scaling = bias_scaling
        self.fixed_bias_val = fixed_bias_val
        self.hidden_size = hidden_size
        
        self.w = torch.nn.Parameter(torch.randn(self.output_size, hidden_size, self.index_dim_per_layer))
        self.b = torch.nn.Parameter(torch.randn(self.output_size, self.index_dim_per_layer))
        self.functional_linear = BatchedFunctionalLinear()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        
        w = self.w / torch.norm(self.w, dim=-1, keepdim=True)
        b = self.b / torch.norm(self.b, dim=-1, keepdim=True)

        w = w * torch.sqrt(torch.tensor(self.weight_scaling / self.hidden_size, device=w.device))
        b = b * torch.sqrt(torch.tensor(self.bias_scaling, device=b.device)) + self.fixed_bias_val

        weights = torch.einsum("ohi,si->soh", w, z)
        bias = torch.einsum("oi,si->so", b, z)

        result = self.functional_linear(x, weights, bias)

        return result



class PriorMLPIndependentLayers(torch.nn.Module):
    """Prior MLP with each layer generated by an independent index."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        index_dim: int,
        weight_scaling: float = 1.0,
        bias_scaling: float = 1.0,
        fixed_bias_val: float = 0.0,
    ):
        super().__init__()
        self.output_sizes = output_sizes
        self.num_layers = len(self.output_sizes) - 1
        self.index_dim = index_dim
        self.weight_scaling = weight_scaling
        self.bias_scaling = bias_scaling
        self.fixed_bias_val = fixed_bias_val

        if self.index_dim < self.num_layers:
            # Assigning all index dimensions to all layers
            self.layers_indices = [np.arange(self.index_dim)] * self.num_layers

        else:
            # Spliting index dimension into num_layers chunks
            self.layers_indices = np.array_split(
                np.arange(self.index_dim), self.num_layers
            )

        # Defining layers of the prior MLP and associating each layer with a set of
        # indices
        self.layers = nn.ModuleList()
        for i in range(1, len(self.output_sizes)):
            index_dim_per_layer = len(self.layers_indices[i - 1])
            layer = PriorHyperLinear(
                self.output_sizes[i],
                self.output_sizes[i - 1],
                index_dim_per_layer,
                self.weight_scaling,
                self.bias_scaling,
                self.fixed_bias_val,
            )
            self.layers.append(layer)

    def __call__(self, x: torch.Tensor, z: base.Index) -> torch.Tensor:
        if self.index_dim < self.num_layers:
            # Assigning all index dimensions to all layers
            index_layers = [z] * self.num_layers
        else:
            # Spliting index dimension into num_layers chunks
            index_layers = torch.tensor_split(z, self.num_layers, dim=-1)

        out = x
        for i, layer in enumerate(self.layers):
            index_layer = index_layers[i]
            out = layer(out, index_layer)
            if i < self.num_layers - 1:
                out = torch.nn.functional.relu(out)
        return out
