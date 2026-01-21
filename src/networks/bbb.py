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
from typing import List, Optional, Sequence

from src import base
from src.networks import indexers
from src.networks.functional import BatchedFunctionalLinear
import torch
import torch.nn as nn


class MlpBbbEnn(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        sigma_0: float = 1.0,
        sigma_init: Optional[callable] = None,
        mu_init: Optional[callable] = None,
        scale_down_weights=True,
        use_double_precision: bool = True,
    ):

        mu_init = mu_init if mu_init is not None else torch.nn.init.zeros_

        class BbbMlp(nn.Module):
            def __init__(self, output_sizes, sigma_init, mu_init, use_double_precision):
                super().__init__()

                self.functional_linear = BatchedFunctionalLinear()

                self.weight_sigmas = nn.ParameterList()
                self.bias_sigmas = nn.ParameterList()

                self.weight_mus = nn.ParameterList()
                self.bias_mus = nn.ParameterList()

                for i in range(1, len(output_sizes)):
                    weight_sigma = nn.Parameter(
                        torch.Tensor(output_sizes[i], output_sizes[i - 1]).to(torch.float64 if use_double_precision else torch.float32)
                    )
                    bias_sigma = nn.Parameter(torch.Tensor(output_sizes[i]).to(torch.float64 if use_double_precision else torch.float32))
                    weight_mu = nn.Parameter(
                        torch.Tensor(output_sizes[i], output_sizes[i - 1]).to(torch.float64 if use_double_precision else torch.float32)
                    )
                    bias_mu = nn.Parameter(torch.Tensor(output_sizes[i]).to(torch.float64 if use_double_precision else torch.float32))

                    if sigma_init is not None:
                        sigma_init(weight_sigma)
                        sigma_init(bias_sigma)
                    else:
                        stddev = 1.0 / math.sqrt(output_sizes[i] * output_sizes[i - 1])
                        torch.nn.init.trunc_normal_(weight_sigma, std=stddev)
                        torch.nn.init.trunc_normal_(bias_sigma, std=stddev)

                    if mu_init is not None:
                        mu_init(bias_mu)
                        mu_init(weight_mu)
                    else:
                        torch.nn.init.zeros_(weight_mu)
                        torch.nn.init.zeros_(bias_mu)

                    self.weight_sigmas.append(weight_sigma)
                    self.bias_sigmas.append(bias_sigma)
                    self.weight_mus.append(weight_mu)
                    self.bias_mus.append(bias_mu)

            def forward(
                self, x: torch.Tensor, indices: List[List[base.DataIndex]]
            ) -> base.Output:

                for i, (weight_mu, weight_sigma, bias_mu, bias_sigma) in enumerate(
                    zip(
                        self.weight_mus,
                        self.weight_sigmas,
                        self.bias_mus,
                        self.bias_sigmas,
                    )
                ):

                    batched_layer_index = [a[i] for a in indices]

                    weight_index = torch.stack([a[0] for a in batched_layer_index])
                    bias_index = torch.stack([a[1] for a in batched_layer_index])

                    weight = weight_mu + weight_index * torch.nn.functional.softplus(
                        weight_sigma
                    )
                    bias = bias_mu + bias_index * torch.nn.functional.softplus(
                        bias_sigma
                    )

                    if scale_down_weights:
                        weight = weight / math.sqrt(weight.shape[-1])

                    x = self.functional_linear(x, weight, bias)

                    if i < len(self.weight_mus) - 1:
                        x = torch.relu(x)

                return x

            def get_params_tuple(self):
                return (
                    self.weight_sigmas,
                    self.bias_sigmas,
                    self.weight_mus,
                    self.bias_mus,
                )

        indexer = indexers.SetScaledGaussianIndexer(
            index_dims_list=(
                [
                    (output_sizes[i], output_sizes[i - 1])
                    for i in range(1, len(output_sizes))
                ]
                + [(output_sizes[i],) for i in range(1, len(output_sizes))]
            ),
            scale=sigma_0,
        )

        def indexer_fn(key, device) -> base.EpistemicIndexer:
            index = indexer(key, device)

            weight_index = index[: len(output_sizes) - 1]
            bias_index = index[len(output_sizes) - 1 :]

            total_index = [*zip(weight_index, bias_index)]
            return total_index

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            return model(inputs, index)

        def init_fn(seed: int) -> nn.Module:

            torch.manual_seed(seed)
            model = BbbMlp(output_sizes, sigma_init, mu_init, use_double_precision)

            return model

        super().__init__(apply_fn, init_fn, indexer_fn)
