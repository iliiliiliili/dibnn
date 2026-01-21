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
"""Implementing Dropout as an ENN in PyTorch."""
import math
from typing import Optional, Sequence

from src import base
from src.networks import indexers
import torch
import torch.nn as nn


class MLPDropoutENN(base.EpistemicNetwork):
    """MLP with dropout as an ENN."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        dropout_rate: float = 0.05,
        dropout_input: bool = True,
        w_init: Optional[callable] = None,
        b_init: Optional[callable] = None,
        use_double_precision: bool = False,
    ):
        """MLP with dropout as an ENN."""

        class DropoutMLP(nn.Module):
            def __init__(
                self, output_sizes, dropout_rate, dropout_input, w_init, b_init
            ):
                super().__init__()
                self.dropout_rate = dropout_rate
                self.dropout_input = dropout_input
                self.flatten = nn.Flatten()

                # Build layers
                layers = []
                sizes = list(output_sizes)
                for i in range(1, len(sizes)):
                    layers.append(nn.Linear(sizes[i - 1], sizes[i], dtype=torch.float64 if use_double_precision else torch.float32))

                    # Add dropout between layers (not after last layer)
                    if i < len(sizes) - 1:
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(p=dropout_rate))

                self.layers = nn.ModuleList(layers)

                for module in self.layers:
                    if isinstance(module, (nn.Linear)):
                        if w_init is not None:
                            w_init(module.weight)
                        else:
                            stddev = 1.0 / math.sqrt(sizes[i - 1])
                            torch.nn.init.trunc_normal_(module.weight, std=stddev)
                        
                        if module.bias is not None:
                            if b_init is not None:
                                b_init(module.bias)
                            else:
                                torch.nn.init.zeros_(module.bias)

            def forward(self, inputs: torch.Tensor, index: base.Index) -> base.Output:
                torch.manual_seed(index)

                x = self.flatten(inputs)

                # Apply input dropout if specified
                if self.dropout_input:
                    x = nn.functional.dropout(x, p=self.dropout_rate, training=True)

                # Forward through layers
                for layer in self.layers:
                    x = layer(x)

                return x

        indexer_fn = indexers.PrngIndexer()

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            # Always use training mode for dropout
            model.train()

            if isinstance(index, int):
                result = model(inputs, index)
            else:
                result = torch.stack(
                    [model(inputs, i)[:, 0] for i in index]
                )

            return result
        def init_fn(seed: int) -> nn.Module:

            torch.manual_seed(seed)
            model = DropoutMLP(
                output_sizes, dropout_rate, dropout_input, w_init, b_init
            )

            return model

        super().__init__(apply_fn, init_fn, indexer_fn)
