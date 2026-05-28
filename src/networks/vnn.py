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
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

from src import base
from src.networks import indexers
from src.networks.functional import BatchedFunctionalLinear
import torch
import torch.nn as nn

import torch.nn.init as init


def create_initializer(names):
    result = []

    for name in names:
        if name is None:
            result.append((None, None))
        elif name == "he_uniform":
            result.append(
                (
                    lambda t: init.kaiming_uniform_(t, a=0, mode="fan_in"),
                    lambda t: init.kaiming_uniform_(t, a=0, mode="fan_in"),
                )
            )
        elif name == "he_normal":
            result.append(
                (
                    lambda t: init.kaiming_normal_(t, a=0, mode="fan_in"),
                    lambda t: init.kaiming_normal_(t, a=0, mode="fan_in"),
                )
            )
        elif name == "glorot_normal":
            result.append((init.xavier_normal_, init.xavier_normal_))
        elif name == "glorot_uniform":
            result.append((init.xavier_uniform_, init.xavier_uniform_))
        elif name == "1":
            result.append((None, lambda t: init.constant_(t, 1.0)))
        elif name == "2":
            result.append((None, lambda t: init.constant_(t, 2.0)))
        else:
            raise ValueError(str(name) + " is an unknown initializer name")

    return result


class VariationalBase(nn.Module):

    GLOBAL_STD: float = 0
    LOG_STDS = False

    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        means: Any,
        stds: Any,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[Union[Callable, List[Callable]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
    ) -> None:

        super().__init__()

        self.means = means
        self.stds = stds
        self.batch_norm_module = batch_norm_module
        self.batch_norm_mode = batch_norm_mode
        self.batch_norm_size = batch_norm_size
        self.activation = activation
        self.activation_mode = activation_mode
        self.use_batch_norm = use_batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.global_std_mode = global_std_mode

    def __call__(self, x, index, global_std=0.2):

        end_activation = None
        end_batch_norm = None

        means = self.means
        stds = self.stds

        if self.use_batch_norm:

            batch_norm_targets = self.batch_norm_mode.split("+")

            for i, target in enumerate(batch_norm_targets):

                if target == "mean":
                    means = torch.nn.Sequential(
                        means,
                        self.batch_norm_module(
                            True,
                            True,
                            eps=self.batch_norm_eps,
                            decay_rate=self.batch_norm_momentum,
                        ),
                    )
                elif target == "std":
                    if stds is not None:
                        stds = torch.nn.Sequential(
                            stds,
                            self.batch_norm_module(
                                self.batch_norm_size,
                                eps=self.batch_norm_eps,
                                momentum=self.batch_norm_momentum,
                            ),
                        )
                elif target == "end":
                    self.end_batch_norm = (
                        self.batch_norm_module(
                            self.batch_norm_size,
                            eps=self.batch_norm_eps,
                            momentum=self.batch_norm_momentum,
                        ),
                    )
                else:
                    raise ValueError("Unknown batch norm target: " + target)

        if self.activation is not None:

            activation_targets = self.activation_mode.split("+")

            for i, target in enumerate(activation_targets):

                if len(activation_targets) == 1:
                    current_activation: Callable = self.activation  # type: ignore
                else:
                    current_activation: Callable = self.activation[i]  # type: ignore

                if target == "mean":
                    means = torch.nn.Sequential(means, current_activation)
                elif target == "std":
                    if stds is not None:
                        stds = torch.nn.Sequential(
                            stds,
                            current_activation,
                        )
                elif target == "end":
                    end_activation = current_activation
                elif target == "none":
                    pass
                else:
                    raise ValueError("Unknown activation target: " + target)

        mean_values = means(x)

        if stds:
            std_values = stds(x)
        else:
            std_values = 0

        if self.global_std_mode == "replace":
            std_values = global_std
        elif self.global_std_mode == "multiply":
            std_values = global_std * std_values

        if len(mean_values.shape) == 2:
            mean_values = mean_values.unsqueeze(0)
        if hasattr(std_values, "shape") and len(std_values.shape) == 2:
            std_values = std_values.unsqueeze(0)

        result = mean_values + std_values * index.unsqueeze(1)

        if end_batch_norm is not None:
            result = end_batch_norm(result)

        if end_activation is not None:
            result = end_activation(result)

        return result


class VariationalLinear(VariationalBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Union[Callable, List[Callable]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        bias=True,
        initializer: Tuple[
            Union[
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
            Union[
                Literal["1"],
                Literal["2"],
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
        ] = (None, None),
        **kwargs,
    ) -> None:

        super().__init__()

        if use_batch_norm:
            bias = False

        initializers_mean, initializers_std = create_initializer(initializer)

        means = nn.Linear(
            in_features,
            out_features,
            bias=bias,
            **kwargs,
        )

        initializers_mean[0](means.weight) if initializers_mean[0] is not None else None
        (
            initializers_mean[1](means.bias)
            if bias and initializers_mean[1] is not None
            else None
        )

        if global_std_mode == "replace":
            stds = None
        else:
            stds = nn.Linear(
                in_features,
                out_features,
                bias=bias,
                **kwargs,
            )

            (
                initializers_std[0](means.weight)
                if initializers_std[0] is not None
                else None
            )
            (
                initializers_std[1](means.bias)
                if bias and initializers_std[1] is not None
                else None
            )

        super().build(
            means,
            stds,
            None,
            out_features,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


class MLPVariationalENN(base.EpistemicNetwork):

    def __init__(
        self,
        output_sizes: Sequence[int],
        activation: Optional[Union[Callable, List[Callable]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        seed: int = 0,
        initializer: Tuple[
            Union[
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
            Union[
                Literal["1"],
                Literal["2"],
                Literal["he_uniform"],
                Literal["he_normal"],
                Literal["glorot_normal"],
                Literal["glorot_uniform"],
                None,
            ],
        ] = (None, None),
        use_double_precision: bool = False,
    ):

        class VnnMlp(nn.Module):
            def __init__(self):
                super().__init__()

                self.layers = nn.ModuleList()

                for i in range(1, len(output_sizes)):
                    layer = VariationalLinear(
                        output_sizes[i - 1],
                        output_sizes[i],
                        activation,
                        activation_mode,
                        use_batch_norm,
                        batch_norm_mode,
                        global_std_mode=global_std_mode,
                        initializer=initializer,
                    )

                    self.layers.append(layer)

            def forward(
                self, x: torch.Tensor, full_index: base.DataIndex
            ) -> base.Output:

                indices = []
                i = 0
                for output_size in output_sizes[1:]:
                    indices.append(full_index[:, i : i + output_size])
                    i += output_size

                for layer, index in zip(self.layers, indices):
                    x = layer(x, index)

                return x

        index_dim = sum(output_sizes)
        indexer = indexers.ScaledGaussianIndexer(
            index_dims=[index_dim],
            scale=math.sqrt(index_dim),
        )

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.Output:
            return model(inputs, index)

        def init_fn(seed: int) -> nn.Module:

            torch.manual_seed(seed)
            model = VnnMlp()

            return model

        super().__init__(apply_fn, init_fn, indexer)
