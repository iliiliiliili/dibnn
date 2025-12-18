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

"""Base classes for Epistemic Neural Network design in PyTorch."""

import abc
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union

import dataclasses
import numpy as np
import torch
import torch.nn as nn
import typing_extensions

DataIndex = torch.Tensor  # Always integer
Index = int  # Epistemic index, paired with network
RngKey = Union[int, torch.Generator]  # Random key/generator


class OutputWithPrior(NamedTuple):
    """Output wrapper for networks with prior functions."""

    train: torch.Tensor
    prior: torch.Tensor = torch.zeros(1)
    extra: Dict[str, torch.Tensor] = {}

    @property
    def preds(self) -> torch.Tensor:
        if isinstance(self.train, torch.Tensor) and isinstance(
            self.prior, torch.Tensor
        ):
            return self.train + self.prior.detach()
        elif isinstance(self.train, torch.Tensor):
            return (
                self.train
                + torch.tensor(
                    self.prior, device=self.train.device, dtype=self.train.dtype
                ).detach()
            )
        else:
            return self.train + self.prior


Output = Union[torch.Tensor, OutputWithPrior]


class EpistemicModule(abc.ABC, nn.Module):
    """Epistemic neural network abstract base class as PyTorch module."""

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, index: Index) -> Output:
        """Forwards the epistemic network y = f(x,z)."""


class ApplyFn(typing_extensions.Protocol):
    """Applies the ENN at given parameters, inputs, index."""

    def __call__(self, model: nn.Module, inputs: torch.Tensor, index: Index) -> Output:
        """Applies the ENN at given model, inputs, index."""


class InitFn(typing_extensions.Protocol):
    """Initializes the ENN."""

    def __call__(self, seed: int) -> nn.Module:
        """Initializes and returns the ENN model."""


class EpistemicIndexer(typing_extensions.Protocol):
    """Generates indices for the ENN from random keys."""

    def __call__(self, key: RngKey) -> Index:
        """Samples a single index for the epistemic network."""


@dataclasses.dataclass
class EpistemicNetwork:
    """Convenient pairing of PyTorch model and index sampler."""

    apply: ApplyFn
    init: InitFn
    indexer: EpistemicIndexer


class Batch(NamedTuple):
    x: torch.Tensor  # Inputs
    y: torch.Tensor  # Targets
    data_index: Optional[DataIndex] = None  # Integer identifiers for data
    weights: Optional[torch.Tensor] = (
        None  # None should default to weights = torch.ones
    )
    extra: Dict[str, torch.Tensor] = {}  # You can put other optional stuff here


BatchIterator = Iterator[Batch]  # Equivalent to the dataset we loop through
LossMetrics = Dict[str, torch.Tensor]


class LossFn(typing_extensions.Protocol):
    """Calculates a loss based on one batch of data per rng_key."""

    def __call__(
        self, enn: EpistemicNetwork, model: nn.Module, batch: Batch, key: RngKey
    ) -> Tuple[torch.Tensor, LossMetrics]:
        """Computes a loss based on one batch of data and a random key."""
