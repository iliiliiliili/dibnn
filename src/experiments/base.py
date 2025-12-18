# python3
# pylint: disable=g-bad-file-header
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
"""Base classes for GP testbed - PyTorch version."""
import abc
from typing import Any, Dict, NamedTuple, Optional

import dataclasses
import typing_extensions
import numpy as np
import torch


# Data class for storing training data
class Data(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor


@dataclasses.dataclass
class PriorKnowledge:
    """Prior knowledge about the problem."""

    input_dim: int
    num_train: int
    num_classes: int = 1
    layers: Optional[int] = None
    noise_std: Optional[float] = None
    temperature: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ENNQuality:
    """Quality metrics for ENN sampler."""

    kl_estimate: float
    extra: Optional[Dict[str, Any]] = None


class EpistemicSampler(typing_extensions.Protocol):
    """Interface for drawing posterior samples from distribution.

    We are considering a model of data: y_i = f(x_i) + e_i.
    In this case the sampler should only model f(x), not aleatoric y.
    """

    def __call__(self, x: np.ndarray, seed: int = 0) -> np.ndarray:
        """Generate a random sample for epistemic f(x)."""


class TestbedAgent(typing_extensions.Protocol):
    """An interface for specifying a testbed agent."""

    def __call__(
        self, data: Data, prior: Optional[PriorKnowledge] = None
    ) -> EpistemicSampler:
        """Sets up a training procedure given ENN prior knowledge."""


class TestbedProblem(abc.ABC):
    """An interface for specifying a generative model of data."""

    @property
    @abc.abstractmethod
    def train_data(self) -> Data:
        """Access training data from the model for ENN training."""

    @abc.abstractmethod
    def evaluate_quality(self, enn_sampler: EpistemicSampler) -> ENNQuality:
        """Evaluate the quality of a posterior sampler."""

    @property
    @abc.abstractmethod
    def prior_knowledge(self) -> PriorKnowledge:
        """Information describing the problem instance."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Information describing the problem instance."""

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> "TestbedProblem":
        """Information describing the problem instance."""
