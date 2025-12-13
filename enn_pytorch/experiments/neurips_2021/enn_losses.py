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
"""Helpful losses for the ENN agent - PyTorch version."""

from typing import Callable, Optional

from enn_pytorch import base as enn_base
from enn_pytorch import losses
from enn_pytorch.experiments.neurips_2021 import base as testbed_base

import torch
import torch.nn as nn

from enn_pytorch.losses.utils import add_l2_weight_decay


EnnCtor = Callable[[testbed_base.PriorKnowledge], enn_base.EpistemicNetwork]
LossCtor = Callable[
    [testbed_base.PriorKnowledge, enn_base.EpistemicNetwork], enn_base.LossFn
]


def default_enn_loss(
    num_index_samples: int = 10,
    distribution: str = "none",
    seed: int = 0,
    weight_reg_scale: Optional[float] = None,
) -> LossCtor:
    """Constructs a default loss suitable for classification or regression."""

    def loss_ctor(
        prior: testbed_base.PriorKnowledge, enn: enn_base.EpistemicNetwork
    ) -> enn_base.LossFn:
        single_loss = losses.L2Loss()
        loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
        return loss_fn

    return loss_ctor


def gaussian_regression_loss(
    num_index_samples: int,
    noise_scale: float = 1,
    l2_weight_decay: float = 0,
    exclude_bias_l2: bool = True,
) -> LossCtor:
    """Constructs a loss for Gaussian regression."""

    def loss_ctor(
        prior: testbed_base.PriorKnowledge, enn: enn_base.EpistemicNetwork
    ) -> enn_base.LossFn:
        """Add a matching Gaussian noise to the target y."""
        single_loss = losses.L2Loss()
        loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
        return loss_fn

    return loss_ctor


def regularized_dropout_loss(
    num_index_samples: int = 10,
    dropout_rate: float = 0.05,
    scale: float = 1e-2,
    tau: float = 1.0,
) -> LossCtor:
    """Constructs a regularized loss for dropout-based uncertainty."""

    def loss_ctor(
        prior: testbed_base.PriorKnowledge, enn: enn_base.EpistemicNetwork
    ) -> enn_base.LossFn:
        single_loss = losses.L2Loss()
        reg = (scale ** 2) * (1 - dropout_rate) / (2.0 * prior.num_train * tau)
        loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
        loss_fn = add_l2_weight_decay(loss_fn, scale=reg)
        return loss_fn

    return loss_ctor
