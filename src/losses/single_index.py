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

"""Collection of simple losses applied to one single index - PyTorch version."""
from typing import Callable, Tuple

import dataclasses
from src import base
from src import data_noise
from src import utils
import torch
import torch.nn as nn
import typing_extensions


class SingleIndexLossFn(typing_extensions.Protocol):
    """Calculates a loss based on one batch of data per index.

    You can use average_single_index_loss to make a LossFn out of the
    SingleIndexLossFn.
    """

    def __call__(
        self,
        apply: base.ApplyFn,
        model: nn.Module,
        batch: base.Batch,
        index: base.Index,
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        """Computes a loss based on one batch of data and one index."""


def average_single_index_loss(
    single_loss: SingleIndexLossFn, num_index_samples: int = 1
) -> base.LossFn:
    """Average a single index loss over multiple index samples.

    Args:
        single_loss: loss function applied per epistemic index.
        num_index_samples: number of index samples to average.

    Returns:
        LossFn that comprises the mean of both the loss and the metrics.
    """

    def loss_fn(
        enn: base.EpistemicNetwork,
        model: nn.Module,
        batch: base.Batch,
        key: base.RngKey,
        device: str,
    ) -> torch.Tensor:
        batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
        indices = batched_indexer(key, device)

        losses = []
        metrics_list = []

        # Process each index
        if isinstance(indices, torch.Tensor):
            for i in range(indices.shape[0]):
                idx = indices[i] if indices.ndim > 0 else indices
                loss, metrics = single_loss(enn.apply, model, batch, idx)
                losses.append(loss)
                metrics_list.append(metrics)
        else:
            for idx in indices:
                loss, metrics = single_loss(enn.apply, model, batch, idx)
                losses.append(loss)
                metrics_list.append(metrics)

        # Average losses
        mean_loss = (
            torch.stack(losses).mean()
            if isinstance(losses[0], torch.Tensor)
            else sum(losses) / len(losses)
        )

        # Average metrics
        mean_metrics = {}
        if metrics_list and metrics_list[0]:
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                if isinstance(values[0], torch.Tensor):
                    mean_metrics[key] = torch.stack(values).mean()
                else:
                    mean_metrics[key] = sum(values) / len(values)

        return mean_loss, mean_metrics

    return loss_fn


def batched_average_single_index_loss(
    single_loss: SingleIndexLossFn, num_index_samples: int = 1
) -> base.LossFn:
    """Average a single index loss over multiple index samples.

    Args:
        single_loss: loss function applied per epistemic index.
        num_index_samples: number of index samples to average.

    Returns:
        LossFn that comprises the mean of both the loss and the metrics.
    """

    def loss_fn(
        enn: base.EpistemicNetwork,
        model: nn.Module,
        batch: base.Batch,
        key: base.RngKey,
    ) -> torch.Tensor:
        batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
        loss, metrics = single_loss(enn.apply, model, batch, batched_indexer(key))
        return loss, metrics

    return loss_fn


def add_data_noise(
    single_loss: SingleIndexLossFn, noise_fn: data_noise.DataNoise
) -> SingleIndexLossFn:
    """Applies a DataNoise function to each batch of data."""

    def noisy_loss(
        apply: base.ApplyFn, model: nn.Module, batch: base.Batch, index: base.Index
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        noisy_batch = noise_fn(batch, index)
        return single_loss(apply, model, noisy_batch, index)

    return noisy_loss


@dataclasses.dataclass
class BatchedL2Loss(SingleIndexLossFn):
    """L2 regression applied to a single epistemic index."""

    def __call__(
        self,
        apply: base.ApplyFn,
        model: nn.Module,
        batch: base.Batch,
        index: base.Index,
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        """L2 regression applied to a single epistemic index."""
        net_out = utils.parse_net_output(apply(model, batch.x, index))

        # Convert to tensors if needed
        if isinstance(batch.y, torch.Tensor):
            y = batch.y
        else:
            y = torch.tensor(batch.y, dtype=torch.float32)

        if isinstance(net_out, torch.Tensor):
            net_tensor = net_out
        else:
            net_tensor = torch.tensor(net_out, dtype=torch.float32)

        sq_loss = torch.square(net_tensor - y)

        if batch.weights is None:
            batch_weights = torch.ones_like(y)
        else:
            batch_weights = (
                batch.weights
                if isinstance(batch.weights, torch.Tensor)
                else torch.tensor(batch.weights, dtype=torch.float32)
            )

        result = torch.mean(batch_weights * sq_loss)
        return result, {}


@dataclasses.dataclass
class L2Loss(SingleIndexLossFn):
    """L2 regression applied to a batched epistemic index."""

    def __call__(
        self,
        apply: base.ApplyFn,
        model: nn.Module,
        batch: base.Batch,
        index: base.Index,
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        """L2 regression applied to a single epistemic index."""
        net_out = utils.parse_net_output(apply(model, batch.x, index))

        sq_loss = torch.square(net_out - batch.y)

        if batch.weights is None:
            batch_weights = torch.ones_like(batch.y)
        else:
            batch_weights = (
                batch.weights
                if isinstance(batch.weights, torch.Tensor)
                else torch.tensor(batch.weights, dtype=torch.float32)
            )
        
        result = torch.mean(batch_weights * sq_loss)

        return result, {}


@dataclasses.dataclass
class XentLoss(SingleIndexLossFn):
    """Cross-entropy classification single index loss."""

    num_classes: int

    def __post_init__(self):
        assert self.num_classes >= 2, "num_classes must be at least 2"

    def __call__(
        self,
        apply: base.ApplyFn,
        model: nn.Module,
        batch: base.Batch,
        index: base.Index,
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        net_out = apply(model, batch.x, index)
        logits = utils.parse_net_output(net_out)

        # Convert to tensors if needed
        if isinstance(batch.y, torch.Tensor):
            labels = batch.y[:, 0].long()
        else:
            labels = torch.tensor(batch.y[:, 0], dtype=torch.long)

        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)

        # One-hot encoding
        labels_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()

        # Compute cross-entropy
        log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
        softmax_xent = -torch.sum(labels_one_hot * log_softmax, dim=1, keepdim=True)

        if batch.weights is None:
            batch_weights = torch.ones_like(softmax_xent)
        else:
            batch_weights = (
                batch.weights
                if isinstance(batch.weights, torch.Tensor)
                else torch.tensor(batch.weights, dtype=torch.float32)
            )

        return torch.mean(batch_weights * softmax_xent), {}


@dataclasses.dataclass
class AccuracyErrorLoss(SingleIndexLossFn):
    """Evaluates the accuracy error of a greedy logit predictor."""

    num_classes: int

    def __call__(
        self,
        apply: base.ApplyFn,
        model: nn.Module,
        batch: base.Batch,
        index: base.Index,
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        net_out = apply(model, batch.x, index)
        logits = utils.parse_net_output(net_out)

        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)

        preds = torch.argmax(logits, dim=1)

        if isinstance(batch.y, torch.Tensor):
            labels = batch.y[:, 0]
        else:
            labels = torch.tensor(batch.y[:, 0])

        correct = (preds == labels).float()
        accuracy = torch.mean(correct)
        return 1 - accuracy, {"accuracy": accuracy}


@dataclasses.dataclass
class NElboLoss(SingleIndexLossFn):
    """Standard VI loss (negative of evidence lower bound).

    Given latent variable u with model density q(u), prior density p_0(u)
    and likelihood function p(D|u) the evidence lower bound is defined as
        ELBO(q) = E[log(p(D|u))] - KL(q(u)||p_0(u))
    In other words, maximizing ELBO is equivalent to regularized log likelihood
    maximization where regularization is encouraging the learned latent
    distribution to be close to the latent prior as measured by KL.
    """

    log_likelihood_fn: Callable[[base.Output, base.Batch], float]
    model_prior_kl_fn: Callable[[base.Output, nn.Module, base.Index], float]
    num_index_samples: int

    def __call__(
        self,
        enn: base.EpistemicNetwork,
        model: nn.Module,
        batch: base.Batch,
        key: base.RngKey,
        device: str,
    ) -> Tuple[torch.Tensor, base.LossMetrics]:
        """This function returns a one-sample MC estimate of the ELBO."""

        batched_indexer = utils.make_batch_indexer(enn.indexer, self.num_index_samples)
        indices = batched_indexer(key, device)

        out = enn.apply(model, batch.x, indices)
        log_likelihood = self.log_likelihood_fn(out, batch)
        model_prior_kl = self.model_prior_kl_fn(out, model, indices)
        return model_prior_kl - log_likelihood, {}
