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
"""Utility functions - PyTorch version."""
from typing import Callable, Optional

from absl import flags
from pandas import DataFrame
from src import base
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.seeds import split_seed

FLAGS = flags.FLAGS


float_fields = [
    "noise_scale",
    "prior_scale",
    "dropout_rate",
    "regularization_scale",
    "sigma_0",
    "learning_rate",
    "mean_error",
    "std_error",
]
int_fields = [
    "num_ensemble",
    "num_layers",
    "hidden_size",
    "index_dim",
    "num_index_samples",
    "indexer",
    # "num_batches",
]
int_list_fields = [
    "num_ensembles",
]

rename = {"num_ensembles": "num_ensemble"}


def epistemic_network_from_module(
    model: nn.Module,
    indexer: base.EpistemicIndexer,
) -> base.EpistemicNetwork:
    """Convert an ENN module to epistemic network with paired index."""

    def apply_fn(
        model: nn.Module, inputs: torch.Tensor, index: base.Index
    ) -> base.Output:
        return model(inputs, index)

    def init_fn() -> nn.Module:
        return model

    return base.EpistemicNetwork(apply_fn, init_fn, indexer)


def wrap_model_as_enn(model: nn.Module) -> base.EpistemicNetwork:
    """Wraps a simple model y = f(x) as an ENN."""
    return base.EpistemicNetwork(
        apply=lambda m, x, z: m(x),
        init=lambda: model,
        indexer=lambda key: key,
    )


def parse_net_output(net_out: base.Output) -> torch.Tensor:
    """Convert potential dict of network outputs to scalar prediction value."""
    if isinstance(net_out, base.OutputWithPrior):
        return net_out.preds
    else:
        return net_out


def make_batch_indexer(
    indexer: base.EpistemicIndexer, batch_size: int
) -> base.EpistemicIndexer:
    """Batches an EpistemicIndexer to produce batch_size index samples."""

    def batch_indexer(key: base.RngKey, device) -> base.Index:
        keys = split_seed(key, batch_size)
        indices = [indexer(k, device) for k in keys]
        if isinstance(indices[0], torch.Tensor):
            return torch.stack(indices)
        else:
            return indices

    return batch_indexer


def _clean_batch_data(data: base.Batch) -> base.Batch:
    """Checks some of the common shape/index issues for dummy data."""
    # Make sure that the data has a separate batch dimension
    if isinstance(data.y, torch.Tensor):
        if data.y.ndim == 1:
            data = data._replace(y=data.y.unsqueeze(-1))
    elif isinstance(data.y, np.ndarray):
        if data.y.ndim == 1:
            data = data._replace(y=data.y[:, None])

    # Data index to identify each instance
    if data.data_index is None:
        n_data = len(data.y)
        data = data._replace(data_index=np.arange(n_data)[:, None])

    # Weights to say how much each data point is worth
    if data.weights is None:
        n_data = len(data.y)
        if isinstance(data.y, torch.Tensor):
            data = data._replace(weights=torch.ones(n_data, 1))
        else:
            data = data._replace(weights=np.ones((n_data, 1)))
    return data


def make_batch_iterator(
    data: base.Batch,
    batch_size: Optional[int] = None,
    seed: int = 0,
    num_workers: int = 0,
    device: str = "cuda:0",
    minimum_batch_size: int = 100,
) -> base.BatchIterator:
    """Converts toy-like training data to batch_iterator for sgd training."""
    data = _clean_batch_data(data)
    n_data = len(data.y)
    if not batch_size:
        batch_size = max(n_data, minimum_batch_size)

    # Convert to torch tensors if needed
    x = (
        torch.tensor(data.x, device=device)
        if isinstance(data.x, np.ndarray)
        else data.x.to(device)
    )
    y = (
        torch.tensor(data.y, device=device)
        if isinstance(data.y, np.ndarray)
        else data.y.to(device)
    )
    data_index = (
        torch.tensor(data.data_index, device=device)
        if isinstance(data.data_index, np.ndarray)
        else data.data_index.to(device)
    )
    weights = (
        torch.tensor(data.weights, device=device)
        if isinstance(data.weights, np.ndarray)
        else data.weights.to(device)
    )

    # If number of samples is less than minimum_batch_size, repeat elements
    if n_data < minimum_batch_size:
        repeat_factor = (minimum_batch_size + n_data - 1) // n_data
        x = x.repeat(repeat_factor, *([1] * (x.ndim - 1)))
        y = y.repeat(repeat_factor, *([1] * (y.ndim - 1)))
        data_index = data_index.repeat(repeat_factor, *([1] * (data_index.ndim - 1)))
        weights = weights.repeat(repeat_factor, *([1] * (weights.ndim - 1)))
        n_data = len(x)

    dataset = TensorDataset(x, y, data_index, weights)

    # Create DataLoader with shuffling
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False
    )

    # Create infinite iterator
    def infinite_iterator():
        while True:
            for batch_data in dataloader:
                x_batch, y_batch, idx_batch, w_batch = batch_data
                yield base.Batch(
                    x=x_batch,
                    y=y_batch,
                    data_index=idx_batch,
                    weights=w_batch,
                    extra=data.extra,
                )

    return infinite_iterator()


def make_test_data(n_samples: int = 20) -> base.BatchIterator:
    """Generate a simple dataset suitable for classification or regression."""
    x, y = datasets.make_moons(n_samples, noise=0.1, random_state=0)
    return make_batch_iterator(base.Batch(x, y))


def read_results_file(file):
    with open(file, "r") as f:
        lines = f.readlines()

        agent_frames = {}

        for line in lines:
            id, kl, *params = line.replace("\n", "").split(" ")

            f = []
            for p in params:
                if "=" in p:
                    f.append(p)
                else:
                    f[-1] += " " + p
            raw_params = f

            params = []

            agent = None

            for p in raw_params:
                k, v = p.split("=")
                if k == "agent":
                    agent = v
                else:
                    params.append(p)

            id = int(id)
            kl = float(kl)

            if agent not in agent_frames:
                agent_frames[agent] = {"kl": []}

            agent_frames[agent]["kl"].append(kl)
            # agent_frames[agent]["kl"].append(min(2, kl))

            for p in params:
                k, v = p.split("=")

                if k in float_fields:
                    v = float(v)
                elif k in int_fields:
                    v = int(v)
                elif k in int_list_fields:
                    v = int(v.split("]")[0].split(" ")[-1])

                if k in rename:
                    k = rename[k]

                if k not in agent_frames[agent]:
                    agent_frames[agent][k] = []

                agent_frames[agent][k].append(v)

        for agent in agent_frames.keys():
            agent_frames[agent] = DataFrame(agent_frames[agent])

        return agent_frames
