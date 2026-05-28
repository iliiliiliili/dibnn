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
"""Loading a GP regression instance for the testbed - PyTorch implementation."""

import os
from typing import Tuple, Callable
import dataclasses
import torch
import numpy as np

from src.experiments import base as testbed_base
from src.experiments import testbed
from src.ntk import Dense, Relu, Serial
from enn.experiments.neurips_2021.load import (
    make_benchmark_kernel as jax_make_benchmark_kernel,
)


@dataclasses.dataclass
class MLPKernelCtor:
    """Generates a GP kernel corresponding to an infinitely-wide MLP."""

    num_hidden_layers: int
    activation: Relu

    def __post_init__(self):
        assert self.num_hidden_layers >= 1, "Must have at least one hidden layer."

    def __call__(self, input_dim: int = 1):
        """Generates a kernel for a given input dimension."""
        limit_width = 50  # Implementation detail of neural_testbed, unused.
        layers = [Dense(limit_width, W_std=1, b_std=1 / np.sqrt(input_dim))]
        for _ in range(self.num_hidden_layers - 1):
            layers.append(self.activation())
            layers.append(Dense(limit_width, W_std=1, b_std=0))
        layers.append(self.activation())
        layers.append(Dense(1, W_std=1, b_std=0))
        kernel = Serial(layers)
        return kernel


def make_benchmark_kernel(input_dim: int = 1):
    """Creates the benchmark kernel used in the testbed = 2-layer ReLU."""
    kernel_ctor = MLPKernelCtor(num_hidden_layers=2, activation=Relu)
    return kernel_ctor(input_dim)


def gaussian_data(
    seed: int, num_train: int, input_dim: int, num_test: int, use_double_precision: bool = True, val_data_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Gaussian training and test data.

    Args:
        seed: Random seed
        num_train: Number of training samples
        input_dim: Input dimension
        num_test: Number of test samples
        val_data_ratio: Ratio of validation data to training data

    Returns:
        Tuple of (x_train, x_test, x_val) tensors
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Generate training data
    x_train = torch.randn(num_train, input_dim, generator=generator, dtype=torch.float64 if use_double_precision else torch.float32)

    # Generate test data
    x_test = torch.randn(num_test, input_dim, generator=generator, dtype=torch.float64 if use_double_precision else torch.float32)

    # Generate validation data
    x_val = torch.randn(max(1, int(num_train * val_data_ratio)), input_dim, generator=generator, dtype=torch.float64 if use_double_precision else torch.float32)

    return x_train, x_test, x_val


@dataclasses.dataclass
class RegressionTestbedConfig:
    """Configuration options for regression testbed instance."""

    num_train: int
    input_dim: int
    seed: int
    noise_std: float
    tau: int = 1
    num_test_cache: int = 1000
    target_test_seeds: int = 1000  # num_test_seeds = target_test_seeds / tau
    num_enn_samples: int = 100
    kernel_ctor: Callable[[int], object] = lambda input_dim: make_benchmark_kernel(
        input_dim
    )
    num_layers: int = 1  # Output to prior knowledge


def regression_load_from_config(
    config: RegressionTestbedConfig,
    use_double_precision: bool = True,
) -> testbed.TestbedGPRegression:
    """Loads regression problem from config.

    Args:
        config: Configuration for the regression testbed

    Returns:
        TestbedGPRegression instance
    """
    x_train, x_test, x_val = gaussian_data(
        seed=config.seed,
        num_train=config.num_train,
        input_dim=config.input_dim,
        num_test=config.num_test_cache,
        use_double_precision=use_double_precision,
    )
    data_sampler = testbed.GPRegression(
        kernel_fn=config.kernel_ctor(config.input_dim),
        x_train=x_train,
        x_test=x_test,
        x_val=x_val,
        tau=config.tau,
        noise_std=config.noise_std,
        seed=config.seed,
    )
    prior_knowledge = testbed_base.PriorKnowledge(
        input_dim=config.input_dim,
        num_train=config.num_train,
        num_classes=1,
        layers=config.num_layers,
        noise_std=config.noise_std,
    )
    assert config.tau == 1, "Only works for tau=1"
    return testbed.TestbedGPRegression(
        data_sampler, prior_knowledge, num_enn_samples=config.num_enn_samples
    )


def regression_load(
    input_dim: int,
    data_ratio: float,
    seed: int,
    noise_std: float,
    dataset_folder: str = "datasets",
    use_double_precision: bool = True,
) -> testbed.TestbedGPRegression:
    """Load GP regression from sweep hyperparameters.

    Args:
        input_dim: Input dimension
        data_ratio: Ratio of training data to input dimension
        seed: Random seed
        noise_std: Standard deviation of observation noise

    Returns:
        TestbedGPRegression instance
    """
    num_train = int(data_ratio * input_dim)
    config = RegressionTestbedConfig(num_train, input_dim, seed, noise_std)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)

    filepath = os.path.join(
        dataset_folder,
        f"regression_id{input_dim}dr{data_ratio}ns{noise_std}seed{seed}{'_double' if use_double_precision else ''}.pt",
    )

    if os.path.exists(filepath):
        tb = testbed.TestbedGPRegression.load(filepath)
        return tb
    else:
        tb = regression_load_from_config(config, use_double_precision)
        tb.save(filepath)
        return tb
