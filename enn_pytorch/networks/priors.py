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

"""Implementing some mechanisms for prior functions with ENNs in PyTorch."""
from typing import Callable, Iterable, Optional, Tuple, Union

from absl import logging

import dataclasses
from enn_pytorch import base
import torch
import torch.nn as nn
import numpy as np

PriorFn = Callable[[torch.Tensor, base.Index], torch.Tensor]


class EnnWithAdditivePrior(base.EpistemicNetwork):
    """Create an ENN with additive prior_fn applied to outputs."""

    def __init__(
        self, enn: base.EpistemicNetwork, prior_fn: PriorFn, prior_scale: float = 1.0
    ):
        """Create an ENN with additive prior_fn applied to outputs."""

        def apply_fn(
            model: nn.Module, inputs: torch.Tensor, index: base.Index
        ) -> base.OutputWithPrior:
            net_out = enn.apply(model, inputs, index)
            prior = prior_scale * prior_fn(inputs, index)
            if isinstance(net_out, base.OutputWithPrior):
                return net_out._replace(prior=net_out.prior + prior)
            else:
                return base.OutputWithPrior(train=net_out, prior=prior)

        super().__init__(
            apply=apply_fn, init=enn.init, indexer=enn.indexer,
        )


def convert_enn_to_prior_fn(
    enn: base.EpistemicNetwork, key: base.RngKey
) -> PriorFn:
    """Convert an ENN to a prior function."""
    if isinstance(key, int):
        index_key = key
        init_key = key + 1
    else:
        index_key = key
        init_key = key + 1
    
    index = enn.indexer(index_key)
    model = enn.init()
    model.eval()

    def prior_fn(x: torch.Tensor, z: base.Index) -> base.Output:
        with torch.no_grad():
            return enn.apply(model, x, z)

    return prior_fn


def make_null_prior(output_dim: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a null prior that returns zeros."""
    def null_prior(inputs: torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return torch.zeros([inputs.shape[0], output_dim], dtype=torch.float32, device=inputs.device)
        else:
            return np.zeros([inputs.shape[0], output_dim], dtype=np.float32)

    return null_prior


# Configure GP kernel via float=gamma or uniform between gamma_min, gamma_max.
GpGamma = Union[float, Tuple[float, float]]


def _parse_gamma(
    gamma: GpGamma, num_feat: int, key: base.RngKey
) -> Union[float, torch.Tensor]:
    """Parse gamma configuration for GP kernel."""
    if isinstance(gamma, float) or isinstance(gamma, int):
        return float(gamma)
    else:
        gamma_min, gamma_max = gamma
        if isinstance(key, int):
            torch.manual_seed(key)
            return gamma_min + (gamma_max - gamma_min) * torch.rand(1, num_feat, 1)
        else:
            return gamma_min + (gamma_max - gamma_min) * torch.rand(1, num_feat, 1, generator=key)


def make_random_feat_gp(
    input_dim: int,
    output_dim: int,
    num_feat: int,
    key: base.RngKey,
    gamma: GpGamma = 1.0,
    scale: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Generate a random features GP realization via random features.

    This is based on the "random kitchen sink" approximation from Rahimi, Recht.
    See blog post/paper: http://www.argmin.net/2017/12/05/kitchen-sinks/.

    Args:
        input_dim: dimension of input.
        output_dim: dimension of output.
        num_feat: number of random features used to approximate GP.
        key: random number key.
        gamma: gaussian kernel variance = gamma^2 (higher = more wiggle).
            If you pass a tuple we generate uniform between gamma_min, gamma_max.
        scale: scale of the output in each dimension.

    Returns:
        A callable gp_instance: inputs -> outputs.
    """
    if isinstance(key, int):
        torch.manual_seed(key)
        weights = torch.randn(num_feat, input_dim, output_dim)
        bias = 2 * np.pi * torch.rand(1, num_feat, output_dim)
        alpha = torch.randn(num_feat) / np.sqrt(num_feat)
        parsed_gamma = _parse_gamma(gamma, num_feat, key + 3)
    else:
        weights = torch.randn(num_feat, input_dim, output_dim, generator=key)
        bias = 2 * np.pi * torch.rand(1, num_feat, output_dim, generator=key)
        alpha = torch.randn(num_feat, generator=key) / np.sqrt(num_feat)
        parsed_gamma = _parse_gamma(gamma, num_feat, key)

    def gp_instance(inputs: torch.Tensor) -> torch.Tensor:
        """Assumes one batch dimension and flattens input to match that."""
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Flatten inputs keeping batch dimension
        flat_inputs = inputs.reshape(inputs.shape[0], -1)
        
        # Compute input embedding: [batch, input_dim] @ [num_feat, input_dim, output_dim]
        # Result: [batch, num_feat, output_dim]
        input_embedding = torch.einsum("bi,kio->bko", flat_inputs, weights)
        
        # Apply cosine with gamma scaling
        if isinstance(parsed_gamma, torch.Tensor):
            random_feats = torch.cos(parsed_gamma * input_embedding + bias)
        else:
            random_feats = torch.cos(parsed_gamma * input_embedding + bias)
        
        # Final output: [batch, output_dim]
        output = scale * torch.einsum("bko,k->bo", random_feats, alpha)
        return output

    return gp_instance


def get_random_mlp_with_index(
    x_sample: torch.Tensor,
    z_sample: torch.Tensor,
    rng: int,
    prior_output_sizes: Optional[Iterable[int]] = None,
    prior_weight_std: float = 3,
    prior_bias_std: float = 0.1,
) -> PriorFn:
    """Construct a prior func f(x, z) based on a random MLP.

    The returned function assumes the data input, x, to include a batch dimension
    but the index input, z, to not include a batch dimension.

    Args:
        x_sample: a sample data input.
        z_sample: a sample index input.
        rng: random seed.
        prior_output_sizes: output sizes for the MLP.
        prior_weight_std: unscaled std of the random weights.
        prior_bias_std: std of the random biases.

    Returns:
        a random function of two inputs x, and z.
    """
    torch.manual_seed(rng)
    
    if prior_output_sizes is None:
        prior_output_sizes = [10, 10, 1]

    # Determine input size
    if isinstance(x_sample, torch.Tensor):
        x_flat = x_sample.flatten(1)
    else:
        x_flat = torch.tensor(x_sample).flatten(1)
    
    if isinstance(z_sample, torch.Tensor):
        z_flat = z_sample.flatten()
    else:
        z_flat = torch.tensor(z_sample).flatten()
    
    input_size = x_flat.shape[-1] + z_flat.shape[-1]
    
    # Build MLP
    layers = []
    sizes = [input_size] + list(prior_output_sizes)
    for i in range(len(sizes) - 1):
        layer = nn.Linear(sizes[i], sizes[i + 1])
        # Custom initialization
        fan_in = sizes[i]
        std = prior_weight_std / np.sqrt(fan_in)
        nn.init.trunc_normal_(layer.weight, mean=0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(layer.bias, mean=0, std=prior_bias_std, a=-2*prior_bias_std, b=2*prior_bias_std)
        layers.append(layer)
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    
    mlp = nn.Sequential(*layers)
    mlp.eval()
    
    def prior_fn(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32)
            
            # Flatten and concatenate
            x_flat = x.reshape(x.shape[0], -1)
            z_flat = z.flatten()
            
            # Repeat z along batch dimension
            z_repeated = z_flat.unsqueeze(0).repeat(x.shape[0], 1)
            xz = torch.cat([x_flat, z_repeated], dim=1)
            
            return mlp(xz)
    
    return prior_fn


WARN = (
    "WARNING: prior parameters will be included as model parameters."
    "If possible, you should use EnnWithAdditivePrior instead."
)


@dataclasses.dataclass
class NetworkWithAdditivePrior(nn.Module):
    """Combines network and a prior using a specified function."""

    net: nn.Module
    prior_net: nn.Module
    prior_scale: float = 1.0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        logging.warning(WARN)
        with torch.no_grad():
            prior = self.prior_net(*args, **kwargs)
        net_out = self.net(*args, **kwargs)
        return net_out + prior * self.prior_scale
