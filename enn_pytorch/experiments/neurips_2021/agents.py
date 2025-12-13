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
"""A minimalist wrapper around ENN experiment for testbed submission - PyTorch version."""

from typing import Callable, Dict, List, Optional

import dataclasses
from enn_pytorch import base as enn_base
from enn_pytorch import utils
from enn_pytorch.experiments.neurips_2021 import base as testbed_base
from enn_pytorch.experiments.neurips_2021 import enn_losses
import torch
import torch.optim as optim

from enn_pytorch.experiments.neurips_2021.random import split_seed


def logging_freq(num_steps: int, log_freq: Optional[int] = None) -> int:
    """Compute a sensible logging frequency."""
    if log_freq is None:
        return max(1, num_steps // 10)
    return log_freq


@dataclasses.dataclass
class VanillaEnnConfig:
    """Configuration options for the VanillaEnnAgent."""

    enn_ctor: enn_losses.EnnCtor
    loss_ctor: enn_losses.LossCtor = enn_losses.default_enn_loss()
    optimizer_ctor: Callable = None  # Function returning torch optimizer
    training_steps: Optional[int] = 100 * 1000
    batch_size: Optional[int] = None
    eval_batch_size: Optional[int] = None
    logger: Optional[object] = None
    train_log_freq: Optional[int] = None
    eval_log_freq: Optional[int] = None
    indexers: Optional[Dict] = None
    inference_samples: Optional[List] = None
    train_num_samples: Optional[int] = None
    batched_inference: Optional[bool] = False
    max_num_samples: Optional[int] = None

    def __post_init__(self):
        if self.optimizer_ctor is None:
            self.optimizer_ctor = lambda params: optim.Adam(params, lr=1e-3)
        if self.training_steps is None:
            self.training_steps = 100 * 1000


def extract_enn_sampler(
    model: torch.nn.Module,
    enn: enn_base.EpistemicNetwork,
) -> testbed_base.EpistemicSampler:
    """Extract an epistemic sampler from a trained ENN."""
    def enn_sampler(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Generate a random sample from posterior distribution at x."""
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                inputs = x
            else:
                inputs = torch.tensor(x, dtype=torch.float32)
            
            # Sample an index and get predictions
            index = enn.indexer(seed)
            net_out = enn.apply(model, inputs, index)
            net_out = utils.parse_net_output(net_out)
            
            return net_out

    return enn_sampler


@dataclasses.dataclass
class VanillaEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    config: VanillaEnnConfig

    def __call__(
        self, 
        data: testbed_base.Data, 
        seed: int,
        prior: Optional[testbed_base.PriorKnowledge] = None,
        device: str = 'cuda:0'
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        # Create the ENN
        enn = self.config.enn_ctor(prior)

        init_seed, dataset_seed, train_seed = split_seed(seed, 3)

        model = enn.init(seed=init_seed)
        
        # Move to device
        model = model.to(device)
        
        # Create data batch
        enn_data = enn_base.Batch(data.x, data.y)
        
        # Create data loader
        dataset = utils.make_batch_iterator(
            enn_data, self.config.batch_size, dataset_seed, device=device
        )
        
        # Create optimizer
        optimizer = self.config.optimizer_ctor(model.parameters())
        
        # Create loss function
        loss_fn = self.config.loss_ctor(prior, enn)
        
        # Training loop
        model.train()

        steps = 0

        while steps < self.config.training_steps:
            batch = next(dataset)
            
            # Compute loss
            train_seed, run_seed = split_seed(train_seed, 2)
            loss, metrics = loss_fn(enn, model, batch, run_seed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (steps // batch.x.shape[0] + 1) % (logging_freq(self.config.training_steps, self.config.train_log_freq) // batch.x.shape[0]) == 0:
                print(f"Step {steps}/{self.config.training_steps}, Loss: {loss.item():.4f}")

            steps += batch.x.shape[0]
        
        model.eval()
        return extract_enn_sampler(model, enn)
