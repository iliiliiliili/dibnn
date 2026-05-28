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
from src import base as enn_base
from src import utils
from src.experiments import base as testbed_base
from src.experiments import enn_losses
import torch
import torch.optim as optim

from src.experiments.seeds import split_seed


def logging_freq(num_steps: int, log_freq: Optional[int] = None) -> int:
    """Compute a sensible logging frequency."""
    if log_freq is None:
        return max(1, num_steps // 10)
    return log_freq


@dataclasses.dataclass
class VanillaEnnConfig:
    """Configuration options for the VanillaEnnAgent."""

    enn_ctor: enn_losses.EnnCtor
    loss_ctor: enn_losses.LossCtor
    log_likelihood_ctor: bool = False
    optimizer_ctor: Callable = None  # Function returning torch optimizer
    training_epochs: Optional[int] = 1000
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
    early_stopping_patience: int = 0
    early_stopping_mode: str = "loss"
    early_stopping_min_delta: float = 0.0
    early_stopping_eval_freq: Optional[int] = None

    def __post_init__(self):
        if self.optimizer_ctor is None:
            print("Using default Adam optimizer with lr=1e-3")
            self.optimizer_ctor = lambda params: optim.Adam(params, lr=1e-3)
        if self.training_epochs is None:
            self.training_epochs = 1000


def extract_enn_sampler(
    model: torch.nn.Module, enn: enn_base.EpistemicNetwork, device
) -> testbed_base.EpistemicSampler:
    """Extract an epistemic sampler from a trained ENN."""

    def enn_sampler(
        x: torch.Tensor, seed: int = 0, num_samples: int = 1
    ) -> torch.Tensor:
        """Generate a random sample from posterior distribution at x."""
        with torch.no_grad():

            batched_indexer = utils.make_batch_indexer(enn.indexer, num_samples)
            indices = batched_indexer(seed, device)
            net_out = enn.apply(model, x, indices)
            net_out = utils.parse_net_output(net_out)

            return net_out

    return enn_sampler


def extract_fixed_enn_sampler(
    model: torch.nn.Module, enn: enn_base.EpistemicNetwork, device
) -> testbed_base.EpistemicSampler:
    """Extract an epistemic sampler from a trained ENN."""

    def enn_sampler(
        x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Generate a random sample from posterior distribution at x."""
        with torch.no_grad():

            net_out = enn.apply(model, x, indices)
            net_out = utils.parse_net_output(net_out)

            return net_out

    return enn_sampler


@dataclasses.dataclass
class VanillaEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    config: VanillaEnnConfig
    use_double_precision: bool = True
    fixed_sampler: bool = False
    best_epoch: Optional[int] = None

    def _evaluate_validation_metric(
        self,
        *,
        enn: enn_base.EpistemicNetwork,
        model: torch.nn.Module,
        loss_fn: enn_base.LossFn,
        val_dataset_iterator_creator: Optional[Callable[[], enn_base.BatchIterator]],
        evaluate_quality_val_fn: Optional[Callable[[testbed_base.EpistemicSampler, int], testbed_base.ENNQuality]],
        eval_seed: int,
        device: str,
        metric: str,
    ) -> float:
        """Evaluate validation metric used by early stopping."""
        with torch.no_grad():
            if metric == "loss":

                total_loss = 0
                total_batches = 0

                for val_batch in val_dataset_iterator_creator():
                    val_loss, _ = loss_fn(enn, model, val_batch, eval_seed, device)
                    total_loss += val_loss.item()
                    total_batches += 1

                return total_loss / total_batches
            elif metric == "kl":

                val_sampler = extract_enn_sampler(model, enn, device)
                kl_quality = evaluate_quality_val_fn(val_sampler, eval_seed, device=device)

                return kl_quality.kl_estimate
            else: 
                raise ValueError(f"Invalid early_stopping_mode: {metric}")


    def __call__(
        self,
        data: testbed_base.Data,
        seed: int,
        prior: Optional[testbed_base.PriorKnowledge] = None,
        device: str = "cuda:0",
        logging: str = "default",
        val_data: Optional[testbed_base.Data] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_mode: Optional[str] = None,
        early_stopping_min_delta: Optional[float] = None,
        early_stopping_eval_freq: Optional[int] = None,
        evaluate_quality_val_fn: Optional[Callable[[testbed_base.EpistemicSampler, int, str], testbed_base.ENNQuality]] = None,
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        # Create the ENN
        enn = self.config.enn_ctor(prior, use_double_precision=self.use_double_precision)
        self.enn = enn

        init_seed, dataset_seed, val_dataset_seed, train_seed = split_seed(seed, 4)

        model = enn.init(seed=init_seed)

        # Move to device
        model = model.to(device)

        # Create data batch
        enn_data = enn_base.Batch(data.x, data.y)

        if self.config.training_epochs < 1:
            raise ValueError("training_epochs must be set to a positive integer")

        # Create data loader
        dataset_iterator_creator = utils.make_batch_iterator(
            enn_data, self.config.batch_size, dataset_seed, device=device
        )

        # Create optimizer
        optimizer = self.config.optimizer_ctor(model.parameters())

        # Create loss function
        loss_fn = self.config.loss_ctor(prior, enn)

        # Training loop
        model.train()

        if early_stopping_patience is None:
            early_stopping_patience = self.config.early_stopping_patience
        if early_stopping_mode is None:
            early_stopping_mode = self.config.early_stopping_mode
        if early_stopping_min_delta is None:
            early_stopping_min_delta = self.config.early_stopping_min_delta
        if early_stopping_eval_freq is None:
            early_stopping_eval_freq = self.config.early_stopping_eval_freq

        use_early_stopping = early_stopping_patience is not None and early_stopping_patience > 0
        if use_early_stopping:
            if early_stopping_mode not in ["loss", "kl"]:
                raise ValueError(
                    "early_stopping_mode must be either 'loss' or 'kl'"
                )
            if early_stopping_mode == "loss" and val_data is None:
                raise ValueError("val_data must be provided when early_stopping_mode='loss'")
            if early_stopping_mode == "kl" and evaluate_quality_val_fn is None:
                raise ValueError(
                    "evaluate_quality_val_fn must be provided when early_stopping_mode='kl'"
                )

        if early_stopping_eval_freq is None:
            early_stopping_eval_freq = logging_freq(
                self.config.training_epochs, self.config.eval_log_freq
            )

        val_dataset_iterator_creator = None
        if val_data is not None:
            val_dataset_iterator_creator = utils.make_batch_iterator(
                enn_base.Batch(val_data.x, val_data.y), self.config.batch_size, val_dataset_seed, minimum_batch_size=1, device=device
            )

        best_metric = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state_dict = None
        early_stopping_seed, train_seed = split_seed(train_seed, 2)

        epoch = 0

        for epoch in range(self.config.training_epochs):

            for i, batch in enumerate(dataset_iterator_creator()):

                # Compute loss
                train_seed, run_seed = split_seed(train_seed, 2)
                loss, metrics = loss_fn(enn, model, batch, run_seed, device)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}", end="\r")

            if use_early_stopping and ((epoch % early_stopping_eval_freq == 0) or (epoch >= self.config.training_epochs - 1)):
                early_stopping_seed, eval_seed = split_seed(early_stopping_seed, 2)
                monitored_value = self._evaluate_validation_metric(
                    enn=enn,
                    model=model,
                    loss_fn=loss_fn,
                    val_dataset_iterator_creator=val_dataset_iterator_creator,
                    evaluate_quality_val_fn=evaluate_quality_val_fn,
                    eval_seed=eval_seed,
                    device=device,
                    metric=early_stopping_mode,
                )

                improved = monitored_value < (best_metric - early_stopping_min_delta)
                if improved:
                    best_metric = monitored_value
                    best_epoch = epoch
                    patience_counter = 0
                    best_state_dict = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if (
                    (logging != "none")
                    and (epoch % logging_freq(self.config.training_epochs, self.config.train_log_freq) == 0)
                ):
                    metric_name = "val_loss" if early_stopping_mode == "loss" else "val_kl"
                    print(
                        f"  EarlyStopping {metric_name}: {monitored_value:.6f}, best={best_metric:.6f}, wait={patience_counter}/{early_stopping_patience}"
                    )

                if patience_counter >= early_stopping_patience:
                    if logging != "none":
                        print(
                            f"Stopping early at epoch {epoch} (best epoch={best_epoch}, best metric={best_metric:.6f})"
                        )
                    break

            if (
                (epoch)
                % logging_freq(self.config.training_epochs, self.config.train_log_freq)
                == 0
            ) and (logging != "none"):
                print(
                    f"Epoch {epoch}/{self.config.training_epochs}, Loss: {loss.item():.4f}"
                )

        if use_early_stopping and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            self.best_epoch = best_epoch
        else:
            self.best_epoch = self.config.training_epochs - 1

        model.eval()

        sampler = extract_fixed_enn_sampler(model, enn, device) if self.fixed_sampler else extract_enn_sampler(model, enn, device)

        val_loss = self._evaluate_validation_metric(
            enn=enn,
            model=model,
            loss_fn=loss_fn,
            val_dataset_iterator_creator=val_dataset_iterator_creator,
            evaluate_quality_val_fn=evaluate_quality_val_fn,
            eval_seed=eval_seed,
            device=device,
            metric="loss",
        )

        val_kl = self._evaluate_validation_metric(
            enn=enn,
            model=model,
            loss_fn=loss_fn,
            val_dataset_iterator_creator=val_dataset_iterator_creator,
            evaluate_quality_val_fn=evaluate_quality_val_fn,
            eval_seed=eval_seed,
            device=device,
            metric="kl",
        )

        return sampler, val_loss, val_kl

    def create_all_indices(
        self,
        max_samples: int,
        seed: int,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        """Create all indices for fixed sampler."""
        indices = self.enn.indexer.batched(seed, max_samples, device)
        return indices
