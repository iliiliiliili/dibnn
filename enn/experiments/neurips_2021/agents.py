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
"""A minimalist wrapper around ENN experiment for testbed submission."""

from typing import Callable, Dict, List, Optional

from acme.utils import loggers
import dataclasses
from enn import base as enn_base
from enn import supervised
from enn import utils
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import enn_losses
import jax
import optax


@dataclasses.dataclass
class VanillaEnnConfig:
    """Configuration options for the VanillaEnnAgent."""

    enn_ctor: enn_losses.EnnCtor
    loss_ctor: enn_losses.LossCtor = enn_losses.default_enn_loss()
    optimizer: optax.GradientTransformation = optax.adam(1e-3)
    num_batches: int = 1000
    seed: int = 0
    batch_size: int = 100
    eval_batch_size: Optional[int] = None
    logger: Optional[loggers.Logger] = None
    train_log_freq: Optional[int] = None
    eval_log_freq: Optional[int] = None
    indexers: Optional[Dict] = None
    inference_samples: Optional[List] = None
    train_num_samples: Optional[int] = None
    batched_inference: Optional[bool] = False
    max_num_samples: Optional[int] = None


def extract_enn_sampler(
    experiment: supervised.Experiment,
) -> testbed_base.EpistemicSampler:
    def enn_sampler(x: enn_base.Array, seed: int = 0) -> enn_base.Array:
        """Generate a random sample from posterior distribution at x."""
        net_out = experiment.predict(x, seed)
        return utils.parse_net_output(net_out)

    return jax.jit(enn_sampler)


def extract_multi_indexer_enn_sampler(
    experiment: supervised.Experiment, indexer_id,
) -> testbed_base.EpistemicSampler:
    def enn_sampler(x: enn_base.Array, seed: int = 0) -> enn_base.Array:
        """Generate a random sample from posterior distribution at x."""
        net_out = experiment.predict(x, seed, indexer_id)
        return utils.parse_net_output(net_out)

    return jax.jit(enn_sampler)


def extract_batched_enn_sampler(
    experiment: supervised.BatchedExperiment,
) -> testbed_base.EpistemicSampler:
    def enn_sampler(x: enn_base.Array, num_samples: int, seed: int = 0) -> enn_base.Array:
        """Generate a random sample from posterior distribution at x."""
        net_out = experiment.predict(x, seed, num_samples)
        return utils.parse_net_output(net_out)

    return jax.jit(enn_sampler, static_argnums=(1, ))


def extract_batched_fixed_enn_sampler(
    experiment: supervised.BatchedRankedExperiment,
) -> testbed_base.EpistemicSampler:
    def enn_sampler(x: enn_base.Array, samples) -> enn_base.Array:
        """Generate a random sample from posterior distribution at x."""
        net_out = experiment.predict(x, samples)
        return utils.parse_net_output(net_out)

    return jax.jit(enn_sampler)


@dataclasses.dataclass
class VanillaEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    config: VanillaEnnConfig
    eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None
    experiment: Optional[supervised.Experiment] = None

    def __call__(
        self, data: testbed_base.Data, prior: testbed_base.PriorKnowledge, evaluate: Callable = None, log_file_name: str = None,
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        enn = self.config.enn_ctor(prior)
        enn_data = enn_base.Batch(data.x, data.y)
        self.experiment = supervised.Experiment(
            enn=enn,
            loss_fn=self.config.loss_ctor(prior, enn),
            optimizer=self.config.optimizer,
            dataset=utils.make_batch_iterator(
                enn_data, self.config.batch_size, self.config.seed
            ),
            seed=self.config.seed,
            logger=self.config.logger,
            train_log_freq=logging_freq(
                self.config.num_batches, log_freq=self.config.train_log_freq
            ),
            eval_datasets=self.eval_datasets,
            eval_log_freq=200,
        )

        self.best_kl = None

        def log_evaluate():
            kl_quality = evaluate(extract_enn_sampler(self.experiment))
            print(
                f"kl_estimate={kl_quality.kl_estimate}"
                + "mean_error="
                + str(kl_quality.extra["mean_error"])
                + " "
                + "std_error="
                + str(kl_quality.extra["std_error"])
            )

            with open(
                log_file_name,
                "a",
            ) as f:

                f.write(
                    f"kl_estimate={kl_quality.kl_estimate}"
                    + " mean_error="
                    + str(kl_quality.extra["mean_error"])
                    + " "
                    + "std_error="
                    + str(kl_quality.extra["std_error"])
                    + "\n"
                )
            
            if self.best_kl is None or self.best_kl.kl_estimate > kl_quality.kl_estimate:
                self.best_kl = kl_quality
                return False
            else:
                return False # True

        loss = self.experiment.train(self.config.num_batches, None if evaluate is None else log_evaluate, log_file_name)
        return extract_enn_sampler(self.experiment)


@dataclasses.dataclass
class MultiIndexerEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    config: VanillaEnnConfig
    eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None
    experiment: Optional[supervised.Experiment] = None

    def __call__(
        self, data: testbed_base.Data, prior: testbed_base.PriorKnowledge, evaluate: Callable = None, log_file_name: str = None,
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        enn = self.config.enn_ctor(prior)
        enn_data = enn_base.Batch(data.x, data.y)
        self.experiment = supervised.MultiIndexerExperiment(
            enn=enn,
            indexers=self.config.indexers,
            loss_fn=self.config.loss_ctor(prior, enn),
            optimizer=self.config.optimizer,
            dataset=utils.make_batch_iterator(
                enn_data, self.config.batch_size, self.config.seed
            ),
            seed=self.config.seed,
            logger=self.config.logger,
            train_log_freq=logging_freq(
                self.config.num_batches, log_freq=self.config.train_log_freq
            ),
            eval_datasets=self.eval_datasets,
            eval_log_freq=200,
        )

        self.best_kl = None

        loss = self.experiment.train(self.config.num_batches, None, log_file_name)

        samplers = {}

        for id in self.config.indexers.keys():
            samplers[id] = extract_multi_indexer_enn_sampler(self.experiment, id)

        return samplers


@dataclasses.dataclass
class BatchedEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    config: VanillaEnnConfig
    eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None
    experiment: Optional[supervised.Experiment] = None

    def __call__(
        self, data: testbed_base.Data, prior: testbed_base.PriorKnowledge, evaluate: Callable = None, log_file_name: str = None,
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        enn = self.config.enn_ctor(prior)
        enn_data = enn_base.Batch(data.x, data.y)
        self.experiment = supervised.BatchedExperiment(
            enn=enn,
            loss_fn=self.config.loss_ctor(prior, enn),
            optimizer=self.config.optimizer,
            dataset=utils.make_batch_iterator(
                enn_data, self.config.batch_size, self.config.seed
            ),
            seed=self.config.seed,
            logger=self.config.logger,
            train_log_freq=logging_freq(
                self.config.num_batches, log_freq=self.config.train_log_freq
            ),
            eval_datasets=self.eval_datasets,
            eval_log_freq=200,
            batched_inference=self.config.batched_inference,
            train_num_samples=self.config.train_num_samples,
        )

        self.best_kl = None

        loss = self.experiment.train(self.config.num_batches, None, log_file_name)

        batched_sampler = extract_batched_enn_sampler(self.experiment)

        return batched_sampler


@dataclasses.dataclass
class BatchedRankedEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    config: VanillaEnnConfig
    eval_datasets: Optional[Dict[str, enn_base.BatchIterator]] = None
    experiment: Optional[supervised.Experiment] = None

    def __call__(
        self, data: testbed_base.Data, prior: testbed_base.PriorKnowledge, evaluate: Callable = None, log_file_name: str = None, max_num_samples = None,
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        enn = self.config.enn_ctor(prior)
        enn_data = enn_base.Batch(data.x, data.y)
        self.experiment = supervised.BatchedRankedExperiment(
            enn=enn,
            loss_fn=self.config.loss_ctor(prior, enn),
            optimizer=self.config.optimizer,
            dataset=utils.make_batch_iterator(
                enn_data, self.config.batch_size, self.config.seed
            ),
            seed=self.config.seed,
            logger=self.config.logger,
            train_log_freq=logging_freq(
                self.config.num_batches, log_freq=self.config.train_log_freq
            ),
            eval_datasets=self.eval_datasets,
            eval_log_freq=200,
            batched_inference=self.config.batched_inference,
            train_num_samples=self.config.train_num_samples,
        )

        self.best_kl = None

        loss = self.experiment.train(self.config.num_batches, None, log_file_name)

        batched_fixed_sampler = extract_batched_fixed_enn_sampler(self.experiment)

        all_indices = enn.indexer.batched(jax.random.PRNGKey(0), self.config.max_num_samples if max_num_samples is None else max_num_samples)

        return batched_fixed_sampler, all_indices


def _round_to_integer(x: float) -> int:
    """Utility function to round a float to integer, or 1 if it would be 0."""
    assert x > 0
    x = int(x)
    if x == 0:
        return 1
    else:
        return x


def logging_freq(
    num_batches: int, num_points: int = 100, log_freq: Optional[int] = None
) -> int:
    """Computes a logging frequency from num_batches, optionally log_freq."""
    if log_freq is None:
        log_freq = _round_to_integer(num_batches / num_points)
    return log_freq
