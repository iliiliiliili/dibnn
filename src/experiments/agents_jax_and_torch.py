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

import functools
from typing import Callable, Dict, List, Optional

import dataclasses

import jax
import numpy as np
from enn.supervised.sgd_experiment import TrainingState as JaxTrainingState
from src import base as enn_base
from src import utils as torch_utils
from enn import utils as jax_utils
from src.experiments import base as testbed_base
from src.experiments import enn_losses
import torch
import torch.optim as optim
import optax
from src.experiments.agents import VanillaEnnConfig as TorchVanillaEnnConfig
from enn.experiments.neurips_2021.agents import VanillaEnnConfig as JaxVanillaEnnConfig
from enn import supervised as jax_supervised

from src.experiments.seeds import split_seed
import haiku as hk
from enn.base import Batch as JaxBatch
import jax.numpy as jnp
from acme.utils import loggers


def logging_freq(num_steps: int, log_freq: Optional[int] = None) -> int:
    """Compute a sensible logging frequency."""
    if log_freq is None:
        return max(1, num_steps // 10)
    return log_freq


def torch_extract_enn_sampler(
    model: torch.nn.Module, enn: enn_base.EpistemicNetwork, device
) -> testbed_base.EpistemicSampler:
    """Extract an epistemic sampler from a trained ENN."""

    def enn_sampler(
        x: torch.Tensor, seed: int = 0, num_samples: int = 1
    ) -> torch.Tensor:
        """Generate a random sample from posterior distribution at x."""
        with torch.no_grad():

            batched_indexer = torch_utils.make_batch_indexer(enn.indexer, num_samples)
            indices = batched_indexer(seed, device)
            net_out = enn.apply(model, x, indices)
            net_out = torch_utils.parse_net_output(net_out)

            return net_out

    return enn_sampler

def jax_extract_enn_sampler(
    predict_fn
) -> testbed_base.EpistemicSampler:
    def enn_sampler(x, seed: int = 0):
        """Generate a random sample from posterior distribution at x."""
        net_out = predict_fn(x, seed)
        return jax_utils.parse_net_output(net_out)

    return enn_sampler


def jax_and_torch_extract_enn_samplers(
    jax_enn,
    jax_state,
    torch_model: torch.nn.Module,
    torch_enn: enn_base.EpistemicNetwork,
    device,
):
    """Extract an epistemic sampler from a trained ENN."""

    def jax_and_torch_enn_sampler(jax_x, torch_x, seed: int = 0, num_samples: int = 1):
        """Generate a random sample from posterior distribution at x."""

        def jax_enn_sampler(x, seed: int = 0):
            jax_index = jax_enn.indexer(jax.random.PRNGKey(seed))
            jax_net_out = jax_enn.apply(jax_state.params, x, jax_index)
            jax_result = jax_utils.parse_net_output(jax_net_out)
            return jax_result

        
        jax_batched_sampler = jax.jit(jax.vmap(jax_enn_sampler, in_axes=[None, 0]))
        jax_batched_result = jax_batched_sampler(jax_x, jnp.arange(num_samples))

        torch_batched_indexer = torch_utils.make_batch_indexer(torch_enn.indexer, num_samples)
        torch_indices = torch_batched_indexer(seed, device)
        torch_net_out = torch_enn.apply(torch_model, torch_x, torch_indices)
        torch_result = torch_utils.parse_net_output(torch_net_out)

        return jax_batched_result, torch_result

    return jax_and_torch_enn_sampler

class JaxExperiment():
    """Class to handle supervised training.

    Optional eval_datasets which is a collection of datasets to *evaluate*
    the loss on every eval_log_freq steps.
    """

    def __init__(
        self,
        enn,
        loss_fn,
        optimizer: optax.GradientTransformation,
        dataset,
        seed: int = 0,  
        logger = None,
        train_log_freq: int = 1,
        eval_datasets = None,
        eval_log_freq: int = 1,
    ):
        self.enn = enn
        self.dataset = dataset
        self.rng = hk.PRNGSequence(seed)

        # Internalize the loss_fn
        self._loss = functools.partial(loss_fn, self.enn)

        # Internalize the eval datasets
        self._eval_datasets = eval_datasets
        self._eval_log_freq = eval_log_freq

        # Forward network at random index
        def forward(
            params: hk.Params, inputs, key
        ):
            index = self.enn.indexer(key)
            return self.enn.apply(params, inputs, index)

        self._forward = forward

        # Define the SGD step on the loss
        def sgd_step(
            state,
            batch,
            key,
        ):
            # Calculate the loss, metrics and gradients
            (loss, metrics), grads = jax.value_and_grad(self._loss, has_aux=True)(
                state.params, batch, key
            )
            metrics.update({"loss": loss})
            updates, new_opt_state = optimizer.update(grads, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)
            new_state = JaxTrainingState(
                params=new_params,
                opt_state=new_opt_state,
            )
            return new_state, metrics

        # self._sgd_step = jax.jit(sgd_step)
        self._sgd_step = sgd_step

        # Initialize networks
        batch = next(self.dataset)
        index = self.enn.indexer(next(self.rng))
        params = self.enn.init(next(self.rng), batch.x, index)
        opt_state = optimizer.init(params)
        self.state = JaxTrainingState(params, opt_state)
        self.step = 0
        self.logger = logger or loggers.make_default_logger("experiment", time_delta=0)
        self._train_log_freq = train_log_freq

    def train(
        self, num_batches: int, evaluate: Callable = None, log_file_name: str = None
    ):
        """Train the ENN for num_batches."""
        for _ in range(num_batches):
            self.step += 1
            self.state, loss_metrics = self._sgd_step(
                self.state, next(self.dataset), next(self.rng)
            )

            # Periodically log this performance as dataset=train.
            if self.step % self._train_log_freq == 0:
                loss_metrics.update(
                    {"dataset": "train", "step": self.step, "sgd": True}
                )
                self.logger.write(loss_metrics)
                print(loss_metrics)
                if log_file_name is not None:
                    with open(log_file_name, "a") as f:
                        f.write(
                            "step="
                            + str(loss_metrics["step"])
                            + " loss="
                            + str(float(loss_metrics["loss"]))
                            + "\n"
                        )

            if evaluate is not None and self.step % self._eval_log_freq == 0:
                if evaluate():  # KL started decreasing
                    return

            # Periodically evaluate the other datasets.
            if self._eval_datasets and self.step % self._eval_log_freq == 0:
                for name, dataset in self._eval_datasets.items():
                    loss, metrics = self._loss(
                        self.state.params, next(dataset), next(self.rng)
                    )
                    metrics.update(
                        {
                            "dataset": name,
                            "step": self.step,
                            "sgd": False,
                            "loss": loss,
                        }
                    )
                    self.logger.write(metrics)

        return float(loss_metrics["loss"])

    def predict(self, inputs, seed):
        """Evaluate the trained model at given inputs."""
        return self._forward(self.state.params, inputs, jax.random.PRNGKey(seed))

    def loss(self, batch, seed):
        """Evaluate the loss for one batch of data."""
        return self._loss(self.state.params, batch, jax.random.PRNGKey(seed))



def write_jax_to_torch_ensembles(jax_params, jax_prior_params, torch_model, device, use_double_precision):

    print(":::Writing JAX ensemble parameters to Torch model:::")
    
    jax_to_torch_weights = [torch.tensor(np.array(b["w"])).permute(2,1,0) for a,b in jax_params.items()]
    jax_to_torch_biases = [torch.tensor(np.array(b["b"])).permute(1,0) for a,b in jax_params.items()]

    jax_to_torch_prior_weights = [torch.tensor(np.array(b["w"])).permute(2,1,0) for a,b in jax_prior_params.items()]
    jax_to_torch_prior_biases = [torch.tensor(np.array(b["b"])).permute(1,0) for a,b in jax_prior_params.items()]

    for i in range(len(torch_model.model.ensembled_weights)):
        torch_model.model.ensembled_weights[i].data = jax_to_torch_weights[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        torch_model.model.ensembled_biases[i].data = jax_to_torch_biases[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        torch_model.prior_model.ensembled_weights[i].data = jax_to_torch_prior_weights[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        torch_model.prior_model.ensembled_biases[i].data = jax_to_torch_prior_biases[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)



def compare_jax_and_torch_ensemble_weights(jax_params, jax_prior_params, torch_model, device, use_double_precision):

    jax_to_torch_weights = [torch.tensor(np.array(b["w"])).permute(2,1,0) for a,b in jax_params.items()]
    jax_to_torch_biases = [torch.tensor(np.array(b["b"])).permute(1,0) for a,b in jax_params.items()]

    jax_to_torch_prior_weights = [torch.tensor(np.array(b["w"])).permute(2,1,0) for a,b in jax_prior_params.items()]
    jax_to_torch_prior_biases = [torch.tensor(np.array(b["b"])).permute(1,0) for a,b in jax_prior_params.items()]

    total_difference = 0.0

    for i in range(len(torch_model.model.ensembled_weights)):

        tw = torch_model.model.ensembled_weights[i].data
        jw = jax_to_torch_weights[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        tb = torch_model.model.ensembled_biases[i].data
        jb = jax_to_torch_biases[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)

        dw = tw - jw
        db = tb - jb
        prior_tw = torch_model.prior_model.ensembled_weights[i].data
        prior_jw = jax_to_torch_prior_weights[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        prior_tb = torch_model.prior_model.ensembled_biases[i].data
        prior_jb = jax_to_torch_prior_biases[i].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        prior_dw = prior_tw - prior_jw
        prior_db = prior_tb - prior_jb

        total_difference += torch.norm(dw).item() + torch.norm(db).item() + torch.norm(prior_dw).item() + torch.norm(prior_db).item()

        print(f"Layer {i} weight difference norm: {torch.norm(dw).item()}, bias difference norm: {torch.norm(db).item()}")
        print(f"Layer {i} prior weight difference norm: {torch.norm(prior_dw).item()}, prior bias difference norm: {torch.norm(prior_db).item()}")
    
    print(f"Total difference norm across all layers: {total_difference}")
    print()

def compare_jax_and_torch_hypermodels_weights(jax_params, jax_prior_params, torch_model, device, use_double_precision):

    jax_to_torch_weights = [torch.tensor(np.array(b["w"])).permute(1,0).to(device, dtype=torch.float64 if use_double_precision else torch.float32) for a,b in jax_params.items()]
    jax_to_torch_biases = [torch.tensor(np.array(b["b"])).to(device, dtype=torch.float64 if use_double_precision else torch.float32) for a,b in jax_params.items()]

    # prior_jax_to_torch_weights = [torch.tensor(np.array(b["w"])).permute(1,0).to(device, dtype=torch.float64 if use_double_precision else torch.float32) for a,b in jax_prior_params.items()]
    # prior_jax_to_torch_biases = [torch.tensor(np.array(b["b"])).to(device, dtype=torch.float64 if use_double_precision else torch.float32) for a,b in jax_prior_params.items()]

    total_difference = 0.0

    for i in range(len(torch_model.model.weight_hyper_layers)):

        tww = torch_model.model.weight_hyper_layers[i].weight.data
        jww = jax_to_torch_weights[i * 2 + 1]

        twb = torch_model.model.weight_hyper_layers[i].bias.data
        jwb = jax_to_torch_biases[i * 2 + 1]

        tbw = torch_model.model.bias_hyper_layers[i].weight.data
        jbw = jax_to_torch_weights[i * 2]

        tbb = torch_model.model.bias_hyper_layers[i].bias.data
        jbb = jax_to_torch_biases[i * 2]

        dww = tww - jww
        dwb = twb - jwb

        dbw = tbw - jbw
        dbb = tbb - jbb

        total_difference += torch.norm(dww).item() + torch.norm(dwb).item() +  torch.norm(dbw).item() + torch.norm(dbb).item()

        print(f"Layer {i} weight weight difference norm: {torch.norm(dww).item()}, weight bias difference norm: {torch.norm(dwb).item()}")
        print(f"Layer {i} bias weight difference norm: {torch.norm(dbw).item()}, bias bias difference norm: {torch.norm(dbb).item()}")
    
    print(f"Total difference norm across all layers: {total_difference}")
    print()


def write_jax_to_torch_dropout(jax_params, torch_model, device, use_double_precision):

    print(":::Writing JAX dropout parameters to Torch model:::")

    jax_to_torch_weights = [torch.tensor(np.array(b["w"])).permute(1,0) for a,b in jax_params.items()]
    jax_to_torch_biases = [torch.tensor(np.array(b["b"])) for a,b in jax_params.items()]

    for i, param in enumerate(torch_model.parameters()):
        
        is_weight = (i % 2 == 0)

        if is_weight:
            jax_param = jax_to_torch_weights[i // 2].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        else:
            jax_param = jax_to_torch_biases[i // 2].to(device, dtype=torch.float64 if use_double_precision else torch.float32)

        param.data = jax_param


def compare_jax_and_torch_dropout_weights(jax_params, torch_model, device, use_double_precision):

    jax_to_torch_weights = [torch.tensor(np.array(b["w"])).permute(1,0) for a,b in jax_params.items()]
    jax_to_torch_biases = [torch.tensor(np.array(b["b"])) for a,b in jax_params.items()]

    total_difference = 0.0

    for i, param in enumerate(torch_model.parameters()):

        is_weight = (i % 2 == 0)

        if is_weight:
            jax_param = jax_to_torch_weights[i // 2].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        else:
            jax_param = jax_to_torch_biases[i // 2].to(device, dtype=torch.float64 if use_double_precision else torch.float32)
        
        delta = param.data - jax_param
        total_difference += torch.norm(delta).item()

        print(f"Layer {i} difference norm: {torch.norm(delta).item()}")
    
    print(f"Total difference norm across all layers: {total_difference}")
    print()


@dataclasses.dataclass
class JaxAndTorchVanillaEnnAgent(testbed_base.TestbedAgent):
    """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""

    jax_config: JaxVanillaEnnConfig
    torch_config: TorchVanillaEnnConfig
    use_double_precision: bool = True

    def __call__(
        self,
        jax_data: testbed_base.Data,
        torch_data: testbed_base.Data,
        seed: int,
        jax_prior: Optional[testbed_base.PriorKnowledge] = None,
        torch_prior: Optional[testbed_base.PriorKnowledge] = None,
        device: str = "cuda:0",
        logging: str = "default",
    ) -> testbed_base.EpistemicSampler:
        """Wraps an ENN as a testbed agent, using sensible loss/bootstrapping."""
        # Create the ENN
        jax_enn = self.jax_config.enn_ctor(jax_prior)
        torch_enn = self.torch_config.enn_ctor(torch_prior, use_double_precision=self.use_double_precision)

        init_seed, dataset_seed, train_seed = split_seed(seed, 3)

        torch_model = torch_enn.init(seed=init_seed)
        torch_model = torch_model.to(device)

        # Create data batch
        torch_enn_data = enn_base.Batch(torch_data.x, torch_data.y)
        jax_enn_data = JaxBatch(jax_data.x, jax_data.y)

        # Create data loader
        torch_dataset = torch_utils.make_batch_iterator(
            torch_enn_data, self.torch_config.batch_size, dataset_seed, device=device
        )

        jax_dataset = jax_utils.make_batch_iterator(
            jax_enn_data, self.jax_config.batch_size, self.jax_config.seed
        )

        # external_jax_experiment = JaxExperiment(
        #     enn=jax_enn,
        #     loss_fn=self.jax_config.loss_ctor(jax_prior, jax_enn),
        #     optimizer=self.jax_config.optimizer,
        #     dataset=jax_utils.make_batch_iterator(
        #         jax_enn_data, self.jax_config.batch_size, self.jax_config.seed
        #     ),
        #     seed=self.jax_config.seed,
        #     logger=self.jax_config.logger,
        #     train_log_freq=logging_freq(
        #         self.jax_config.num_batches, log_freq=self.jax_config.train_log_freq
        #     ),
        #     eval_datasets=None,
        #     eval_log_freq=200,
        # )

        # external_jax_experiment.train(
        #     self.jax_config.num_batches,
        #     None,
        #     None,
        # )

        # Create optimizer
        torch_optimizer = self.torch_config.optimizer_ctor(torch_model.parameters())
        jax_optimizer = self.jax_config.optimizer
        # Create loss function
        torch_loss_fn = self.torch_config.loss_ctor(torch_prior, torch_enn)
        
        exported_jax_index = None

        def export_jax_index(index):
            nonlocal exported_jax_index
            exported_jax_index = index
        
        
        jax_loss_fn = self.jax_config.loss_ctor(jax_prior, jax_enn)
        # jax_loss_fn = self.jax_config.loss_ctor(jax_prior, jax_enn, export_jax_index)
        jax_partial_loss_fn = functools.partial(jax_loss_fn, jax_enn)


        # Training loop
        torch_model.train()
        jax_rng = hk.PRNGSequence(seed)

        steps = 0

        jax_batch = next(jax_dataset)
        jax_index = jax_enn.indexer(next(jax_rng))
        jax_params = jax_enn.init(next(jax_rng), jax_batch.x, jax_index)

        jax_opt_state = jax_optimizer.init(jax_params)
        jax_state = JaxTrainingState(jax_params, jax_opt_state)

        jax_num_batches = self.jax_config.num_batches

        # write_jax_to_torch_ensembles(jax_state.params, jax_enn.prior_params, torch_model, device, self.use_double_precision)
        # compare_jax_and_torch_ensemble_weights(jax_state.params, jax_enn.prior_params, torch_model, device, self.use_double_precision)

        # write_jax_to_torch_dropout(jax_state.params, torch_model, device, self.use_double_precision)
        # compare_jax_and_torch_dropout_weights(jax_state.params, torch_model, device, self.use_double_precision)

        compare_jax_and_torch_hypermodels_weights(jax_state.params, jax_enn.prior_params, torch_model, device, self.use_double_precision)

        # external_jax_experiment.state = jax_state
        # external_jax_experiment._loss = jax_partial_loss_fn

        jax_loss_metrics = {"loss": -2605.0}

        while steps < self.torch_config.training_steps:
        # for steps in range(jax_num_batches):
            torch_batch = next(torch_dataset)
            jax_batch = next(jax_dataset)
            jax_to_torch_batch = enn_base.Batch(
                x=torch.tensor(np.array(jax_batch.x), device=device, dtype=torch.float64 if self.use_double_precision else torch.float32),
                y=torch.tensor(np.array(jax_batch.y), device=device, dtype=torch.float64 if self.use_double_precision else torch.float32),
                data_index=torch.tensor(np.array(jax_batch.data_index), device=device),
                weights=torch.tensor(np.array(jax_batch.weights), device=device, dtype=torch.float64 if self.use_double_precision else torch.float32),
            )
            torch_to_jax_batch = JaxBatch(
                x=torch_batch.x.cpu().numpy(),
                y=torch_batch.y.cpu().numpy(),
                data_index=torch_batch.data_index.cpu().numpy(),
                weights=torch_batch.weights.cpu().numpy(),
            )
            jax_next_rng = next(jax_rng)
            # torch_batch = jax_to_torch_batch
            # jax_batch = torch_to_jax_batch

            jax_loss_again = jax_loss_fn(jax_enn, jax_state.params, jax_batch, jax_next_rng)
            (jax_loss, jax_metrics), jax_grads = jax.value_and_grad(jax_partial_loss_fn, has_aux=True)(
                jax_state.params, jax_batch, jax_next_rng
            )
            assert (jax_loss - jax_loss_again[0]) < 0.0001
            jax_metrics.update({"loss": jax_loss})
            jax_updates, jax_new_opt_state = jax_optimizer.update(jax_grads, jax_state.opt_state)
            new_params = optax.apply_updates(jax_state.params, jax_updates)
            new_state = JaxTrainingState(
                params=new_params,
                opt_state=jax_new_opt_state,
            )
            jax_state = new_state
            jax_loss_metrics = jax_metrics

            self.jax_config.logger.write(jax_loss_metrics)

            train_seed, run_seed = split_seed(train_seed, 2)
            torch_loss, torch_metrics = torch_loss_fn(torch_enn, torch_model, torch_batch, run_seed, device)
            # jax_to_torch_index = torch.tensor(np.array(exported_jax_index), device=device)
            # torch_loss, torch_metrics = torch_loss_fn(torch_enn, torch_model, torch_batch, run_seed, device, replace_indices=jax_to_torch_index)
            # Backward pass
            torch_optimizer.zero_grad()
            torch_loss.backward()
            torch_optimizer.step()

            
            # compare_jax_and_torch_ensemble_weights(jax_state.params, jax_enn.prior_params, torch_model, device, self.use_double_precision)
            # print(
            #     f"Step {steps}/{self.torch_config.training_steps}, Torch Loss: {torch_loss.item():.4f}, Jax Loss: {jax_loss_metrics['loss']:.4f}"
            # )
            # print(end="")


            if (
                (steps)
                % logging_freq(self.torch_config.training_steps, self.torch_config.train_log_freq)
                == 0
            ) and (logging != "none"):
                print(
                    f"Step {steps}/{self.torch_config.training_steps}, Torch Loss: {torch_loss.item():.4f}, Jax Loss: {jax_loss_metrics['loss']:.4f}"
                )
            # if (
            #     (steps)
            #     % logging_freq(jax_num_batches, self.jax_config.train_log_freq)
            #     == 0
            # ) and (logging != "none"):
            #     print(
            #         f"Step {steps}/{jax_num_batches}, Torch Loss: {torch_loss.item():.4f}, Jax Loss: {jax_loss_metrics['loss']:.4f}"
            #     )

            steps += max(1, torch_batch.x.shape[0] // 100)

        torch_model.eval()

        def jax_predict_fn(x, seed: int):
            index = jax_enn.indexer(jax.random.PRNGKey(seed))
            return jax_enn.apply(jax_state.params, x, index)
    
        # write_jax_to_torch_ensembles(jax_state.params, jax_enn.prior_params, torch_model, device, self.use_double_precision)

        torch_sampler = torch_extract_enn_sampler(torch_model, torch_enn, device)
        jax_sampler = jax_extract_enn_sampler(jax_predict_fn)

        jax_and_torch_enn_sampler = jax_and_torch_extract_enn_samplers(
            jax_enn,
            jax_state,
            torch_model,
            torch_enn,
            device,
        )
        
        # external_sampler = jax_extract_enn_sampler(
        #     external_jax_experiment.predict
        # )

        return jax_sampler, torch_sampler, jax_and_torch_enn_sampler