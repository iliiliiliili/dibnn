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
"""Agent factory methods - PyTorch version."""

import dataclasses
from functools import reduce
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from src import networks
from src import base as enn_base
from src.experiments import agents, enn_losses
from src.experiments import base as testbed_base
import torch
import torch.optim as optim


ConfigCtor = Callable[[], agents.VanillaEnnConfig]


@dataclasses.dataclass
class AgentCtorConfig:
    settings: Dict[str, Any]  # Hyperparameters to work out which agent it is
    config_ctor: ConfigCtor  # Constructor for the agent config.


# def make_ensemble_agent(
#     num_ensemble: int = 5,
#     hidden_sizes: Sequence[int] = (50, 50),
#     num_batches: int = 1000,
#     batch_size: int = 32,
#     learning_rate: float = 1e-3,
#     seed: int = 0,
# ) -> testbed_base.TestbedAgent:
#     """Factory for creating an ensemble-based agent."""
#     from enn_pytorch.experiments.neurips_2021 import agents

#     def enn_ctor(prior: testbed_base.PriorKnowledge):
#         output_sizes = list(hidden_sizes) + [1]
#         return networks.MLPEnsembleEnn(output_sizes, num_ensemble)

#     def optimizer_ctor(params):
#         return optim.Adam(params, lr=learning_rate)

#     config = agents.VanillaEnnConfig(
#         enn_ctor=enn_ctor,
#         loss_ctor=enn_losses.default_enn_loss(num_index_samples=10),
#         optimizer_ctor=optimizer_ctor,
#         num_batches=num_batches,
#         batch_size=batch_size,
#         seed=seed,
#     )

#     return agents.VanillaEnnAgent(config)


def make_dropout_ctor(
    dropout_rate: float,
    regularization_scale: float,
    hidden_size: int = 50,
    num_layers: int = 2,
    dropout_input: bool = True,
    regularization_tau: float = 1,
    learning_rate: float = 1e-3,
    training_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> testbed_base.TestbedAgent:
    """Factory for creating a dropout-based agent."""

    def enn_ctor(prior: testbed_base.PriorKnowledge, use_double_precision: bool = False):
        output_sizes = (
            [prior.input_dim] + list([hidden_size] * num_layers) + [prior.num_classes]
        )
        return networks.MLPDropoutENN(
            output_sizes=output_sizes,
            dropout_rate=dropout_rate,
            dropout_input=dropout_input,
            use_double_precision=use_double_precision,
        )

    def optimizer_ctor(params):
        return optim.Adam(params, lr=learning_rate)

    def make_agent_config() -> agents.VanillaEnnConfig:
        config = agents.VanillaEnnConfig(
            enn_ctor=enn_ctor,
            loss_ctor=enn_losses.regularized_dropout_loss(
                num_index_samples=1,
                dropout_rate=dropout_rate,
                scale=regularization_scale,
                tau=regularization_tau,
            ),
            optimizer_ctor=optimizer_ctor,
            training_steps=training_steps,
            batch_size=batch_size,
        )

        return config

    return make_agent_config


def make_bbb_ctor(
    sigma_0: float,
    learning_rate: float,
    hidden_size: int = 50,
    num_layers: int = 2,
    num_index_samples: int = 64,
    training_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> ConfigCtor:
    """Generate an ensemble agent config."""

    def enn_ctor(prior: testbed_base.PriorKnowledge, use_double_precision: bool = True):
        output_sizes = (
            [prior.input_dim] + list([hidden_size] * num_layers) + [prior.num_classes]
        )
        return networks.MlpBbbEnn(
            output_sizes=output_sizes,
            sigma_0=sigma_0,
            use_double_precision=use_double_precision,
        )

    def optimizer_ctor(params):
        return optim.Adam(params, lr=learning_rate)

    def make_agent_config() -> agents.VanillaEnnConfig:
        config = agents.VanillaEnnConfig(
            enn_ctor=enn_ctor,
            loss_ctor=enn_losses.bbb_loss(
                sigma_0=sigma_0, num_index_samples=num_index_samples
            ),
            optimizer_ctor=optimizer_ctor,
            training_steps=training_steps,
            batch_size=batch_size,
        )

        return config

    return make_agent_config


def make_hypermodel_ctor(
    index_dim: int,
    noise_scale: float,
    prior_scale: float,
    hidden_size: int = 50,
    num_layers: int = 2,
    num_index_samples: Optional[int] = None,
    training_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-3,
) -> ConfigCtor:
    """Generate an ensemble agent config."""

    def enn_ctor(prior: testbed_base.PriorKnowledge, use_double_precision: bool = True):
        output_sizes = (
            [prior.input_dim] + list([hidden_size] * num_layers) + [prior.num_classes]
        )
        return networks.MlpLinearHypermodelEnnWithAdditivePriorIndependentLayers(
            output_sizes=output_sizes,
            index_dim=index_dim,
            prior_scale=prior_scale,
            use_double_precision=use_double_precision,
        )

    def optimizer_ctor(params):
        return optim.Adam(params, lr=learning_rate)

    def make_agent_config() -> agents.VanillaEnnConfig:
        config = agents.VanillaEnnConfig(
            enn_ctor=enn_ctor,
            loss_ctor=enn_losses.gaussian_regression_loss(
                num_index_samples, noise_scale, l2_weight_decay=0
            ),
            optimizer_ctor=optimizer_ctor,
            training_steps=training_steps,
            batch_size=batch_size,
        )

        return config

    return make_agent_config


def make_ensemble_ctor(
    num_ensemble: int,
    noise_scale: float,
    prior_scale: float,
    learning_rate: float,
    hidden_size: int = 50,
    num_layers: int = 2,
    training_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> ConfigCtor:
    """Generate an ensemble agent config."""

    def enn_ctor(prior: testbed_base.PriorKnowledge, use_double_precision: bool = True):
        output_sizes = (
            [prior.input_dim] + list([hidden_size] * num_layers) + [prior.num_classes]
        )
        return networks.MlpEnsembleEnnWithAdditivePrior(
            output_sizes=output_sizes,
            num_ensemble=num_ensemble,
            prior_scale=prior_scale,
            use_double_precision=use_double_precision,
        )

    def optimizer_ctor(params):
        return optim.Adam(params, lr=learning_rate)

    def make_agent_config() -> agents.VanillaEnnConfig:
        config = agents.VanillaEnnConfig(
            enn_ctor=enn_ctor,
            loss_ctor=enn_losses.gaussian_regression_loss(
                num_ensemble, noise_scale, l2_weight_decay=0
            ),
            optimizer_ctor=optimizer_ctor,
            training_steps=training_steps,
            batch_size=batch_size,
        )

        return config

    return make_agent_config



def make_vnn_ctor(
    activation: Optional[Union[Callable, List[Callable]]] = None,
    activation_mode: Union[
        Literal["mean"],
        Literal["std"],
        Literal["mean+std"],
        Literal["end"],
        Literal["mean+end"],
        Literal["std+end"],
        Literal["mean+std+end"],
    ] = "mean",
    use_batch_norm: bool = False,
    batch_norm_mode: Union[
        Literal["mean"],
        Literal["std"],
        Literal["mean+std"],
        Literal["end"],
        Literal["mean+end"],
        Literal["std+end"],
        Literal["mean+std+end"],
    ] = "mean",
    global_std_mode: Union[
        Literal["none"], Literal["replace"], Literal["multiply"]
    ] = "none",
    num_index_samples: int = 10,
    hidden_size: int = 50,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
    training_steps: int = 1000,
    initializer: Tuple[Optional[str], Optional[str]] = (None, None),
    loss_function: str = "gaussian",
    noise_scale: float = 1,
    batch_size: Optional[int] = None,
) -> ConfigCtor:
    """Generate a dropout agent config."""

    # def get_cosine_lr_scheduler(init_lr, final_lr, n_epoch=1000):
    #     import numpy as np

    #     def lr_scheduler(epoch_idx):
    #         lr = final_lr + 0.5 * (init_lr - final_lr) * (
    #             1 + torch.cos(torch.pi * epoch_idx / n_epoch)
    #         )
    #         return lr

    #     return lr_scheduler

    def make_enn(prior: testbed_base.PriorKnowledge, use_double_precision: bool = False) -> enn_base.EpistemicNetwork:
        output_sizes = (
            [prior.input_dim] + list([hidden_size] * num_layers) + [prior.num_classes]
        )
        return networks.MLPVariationalENN(
            output_sizes=output_sizes,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            global_std_mode=global_std_mode,
            initializer=initializer,
            seed=seed,
            use_double_precision=use_double_precision,
        )

    def optimizer_ctor(params):
        return optim.Adam(params, lr=learning_rate)

    def make_agent_config() -> agents.VanillaEnnConfig:
        """Factory method to create agent_config, swap this for different agents."""

        if loss_function == "gaussian":
            loss_ctor = enn_losses.gaussian_regression_loss(
                num_index_samples, noise_scale, l2_weight_decay=0
            )
        # elif loss_function == "nelbo":
        #     loss_ctor = enn_losses.bbb_loss(
        #         sigma_0=sigma_0, num_index_samples=num_index_samples
        #     )
        else:
            raise ValueError(loss_function + "is an unknown loss_function")

        return agents.VanillaEnnConfig(
            enn_ctor=make_enn,
            loss_ctor=loss_ctor,
            optimizer_ctor=optimizer_ctor,
            training_steps=training_steps,
            batch_size=batch_size,
        )

    return make_agent_config


def make_layer_ensembles_ctor(
    num_ensembles: List[int],
    noise_scale: float,
    prior_scale: float,
    learning_rate: float,
    hidden_size: int = 50,
    num_layers: int = 2,
    inference_samples: List[int] = ["full"],
    training_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> ConfigCtor:
    """Generate an ensemble agent config."""

    num_samples = reduce(lambda x, y: x * y, num_ensembles)

    inference_samples = [
        (int(x) * num_ensembles[0] if x != "full" else num_samples)
        for x in inference_samples
    ]

    def make_enn(prior: testbed_base.PriorKnowledge, use_double_precision: bool = False) -> enn_base.EpistemicNetwork:
        output_sizes = (
            [prior.input_dim] + list([hidden_size] * num_layers) + [prior.num_classes]
        )
        return networks.MlpLayerEnsembleEnnWithAdditivePrior(
            output_sizes=output_sizes,
            num_ensembles=num_ensembles,
            prior_scale=prior_scale,
            use_double_precision=use_double_precision,
        )

    def optimizer_ctor(params):
        return optim.Adam(params, lr=learning_rate)

    def make_agent_config() -> agents.VanillaEnnConfig:
        config = agents.VanillaEnnConfig(
            enn_ctor=make_enn,
            loss_ctor=enn_losses.gaussian_regression_loss(
                num_samples, noise_scale, l2_weight_decay=0
            ),
            optimizer_ctor=optimizer_ctor,
            training_steps=training_steps,
            batch_size=batch_size,
            inference_samples=inference_samples,
            max_num_samples=num_samples,
        )

        return config

    return make_agent_config


def make_dropout_sweep() -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting dropout agents
    for dropout_rate in [0.05, 0.1, 0.2]:
        for regularization_scale in [0, 1e-6, 1e-4]:
            for num_layers in [2, 3]:
                for hidden_size in [50, 100]:
                    settings = {
                        "agent": "dropout",
                        "dropout_rate": dropout_rate,
                        "regularization_scale": regularization_scale,
                        "num_layers": num_layers,
                        "hidden_size": hidden_size,
                    }
                    config_ctor = make_dropout_ctor(
                        dropout_rate, regularization_scale, hidden_size, num_layers
                    )
                    sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep

def make_dropout_best_sweep() -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting dropout agents
    for dropout_rate in [0.05]:
        for regularization_scale in [1e-6]:
            for num_layers in [2]:
                for hidden_size in [50]:
                    settings = {
                        "agent": "dropout",
                        "dropout_rate": dropout_rate,
                        "regularization_scale": regularization_scale,
                        "num_layers": num_layers,
                        "hidden_size": hidden_size,
                    }
                    config_ctor = make_dropout_ctor(
                        dropout_rate, regularization_scale, hidden_size, num_layers
                    )
                    sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_bbb_sweep() -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting bbb agents
    for sigma_0 in [1, 10, 100, 200]:
        for learning_rate in [1e-3, 3e-4, 1e-4]:
            for num_layers in [2, 3]:
                for hidden_size in [50, 100]:
                    for training_steps in [1000]:
                        settings = {
                            "agent": "bbb",
                            "sigma_0": sigma_0,
                            "learning_rate": learning_rate,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                        }
                        config_ctor = make_bbb_ctor(
                            sigma_0, learning_rate, hidden_size, num_layers, training_steps=training_steps
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_bbb_selected_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting bbb agents
    for sigma_0 in [100, 200]:
        for learning_rate in [1e-3, 1e-4]:
            for num_layers in [2, 3]:
                for hidden_size in [50, 100]:
                    for training_steps in [1000, 1500]:
                        settings = {
                            "agent": "bbb",
                            "sigma_0": sigma_0,
                            "learning_rate": learning_rate,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "training_steps": training_steps,
                        }
                        config_ctor = make_bbb_ctor(
                            sigma_0,
                            learning_rate,
                            hidden_size,
                            num_layers,
                            training_steps=training_steps,
                            batch_size=1000 if reduce_batch else None,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_bbb_best_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting bbb agents
    for sigma_0 in [100]:
        for learning_rate in [1e-3]:
            for num_layers in [2]:
                for hidden_size in [50]:
                    for training_steps in [1000]:
                        settings = {
                            "agent": "bbb",
                            "sigma_0": sigma_0,
                            "learning_rate": learning_rate,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "training_steps": training_steps,
                        }
                        config_ctor = make_bbb_ctor(
                            sigma_0,
                            learning_rate,
                            hidden_size,
                            num_layers,
                            training_steps=training_steps,
                            batch_size=1000 if reduce_batch else None,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_bbb_debug(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting bbb agents
    for sigma_0 in [100]:
        for learning_rate in [1e-3]:
            for num_layers in [2]:
                for hidden_size in [50]:
                    for training_steps in [1000]:
                        settings = {
                            "agent": "bbb",
                            "sigma_0": sigma_0,
                            "learning_rate": learning_rate,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "training_steps": training_steps,
                        }
                        config_ctor = make_bbb_ctor(
                            sigma_0,
                            learning_rate,
                            hidden_size,
                            num_layers,
                            training_steps=training_steps,
                            batch_size=1000 if reduce_batch else None,
                            num_index_samples = 64,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_ensemble_sweep() -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting ensemble agents
    for num_ensemble in [3, 10, 30]:
    # for num_ensemble in [1, 3, 10, 30]:
        for noise_scale in [0, 1]:
            for prior_scale in [0, 1]:
                for num_layers in [2, 3]:
                    for hidden_size in [50]:
                        for learning_rate in [1e-3]:
                            # for training_steps in [1000, 1500, 2000, 3000]:
                            for training_steps in [1000]:
                                settings = {
                                    "agent": "ensemble",
                                    "num_ensemble": num_ensemble,
                                    "noise_scale": noise_scale,
                                    "prior_scale": prior_scale,
                                    "num_layers": num_layers,
                                    "hidden_size": hidden_size,
                                    "learning_rate": learning_rate,
                                    "training_steps": training_steps,
                                    "max_num_samples": num_ensemble,
                                }
                                config_ctor = make_ensemble_ctor(
                                    num_ensemble = num_ensemble,
                                    noise_scale = noise_scale,
                                    prior_scale = prior_scale,
                                    learning_rate = learning_rate,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    training_steps=training_steps,
                                    # batch_size=100,
                                )
                                sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_ensemble_best_sweep() -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting ensemble agents
    for num_ensemble in [30]:
        for noise_scale in [1]:
            for prior_scale in [1]:
                for num_layers in [2]:
                    for hidden_size in [50]:
                        for learning_rate in [1e-3]:
                            for training_steps in [1000]:
                                settings = {
                                    "agent": "ensemble",
                                    "num_ensemble": num_ensemble,
                                    "noise_scale": noise_scale,
                                    "prior_scale": prior_scale,
                                    "num_layers": num_layers,
                                    "hidden_size": hidden_size,
                                    "learning_rate": learning_rate,
                                    "training_steps": training_steps,
                                    "max_num_samples": num_ensemble,
                                }
                                config_ctor = make_ensemble_ctor(
                                    num_ensemble = num_ensemble,
                                    noise_scale = noise_scale,
                                    prior_scale = prior_scale,
                                    learning_rate = learning_rate,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    training_steps=training_steps,
                                    # batch_size=100,
                                )
                                sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_hypermodel_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting hypermodel agents
    for index_dim in [5, 10, 20]:
        for noise_scale in [0, 1]:
            for prior_scale in [0, 5]:
                for num_layers in [2, 3]:
                    for hidden_size in [50]:
                        settings = {
                            "agent": "hypermodel",
                            "index_dim": index_dim,
                            "noise_scale": noise_scale,
                            "prior_scale": prior_scale,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                        }
                        config_ctor = make_hypermodel_ctor(
                            index_dim, noise_scale, prior_scale, hidden_size, num_layers, num_index_samples=index_dim * 20,
                            batch_size=100 if reduce_batch else None,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_hypermodel_best_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting hypermodel agents
    for index_dim in [20]:
        for noise_scale in [1]:
            for prior_scale in [5]:
                for num_layers in [2]:
                    for hidden_size in [50]:
                        settings = {
                            "agent": "hypermodel",
                            "index_dim": index_dim,
                            "noise_scale": noise_scale,
                            "prior_scale": prior_scale,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                        }
                        config_ctor = make_hypermodel_ctor(
                            index_dim, noise_scale, prior_scale, hidden_size, num_layers, num_index_samples=index_dim * 20,
                            batch_size=100 if reduce_batch else None,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_vnn_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    for activation in ["lrelu", "relu", "tanh"]:
        for learning_rate in [1e-3, 1e-4, 5e-5]:
            for num_layers in [2, 3]:
                for hidden_size in [50, 100]:
                    for activation_mode in [
                        "mean",
                        "mean+std",
                        "mean+end",
                        "end",
                        "none",
                    ]:
                        for use_batch_norm in [False]:
                            for global_std_mode in ["none", "replace", "multiply"]:
                                for num_index_samples in [10, 100]:
                                    for training_steps in [1000, 3000, -1]:
                                        batch_norm_mode = activation_mode

                                        current_activation = {
                                            "relu": torch.nn.ReLU(),
                                            "tanh": torch.nn.Tanh(),
                                            "lrelu": torch.nn.LeakyReLU(),
                                        }[activation]

                                        if len(activation_mode.split("+")) > 1:
                                            current_activation = [
                                                current_activation
                                            ] * len(activation_mode.split("+"))

                                        settings = {
                                            "agent": "vnn",
                                            "activation": activation,
                                            "learning_rate": learning_rate,
                                            "num_layers": num_layers,
                                            "hidden_size": hidden_size,
                                            "activation_mode": activation_mode,
                                            "batch_norm_mode": batch_norm_mode,
                                            "use_batch_norm": use_batch_norm,
                                            "global_std_mode": global_std_mode,
                                            "training_steps": training_steps,
                                            "num_index_samples": num_index_samples,
                                        }
                                        config_ctor = make_vnn_ctor(
                                            current_activation,
                                            activation_mode,
                                            use_batch_norm,
                                            batch_norm_mode,
                                            global_std_mode,
                                            num_index_samples,
                                            hidden_size,
                                            training_steps=training_steps,
                                            batch_size=1000 if reduce_batch else None,
                                        )
                                        sweep.append(
                                            AgentCtorConfig(settings, config_ctor)
                                        )

    return sweep


def make_vnn_best_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    for activation in ["relu"]:
        for learning_rate in [1e-3]:
            for num_layers in [3]:
                for hidden_size in [100]:
                    for activation_mode in [
                        "mean",
                    ]:
                        for use_batch_norm in [False]:
                            for global_std_mode in ["multiply"]:
                                for num_index_samples in [100]:
                                    for training_steps in [-1]:
                                        batch_norm_mode = activation_mode

                                        current_activation = {
                                            "relu": torch.nn.ReLU(),
                                            "tanh": torch.nn.Tanh(),
                                            "lrelu": torch.nn.LeakyReLU(),
                                        }[activation]

                                        if len(activation_mode.split("+")) > 1:
                                            current_activation = [
                                                current_activation
                                            ] * len(activation_mode.split("+"))

                                        settings = {
                                            "agent": "vnn",
                                            "activation": activation,
                                            "learning_rate": learning_rate,
                                            "num_layers": num_layers,
                                            "hidden_size": hidden_size,
                                            "activation_mode": activation_mode,
                                            "batch_norm_mode": batch_norm_mode,
                                            "use_batch_norm": use_batch_norm,
                                            "global_std_mode": global_std_mode,
                                            "training_steps": training_steps,
                                            "num_index_samples": num_index_samples,
                                        }
                                        config_ctor = make_vnn_ctor(
                                            current_activation,
                                            activation_mode,
                                            use_batch_norm,
                                            batch_norm_mode,
                                            global_std_mode,
                                            num_index_samples,
                                            hidden_size,
                                            training_steps=training_steps,
                                            batch_size=1000 if reduce_batch else None,
                                        )
                                        sweep.append(
                                            AgentCtorConfig(settings, config_ctor)
                                        )

    return sweep


def make_layer_ensembles_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting ensemble agents
    # for num_ensemble in [2, 3]:
    for num_ensemble, inference_samples in [
        (2, ["full"]),
        (3, ["full"]),
        (5, ["full"]),
        (6, ["full"]),
        (8, ["full"]),
        (10, ["full"]),
    ]:
        for noise_scale in [1]:
            for prior_scale in [1]:
                for num_layers in [2]:
                    for hidden_size in [50]:
                        num_ensembles = [num_ensemble for _ in range(num_layers + 1)]
                        max_num_samples = reduce(lambda x, y: x * y, num_ensembles)

                        settings = {
                            "agent": "layer_ensembles",
                            "num_ensembles": num_ensemble,
                            "inference_samples": str(inference_samples),
                            "noise_scale": noise_scale,
                            "prior_scale": prior_scale,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "max_num_samples": max_num_samples,
                        }
                        config_ctor = make_layer_ensembles_ctor(
                            num_ensembles=num_ensembles,
                            noise_scale=noise_scale,
                            prior_scale=prior_scale,
                            learning_rate=1e-3,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            inference_samples=inference_samples,
                            training_steps=1000,
                            batch_size=1000 if reduce_batch else None,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep


def make_layer_ensembles_best_sweep(reduce_batch=False) -> List[AgentCtorConfig]:
    """Generates the benchmark sweep for paper results."""
    sweep = []

    # Adding reasonably interesting ensemble agents
    # for num_ensemble in [2, 3]:
    for num_ensemble, inference_samples in [
        (5, ["full"]),
    ]:
        for noise_scale in [1]:
            for prior_scale in [1]:
                for num_layers in [2]:
                    for hidden_size in [50]:
                        num_ensembles = [num_ensemble for _ in range(num_layers + 1)]
                        max_num_samples = reduce(lambda x, y: x * y, num_ensembles)

                        settings = {
                            "agent": "layer_ensembles",
                            "num_ensembles": num_ensemble,
                            "inference_samples": str(inference_samples),
                            "noise_scale": noise_scale,
                            "prior_scale": prior_scale,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "max_num_samples": max_num_samples,
                        }
                        config_ctor = make_layer_ensembles_ctor(
                            num_ensembles=num_ensembles,
                            noise_scale=noise_scale,
                            prior_scale=prior_scale,
                            learning_rate=1e-3,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            inference_samples=inference_samples,
                            training_steps=1000,
                            batch_size=1000 if reduce_batch else None,
                        )
                        sweep.append(AgentCtorConfig(settings, config_ctor))

    return sweep



def make_agent_sweep(agent: str = "all", reduce_batch=False) -> Sequence[AgentCtorConfig]:

    if agent == "all":
        agent_sweep = make_dropout_sweep() + make_bbb_sweep() + make_ensemble_sweep() + make_hypermodel_sweep()
    if agent == "all_best":
        agent_sweep = make_dropout_best_sweep() +make_bbb_best_sweep(reduce_batch=reduce_batch) + make_ensemble_best_sweep() + make_hypermodel_best_sweep(reduce_batch=reduce_batch)
    elif agent == "dropout":
        agent_sweep = make_dropout_sweep()
    elif agent == "dropout_best":
        agent_sweep = make_dropout_best_sweep()
    elif agent == "bbb":
        agent_sweep = make_bbb_selected_sweep(reduce_batch=reduce_batch)
    elif agent == "bbb_best":
        agent_sweep = make_bbb_best_sweep(reduce_batch=reduce_batch)
    elif agent == "bbb_debug":
        agent_sweep = make_bbb_debug()
    elif agent == "ensemble":
        agent_sweep = make_ensemble_sweep()
    elif agent == "ensemble_best":
        agent_sweep = make_ensemble_best_sweep()
    elif agent == "hypermodel":
        agent_sweep = make_hypermodel_sweep()
    elif agent == "hypermodel_best":
        agent_sweep = make_hypermodel_best_sweep(reduce_batch=reduce_batch)
    elif agent == "vnn":
        agent_sweep = make_vnn_sweep(reduce_batch=reduce_batch)
    elif agent == "vnn_best":
        agent_sweep = make_vnn_best_sweep(reduce_batch=reduce_batch)
    elif agent == "layer_ensembles":
        agent_sweep = make_layer_ensembles_sweep(reduce_batch=reduce_batch)
    elif agent == "layer_ensembles_best":
        agent_sweep = make_layer_ensembles_best_sweep(reduce_batch=reduce_batch)
    else:
        raise ValueError(f"agent={agent} is not valid!")

    return tuple(agent_sweep)


def load_agent_config(agent_id: int, agent: str = "all", reduce_batch=False) -> agents.VanillaEnnConfig:
    sweep = make_agent_sweep(agent, reduce_batch=reduce_batch)
    return sweep[agent_id]


def load_agent_config_sweep(agent: str = "all", reduce_batch=False):
    sweep = make_agent_sweep(agent, reduce_batch=reduce_batch)
    return sweep
