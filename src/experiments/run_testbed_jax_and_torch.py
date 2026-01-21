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
"""Example running an ENN on Thompson bandit task (Fire CLI)."""

import fire
import jax

from src.experiments import agent_factories as torch_agent_factories
from src.experiments import agents as torch_agents
from src.experiments import load as torch_load
from src.experiments import testbed as torch_testbed
from src.experiments.seeds import split_seed
import os
from enn.experiments.neurips_2021 import agent_factories as jax_agent_factories
from enn.experiments.neurips_2021 import agents as jax_agents
from enn.experiments.neurips_2021 import load as jax_load
from jax.config import config
import numpy as np
from src.experiments.base import Data

from src.experiments import agents_jax_and_torch as jax_and_torch_agents
from src.experiments.testbed import _kl_gaussian as torch_kl_gaussian, ENNQuality
from enn.experiments.neurips_2021.testbed import _kl_gaussian as jax_kl_gaussian

import jax.numpy as jnp
import jax


import torch
from src import torch_repr

def jax_problem_to_torch(jax_problem, seed, device, use_double_precision):
    """Convert a JAX testbed problem to a Torch one."""

    data_sampler = torch_testbed.GPRegression.restore(
        
        tau=jax_problem.data_sampler._tau,
        input_dim = jax_problem.data_sampler._input_dim,
        x_train=torch.tensor(np.array(jax_problem.data_sampler._x_train), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        x_test=torch.tensor(np.array(jax_problem.data_sampler._x_test), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        x_val=torch.tensor(np.array(jax_problem.data_sampler._x_val), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        train_y = torch.tensor(np.array(jax_problem.train_data.y), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        test_mean = torch.tensor(np.array(jax_problem.data_sampler._test_mean), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        test_cov = torch.tensor(np.array(jax_problem.data_sampler._test_cov), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        val_mean = torch.tensor(np.array(jax_problem.data_sampler._val_mean), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        val_cov = torch.tensor(np.array(jax_problem.data_sampler._val_cov), device=device, dtype=torch.float64 if use_double_precision else torch.float32),
        noise_std=jax_problem.data_sampler._noise_std,
        kernel_ridge=jax_problem.data_sampler._kernel_ridge,
    )

    result = torch_testbed.TestbedGPRegression(
        data_sampler=data_sampler,
        prior=jax_problem.prior_knowledge,
        num_enn_samples=jax_problem.num_enn_samples,
        std_ridge=jax_problem.std_ridge,
    )
    
    return result

def evaluate_quality_jax_and_torch(
    jax_and_torch_enn_sampler,
    jax_data_sampler,
    torch_data_sampler,
    seed,
    num_samples,
    device: str = "cuda:0",
    std_ridge = 1e-3,
):
    """Computes KL estimate on mean functions for tau=1 only."""

    torch_x_test = torch_data_sampler.x_test.to(device)
    torch_num_test = torch_x_test.shape[0]
    torch_posterior_mean = torch_data_sampler.test_mean[:, 0].to(device)
    torch_posterior_std = torch.sqrt(torch.diag(torch_data_sampler.test_cov)).to(device)
    torch_posterior_std += std_ridge

    jax_x_test = jax_data_sampler.x_test
    jax_num_test = jax_x_test.shape[0]
    jax_posterior_mean = jax_data_sampler.test_mean[:, 0]
    jax_posterior_std = jnp.sqrt(jnp.diag(jax_data_sampler.test_cov))
    jax_posterior_std += std_ridge

    jax_enn_samples, torch_enn_samples = jax_and_torch_enn_sampler(
        jax_x_test,
        torch_x_test,
        seed,
        num_samples,
    )
    torch_enn_samples = torch_enn_samples.squeeze(-1)
    # torch_enn_samples = torch_enn_sampler(torch_x_test, seed, num_samples).squeeze(-1)

    assert torch_enn_samples.shape == (num_samples, torch_num_test)
    torch_enn_mean = torch.mean(torch_enn_samples, dim=0)
    torch_enn_std = torch.std(torch_enn_samples, dim=0) + std_ridge

    # # Compute the mean and std of ENN posterior
    # jax_batched_sampler = jax.jit(jax.vmap(jax_enn_sampler, in_axes=[None, 0]))
    # jax_enn_samples = jax_batched_sampler(jax_x_test, jnp.arange(num_samples))
    jax_enn_samples = jax_enn_samples[:, :, 0]
    jax_enn_mean = jnp.mean(jax_enn_samples, axis=0)
    jax_enn_std = jnp.std(jax_enn_samples, axis=0) + std_ridge

    torch_kl_estimates = torch.stack(
        [
            torch_kl_gaussian(
                torch_posterior_mean[i], torch_posterior_std[i], torch_enn_mean[i], torch_enn_std[i]
            )
            for i in range(torch_num_test)
        ]
    )
    torch_kl_estimate = torch.mean(torch_kl_estimates)

    
    # Compute the KL divergence between this and reference posterior
    jax_batched_kl = jax.jit(jax.vmap(jax_kl_gaussian))
    jax_kl_estimates = jax_batched_kl(jax_posterior_mean, jax_posterior_std, jax_enn_mean, jax_enn_std)
    jax_kl_estimate = jnp.mean(jax_kl_estimates)

    torch_error_mean = torch.mean(torch.abs((torch_posterior_mean - torch_enn_mean) / torch_posterior_mean))
    torch_error_std = torch.mean(torch.abs((torch_posterior_std - torch_enn_std) / torch_posterior_std))

    torch_result = ENNQuality(
        torch_kl_estimate.item(),
        {"mean_error": torch_error_mean.item(), "std_error": torch_error_std.item()},
    )

    jax_error_mean = jnp.mean(jnp.abs((jax_posterior_mean - jax_enn_mean) / jax_posterior_mean))
    jax_error_std = jnp.mean(jnp.abs((jax_posterior_std - jax_enn_std) / jax_posterior_std))

    jax_result = ENNQuality(
        jax_kl_estimate,
        {
            "mean_error": jax_error_mean,
            "std_error": jax_error_std,
        },
    )

    print(
        f"combined jax kl_estimate={jax_result.kl_estimate}"
        + " mean_error="
        + str(jax_result.extra["mean_error"])
        + " "
        + "std_error="
        + str(jax_result.extra["std_error"])
    )
    print(
        f"combined torch kl_estimate={torch_result.kl_estimate}"
        + " mean_error="
        + str(torch_result.extra["mean_error"])
        + " "
        + "std_error="
        + str(torch_result.extra["std_error"])
    )

    return jax_result, torch_result
def main(
    input_dim=(1, 10, 100),
    data_ratio=(1.0, 10.0, 100.0),
    noise_std=(0.01, 0.1, 1.0),
    seed=1,
    agent_id_start=0,
    agent_id_end=-1,
    agent_name="all",
    experiment_group="",
    device="cuda:0",
    results_folder="results",
    use_double_precision=False,
):
    """Run testbed sweep.

    Args:
        input_dim: Iterable of input dimensions.
        data_ratio: Iterable of ratios of num_train to input_dim.
        noise_std: Iterable of additive noise std deviations.
        seed: Seed for testbed problem.
        agent_id_start: Start index for agent sweep.
        agent_id_end: End index for agent sweep (-1 means until end).
        agent: Which agent family to use.
        experiment_group: Name of the experiment group.
    """

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/{agent_name}", exist_ok=True)

    if isinstance(input_dim, int):
        input_dim = [input_dim]
    if isinstance(data_ratio, float) or isinstance(data_ratio, int):
        data_ratio = [float(data_ratio)]
    if isinstance(noise_std, float) or isinstance(noise_std, int):
        noise_std = [float(noise_std)]

    for ind in input_dim:
        for dr in data_ratio:
            for ns in noise_std:
                # Load the appropriate testbed problem
                # torch_problem = torch_load.regression_load(
                #     input_dim=ind,
                #     data_ratio=dr,
                #     seed=seed,
                #     noise_std=ns,
                # )
                jax_problem = jax_load.regression_load(
                    input_dim=ind,
                    data_ratio=dr,
                    seed=seed,
                    noise_std=ns,
                )

                torch_problem = jax_problem_to_torch(jax_problem, seed, device, use_double_precision)

                all_results = []

                torch_sweep = torch_agent_factories.load_agent_config_sweep(agent_name)
                torch_sweep = (
                    torch_sweep[agent_id_start:]
                    if agent_id_end == -1
                    else torch_sweep[agent_id_start:agent_id_end]
                )
                jax_sweep = jax_agent_factories.load_agent_config_sweep(agent_name)
                jax_sweep = (
                    jax_sweep[agent_id_start:]
                    if agent_id_end == -1
                    else jax_sweep[agent_id_start:agent_id_end]
                )

                agent_seeds = split_seed(seed, len(torch_sweep))

                for i, (torch_agent_config, jax_agent_config, agent_seed) in enumerate(zip(torch_sweep, jax_sweep, agent_seeds)):

                    agent_id = agent_id_start + i

                    print(
                        "input_dim",
                        ind,
                        "data_ratio",
                        dr,
                        "noise_std",
                        ns,
                    )
                    print("agent_id", agent_id, "of", len(torch_sweep))

                    # Form the appropriate agent for training
                    # torch_agent = torch_agents.VanillaEnnAgent(torch_agent_config.config_ctor(), use_double_precision=True)
                    jax_agent = jax_agents.VanillaEnnAgent(jax_agent_config.config_ctor())
                    jax_and_torch_agent = jax_and_torch_agents.JaxAndTorchVanillaEnnAgent(
                        jax_config=jax_agent_config.config_ctor(),
                        torch_config=torch_agent_config.config_ctor(),
                        use_double_precision=use_double_precision,
                    )

                    log_file_name = (
                        "single_runs/single_run_"
                        + experiment_group
                        + ("_" if len(experiment_group) > 0 else "")
                        + agent_name
                        + "_aid"
                        + str(agent_id)
                        + "_id"
                        + str(input_dim)
                        + "dr"
                        + str(data_ratio)
                        + "ns"
                        + str(noise_std)
                        + "sd"
                        + str(seed)
                        + ".txt"
                    )

                    train_seed, evaluation_seed = split_seed(agent_seed, 2)

                    # single_jax_enn_sampler = jax_agent(
                    #     jax_problem.train_data,
                    #     jax_problem.prior_knowledge,
                    #     jax_problem.evaluate_quality_val,
                    #     log_file_name,
                    # )

                    # torch_enn_sampler = torch_agent(
                    #     torch_problem.train_data,
                    #     train_seed,
                    #     torch_problem.prior_knowledge,
                    #     device=device,
                    # )

                    jax_enn_sampler, torch_enn_sampler, jax_and_torch_sampler = jax_and_torch_agent(
                        jax_problem.train_data,
                        torch_problem.train_data,
                        train_seed,
                        jax_problem.prior_knowledge,
                        torch_problem.prior_knowledge,
                        device=device,
                    )

                    # # Evaluate the quality of the ENN sampler after training
                    # single_jax_kl_quality = jax_problem.evaluate_quality(
                    #     single_jax_enn_sampler
                    # )
                    # external_jax_kl_quality = jax_problem.evaluate_quality(
                    #     external_jax_enn_sampler
                    # )
                    jax_kl_quality = jax_problem.evaluate_quality(
                        jax_enn_sampler
                    )
                    torch_kl_quality = torch_problem.evaluate_quality(
                        torch_enn_sampler, seed=evaluation_seed, device=device
                    )

                    combined_jax_quality, combined_torch_quality = evaluate_quality_jax_and_torch(
                        jax_and_torch_sampler,
                        # jax_enn_sampler,
                        # torch_enn_sampler,
                        jax_problem.data_sampler,
                        torch_problem.data_sampler,
                        evaluation_seed,
                        jax_problem.num_enn_samples,
                        device=device,
                        std_ridge=jax_problem.std_ridge,
                    )

                    # print(
                    #     f"single jax kl_estimate={single_jax_kl_quality.kl_estimate}"
                    #     + " mean_error="
                    #     + str(single_jax_kl_quality.extra["mean_error"])
                    #     + " "
                    #     + "std_error="
                    #     + str(single_jax_kl_quality.extra["std_error"])
                    # )
                    # all_results.append(single_jax_kl_quality)
                    # print(
                    #     f"external jax kl_estimate={external_jax_kl_quality.kl_estimate}"
                    #     + " mean_error="
                    #     + str(external_jax_kl_quality.extra["mean_error"])
                    #     + " "
                    #     + "std_error="
                    #     + str(external_jax_kl_quality.extra["std_error"])
                    # )
                    # all_results.append(external_jax_kl_quality)
                    print(
                        f"jax kl_estimate={jax_kl_quality.kl_estimate}"
                        + " mean_error="
                        + str(jax_kl_quality.extra["mean_error"])
                        + " "
                        + "std_error="
                        + str(jax_kl_quality.extra["std_error"])
                    )
                    all_results.append(jax_kl_quality)
                    print(
                        f"torch kl_estimate={torch_kl_quality.kl_estimate}"
                        + " mean_error="
                        + str(torch_kl_quality.extra["mean_error"])
                        + " "
                        + "std_error="
                        + str(torch_kl_quality.extra["std_error"])
                    )
                    all_results.append(torch_kl_quality)
                    print(
                        f"combined jax kl_estimate={combined_jax_quality.kl_estimate}"
                        + " mean_error="
                        + str(combined_jax_quality.extra["mean_error"])
                        + " "
                        + "std_error="
                        + str(combined_jax_quality.extra["std_error"])
                    )
                    all_results.append(combined_jax_quality)
                    print(
                        f"combined torch kl_estimate={combined_torch_quality.kl_estimate}"
                        + " mean_error="
                        + str(combined_torch_quality.extra["mean_error"])
                        + " "
                        + "std_error="
                        + str(combined_torch_quality.extra["std_error"])
                    )
                    all_results.append(combined_torch_quality)

                    with open(
                        f"{results_folder}/{agent_name}/results_jax_and_torch_"
                        + experiment_group
                        + ("_" if len(experiment_group) > 0 else "")
                        + agent_name
                        + "_id"
                        + str(ind)
                        + "dr"
                        + str(dr)
                        + "ns"
                        + str(ns)
                        + ".txt",
                        "a",
                    ) as f:

                        f.write(
                            str(agent_id)
                            + " "
                            + str(jax_kl_quality.kl_estimate)
                            + " "
                            + "mean_error="
                            + str(jax_kl_quality.extra["mean_error"])
                            + " "
                            + "std_error="
                            + str(jax_kl_quality.extra["std_error"])
                            + " "
                            + " ".join(
                                [
                                    str(k) + "=" + str(v)
                                    for (
                                        k,
                                        v,
                                    ) in jax_agent_config.settings.items()
                                ]
                            )
                            + "\n"
                        )
                        f.write(
                            str(agent_id)
                            + " "
                            + str(torch_kl_quality.kl_estimate)
                            + " "
                            + "mean_error="
                            + str(torch_kl_quality.extra["mean_error"])
                            + " "
                            + "std_error="
                            + str(torch_kl_quality.extra["std_error"])
                            + " "
                            + " ".join(
                                [
                                    str(k) + "=" + str(v)
                                    for (
                                        k,
                                        v,
                                    ) in torch_agent_config.settings.items()
                                ]
                            )
                            + "\n"
                        )

                print(all_results)


if __name__ == "__main__":
    fire.Fire(main)
