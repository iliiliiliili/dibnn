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

from enn_pytorch.experiments.neurips_2021 import agent_factories
from enn_pytorch.experiments.neurips_2021 import agents
from enn_pytorch.experiments.neurips_2021 import load
from enn_pytorch.experiments.neurips_2021.random import split_seed
import os
from multiprocessing import Pool
import random


def single_run(ind, dr, ns, seed, agent_id, agent_seed, agent_name, experiment_group, device):

    dr = float(dr)
    ns = float(ns)

    # Load the appropriate testbed problem
    problem = load.regression_load(
        input_dim=ind,
        data_ratio=dr,
        seed=seed,
        noise_std=ns,
        # device=device,
    )

    # print(
    #     "input_dim",
    #     ind,
    #     "data_ratio",
    #     dr,
    #     "noise_std",
    #     ns,
    #     "agent_id", agent_id,
    # )

    agent_config = agent_factories.load_agent_config(agent_id, agent_name)

    # Form the appropriate agent for training
    agent = agents.VanillaEnnAgent(agent_config.config_ctor())

    # Train
    enn_sampler = agent(
        problem.train_data, agent_seed, problem.prior_knowledge, device=device, logging="none"
    )

    # Evaluate the quality of the ENN sampler after training
    kl_quality = problem.evaluate_quality(enn_sampler, device=device)

    print("#", end="")

    # print(
    #     f"kl_estimate={kl_quality.kl_estimate}"
    #     + " mean_error="
    #     + str(kl_quality.extra["mean_error"])
    #     + " "
    #     + "std_error="
    #     + str(kl_quality.extra["std_error"])
    # )

    result_folder = (
        "results/results_"
        + experiment_group
        + ("_" if len(experiment_group) > 0 else "")
        + agent_name
        + "_id"
        + str(ind)
        + "dr"
        + str(dr)
        + "ns"
        + str(ns)
    )

    os.makedirs(result_folder, exist_ok=True)

    with open(
        result_folder
        + "/agentid"
        + str(agent_id)
        + ".txt",
        "w",
    ) as f:

        f.write(
            str(agent_id)
            + " "
            + str(kl_quality.kl_estimate)
            + " "
            + "mean_error="
            + str(kl_quality.extra["mean_error"])
            + " "
            + "std_error="
            + str(kl_quality.extra["std_error"])
            + " "
            + " ".join(
                [
                    str(k) + "=" + str(v)
                    for (
                        k,
                        v,
                    ) in agent_config.settings.items()
                ]
            )
            + "\n"
        )


def run_experiments(experiments, devices, processes_per_device, debug=False):
    print(f"Starting {len(experiments)} experiments")
    
    if not isinstance(devices, list):
        devices = [f"cuda:{d}" for d in range(devices)]

    experiments_per_device = [0 for _ in devices]

    experiments_to_assign = len(experiments)

    for i in range(len(experiments_per_device)):
        experiments_per_device[i] = min(experiments_to_assign, processes_per_device)
        experiments_to_assign -= experiments_per_device[i]

    while experiments_to_assign > 0:
        for i in range(len(experiments_per_device)):
            experiments_per_device[i] += 1
            experiments_to_assign -= 1

            if experiments_to_assign <= 0:
                break

    pools = [Pool(processes_per_device) for _ in devices]

    for exp_count, device, pool in zip(experiments_per_device, devices, pools):

        for i in range(exp_count):
            experiment_args = experiments.pop(0)

            if debug:
                print("DEBUG")
                print("DEBUG")
                print("DEBUG")
                single_run(
                    *experiment_args,
                    device=device,
                )
            else:
                # pool.apply(
                pool.apply_async(
                    single_run,
                    args=(*experiment_args,),
                    kwds={"device": device},
                    error_callback=lambda e, args=experiment_args: open(
                        "./errors.log", "a"
                    ).write(
                        f"Error: {str(e)}\nArgs: {args}\n"
                    ),
                )

    for pool in pools:
        pool.close()
        pool.join()

    print("Done")


def combine_results(experiment_group: str, agent_name: str, input_dims: list, data_ratios: list, noise_stds: list, agent_id_start: int, agent_id_end: int):
    """Combine results from multiple runs into single files."""
    for ind in input_dims:
        for dr in data_ratios:
            for ns in noise_stds:
                result_folder = (
                    "results/results_"
                    + experiment_group
                    + ("_" if len(experiment_group) > 0 else "")
                    + agent_name
                    + "_id"
                    + str(ind)
                    + "dr"
                    + str(dr)
                    + "ns"
                    + str(ns)
                )

                combined_filepath = result_folder + ".txt"

                with open(combined_filepath, "w") as combined_file:
                    for agent_id in range(agent_id_start, agent_id_end + 1):
                        agent_filepath = (
                            result_folder
                            + "/agentid"
                            + str(agent_id)
                            + ".txt"
                        )
                        if os.path.exists(agent_filepath):
                            with open(agent_filepath, "r") as agent_file:
                                combined_file.write(agent_file.read())


def main(
    input_dim=(1, 10, 100),
    data_ratio=(1.0, 10.0, 100.0),
    noise_std=(0.01, 0.1, 1.0),
    seed=1,
    agent_id_start=0,
    agent_id_end=-1,
    agent_name="all",
    experiment_group="",
    devices=2,
    processes_per_device=6,
    debug=False,
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

    os.makedirs("results", exist_ok=True)

    if isinstance(input_dim, int):
        input_dim = [input_dim]
    if isinstance(data_ratio, float) or isinstance(data_ratio, int):
        data_ratio = [float(data_ratio)]
    if isinstance(noise_std, float) or isinstance(noise_std, int):
        noise_std = [float(noise_std)]

    experiments = []

    for ind in input_dim:
        for dr in data_ratio:
            for ns in noise_std:

                problem = load.regression_load(
                    input_dim=ind,
                    data_ratio=dr,
                    seed=seed,
                    noise_std=ns,
                )

                print("Created problem for ind", ind, "dr", dr, "ns", ns)

                sweep = agent_factories.load_agent_config_sweep(agent_name)
                sweep = (
                    sweep[agent_id_start :]
                    if agent_id_end == -1
                    else sweep[agent_id_start : agent_id_end]
                )

                agent_seeds = split_seed(seed, len(sweep))

                for i, (agent_config, agent_seed) in enumerate(zip(sweep, agent_seeds)):

                    agent_id = agent_id_start + i

                    arguments = (
                        ind,
                        dr,
                        ns,
                        seed,
                        agent_id,
                        agent_seed,
                        agent_name,
                        experiment_group,
                    )

                    experiments.append(arguments)


    random.shuffle(experiments)

    run_experiments(experiments, devices, processes_per_device=processes_per_device, debug=debug)
    print("Finished all runs")

    max_agent_id = (
        agent_id_end
        if agent_id_end != -1
        else len(agent_factories.load_agent_config_sweep(agent_name))
    )

    combine_results(
        experiment_group,
        agent_name,
        input_dim,
        data_ratio,
        noise_std,
        agent_id_start,
        max_agent_id,
    )

    print("Combined results")


if __name__ == "__main__":
    fire.Fire(main)
