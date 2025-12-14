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


def single_run(ind, dr, ns, seed, agent_id_start, agent_id_end, agent_name, experiment_group, device):
    
    dr = float(dr)
    ns = float(ns)
    
    # Load the appropriate testbed problem
    problem = load.regression_load(
        input_dim=ind,
        data_ratio=dr,
        seed=seed,
        noise_std=ns,
    )

    all_results = []

    sweep = agent_factories.load_agent_config_sweep(agent_name)
    sweep = (
        sweep[agent_id_start :]
        if agent_id_end == -1
        else sweep[agent_id_start : agent_id_end]
    )

    agent_seeds = split_seed(seed, len(sweep))

    for i, (agent_config, agent_seed) in enumerate(zip(sweep, agent_seeds)):

        agent_id = agent_id_start + i

        print(
            "input_dim",
            ind,
            "data_ratio",
            dr,
            "noise_std",
            ns,
        )
        print("agent_id", agent_id, "of", len(sweep))

        # Form the appropriate agent for training
        agent = agents.VanillaEnnAgent(agent_config.config_ctor())

        # Train
        enn_sampler = agent(
            problem.train_data, agent_seed, problem.prior_knowledge, device=device
        )

        # Evaluate the quality of the ENN sampler after training
        kl_quality = problem.evaluate_quality(enn_sampler)
        print(
            f"kl_estimate={kl_quality.kl_estimate}"
            + " mean_error="
            + str(kl_quality.extra["mean_error"])
            + " "
            + "std_error="
            + str(kl_quality.extra["std_error"])
        )
        all_results.append(kl_quality)

        with open(
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
            + ".txt",
            "a",
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

    print(all_results)


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
    processes=30,
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

    pool = Pool(processes)

    for ind in input_dim:
        for dr in data_ratio:
            for ns in noise_std:
                print(
                    "input_dim",
                    input_dim,
                    "data_ratio",
                    data_ratio,
                    "noise_std",
                    noise_std,
                )
                pool.apply_async(
                    single_run,
                    args=(ind, dr, ns, seed, agent_id_start, agent_id_end, agent_name, experiment_group, device),
                )
    pool.close()
    pool.join()
    print("Finished all runs")                


if __name__ == "__main__":
    fire.Fire(main)
