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

from src.experiments import agent_factories
from src.experiments import agents
from src.experiments import load
from src.experiments.seeds import split_seed
import os
import torch
from src import torch_repr


def main(
    input_dim=(1, 10, 100),
    data_ratio=(1.0, 10.0, 100.0),
    noise_std=(0.01, 0.1, 1.0),
    max_num_samples=(10, 100, 1000),
    seed=1,
    agent_id_start=0,
    agent_id_end=-1,
    agent_name="all",
    experiment_group="",
    device="cuda:0",
    results_folder="results",
    use_double_precision=False,
    test_random_set_count=10,
):
    """Run testbed sweep.

    Args:
        input_dim: Iterable of input dimensions.
        data_ratio: Iterable of ratios of num_train to input_dim.
        noise_std: Iterable of additive noise std deviations.
        max_num_samples: Iterable of maximum number of samples.
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
    if isinstance(max_num_samples, int):
        max_num_samples = [max_num_samples]
    if isinstance(data_ratio, float) or isinstance(data_ratio, int):
        data_ratio = [float(data_ratio)]
    if isinstance(noise_std, float) or isinstance(noise_std, int):
        noise_std = [float(noise_std)]

    for ind in input_dim:
        for dr in data_ratio:
            for ns in noise_std:
                # Load the appropriate testbed problem
                problem = load.regression_load(
                    input_dim=ind,
                    data_ratio=dr,
                    seed=seed,
                    noise_std=ns,
                    use_double_precision=use_double_precision,
                )

                all_results = []

                sweep = agent_factories.load_agent_config_sweep(agent_name)
                sweep = (
                    sweep[agent_id_start:]
                    if agent_id_end == -1
                    else sweep[agent_id_start:agent_id_end]
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
                    agent = agents.VanillaEnnAgent(agent_config.config_ctor(), use_double_precision=use_double_precision, fixed_sampler=True)

                    train_seed, all_indices_seed, best_samples_seed, random_set_evaluation_seed = split_seed(agent_seed, 4)

                    # Train
                    (enn_sampler_fixed, enn_sampler_free), val_loss, val_kl = agent(
                        problem.train_data,
                        train_seed,
                        problem.prior_knowledge,
                        device=device,
                        val_data=problem.val_data,
                        evaluate_quality_val_fn=problem.evaluate_quality_val,
                    )

                    if agent_config.settings["agent"] in ["layer_ensembles", "ensemble"]:
                        mns_list = [agent_config.settings["max_num_samples"]]
                    else:
                        mns_list = [*max_num_samples]

                    for max_samples in mns_list:

                        # Random sets
                        for random_samples_count in range(2, max_samples):

                            random_set_evaluation_seed, current_set_seed = split_seed(random_set_evaluation_seed, 2)

                            random_set_kls = []

                            for _ in range(test_random_set_count):

                                current_set_seed, evaluation_seed = split_seed(current_set_seed, 2)

                                random_set_kl_quality = problem.evaluate_quality(
                                    enn_sampler_free, seed=evaluation_seed, num_samples=random_samples_count, device=device
                                )

                                random_set_kls.append(random_set_kl_quality.kl_estimate)
                            
                            mean_random_set_kl = sum(random_set_kls) / len(random_set_kls)
                            var_random_set_kl = sum((x - mean_random_set_kl) ** 2 for x in random_set_kls) / len(random_set_kls)
                            
                            with open(
                                f"{results_folder}/{agent_name}/random_set_results_"
                                + experiment_group
                                + ("_" if len(experiment_group) > 0 else "")
                                + agent_name
                                + "_id"
                                + str(ind)
                                + "dr"
                                + str(dr)
                                + "ns"
                                + str(ns)
                                + "mns"
                                + str(max_samples)
                                + "_kl.txt",
                                "a",
                            ) as f:

                                f.write(
                                    str(agent_id)
                                    + " "
                                    + str(mean_random_set_kl)
                                    + " "
                                    + "kl_variance="
                                    + str(var_random_set_kl)
                                    + " "
                                    + "indexer="
                                    + str(random_samples_count)
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

                        all_indices = agent.create_all_indices(max_samples, seed=all_indices_seed, device=device)

                        # Best sets based on kl
                        best_samples_kl = problem.find_best_samples(
                            enn_sampler_fixed, all_indices, seed=best_samples_seed, noise_std=ns, device=device, use_log_likelihood=False
                        )

                        for samples, best_kl_dict in best_samples_kl.items():
                            kl_quality = best_kl_dict["best_kl"]
                            val_kl_quality = best_kl_dict["best_val_kl"]
                            print(
                                f"kl_estimate={kl_quality.kl_estimate}"
                                + " val_kl="
                                + str(val_kl_quality.kl_estimate)
                                + " mean_error="
                                + str(kl_quality.extra["mean_error"])
                                + " "
                                + "std_error="
                                + str(kl_quality.extra["std_error"])
                            )
                            all_results.append(kl_quality)

                            with open(
                                f"{results_folder}/{agent_name}/results_"
                                + experiment_group
                                + ("_" if len(experiment_group) > 0 else "")
                                + agent_name
                                + "_id"
                                + str(ind)
                                + "dr"
                                + str(dr)
                                + "ns"
                                + str(ns)
                                + "mns"
                                + str(max_samples)
                                + "_kl.txt",
                                "a",
                            ) as f:

                                f.write(
                                    str(agent_id)
                                    + " "
                                    + str(kl_quality.kl_estimate)
                                    + " "
                                    + "val_kl="
                                    + str(val_kl_quality.kl_estimate)
                                    + " "
                                    + "mean_error="
                                    + str(kl_quality.extra["mean_error"])
                                    + " "
                                    + "std_error="
                                    + str(kl_quality.extra["std_error"])
                                    + " "
                                    + "best_epoch="
                                    + str(agent.best_epoch)
                                    + " "
                                    + "indexer="
                                    + str(samples)
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

                        # Best sets based on ll
                        best_samples_ll = problem.find_best_samples(
                            enn_sampler_fixed, all_indices, seed=best_samples_seed, noise_std=ns, device=device, use_log_likelihood=True
                        )

                        for samples, best_kl_dict in best_samples_ll.items():
                            kl_quality = best_kl_dict["best_kl"]
                            val_ll = best_kl_dict["best_val_log_likelihood"]
                            print(
                                f"kl_estimate={kl_quality.kl_estimate}"
                                + " val_ll="
                                + str(val_ll)
                                + " mean_error="
                                + str(kl_quality.extra["mean_error"])
                                + " "
                                + "std_error="
                                + str(kl_quality.extra["std_error"])
                            )
                            all_results.append(kl_quality)

                            with open(
                                f"{results_folder}/{agent_name}/results_"
                                + experiment_group
                                + ("_" if len(experiment_group) > 0 else "")
                                + agent_name
                                + "_id"
                                + str(ind)
                                + "dr"
                                + str(dr)
                                + "ns"
                                + str(ns)
                                + "mns"
                                + str(max_samples)
                                + "_ll.txt",
                                "a",
                            ) as f:

                                f.write(
                                    str(agent_id)
                                    + " "
                                    + str(kl_quality.kl_estimate)
                                    + " "
                                    + "val_ll="
                                    + str(val_ll)
                                    + " "
                                    + "mean_error="
                                    + str(kl_quality.extra["mean_error"])
                                    + " "
                                    + "std_error="
                                    + str(kl_quality.extra["std_error"])
                                    + " "
                                    + "best_epoch="
                                    + str(agent.best_epoch)
                                    + " "
                                    + "indexer="
                                    + str(samples)
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



if __name__ == "__main__":
    fire.Fire(main)
