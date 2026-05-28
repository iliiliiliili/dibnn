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

from typing import List
import fire
from tqdm import tqdm

from src.experiments import agent_factories
from src.experiments import agents
from src.experiments import load
from src.experiments.seeds import split_seed
import os
from multiprocessing import Pool
import random
from src.utils import read_results_file
from src import torch_repr


def single_run(
    ind,
    dr,
    ns,
    seed,
    agent_id,
    agent_seed,
    agent_name,
    experiment_group,
    results_file,
    use_double_precision,
    reduce_batch,
    device,
):

    dr = float(dr)
    ns = float(ns)

    if os.path.exists(results_file):
        print(".", end="")
        return

    # Load the appropriate testbed problem
    problem = load.regression_load(
        input_dim=ind,
        data_ratio=dr,
        seed=seed,
        noise_std=ns,
        use_double_precision=use_double_precision,
    )

    agent_config = agent_factories.load_agent_config(agent_id, agent_name, reduce_batch=reduce_batch)

    # Form the appropriate agent for training
    agent = agents.VanillaEnnAgent(agent_config.config_ctor(), use_double_precision=use_double_precision)

    train_seed, evaluation_seed, _ = split_seed(agent_seed, 3)

    # Train
    enn_sampler, val_loss, val_kl = agent(
        problem.train_data,
        train_seed,
        problem.prior_knowledge,
        device=device,
        logging="none",
        val_data=problem.val_data,
        evaluate_quality_val_fn=problem.evaluate_quality_val,
    )

    # Evaluate the quality of the ENN sampler after training
    kl_quality = problem.evaluate_quality(
        enn_sampler, seed=evaluation_seed, device=device
    )

    print("#", end="", flush=True)

    # print(
    #     f"kl_estimate={kl_quality.kl_estimate}"
    #     + " mean_error="
    #     + str(kl_quality.extra["mean_error"])
    #     + " "
    #     + "std_error="
    #     + str(kl_quality.extra["std_error"])
    # )

    with open(
        results_file,
        "w",
    ) as f:

        f.write(
            str(agent_id)
            + " "
            + str(kl_quality.kl_estimate)
            + " "
            + "val_loss="
            + str(val_loss)
            + " "
            + "val_kl="
            + str(val_kl)
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

                def error_callback(e, args=experiment_args):
                    print("/ ", end="")
                    with open("./errors.log", "a") as f:
                        f.write(f"Error: {str(e)}\nArgs: {args}\n")

                # pool.apply(
                pool.apply_async(
                    single_run,
                    args=(*experiment_args,),
                    kwds={"device": device},
                    error_callback=error_callback,
                )

    for pool in pools:
        pool.close()
        pool.join()

    print("Done")


def combine_results(
    experiment_group: str,
    seeds: List[int],
    agent_name: str,
    input_dims: list,
    data_ratios: list,
    noise_stds: list,
    agent_id_start: int,
    agent_id_end: int,
    results_folder: str,
):
    """Combine results from multiple runs into single files."""
    total_combinations = (
        len(input_dims)
        * len(data_ratios)
        * len(noise_stds)
    )
    with tqdm(total=total_combinations, desc="Combining results") as pbar:
        for ind in input_dims:
            for dr in data_ratios:
                for ns in noise_stds:
                    single_result_folder = (
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
                    )
                    combined_filepath = (
                        f"{results_folder}/results_"
                        + experiment_group
                        + ("_" if len(experiment_group) > 0 else "")
                        + agent_name
                        + "_id"
                        + str(ind)
                        + "dr"
                        + str(dr)
                        + "ns"
                        + str(ns)
                        + ".txt"
                    )

                    with open(combined_filepath, "w") as combined_file:
                        for agent_id in range(agent_id_start, agent_id_end):

                            agent_results = []

                            for seed in seeds:
                                agent_filepath = (
                                    single_result_folder
                                    + "/agentid"
                                    + str(agent_id)
                                    + "_seed"
                                    + str(seed)
                                    + ".txt"
                                )

                                result = read_results_file(agent_filepath)

                                agent_results.append(result)

                            kls = []
                            val_losses = []
                            val_kls = []
                            mean_errors = []
                            std_errors = []
                            best_epochs = []

                            result_agent_name = list(agent_results[0].keys())[0]

                            for result in agent_results:
                                kls.extend(result[result_agent_name]["kl"])
                                val_losses.extend(result[result_agent_name]["val_loss"])
                                val_kls.extend(result[result_agent_name]["val_kl"])
                                mean_errors.extend(result[result_agent_name]["mean_error"])
                                std_errors.extend(result[result_agent_name]["std_error"])
                                best_epochs.extend(result[result_agent_name]["best_epoch"])

                            agent_settings = agent_factories.load_agent_config(
                                agent_id, agent_name
                            ).settings

                            kl_mean = sum([kl_quality for kl_quality in kls]) / len(kls)
                            kl_variance = sum(
                                [(kl_quality - kl_mean) ** 2 for kl_quality in kls]
                            ) / len(kls)
                            mean_error = sum([me for me in mean_errors]) / len(mean_errors)
                            std_error = sum([se for se in std_errors]) / len(std_errors)
                            val_loss_mean = sum(val_losses) / len(val_losses) if val_losses else None
                            val_kl_mean = sum(val_kls) / len(val_kls) if val_kls else None
                            best_epoch_mean = int(round(sum(best_epochs) / len(best_epochs))) if best_epochs else None

                            combined_file.write(
                                str(agent_id)
                                + " "
                                + str(kl_mean)
                                + " "
                                + "kl_variance="
                                + str(kl_variance)
                                + " "
                                + ("val_loss=" + str(val_loss_mean) + " " if val_loss_mean is not None else "")
                                + ("val_kl=" + str(val_kl_mean) + " " if val_kl_mean is not None else "")
                                + "mean_error="
                                + str(mean_error)
                                + " "
                                + "std_error="
                                + str(std_error)
                                + (" best_epoch=" + str(best_epoch_mean) if best_epoch_mean is not None else "")
                                + " "
                                + " ".join(
                                    [
                                        str(k) + "=" + str(v)
                                        for (
                                            k,
                                            v,
                                        ) in agent_settings.items()
                                    ]
                                )
                                + "\n"
                            )
                    pbar.update(1)


def main(
    input_dim=(1, 10, 100),
    data_ratio=(1.0, 10.0, 100.0),
    noise_std=(0.01, 0.1, 1.0),
    seeds=(2605, 26, 0, 5),
    agent_id_start=0,
    agent_id_end=-1,
    agent_name="all",
    experiment_group="",
    devices=2,
    processes_per_device=5,
    debug=False,
    results_folder="results",
    use_double_precision=False,
    reduce_batch_dims=[],
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

    experiments = []

    for ind in input_dim:
        for dr in data_ratio:
            for ns in noise_std:

                for seed in seeds:
                    problem = load.regression_load(
                        input_dim=ind,
                        data_ratio=dr,
                        seed=seed,
                        noise_std=ns,
                        use_double_precision=use_double_precision,
                    )

                    print(
                        "Created problem for ind", ind, "dr", dr, "ns", ns, "seed", seed
                    )
                
                reduce_batch = ind in reduce_batch_dims

                sweep = agent_factories.load_agent_config_sweep(agent_name, reduce_batch=reduce_batch)
                sweep = (
                    sweep[agent_id_start:]
                    if agent_id_end == -1
                    else sweep[agent_id_start:agent_id_end]
                )

                for seed in seeds:
                    agent_seeds = split_seed(seed, len(sweep))

                    for i, agent_seed in enumerate(agent_seeds):

                        agent_id = agent_id_start + i

                        single_result_folder = (
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
                        )

                        os.makedirs(single_result_folder, exist_ok=True)

                        results_file = (
                            single_result_folder
                            + "/agentid"
                            + str(agent_id)
                            + "_seed"
                            + str(seed)
                            + ".txt"
                        )

                        arguments = (
                            ind,
                            dr,
                            ns,
                            seed,
                            agent_id,
                            agent_seed,
                            agent_name,
                            experiment_group,
                            results_file,
                            use_double_precision,
                            reduce_batch,
                        )

                        if os.path.exists(results_file):
                            print(".", end="")
                        else:
                            experiments.append(arguments)

    if len(experiments) > 0:
        random.shuffle(experiments)

        run_experiments(
            experiments, devices, processes_per_device=processes_per_device, debug=debug
        )
        print("Finished all runs")

    else:
        print("No experiments to run")

    max_agent_id = (
        agent_id_end
        if agent_id_end != -1
        else len(agent_factories.load_agent_config_sweep(agent_name))
    )

    combine_results(
        experiment_group,
        seeds,
        agent_name,
        input_dim,
        data_ratio,
        noise_std,
        agent_id_start,
        max_agent_id,
        results_folder,
    )

    print("Combined results")




if __name__ == "__main__":
    fire.Fire(main)
