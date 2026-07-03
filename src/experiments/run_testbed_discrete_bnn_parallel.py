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

import random
from typing import List
import fire
from tqdm import tqdm

from src.experiments import agent_factories
from src.experiments import agents
from src.experiments import load
from src.experiments.seeds import split_seed
import os
from multiprocessing import Pool
from src.utils import read_results_file


def single_run(
    ind,
    dr,
    ns,
    max_num_samples,
    seed,
    agent_id,
    agent_seed,
    agent_name,
    experiment_group,
    results_file_prefix,
    use_double_precision,
    reduce_batch,
    test_random_set_count,
    device,
):

    problem = load.regression_load(
        input_dim=ind,
        data_ratio=dr,
        seed=seed,
        noise_std=ns,
        use_double_precision=use_double_precision,
    )

    all_results = []

    agent_config = agent_factories.load_agent_config(agent_id, agent_name, reduce_batch=reduce_batch)

    # Form the appropriate agent for training
    agent = agents.VanillaEnnAgent(agent_config.config_ctor(), use_double_precision=use_double_precision, fixed_sampler=True)

    train_seed, all_indices_seed, best_samples_seed, random_set_evaluation_seed = split_seed(agent_seed, 4)

    # Train
    (enn_sampler_fixed, enn_sampler_free), val_loss, val_kl = agent(
        problem.train_data,
        train_seed,
        problem.prior_knowledge,
        device=device,
        logging="none",
        val_data=problem.val_data,
        evaluate_quality_val_fn=problem.evaluate_quality_val,
    )

    if agent_config.settings["agent"] in ["layer_ensembles", "ensemble"]:
        mns_list = [agent_config.settings["max_num_samples"]]
    else:
        mns_list = [*max_num_samples]

    for max_samples in mns_list:
        # Evaluate the quality of the ENN sampler after training

        
        with open(
            results_file_prefix
            + "mns"
            + str(max_samples)
            + "_randomset.txt",
            "w",
        ) as f:

            for random_samples_count in range(2, max_samples + 1):

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

        best_samples_kl = problem.find_best_samples(
            enn_sampler_fixed, all_indices, seed=best_samples_seed, noise_std=ns, device=device, use_log_likelihood=False, verbose=False
        )
        
        with open(
            results_file_prefix
            + "mns"
            + str(max_samples)
            + "_kl.txt",
            "w",
        ) as f:

            for samples, best_kl_dict in best_samples_kl.items():
                kl_quality = best_kl_dict["best_kl"]
                val_kl_quality = best_kl_dict["best_val_kl"]
                # print(
                #     f"kl_estimate={kl_quality.kl_estimate}"
                #     + " val_kl="
                #     + str(val_kl_quality.kl_estimate)
                #     + " mean_error="
                #     + str(kl_quality.extra["mean_error"])
                #     + " "
                #     + "std_error="
                #     + str(kl_quality.extra["std_error"])
                # )
                all_results.append(kl_quality)

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

        best_samples_ll = problem.find_best_samples(
            enn_sampler_fixed, all_indices, seed=best_samples_seed, noise_std=ns, device=device, use_log_likelihood=True
        )

        with open(
            results_file_prefix
            + "mns"
            + str(max_samples)
            + "_ll.txt",
            "w",
        ) as f:

            for samples, best_kl_dict in best_samples_ll.items():
                kl_quality = best_kl_dict["best_kl"]
                val_ll = best_kl_dict["best_val_log_likelihood"]
                # print(
                #     f"kl_estimate={kl_quality.kl_estimate}"
                #     + " val_ll="
                #     + str(val_ll)
                #     + " mean_error="
                #     + str(kl_quality.extra["mean_error"])
                #     + " "
                #     + "std_error="
                #     + str(kl_quality.extra["std_error"])
                # )
                all_results.append(kl_quality)

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

        print("#", end="", flush=True)

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
                    print("/ ", end="", flush=True)
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
    max_num_samples: list,
    agent_id_start: int,
    agent_id_end: int,
    results_folder: str,
    no_file_ok: bool = False,
):
    total_combinations = (
        len(input_dims)
        * len(data_ratios)
        * len(noise_stds)
        * len(max_num_samples)
        * 2
    )
    with tqdm(total=total_combinations, desc="Combining results") as pbar:
        for ind in input_dims:
            for dr in data_ratios:
                for ns in noise_stds:
                    for mns in max_num_samples:
                        for val_type in ["kl", "ll"]:
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
                                + "mns"
                                + str(mns)
                                + ("_" + val_type)
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
                                            + "mns"
                                            + str(mns)
                                            + ("_" + val_type)
                                            + ".txt"
                                        )

                                        if not os.path.exists(agent_filepath):
                                            if no_file_ok:
                                                continue
                                            else:
                                                raise FileNotFoundError(
                                                    f"Expected file {agent_filepath} not found."
                                                )

                                        result = read_results_file(agent_filepath)

                                        agent_results.append(result)

                                    if no_file_ok and len(agent_results) == 0:
                                        continue

                                    all_kls = {}
                                    val_metrics = {}
                                    mean_errors = {}
                                    std_errors = {}

                                    result_agent_name = list(agent_results[0].keys())[0]

                                    for result in agent_results:
                                        for i, indexer in enumerate(result[result_agent_name]["indexer"]):
                                            
                                            if indexer not in all_kls:
                                                all_kls[indexer] = []
                                                val_metrics[indexer] = []
                                                mean_errors[indexer] = []
                                                std_errors[indexer] = []

                                            all_kls[indexer].append(result[result_agent_name]["kl"][i])
                                            val_metrics[indexer].append(result[result_agent_name]["val_" + val_type][i])
                                            mean_errors[indexer].append(result[result_agent_name]["mean_error"][i])
                                            std_errors[indexer].append(result[result_agent_name]["std_error"][i])

                                    agent_settings = agent_factories.load_agent_config(
                                        agent_id, agent_name
                                    ).settings

                                    for indexer, kls in all_kls.items():
                                        kl_mean = sum([kl_quality for kl_quality in kls]) / len(kls)
                                        kl_variance = sum(
                                            [(kl_quality - kl_mean) ** 2 for kl_quality in kls]
                                        ) / len(kls)
                                        val_metric_mean = sum([vm for vm in val_metrics[indexer]]) / len(val_metrics[indexer])
                                        val_metric_variance = sum(
                                            [(vm - val_metric_mean) ** 2 for vm in val_metrics[indexer]]
                                        ) / len(val_metrics[indexer])
                                        mean_error = sum([me for me in mean_errors[indexer]]) / len(mean_errors[indexer])
                                        std_error = sum([se for se in std_errors[indexer]]) / len(std_errors[indexer])

                                        combined_file.write(
                                            str(agent_id)
                                            + " "
                                            + str(kl_mean)
                                            + " "
                                            + "kl_variance="
                                            + str(kl_variance)
                                            + " "
                                            + "val_" + val_type + "_mean="
                                            + str(val_metric_mean)
                                            + " "
                                            + "val_" + val_type + "_variance="
                                            + str(val_metric_variance)
                                            + " "
                                            + "mean_error="
                                            + str(mean_error)
                                            + " "
                                            + "std_error="
                                            + str(std_error)
                                            + " "
                                            + "indexer="
                                            + str(indexer)
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
    max_num_samples=(10, 100, 1000),
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
    lens_max_num_samples=[125],
    test_random_set_count=5,
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

    print("A")

    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(f"{results_folder}/{agent_name}", exist_ok=True)

    if isinstance(input_dim, int):
        input_dim = [input_dim]
    if isinstance(data_ratio, float) or isinstance(data_ratio, int):
        data_ratio = [float(data_ratio)]
    if isinstance(noise_std, float) or isinstance(noise_std, int):
        noise_std = [float(noise_std)]

    single_mns_per_experiment = False
        
    if "ensemble" in agent_name:
        max_num_samples = [10, 30]
        single_mns_per_experiment = True
    
    if "layer_ensembles" in agent_name:
        max_num_samples = lens_max_num_samples
        single_mns_per_experiment = False


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

                        results_file_prefix = (
                            single_result_folder
                            + "/agentid"
                            + str(agent_id)
                            + "_seed"
                            + str(seed)
                        )

                        arguments = (
                            ind,
                            dr,
                            ns,
                            max_num_samples,
                            seed,
                            agent_id,
                            agent_seed,
                            agent_name,
                            experiment_group,
                            results_file_prefix,
                            use_double_precision,
                            reduce_batch,
                            test_random_set_count,
                        )

                        to_add = []

                        for mns in max_num_samples:
                            results_file_kl = (
                                results_file_prefix
                                + "mns"
                                + str(mns)
                                + "_kl.txt"
                            )
                            results_file_ll = (
                                results_file_prefix
                                + "mns"
                                + str(mns)
                                + "_ll.txt"
                            )

                            if os.path.exists(results_file_kl) and os.path.exists(results_file_ll):
                                print(".", end="", flush=True)
                            else:
                                to_add.append(arguments)

                            if single_mns_per_experiment and len(to_add) == len(max_num_samples):
                                experiments.extend(to_add)
                            elif not single_mns_per_experiment:
                                experiments.extend(to_add)

    if len(experiments) > 0:
        random.shuffle(experiments)

        run_experiments(
            experiments, devices, processes_per_device=processes_per_device, debug=debug
        )
        print("Finished all runs")
    else:
        print("No experiments to run.")

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
            max_num_samples,
            agent_id_start,
            max_agent_id,
            results_folder,
            no_file_ok=single_mns_per_experiment
        )

        print("Combined results")

if __name__ == "__main__":
    fire.Fire(main)
