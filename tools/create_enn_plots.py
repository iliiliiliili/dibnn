import os

from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_y_continuous,
    scale_x_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
    theme,
    element_text,
    ylab,
    xlab,
    scale_color_discrete,
    labeller,
)
from plotnine.data import economics
from pandas import Categorical, DataFrame, read_csv
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete
from plotnine.guides import guide_axis, guide_legend, guide
from glob import glob
import re
from fire import Fire

limit_std = 1000

tex_template_file = "tools/tex_table_template.tex"

with open(tex_template_file, "r") as f:
    tex_template = f.read()

# files = glob("results_vnn_selected*")
# files = glob("results/results_*")
# files = glob("results/results_*layer*")
# files = glob("results_all_old*") + glob("results_vnn_selected*")
# files = glob("results_mserr*") + glob("results_lrelu*")
# files = glob("results/results_best_selected_val_*") + glob("results/results_mserr*")
# files = glob("results/results_mserr_layer_ensemble*")
# files = glob("results/results_best_selected_val_*") + glob("results/results_multi_indexed_val_*")
# files = glob("results/results_batched_multi_indexed_val_*")
# files = glob("results/results_ranked_batched_multi_indexed_val_*")
# files = glob("results1/results_best_selected_val_true_layer_ensemble_einsum_cor*")

float_fields = [
    "val_kl",
    "val_ll",
    "val_kl_mean",
    "val_ll_mean",
    "val_kl_variance",
    "val_ll_variance",
    "noise_scale",
    "prior_scale",
    "dropout_rate",
    "regularization_scale",
    "sigma_0",
    "learning_rate",
    "mean_error",
    "std_error",
]
int_fields = [
    "num_ensemble",
    "num_layers",
    "hidden_size",
    "index_dim",
    "num_index_samples",
    "indexer",
    # "num_batches",
    "training_epochs",
]
int_list_fields = [
    "num_ensembles",
]

rename = {"num_ensembles": "num_ensemble"}

field_tex_names = {
    "kl": "KL",
    "kl_variance": "Var[KL|seed]",
    "agent": "Type",
    "agent_full": "TypeF",
    "mean": "Mean[KL]",
    "val_mean": "Mean[val KL]",
    "kl_std": "Var[KL]",
    "std": "Var[KL]",
    "val_std": "Var[val KL]",
    "low_mstd": "Mean[KL]-Var[KL]",
    "high_mstd": "Mean[KL]+Var[KL]",
    "low_vmstd": "Mean[val KL]-Var[val KL]",
    "high_vmstd": "Mean[val KL]+Var[val KL]",
    "low_lmstd": "Mean[last KL]-Var[last KL]",
    "high_lmstd": "Mean[last KL]+Var[last KL]",
    "last_mean": "Mean[last KL]",
    "last_std": "Std[last KL]",
    "noise_scale": "NS",
    "prior_scale": "PS",
    "dropout_rate": "DR",
    "regularization_scale": "RS",
    "sigma_0": "\\sigma_0",
    "learning_rate": "LR",
    "mean_error": "E_{\\mu}",
    "std_error": "E_{\\sigma}",
    "num_ensemble": "Ens",
    "num_ensembles": "Ens",
    "num_layers": "Depth",
    "hidden_size": "Size",
    "index_dim": "D_{index}",
    "activation": "Act",
    "activation_mode": "ActM",
    "use_batch_norm": "BN",
    "batch_norm_mode": "BNM",
    "global_std_mode": "GStdM",
    "num_batches": "Epochs",
    "indexer": "Indexer",
    "num_index_samples": "Samples",
    "initializer": "Init",
    "loss_function": "L_{f}",
    "sample_type": "Samples",
}


agent_plot_params = {
    "ensemble": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "subsample_ensemble": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "dropout": {
        "x": "dropout_rate",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["regularization_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "hypermodel": {
        "x": "index_dim",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "bbb": {
        "x": "sigma_0",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["learning_rate"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "vnn": {
        "x": "num_batches",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "vnn_eskl": {
        "x": "num_batches",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "vnn_esll": {
        "x": "num_batches",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "vnn_lrelu": {
        "x": "num_batches",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "vnn_init": {
        "x": "num_batches",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["activation_mode", "global_std_mode", "loss_function"],
        "colour": "activation",
        "shape": "factor(hidden_size)",
        "fill": "initializer",
    },
    "layer_ensembles": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "layer_ensemble_cor": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "layer_ensemble_einsum_cor": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "true_layer_ensemble_einsum": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "true_layer_ensemble_einsum_cor": {
        "x": "num_ensemble",
        "y": "kl",
        "val_y": "val_kl_mean",
        "val_ll_y": "val_ll_mean",
        "facet": ["noise_scale", "sample_type"],
        "colour": "factor(num_layers)",
        "shape": "factor(prior_scale)",
    },
}


summary_select_agent_params = {
    "ensemble": [
        # {
        #     "agent_suffix": "_10",
        #     "noise_scale": [1.0],
        #     "prior_scale": [1.0],
        #     "num_layers": [2],
        #     "hidden_size": [50],
        #     "num_ensemble": [10],
        #     "training_epochs": [1000],
        # },
        {
            "agent_suffix": "_30",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [30],
            "training_epochs": [1000],
        },
    ],
    "dropout": [
        {
            "dropout_rate": [0.05],
            "regularization_scale": [1e-6],
            "num_layers": [2],
            "hidden_size": [50],
        },
    ],
    "hypermodel": [
        {
            "index_dim": [20],
            "noise_scale": [1.0],
            "prior_scale": [5.0],
            "num_layers": [2],
            "hidden_size": [50],
        },
    ],
    "bbb": [
        {
            "sigma_0": [1e2],
            "learning_rate": [1e-3],
            "num_layers": [2],
            "hidden_size": [50],
        },
    ],
    "vnn": [
        # {
        #     "activation": ["relu"],
        #     "activation_mode": ["mean"],
        #     "global_std_mode": ["multiply"],
        #     "num_layers": [3],
        #     "hidden_size": [100],
        #     "num_index_samples": [100],
        #     "training_epochs": [-1],
        # },
        {
            "agent_suffix": "_eskl",
            "activation": ["relu"],
            "activation_mode": ["mean"],
            "global_std_mode": ["multiply"],
            "num_layers": [3],
            "hidden_size": [100],
            "num_index_samples": [100],
            "early_stopping_mode": ["kl"],
        },
        {
            "agent_suffix": "_esll",
            "activation": ["relu"],
            "activation_mode": ["mean"],
            "global_std_mode": ["multiply"],
            "num_layers": [3],
            "hidden_size": [100],
            "num_index_samples": [100],
            "early_stopping_mode": ["loss"],
        },
    ],
    "layer_ensembles": [
        {
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [5],
        },
    ],
}

summary_select_agent_params_best = {
    "ensemble": [
        # {
        #     "agent_suffix": "_10",
        #     "noise_scale": [1.0],
        #     "prior_scale": [1.0],
        #     "num_layers": [2],
        #     "hidden_size": [50],
        #     "num_ensemble": [10],
        #     "training_epochs": [1000],
        # },
        {
            "agent_suffix": "_30",
            "noise_scale": [],
            "prior_scale": [],
            "num_layers": [],
            "hidden_size": [],
            "num_ensemble": [30],
            "training_epochs": [],
        },
    ],
    "dropout": [
        {
            "dropout_rate": [],
            "regularization_scale": [],
            "num_layers": [],
            "hidden_size": [],
        },
    ],
    "hypermodel": [
        {
            "index_dim": [],
            "noise_scale": [],
            "prior_scale": [],
            "num_layers": [],
            "hidden_size": [],
        },
    ],
    "bbb": [
        {
            "sigma_0": [],
            "learning_rate": [],
            "num_layers": [],
            "hidden_size": [],
        },
    ],
    "vnn": [
        {
            "activation": [],
            "activation_mode": [],
            "global_std_mode": [],
            "num_layers": [],
            "hidden_size": [],
            "num_index_samples": [],
            "training_epochs": [],
        },
        # {
        #     "agent_suffix": "_eskl",
        #     "activation": [],
        #     "activation_mode": [],
        #     "global_std_mode": [],
        #     "num_layers": [],
        #     "hidden_size": [],
        #     "num_index_samples": [],
        #     "early_stopping_mode": ["kl"],
        # },
        # {
        #     "agent_suffix": "_esll",
        #     "activation": [],
        #     "activation_mode": [],
        #     "global_std_mode": [],
        #     "num_layers": [],
        #     "hidden_size": [],
        #     "num_index_samples": [],
        #     "early_stopping_mode": ["loss"],
        # },
    ],
    "layer_ensembles": [
        {
            "noise_scale": [],
            "prior_scale": [],
            "num_layers": [],
            "hidden_size": [],
            "num_ensemble": [],
        },
    ],
}


def add_layer_ensembles_summary_params(use_ranked):

    if use_ranked:
        all_nens_samples = [
            (2, [*range(2, 2**3)]),
            (3, [*range(2, 3**3)]),
            (5, [*range(2, 5**3)]),
            (6, [*range(2, 6**3)]),
            # (8, [*range(2, 8 ** 3)]),
            # (10, [*range(2, 10 ** 3)]),
        ]
    else:
        all_nens_samples = [
            (2, [2, 3, "full"]),
            (3, [2, 3, 4, "full"]),
            (5, [2, 3, 5, "full"]),
            (6, [2, 3, 5, 10, "full"]),
            (8, [2, 3, 5, 10, "full"]),
            (10, [2, 3, 5, 10, 20, "full"]),
        ]

    for num_ensemble, inference_samples in all_nens_samples:
        for samples in inference_samples:

            if use_ranked:
                indexer = samples
            else:
                indexer = (
                    num_ensemble**3 if samples == "full" else samples * num_ensemble
                )

            params = {
                "agent_suffix": "_"
                + str(num_ensemble)
                + "s"
                + str(indexer)
                + ("f" if samples == "full" else ""),
                "noise_scale": [1.0],
                "prior_scale": [1.0],
                "num_layers": [2],
                "hidden_size": [50],
                "num_ensemble": [num_ensemble],
                "indexer": [indexer],
            }
            summary_select_agent_params["layer_ensembles"].append(params)


def make_vnn_ranked_params(num_samples, use_ranked, early_stopping_mode):

    if use_ranked:

        summary_select_agent_params_best["vnn"] = []

        all_nens_samples = [
            (num_samples, [*range(2, num_samples)]),
        ]

        for max_num_samples, inference_samples in all_nens_samples:
            for samples in inference_samples:

                indexer = samples

                params = {
                    "agent_suffix": "_"
                    + str(max_num_samples)
                    + "s"
                    + str(indexer)
                    + ("f" if samples == "full" else ""),
                    "activation": [],
                    "hidden_size": [],
                    "num_index_samples": [],
                    "training_epochs": [],
                    "max_num_samples": [max_num_samples],
                    "indexer": [indexer],
                    "early_stopping_mode": [early_stopping_mode],
                }

                summary_select_agent_params_best["vnn"].append(params)


def make_dropout_ranked_params(num_samples, use_ranked, early_stopping_mode):

    if use_ranked:

        summary_select_agent_params_best["dropout"] = []

        all_nens_samples = [
            (num_samples, [*range(2, num_samples)]),
        ]

        for max_num_samples, inference_samples in all_nens_samples:
            for samples in inference_samples:

                indexer = samples

                params = {
                    "agent_suffix": "_"
                    + str(max_num_samples)
                    + "s"
                    + str(indexer)
                    + ("f" if samples == "full" else ""),
                    "dropout_rate": [],
                    "regularization_scale": [],
                    "num_layers": [],
                    "hidden_size": [],
                    "max_num_samples": [max_num_samples],
                    "indexer": [indexer],
                    "early_stopping_mode": [early_stopping_mode],
                }

                summary_select_agent_params_best["dropout"].append(params)


def make_bbb_ranked_params(num_samples, use_ranked, early_stopping_mode):

    if use_ranked:

        summary_select_agent_params_best["bbb"] = []

        all_nens_samples = [
            (num_samples, [*range(2, num_samples)]),
        ]

        for max_num_samples, inference_samples in all_nens_samples:
            for samples in inference_samples:

                indexer = samples

                params = {
                    "agent_suffix": "_"
                    + str(max_num_samples)
                    + "s"
                    + str(indexer)
                    + ("f" if samples == "full" else ""),
                    "sigma_0": [],
                    "learning_rate": [],
                    "num_layers": [],
                    "hidden_size": [],
                    "max_num_samples": [max_num_samples],
                    "indexer": [indexer],
                    "early_stopping_mode": [early_stopping_mode],
                }

                summary_select_agent_params_best["bbb"].append(params)


def make_hypermodel_ranked_params(num_samples, use_ranked, early_stopping_mode):

    if use_ranked:

        summary_select_agent_params_best["hypermodel"] = []

        all_nens_samples = [
            (num_samples, [*range(2, num_samples)]),
        ]

        for max_num_samples, inference_samples in all_nens_samples:
            for samples in inference_samples:

                indexer = samples

                params = {
                    "agent_suffix": "_"
                    + str(max_num_samples)
                    + "s"
                    + str(indexer)
                    + ("f" if samples == "full" else ""),
                    "index_dim": [],
                    "noise_scale": [],
                    "prior_scale": [],
                    "num_layers": [],
                    "hidden_size": [],
                    "max_num_samples": [max_num_samples],
                    "indexer": [indexer],
                    "early_stopping_mode": [early_stopping_mode],
                }

                summary_select_agent_params_best["hypermodel"].append(params)


def make_ensemble_ranked_params(num_samples, use_ranked, early_stopping_mode):

    if use_ranked:

        summary_select_agent_params_best["ensemble"] = []
        all_nens_samples = [
            (num_samples, [*range(2, num_samples)]),
        ]

        for max_num_samples, inference_samples in all_nens_samples:
            for samples in inference_samples:

                indexer = samples

                params = {
                    "agent_suffix": "_"
                    + str(max_num_samples)
                    + "s"
                    + str(indexer)
                    + ("f" if samples == "full" else ""),
                    "noise_scale": [],
                    "prior_scale": [],
                    "num_layers": [],
                    "hidden_size": [],
                    "num_ensemble": [],
                    "training_epochs": [],
                    "max_num_samples": [max_num_samples],
                    "indexer": [indexer],
                    "early_stopping_mode": [early_stopping_mode],
                }

                summary_select_agent_params_best["ensemble"].append(params)


def make_layer_ensemble_ranked_params(num_samples, use_ranked, num_ensemble=3, early_stopping_mode=None):

    if use_ranked:

        summary_select_agent_params_best["layer_ensembles"] = []
        all_nens_samples = [
            (num_samples, [*range(2, num_samples)]),
        ]

        for max_num_samples, inference_samples in all_nens_samples:
            for samples in inference_samples:

                indexer = samples

                params = {
                    "agent_suffix": "_"
                    + str(max_num_samples)
                    + "s"
                    + str(indexer)
                    + ("f" if samples == "full" else ""),
                    "noise_scale": [],
                    "prior_scale": [],
                    "num_layers": [],
                    "hidden_size": [],
                    "num_ensemble": [num_ensemble],
                    "max_num_samples": [max_num_samples],
                    "indexer": [indexer],
                    "early_stopping_mode": [early_stopping_mode],
                }

                summary_select_agent_params_best["layer_ensembles"].append(params)


def add_subsample_ensemble_summary_params():
    for num_ensemble, inference_samples in [
        (2, [2]),
        (3, [2, 3]),
        (5, [2, 3, 5]),
        (6, [2, 3, 5, 6]),
        (8, [2, 3, 5, 8]),
        (10, [2, 3, 5, 10]),
        (30, [2, 3, 5, 10, 20, 30]),
    ]:
        for samples in inference_samples:
            indexer = samples
            params = {
                "agent_suffix": "_" + str(num_ensemble) + "s" + str(indexer),
                "noise_scale": [],
                "prior_scale": [],
                "num_layers": [],
                "hidden_size": [],
                "num_ensemble": [num_ensemble],
                "indexer": [indexer],
            }
            summary_select_agent_params_best["subsample_ensemble"].append(params)



def read_data(file):
    with open(file, "r") as f:
        lines = f.readlines()

        agent_frames = {}

        for line in lines:
            id, kl, *params = line.replace("\n", "").split(" ")

            f = []
            for p in params:
                if "=" in p:
                    f.append(p)
                else:
                    f[-1] += " " + p
            raw_params = f

            params = []

            agent = None

            for p in raw_params:
                k, v = p.split("=")
                if k == "agent":
                    agent = v
                else:
                    params.append(p)

            id = int(id)
            kl = float(kl)

            if "results_lrelu" in file:
                agent += "_lrelu"

            if agent not in agent_frames:
                agent_frames[agent] = {"kl": []}

            agent_frames[agent]["kl"].append(kl)
            # agent_frames[agent]["kl"].append(min(2, kl))

            for p in params:
                k, v = p.split("=")

                if k in float_fields:
                    v = float(v)
                elif k in int_fields:
                    v = int(v)
                elif k in int_list_fields:
                    v = int(v.split("]")[0].split(" ")[-1])

                if k in rename:
                    k = rename[k]

                # if k == "num_batches":
                #     v //= 1000

                if k not in agent_frames[agent]:
                    agent_frames[agent][k] = []

                agent_frames[agent][k].append(v)

        for agent in agent_frames.keys():
            agent_frames[agent] = DataFrame(agent_frames[agent])

        return agent_frames


def create_tex_table(frame, agent, output_file_name):

    global tex_template

    tex_file = tex_template

    fields = []
    mode = []

    to_describe = []

    for key in frame.keys():
        fields.append(field_tex_names[key])
        mode.append("c")

        if key not in to_describe:
            to_describe.append(key)

    fields = "    " + " & ".join(fields) + "\\\\ [0.5ex] \n        \\hline"
    mode = "|" + " ".join(mode) + "|"
    caption = ", ".join(
        [
            str(field_tex_names[key]) + ":" + str(key).replace("_", " ")
            for key in to_describe
        ]
    )

    table = [fields]

    for i in range(len(frame)):
        line = []
        for key in frame.keys():

            val = frame[key][i]

            if key in ["kl", "mean_error", "std_error", "kl_std"]:
                val = "{:.4f}".format(val)

            line.append(str(val))

        line = " & ".join(line) + " \\\\"
        table.append(line)

    table = "\n        ".join(table)

    tex_file = (
        tex_file.replace(
            "<TITLE>", (agent + " in " + output_file_name).replace("_", " ")
        )
        .replace("<CAPTION>", caption)
        .replace("<MODE>", mode)
        .replace("<TABLE>", table)
    )

    with open("tex/" + output_file_name + ".tex", "w") as f:
        f.write(tex_file)


def plot_single_frame(frame, agent, output_file_name):

    params = agent_plot_params[agent]

    point_aes_params = {}

    for key in ["colour", "shape", "fill"]:
        if key in params:
            point_aes_params[key] = params[key]

    plot = (
        ggplot(frame)
        + aes(x=params["x"], y=params["y"])
        + facet_wrap(params["facet"], nrow=2, labeller="label_both")
        + geom_hline(yintercept=1)
        + ylim(0, 2)
        + geom_point(
            aes(**point_aes_params),
            size=3,
            position=position_dodge(width=0.8),
            stroke=0.2,
        )
    )

    if len(params["facet"]) > 2:
        plot = plot + theme(strip_text_x=element_text(size=5))

    plot.save("plots/" + output_file_name + ".png", dpi=600)

    create_tex_table(frame, agent, output_file_name)


def plot_multiple_frames(frames, agent, output_file_name):

    params = agent_plot_params[agent]

    result = frames[0].copy()
    result[params["y"]] = sum(f[params["y"]] for f in frames) / len(frames)
    std = (
        sum((f[params["y"]] - result[params["y"]]) ** 2 for f in frames) / len(frames)
    ) ** 0.5
    result[params["y"] + "_std"] = std

    point_aes_params = {}

    for key in ["colour", "shape", "fill"]:
        if key in params:
            point_aes_params[key] = params[key]

    dodge = position_dodge(width=0.8)

    plot = (
        ggplot(result)
        + aes(x=params["x"], y=params["y"])
        + facet_wrap(params["facet"], nrow=2, labeller="label_both")
        + geom_hline(yintercept=1)
        + ylim(0, 2)
        + geom_point(
            aes(**point_aes_params),
            size=3,
            position=position_dodge(width=0.8),
            stroke=0.2,
        )
        + geom_errorbar(
            aes(
                **point_aes_params,
                ymin=params["y"] + "-" + params["y"] + "_std",
                ymax=params["y"] + "+" + params["y"] + "_std",
            ),
            position=dodge,
            width=0.8,
        )
    )

    if len(params["facet"]) > 2:
        plot = plot + theme(strip_text_x=element_text(size=5))

    plot.save("plots/" + output_file_name + ".png", dpi=600)
    create_tex_table(result, agent, output_file_name)


def plot_all_single_frames(files):

    for file in files:
        agent_frames = read_data(file)
        for agent, frame in agent_frames.items():

            plot_single_frame(
                frame,
                agent,
                "enn_plot_"
                + agent
                + "_"
                + file.replace(".txt", "")
                .replace("results/", "")
                .replace("results1/", ""),
            )


def plot_all_total_frames(files):

    all_agent_frames = {}

    for file in files:
        agent_frames = read_data(file)
        for agent in agent_frames.keys():
            frame = agent_frames[agent]

            if agent not in all_agent_frames:
                all_agent_frames[agent] = []

            all_agent_frames[agent].append(frame)

    for agent, frames in all_agent_frames.items():
        if len(frames) > 0:
            plot_multiple_frames(frames, agent, "total_enn_plot_" + agent)


def parse_enn_experiment_parameters(file, ranked_file=False):

    param_string = file.split("_")[-2 if ranked_file else -1]
    params = re.findall(r"\d+(?:\.\d+|\d*)", param_string)

    input_dim, data_ratio, noise_std = params[:3]

    max_num_samples = None

    if len(params) > 3:
        max_num_samples = int(params[3])

    input_dim = int(input_dim)
    data_ratio = float(data_ratio)
    noise_std = float(noise_std)

    return {
        "input_dim": input_dim,
        "data_ratio": data_ratio,
        "noise_std": noise_std,
        "max_num_samples": max_num_samples,
    }


def plot_summary_vnn(
    files,
    allowed_input_dims,
    parse_experiment_parameters=parse_enn_experiment_parameters,
):

    all_agent_frames = {}

    for file in files:
        agent_frames = read_data(file)
        experiment_params = parse_experiment_parameters(file)

        if experiment_params["input_dim"] not in allowed_input_dims:
            print("Skipping file", file, "due to input dim filter")
            continue

        for agent in agent_frames.keys():

            if agent not in ["vnn", "vnn_lrelu"]:
                continue

            frame = agent_frames[agent]

            if agent not in all_agent_frames:
                all_agent_frames[agent] = []

            all_agent_frames[agent].append(frame)

    data = {
        "agent": [],
        "mean": [],
        "std": [],
    }

    for agent, all_frames in all_agent_frames.items():

        params = agent_plot_params[agent]
        frames = all_frames

        for id in range(len(frames[0])):

            mean = sum(f[params["y"]][id] for f in frames) / len(frames)
            std = sum((f[params["y"]][id] - mean) ** 2 for f in frames) / len(frames)
            data["agent"].append(agent + str(id))
            data["mean"].append(mean)
            data["std"].append(std)

    frame = DataFrame(data)

    plot = (
        ggplot(frame)
        + aes(x="agent", y="mean")
        + geom_hline(yintercept=1)
        +
        # ylim(0, 2) +
        scale_y_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=3, stroke=0.2)
        + geom_errorbar(
            aes(colour="agent", ymin="mean-std", ymax="mean+std"), width=0.8
        )
    )
    plot.save(
        "plots/summary_vnn_plot_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + ".png",
        dpi=600,
    )
    frame.to_csv(
        "plots/summary_vnn_id" + "_".join([str(a) for a in allowed_input_dims]) + ".csv"
    )
    create_tex_table(
        frame,
        "all",
        "summary_vnn_plot_id" + "_".join([str(a) for a in allowed_input_dims]),
    )


def plot_summary(
    files,
    allowed_input_dims,
    parse_experiment_parameters=parse_enn_experiment_parameters,
):

    all_agent_frames = {}

    for file in files:
        agent_frames = read_data(file)
        experiment_params = parse_experiment_parameters(file)

        if experiment_params["input_dim"] not in allowed_input_dims:
            print("Skipping file", file, "due to input dim filter")
            continue

        for agent in agent_frames.keys():

            # if agent in ["layer_ensemble"]:
            #     continue

            frame = agent_frames[agent]

            if agent not in all_agent_frames:
                all_agent_frames[agent] = []

            all_agent_frames[agent].append(frame)

    data = {
        "agent": [],
        "mean": [],
        "std": [],
    }

    for agent, all_frames in all_agent_frames.items():

        if agent not in summary_select_agent_params:
            print(f"Skipping agent {agent} due to summary_select_agent_params filter")
            continue

        params = agent_plot_params[agent]
        filters = summary_select_agent_params[agent]

        for filter in filters:

            frames = all_frames
            old_frames = None
            agent_suffix = ""

            for key, value in filter.items():

                if key == "agent_suffix":
                    agent_suffix = value
                    continue

                old_frames = frames
                frames = [f[f[key].isin(value)] for f in frames]
                if len(frames[0]) <= 0:
                    raise ValueError(f"Empty frame after filtering for {key}={value}")

            mean = sum(sum(f[params["y"]]) for f in frames) / sum(
                len(f) for f in frames
            )
            std = sum(sum((f[params["y"]] - mean) ** 2) for f in frames) / sum(
                len(f) for f in frames
            )

            data["agent"].append((agent + agent_suffix).replace("_", "\n"))
            data["mean"].append(mean)
            data["std"].append(min(limit_std, std))

    frame = DataFrame(data)

    agent_list = frame["agent"].unique()
    agent_list = sorted(agent_list)

    frame["agent"] = Categorical(
        frame["agent"],
        agent_list,
    )

    plot = (
        ggplot(frame)
        + aes(x="agent", y="mean")
        + geom_hline(yintercept=1)
        +
        # ylim(0, 2) +
        scale_y_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=4, stroke=0.2)
        + geom_errorbar(
            aes(colour="agent", ymin="mean-std", ymax="mean+std"),
            width=0.8,
            size=1.5,
        )
        + theme(axis_title=element_text(size=15), axis_text=element_text(size=4))
        # + scale_color_discrete(guide=False)
        + scale_x_discrete(guide=guide_legend())
        + ylab("Mean KL")
        + xlab("Method")
    )

    os.makedirs("plots", exist_ok=True)

    plot.save(
        "plots/summary_enn_plot_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + ".png",
        dpi=600,
    )
    frame.to_csv(
        "plots/summary_enn_id" + "_".join([str(a) for a in allowed_input_dims]) + ".csv"
    )
    create_tex_table(
        frame,
        "all",
        "summary_enn_plot_id" + "_".join([str(a) for a in allowed_input_dims]),
    )


def plot_ensemble_summary(
    files,
    allowed_input_dims,
    parse_experiment_parameters=parse_enn_experiment_parameters,
):

    all_agent_frames = {}

    for file in files:
        agent_frames = read_data(file)
        experiment_params = parse_experiment_parameters(file)

        if experiment_params["input_dim"] not in allowed_input_dims:
            print("scipping file", file, "due to input dim filter")
            continue

        for agent in agent_frames.keys():

            # if agent in ["layer_ensemble"]:
            #     continue

            frame = agent_frames[agent]

            if agent not in all_agent_frames:
                all_agent_frames[agent] = []

            all_agent_frames[agent].append(frame)

    data = {
        "agent_full": [],
        "agent": [],
        "mean": [],
        "std": [],
        "num_ensemble": [],
        "indexer": [],
    }

    for agent, all_frames in all_agent_frames.items():

        if agent not in summary_select_agent_params:
            print(f"Skipping agent {agent} due to summary_select_agent_params filter")
            continue

        params = agent_plot_params[agent]
        filters = summary_select_agent_params[agent]

        for filter in filters:

            frames = all_frames
            old_frames = None
            agent_suffix = ""

            for key, value in filter.items():

                if key == "agent_suffix":
                    agent_suffix = value
                    continue

                old_frames = frames
                frames = [f[f[key].isin(value)] for f in frames]
                if len(frames[0]) <= 0:
                    raise ValueError(f"Empty frame after filtering for {key}={value}")

            mean = sum(sum(f[params["y"]]) for f in frames) / sum(
                len(f) for f in frames
            )
            std = sum(sum((f[params["y"]] - mean) ** 2) for f in frames) / sum(
                len(f) for f in frames
            )

            data["agent_full"].append((agent + agent_suffix).replace("_", "\n"))
            data["agent"].append(
                agent.replace("subsample_ensemble", "Deep Ensembles").replace(
                    "true_layer_ensemble_einsum_cor", "Layer Ensembles"
                )
            )
            data["mean"].append(mean)
            data["std"].append(min(limit_std, std))
            data["num_ensemble"].append(int(frames[0]["num_ensemble"]))
            data["indexer"].append(int(frames[0]["indexer"]))

    frame = DataFrame(data)
    plot = (
        ggplot(frame)
        + aes(x="indexer", y="mean")
        + geom_hline(yintercept=1)
        # + facet_grid("num_ensemble ~ agent", space="free", scales="free")
        + facet_wrap(
            ["num_ensemble"],
            nrow=2,
            labeller=labeller(cols=lambda x: str(x) + " ensembles"),
        )
        + scale_y_continuous(trans="log10")
        + scale_x_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=2, stroke=0.1)
        + geom_errorbar(
            aes(colour="agent", ymin="mean-std", ymax="mean+std"),
            width=0.3,
            size=0.9,
        )
        + theme(
            axis_title=element_text(size=17),
            axis_text=element_text(size=10),
            figure_size=(6, 4),
        )
        + scale_color_discrete(guide=False)
        # + scale_x_discrete(guide=guide_legend())
        + ylab("Mean KL")
        + xlab("Number of samples")
    )
    plot.save(
        "plots/summary_ensemble_enn_plot_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + ".png",
        dpi=600,
    )
    frame.to_csv(
        "plots/summary_ensemble_enn_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + ".csv"
    )
    create_tex_table(
        frame,
        "all",
        "summary_ensemble_enn_plot_id" + "_".join([str(a) for a in allowed_input_dims]),
    )


def plot_ranked_ensemble_summary(
    files,
    allowed_input_dims,
    parse_experiment_parameters=parse_enn_experiment_parameters,
    allowed_max_num_samples=None,
    allowed_data_ratios=None,
    agent_name=None,
    is_log_likelihood=False,
    is_randomset=False,
    y_limit=(0.1, 100),
):

    all_agent_frames = {}
    all_experiment_params = {}
    all_experiment_files = {}

    max_num_samples = None

    for file in files:
        agent_frames = read_data(file)
        experiment_params = parse_experiment_parameters(file, True)

        if experiment_params["input_dim"] not in allowed_input_dims:
            print("Skipping file", file, "due to input dim filter")
            continue

        if (allowed_data_ratios is not None) and (
            experiment_params["data_ratio"] not in allowed_data_ratios
        ):
            print("Skipping file", file, "due to data ratio filter")
            continue

        if (allowed_max_num_samples is not None) and (
            experiment_params["max_num_samples"] not in allowed_max_num_samples
        ):
            print("Skipping file", file, "due to allowed_max_num_samples filter")
            continue


        if len(agent_frames.keys()) == 0:
            raise ValueError("No agent frames found in file " + file)

        print("Frame", {k: len(v) for k, v in agent_frames.items()}, "from file", file)

        if experiment_params["max_num_samples"] is not None:
            max_num_samples = experiment_params["max_num_samples"]

        for agent in agent_frames.keys():

            # if agent in ["layer_ensemble"]:
            #     continue

            frame = agent_frames[agent]

            if agent not in all_agent_frames:
                all_agent_frames[agent] = []
                all_experiment_params[agent] = []
                all_experiment_files[agent] = []

            all_agent_frames[agent].append(frame)
            all_experiment_params[agent].append(experiment_params)
            all_experiment_files[agent].append(file)

    data = {
        "agent_full": [],
        "agent": [],
        "mean": [],
        "std": [],
        "low_mstd": [],
        "high_mstd": [],
        "low_vmstd": [],
        "high_vmstd": [],
        "val_mean": [],
        "val_std": [],
        "num_ensemble": [],
        "indexer": [],
    }

    for agent, all_frames in all_agent_frames.items():

        if agent not in summary_select_agent_params_best:
            print(f"Skipping agent {agent} due to summary_select_agent_params filter")
            continue

        params = agent_plot_params[agent]
        filters = summary_select_agent_params_best[agent]

        for filter in filters:

            if len(filters) > 20:
                print(",", end="")

            frames = all_frames
            old_frames = None
            agent_suffix = ""

            for key, value in filter.items():

                if key == "agent_suffix":
                    agent_suffix = value
                    continue

                if key == "max_num_samples":
                    if max_num_samples is None:
                        raise ValueError(
                            "max_num_samples is not set in the experiment parameters"
                        )
                    if value[0] != max_num_samples:
                        raise ValueError(f"Empty frame after filtering for {key}={value}")
                    continue

                if len(value) == 0:
                    continue

                old_frames = frames
                frames = [f[f[key].isin(value)] for f in frames]
                if len(frames[0]) <= 0:
                    raise ValueError(f"Empty frame after filtering for {key}={value}")

            mean = sum(sum(f[params["y"]]) for f in frames) / sum(
                len(f) for f in frames
            )
            std = sum(sum((f[params["y"]] - mean) ** 2) for f in frames) / sum(
                len(f) for f in frames
            )

            if is_randomset:
                val_mean = 0
                val_std = 0
            else:
        
                val_mean = sum(
                    sum(f[params["val_ll_y" if is_log_likelihood else "val_y"]])
                    for f in frames
                ) / sum(len(f) for f in frames)
                val_std = sum(
                    sum(
                        (f[params["val_ll_y" if is_log_likelihood else "val_y"]] - val_mean)
                        ** 2
                    )
                    for f in frames
                ) / sum(len(f) for f in frames)

                if is_log_likelihood:
                    val_mean = -val_mean

            mean = min(y_limit[1], max(y_limit[0], mean))
            val_mean = min(y_limit[1], max(y_limit[0], val_mean))
            std = min(limit_std, std)
            val_std = min(limit_std, val_std)

            data["agent_full"].append((agent + agent_suffix).replace("_", "\n"))
            data["agent"].append(agent)
            data["mean"].append(mean)
            data["std"].append(std)
            data["low_mstd"].append(max(y_limit[0], mean - std))
            data["high_mstd"].append(min(y_limit[1], mean + std))
            data["val_mean"].append(val_mean)
            data["val_std"].append(val_std)
            data["low_vmstd"].append(max(y_limit[0], val_mean - val_std))
            data["high_vmstd"].append(min(y_limit[1], val_mean + val_std))
            data["num_ensemble"].append(
                max_num_samples
                if max_num_samples is not None
                else int(frames[0]["num_ensemble"])
            )
            data["indexer"].append(int(frames[0]["indexer"]))

    frame = DataFrame(data)

    if len(frame) == 0:
        raise ValueError("No data to plot")


    plot = (
        ggplot(frame)
        + aes(x="indexer", y="mean")
        + geom_hline(yintercept=1)
        # + facet_grid("num_ensemble ~ agent", space="free", scales="free")
        + facet_wrap(
            ["num_ensemble"],
            ncol=3,
            labeller=labeller(
                cols=lambda x: ("" if agent_name is None else f"{agent_name} with ")
                + str(x)
                + " samples"
            ),
        )
        + scale_y_continuous(trans="log10", limits=y_limit)
        + scale_x_continuous(trans="log10")
    )
    
    if is_randomset:
        plot = (plot
            + geom_point(
                aes(
                    x="indexer",
                    y="val_mean",
                ),
                size=2,
                stroke=0.1,
                shape="^",
            )
            + ylab("Mean Test KL of random samples")
        )
    else:

        plot = (plot
            + geom_errorbar(
                aes(
                    colour=f"'Validation {'NLL' if is_log_likelihood else 'KL'} STD'",
                    ymin="low_vmstd",
                    ymax="high_vmstd",
                ),
                width=0.03,
                size=0.4,
                alpha=0.6,
            )
            + geom_point(
                aes(
                    x="indexer",
                    y="val_mean",
                    colour=f"'Validation {'NLL' if is_log_likelihood else 'KL'}'",
                ),
                size=2,
                stroke=0.1,
                shape="^",
            )
            + ylab("Mean Test KL / Val " + ("NLL" if is_log_likelihood else "KL"))
        )
        
    
    plot = (plot
        + geom_errorbar(
            aes(colour="'Test KL STD'", ymin="low_mstd", ymax="high_mstd"),
            width=0.04,
            size=0.6,
            alpha=0.6,
        )
        + geom_point(aes(colour="agent"), size=3, stroke=0.1)
        + theme(
            axis_title=element_text(size=15),
            axis_text=element_text(size=8),
            figure_size=(12, 4),
        )
        + scale_color_discrete(guide=guide_legend())
        # + scale_x_discrete(guide=guide_legend())
        + xlab("Number of samples")
    )

    name = (
        f"{'' if agent_name is None else agent_name + '_'}summary_ranked_ensemble_enn_plot_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + (
            ""
            if allowed_max_num_samples is None
            else "_mns" + "_".join([str(a) for a in allowed_max_num_samples])
        )
        + (
            ""
            if allowed_data_ratios is None
            else "_dr" + "_".join([str(a) for a in allowed_data_ratios])
        )
    )


    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset"
    else:
        file_prefix = "_ll" if is_log_likelihood else "_kl"


    plot.save(
        "plots/" + name + file_prefix + ".png",
        dpi=600,
    )
    frame.to_csv("plots/" + name + file_prefix + ".csv")
    create_tex_table(
        frame,
        "all",
        name,
    )


def plot_combined_ranked_ensemble_summary(
    files,
    allowed_input_dims,
    parse_experiment_parameters=parse_enn_experiment_parameters,
    allowed_max_num_samples=None,
    allowed_data_ratios=None,
    is_log_likelihood=False,
    y_limit=(0.1, 10),
):

    all_agent_frames = {}
    all_experiment_params = {}
    all_experiment_files = {}

    max_num_samples = None

    for file in files:
        agent_frames = read_data(file)
        experiment_params = parse_experiment_parameters(file, True)

        if experiment_params["input_dim"] not in allowed_input_dims:
            print("Skipping file", file, "due to input dim filter")
            continue

        if (allowed_data_ratios is not None) and (
            experiment_params["data_ratio"] not in allowed_data_ratios
        ):
            print("Skipping file", file, "due to data ratio filter")
            continue

        if (allowed_max_num_samples is not None) and (
            experiment_params["max_num_samples"] not in allowed_max_num_samples
        ):
            print("Skipping file", file, "due to allowed_max_num_samples filter")
            continue

        print("Frame", {k: len(v) for k, v in agent_frames.items()}, "from file", file)

        if experiment_params["max_num_samples"] is not None:
            max_num_samples = experiment_params["max_num_samples"]

        for agent in agent_frames.keys():

            # if agent in ["layer_ensemble"]:
            #     continue

            frame = agent_frames[agent]

            if agent not in all_agent_frames:
                all_agent_frames[agent] = []
                all_experiment_params[agent] = []
                all_experiment_files[agent] = []

            all_agent_frames[agent].append(frame)
            all_experiment_params[agent].append(experiment_params)
            all_experiment_files[agent].append(file)

    data = {
        "agent_full": [],
        "agent": [],
        "mean": [],
        "std": [],
        "low_mstd": [],
        "high_mstd": [],
        "low_lmstd": [],
        "high_lmstd": [],
        "last_mean": [],
        "last_std": [],
        "num_ensemble": [],
        "indexer": [],
    }

    for agent, all_frames in all_agent_frames.items():

        if agent not in summary_select_agent_params:
            print(f"Skipping agent {agent} due to summary_select_agent_params filter")
            continue

        params = agent_plot_params[agent]
        filters = summary_select_agent_params[agent]

        for filter in filters:

            frames = all_frames
            old_frames = None
            agent_suffix = ""

            for key, value in filter.items():

                if key == "agent_suffix":
                    agent_suffix = value
                    continue

                if key == "max_num_samples":
                    if max_num_samples is None:
                        raise ValueError(
                            "max_num_samples is not set in the experiment parameters"
                        )
                    if value[0] not in max_num_samples:
                        raise ValueError(f"Empty frame after filtering for {key}={value}")
                    continue

                old_frames = frames
                frames = [f[f[key].isin(value)] for f in frames]
                if len(frames[0]) <= 0:
                    raise ValueError(f"Empty frame after filtering for {key}={value}")

            y_name = params["y"]
            val_y_name = params["val_ll_y" if is_log_likelihood else "val_y"]

            best_ys = []
            best_val_ys = []
            best_indexers = []
            last_ys = []

            for i in range(len(frames)):
                best_val_idx = (
                    frames[i][val_y_name].idxmax()
                    if is_log_likelihood
                    else frames[i][val_y_name].idxmin()
                )
                best_val_row = frames[i].loc[best_val_idx]
                best_y = best_val_row[y_name]
                best_val_y = best_val_row[val_y_name]
                best_ys.append(best_y)
                best_val_ys.append(best_val_y)
                best_indexers.append(best_val_row["indexer"])
                last_ys.append(frames[i][y_name].iloc[-1])

            mean = sum(best_ys) / len(best_ys)
            std = sum((y - mean) ** 2 for y in best_ys) / len(best_ys)

            val_mean = sum(best_val_ys) / len(best_val_ys)
            val_std = sum((y - val_mean) ** 2 for y in best_val_ys) / len(best_val_ys)

            last_mean = sum(last_ys) / len(last_ys)
            last_std = sum((y - last_mean) ** 2 for y in last_ys) / len(last_ys)

            mean_indexer = sum(best_indexers) / len(best_indexers)

            if is_log_likelihood:
                val_mean = -val_mean

            mean = min(y_limit[1], max(y_limit[0], mean))
            val_mean = min(y_limit[1], max(y_limit[0], val_mean))
            std = min(limit_std, std)
            val_std = min(limit_std, val_std)

            data["agent_full"].append((agent + agent_suffix).replace("_", "\n"))
            data["agent"].append(agent)
            data["mean"].append(mean)
            data["std"].append(std)
            data["low_mstd"].append(max(y_limit[0], mean - std))
            data["high_mstd"].append(min(y_limit[1], mean + std))
            data["last_mean"].append(last_mean)
            data["last_std"].append(last_std)
            data["low_lmstd"].append(max(y_limit[0], last_mean - last_std))
            data["high_lmstd"].append(min(y_limit[1], last_mean + last_std))
            data["num_ensemble"].append(
                max_num_samples
                if max_num_samples is not None
                else int(frames[0]["num_ensemble"])
            )
            data["indexer"].append(mean_indexer)

    frame = DataFrame(data)

    frame["agent_full"] = Categorical(
        frame["agent_full"],
        [
            "dropout",
            "bbb",
            "vnn",
            "hypermodel",
            # "ensemble\n10",
            "ensemble\n30",
            "layer\nensembles",
        ],
    )
    plot = (
        ggplot(frame)
        + aes(x="agent_full", y="mean")
        + geom_hline(yintercept=1)
        # + facet_grid("num_ensemble ~ agent", space="free", scales="free")
        + facet_wrap(
            ["num_ensemble"],
            ncol=3,
            labeller=labeller(
                cols=lambda x: (
                    "Full inference KL vs. "
                    + ("NLL" if is_log_likelihood else "KL")
                    + "-based ranking KL"
                )
            ),
        )
        + scale_y_continuous(trans="log10", limits=y_limit)
        # + scale_x_continuous(trans="log10")
        + geom_errorbar(
            aes(colour="'Full Inference'", ymin="low_mstd", ymax="high_mstd"),
            width=0.12,
            size=0.9,
            alpha=0.6,
        )
        + geom_errorbar(
            aes(colour="'Ranked Selection'", ymin="low_lmstd", ymax="high_lmstd"),
            width=0.08,
            size=0.4,
            alpha=0.6,
        )
        + geom_point(aes(colour="agent_full"), size=6, stroke=0.1)
        + geom_point(
            aes(x="agent_full", y="last_mean", colour="'Last Sample'"),
            size=3,
            stroke=0.1,
            shape="s",
        )
        + theme(
            axis_title=element_text(size=15),
            axis_text=element_text(size=8),
            figure_size=(12, 4),
        )
        + scale_color_discrete(guide=guide_legend())
        + scale_x_discrete(guide=False)
        + ylab("Mean Best Test KL / Full KL")
        + xlab("Method")
    )

    name = (
        f"combined_summary_ranked_ensemble_enn_plot_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + (
            ""
            if allowed_max_num_samples is None
            else "_mns" + "_".join([str(a) for a in allowed_max_num_samples])
        )
        + (
            ""
            if allowed_data_ratios is None
            else "_dr" + "_".join([str(a) for a in allowed_data_ratios])
        )
    )

    plot.save(
        "plots/" + name + ("_ll" if is_log_likelihood else "_kl") + ".png",
        dpi=600,
    )
    frame.to_csv("plots/" + name + ("_ll" if is_log_likelihood else "_kl") + ".csv")
    create_tex_table(
        frame,
        "all",
        name + ("_ll" if is_log_likelihood else "_kl"),
    )


def plot_all_hyperexperiment_frames(
    files, parse_experiment_parameters=parse_enn_experiment_parameters
):

    all_experiment_agent_frames = {}

    for file in files:
        agent_frames = read_data(file)
        for agent in agent_frames.keys():
            frame = agent_frames[agent]

            experiment_params = parse_experiment_parameters(file)

            for name, value in experiment_params.items():
                key = str(name) + ":" + str(value)
                if key not in all_experiment_agent_frames:
                    all_experiment_agent_frames[key] = {}

                if agent not in all_experiment_agent_frames[key]:
                    all_experiment_agent_frames[key][agent] = []

                all_experiment_agent_frames[key][agent].append(frame)

            key = "all"
            if key not in all_experiment_agent_frames:
                all_experiment_agent_frames[key] = {}

            if agent not in all_experiment_agent_frames[key]:
                all_experiment_agent_frames[key][agent] = []

            all_experiment_agent_frames[key][agent].append(frame)

    for (
        experiment_param,
        all_agent_frames,
    ) in all_experiment_agent_frames.items():
        for agent, frames in all_agent_frames.items():
            if len(frames) > 0:
                plot_multiple_frames(
                    frames,
                    agent,
                    "hyperexperiment_enn_plot_" + experiment_param + "_" + agent,
                )


def plot_summary_from_csv(
    file,
):

    frame = read_csv(file)
    frame = frame.sort_values(by=["mean"])

    frame["agent"] = Categorical(
        frame["agent"],
        [
            "dropout",
            "bbb",
            "vnn",
            "hypermodel",
            "ensemble",
            "lens",
        ],
    )

    plot = (
        ggplot(frame)
        + aes(x="agent", y="mean")
        + geom_hline(yintercept=1)
        + scale_y_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=4, stroke=0.2)
        + geom_errorbar(
            aes(colour="agent", ymin="mean-std", ymax="mean+std"),
            width=0.8,
            size=1.5,
        )
        + theme(axis_title=element_text(size=15), axis_text=element_text(size=9))
        + scale_color_discrete(guide=False)
        # + scale_x_discrete(guide=guide_legend())
        + ylab("Mean KL")
        + xlab("Method")
    )
    plot.save(
        file + ".png",
        dpi=600,
    )


def create_combined_summary_plots(
    summary_input_dims=[[1, 10, 100, 1000], [10, 100, 1000]], 
    results_folder="results",
):

    files = glob(results_folder + "/results_*.txt")

    for ids in summary_input_dims:
        plot_summary(files, ids)


def create_ranked_vnn_plots(
    num_samples=100,
    summary_input_dims=[[10, 100, 1000]],
    results_folder="results",
    file_filter="results_vnn*",
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    global summary_select_agent_params
    summary_select_agent_params = {}

    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset.txt"
    else:
        file_prefix = "_ll.txt" if is_log_likelihood else "_kl.txt"

    files = glob(
        results_folder
        + "/"
        + file_filter
        + file_prefix
    )
    # early_stopping_mode="loss" if is_log_likelihood else "kl"
    early_stopping_mode="kl"
    make_vnn_ranked_params(num_samples, use_ranked=True, early_stopping_mode=early_stopping_mode)

    for ids in summary_input_dims:
        plot_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=[num_samples],
            agent_name="VNN",
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_ranked_bbb_plots(
    num_samples=100,
    summary_input_dims=[[1, 10, 100, 1000]],
    results_folder="results",
    file_filter="results_bbb*",
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    global summary_select_agent_params
    summary_select_agent_params = {}

    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset.txt"
    else:
        file_prefix = "_ll.txt" if is_log_likelihood else "_kl.txt"

    files = glob(
        results_folder
        + "/"
        + file_filter
        + file_prefix
    )
    early_stopping_mode="kl"
    make_bbb_ranked_params(num_samples, use_ranked=True, early_stopping_mode=early_stopping_mode)

    for ids in summary_input_dims:
        plot_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=[num_samples],
            agent_name="BBB",
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_ranked_dropout_plots(
    num_samples=100,
    summary_input_dims=[[1, 10, 100, 1000]],
    results_folder="results",
    file_filter="results_dropout*",
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    global summary_select_agent_params
    summary_select_agent_params = {}

    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset.txt"
    else:
        file_prefix = "_ll.txt" if is_log_likelihood else "_kl.txt"

    files = glob(
        results_folder
        + "/"
        + file_filter
        + file_prefix
    )
    early_stopping_mode="kl"
    make_dropout_ranked_params(num_samples, use_ranked=True, early_stopping_mode=early_stopping_mode)

    for ids in summary_input_dims:
        plot_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=[num_samples],
            agent_name="Dropout",
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_ranked_hypermodel_plots(
    num_samples=100,
    summary_input_dims=[[1, 10, 100, 1000]],
    results_folder="results",
    file_filter="results_hypermodel*",
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    global summary_select_agent_params
    summary_select_agent_params = {}

    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset.txt"
    else:
        file_prefix = "_ll.txt" if is_log_likelihood else "_kl.txt"

    files = glob(
        results_folder
        + "/"
        + file_filter
        + file_prefix
    )
    early_stopping_mode="kl"
    make_hypermodel_ranked_params(num_samples, use_ranked=True, early_stopping_mode=early_stopping_mode)

    for ids in summary_input_dims:
        plot_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=[num_samples],
            agent_name="Hypermodels",
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_ranked_ensemble_plots(
    num_samples=30,
    summary_input_dims=[[1, 10, 100, 1000]],
    results_folder="results",
    file_filter="results_ensemble*",
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    global summary_select_agent_params
    summary_select_agent_params = {}

    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset.txt"
    else:
        file_prefix = "_ll.txt" if is_log_likelihood else "_kl.txt"

    files = glob(
        results_folder
        + "/"
        + file_filter
        + file_prefix
    )
    early_stopping_mode="kl"
    make_ensemble_ranked_params(num_samples, use_ranked=True, early_stopping_mode=early_stopping_mode)

    for ids in summary_input_dims:
        plot_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=[num_samples],
            agent_name="Ensembles",
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_ranked_layer_ensembles_plots(
    num_samples=125,
    summary_input_dims=[[1, 10, 100, 1000]],
    results_folder="results",
    file_filter="results_layer_ensembles*",
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    global summary_select_agent_params
    summary_select_agent_params = {}

    file_prefix = ""

    if is_randomset:
        file_prefix = "_randomset.txt"
    else:
        file_prefix = "_ll.txt" if is_log_likelihood else "_kl.txt"

    files = glob(
        results_folder
        + "/"
        + file_filter
        + file_prefix
    )
    early_stopping_mode="kl"
    make_layer_ensemble_ranked_params(num_samples, use_ranked=True, early_stopping_mode=early_stopping_mode)

    for ids in summary_input_dims:
        plot_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=[num_samples],
            agent_name="LayerEnsembles",
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_ranked_dbnn_plots(
    num_samples=100,
    max_num_samples_lens=27,
    max_num_samples_ens=30,
    summary_input_dims=[[10, 100, 1000]],
    agents="all",
    results_folder="results",
    file_filter=None,
    is_log_likelihood=False,
    is_randomset=False,
    allowed_data_ratios=None,
):

    extra_params = {}

    if file_filter is not None:
        extra_params["file_filter"] = file_filter

    if agents == "all" or "vnn" in agents:
        create_ranked_vnn_plots(
            num_samples=num_samples,
            summary_input_dims=summary_input_dims,
            results_folder=results_folder,
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )

    if agents == "all" or "bbb" in agents:
        create_ranked_bbb_plots(
            num_samples=num_samples,
            summary_input_dims=summary_input_dims,
            results_folder=results_folder,
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )

    if agents == "all" or "dropout" in agents:
        create_ranked_dropout_plots(
            num_samples=num_samples,
            summary_input_dims=summary_input_dims,
            results_folder=results_folder,
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )

    if agents == "all" or "hypermodel" in agents:
        create_ranked_hypermodel_plots(
            num_samples=num_samples,
            summary_input_dims=summary_input_dims,
            results_folder=results_folder,
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )

    if agents == "all" or "ensemble" in agents and not "layer_ensembles" in agents:
        create_ranked_ensemble_plots(
            num_samples=max_num_samples_ens,
            summary_input_dims=summary_input_dims,
            results_folder=results_folder,
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )

    if agents == "all" or "layer_ensembles" in agents:
        create_ranked_layer_ensembles_plots(
            num_samples=max_num_samples_lens,
            summary_input_dims=summary_input_dims,
            results_folder=results_folder,
            is_log_likelihood=is_log_likelihood,
            is_randomset=is_randomset,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )


def create_all_ranked_dbnn_plots(
    max_num_samples=(10, 100, 1000),
    max_num_samples_lens=27,
    max_num_samples_ens=30,
    summary_input_dims=[[10, 100, 1000]],
    agents="all",
    results_folder="results",
    file_filter=None,
    allowed_data_ratios=None,
):

    extra_params = {}

    if file_filter is not None:
        extra_params["file_filter"] = file_filter

    for num_samples in max_num_samples:
        create_ranked_dbnn_plots(
            num_samples=num_samples,
            max_num_samples_lens=max_num_samples_lens,
            max_num_samples_ens=max_num_samples_ens,
            summary_input_dims=summary_input_dims,
            agents=agents,
            results_folder=results_folder,
            is_log_likelihood=False,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )

        create_ranked_dbnn_plots(
            num_samples=num_samples,
            max_num_samples_lens=max_num_samples_lens,
            max_num_samples_ens=max_num_samples_ens,
            summary_input_dims=summary_input_dims,
            agents=agents,
            results_folder=results_folder,
            is_log_likelihood=True,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )
        
        create_ranked_dbnn_plots(
            num_samples=num_samples,
            max_num_samples_lens=max_num_samples_lens,
            max_num_samples_ens=max_num_samples_ens,
            summary_input_dims=summary_input_dims,
            agents=agents,
            results_folder=results_folder,
            is_randomset=True,
            allowed_data_ratios=allowed_data_ratios,
            **extra_params,
        )


def create_combined_ranked_dbnn_plots(
    num_samples=[30, 100, 125],
    summary_input_dims=[[10, 100, 1000]],
    agents="all",
    results_folder="results",
    file_filters=None,
    is_log_likelihood=False,
    allowed_data_ratios=None,
):

    if file_filters is None:
        file_filters = ["results_*"]

    files = []

    for file_filter in file_filters:
        files += glob(
            results_folder
            + "/"
            + file_filter
            + ("_ll.txt" if is_log_likelihood else "_kl.txt")
        )

    for ids in summary_input_dims:
        plot_combined_ranked_ensemble_summary(
            files,
            ids,
            allowed_max_num_samples=num_samples,
            is_log_likelihood=is_log_likelihood,
            allowed_data_ratios=allowed_data_ratios,
        )


def create_all_combined_ranked_dbnn_plots(
    num_samples=[30, 100, 125],
    summary_input_dims=[[10, 100, 1000]],
    agents="all",
    results_folder="results",
    file_filters=None,
    is_log_likelihood=False,
    allowed_data_ratios=None,
):

    create_combined_ranked_dbnn_plots(
        num_samples=num_samples,
        summary_input_dims=summary_input_dims,
        agents=agents,
        results_folder=results_folder,
        file_filters=file_filters,
        is_log_likelihood=False,
        allowed_data_ratios=allowed_data_ratios,
    )
    create_combined_ranked_dbnn_plots(
        num_samples=num_samples,
        summary_input_dims=summary_input_dims,
        agents=agents,
        results_folder=results_folder,
        file_filters=file_filters,
        is_log_likelihood=True,
        allowed_data_ratios=allowed_data_ratios,
    )


if __name__ == "__main__":
    Fire()
