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
    labeller
)
from plotnine.data import economics
from pandas import Categorical, DataFrame, read_csv
from plotnine.scales.limits import ylim
from plotnine.scales.scale_xy import scale_x_discrete
from plotnine.guides import guide_axis, guide_legend, guide
from glob import glob
import re

limit_std = 100

tex_template_file = "tools/tex_table_template.tex"
use_ranked_layer_enbsemble = True

with open(tex_template_file, "r") as f:
    tex_template = f.read()

# files = glob("results_vnn_selected*")
# files = glob("results/results_*")
files = glob("results/results_*layer*")
# files = glob("results_all_old*") + glob("results_vnn_selected*")
# files = glob("results_mserr*") + glob("results_lrelu*")
# files = glob("results/results_best_selected_val_*") + glob("results/results_mserr*")
# files = glob("results/results_mserr_layer_ensemble*")
# files = glob("results/results_best_selected_val_*") + glob("results/results_multi_indexed_val_*")
# files = glob("results/results_batched_multi_indexed_val_*")
# files = glob("results/results_ranked_batched_multi_indexed_val_*")
# files = glob("results1/results_best_selected_val_true_layer_ensemble_einsum_cor*")

float_fields = [
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
    "kl_std": "Var[KL]",
    "std": "Var[KL]",
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
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "subsample_ensemble": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "dropout": {
        "x": "dropout_rate",
        "y": "kl",
        "facet": ["regularization_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "hypermodel": {
        "x": "index_dim",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "bbb": {
        "x": "sigma_0",
        "y": "kl",
        "facet": ["learning_rate"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "vnn": {
        "x": "num_batches",
        "y": "kl",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "vnn_lrelu": {
        "x": "num_batches",
        "y": "kl",
        "facet": ["activation_mode", "global_std_mode"],
        # "facet": ["activation_mode", "global_std_mode", "num_index_samples"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
        "fill": "activation",
    },
    "vnn_init": {
        "x": "num_batches",
        "y": "kl",
        "facet": ["activation_mode", "global_std_mode", "loss_function"],
        "colour": "activation",
        "shape": "factor(hidden_size)",
        "fill": "initializer",
    },
    "layer_ensemble": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "layer_ensemble_cor": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "layer_ensemble_einsum_cor": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "true_layer_ensemble_einsum": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "prior_scale"],
        "colour": "factor(num_layers)",
        "shape": "factor(hidden_size)",
    },
    "true_layer_ensemble_einsum_cor": {
        "x": "num_ensemble",
        "y": "kl",
        "facet": ["noise_scale", "sample_type"],
        "colour": "factor(num_layers)",
        "shape": "factor(prior_scale)",
    },
}


summary_select_agent_params = {
    "ensemble": [
        # {
        #     "agent_suffix": "_3",
        #     "noise_scale": [1.0],
        #     "prior_scale": [1.0],
        #     "num_layers": [2],
        #     "hidden_size": [50],
        #     "num_ensemble": [3],
        # },
        {
            "agent_suffix": "_10",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [10],
        },
        {
            "agent_suffix": "_30",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [30],
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
        {
            "activation": ["relu", "tanh"],
            # "activation_mode": ["mean"],
            # "global_std_mode": ["multiply"],
            "activation_mode": ["none"],
            "global_std_mode": ["multiply"],
            "num_layers": [3],
            "hidden_size": [100],
            "num_index_samples": [100],
            "num_batches": ["1000"],
        },
    ],
    "layer_ensemble": [
        {
            "agent_suffix": "_2",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [2],
        },
        {
            "agent_suffix": "_3",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [3],
        },
        {
            "agent_suffix": "_5",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [5],
        },
    ],
    "layer_ensemble_cor": [
        {
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [10],
        },
    ],
    "layer_ensemble_einsum_cor": [
        {
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [10],
        },
    ],
    "true_layer_ensemble_einsum": [
        {
            "agent_suffix": "_2",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [2],
        },
        {
            "agent_suffix": "_3",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [3],
        },
        {
            "agent_suffix": "_4",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [4],
        },
        {
            "agent_suffix": "_5",
            "noise_scale": [1.0],
            "prior_scale": [1.0],
            "num_layers": [2],
            "hidden_size": [50],
            "num_ensemble": [5],
        },
    ],
    "true_layer_ensemble_einsum_cor": [],
    "subsample_ensemble": [],
}


def add_true_layer_ensemble_einsum_cor_summary_params():

    if use_ranked_layer_enbsemble:
        all_nens_samples = [
            (2, [*range(2, 2 ** 3)]),
            (3, [*range(2, 3 ** 3)]),
            (5, [*range(2, 5 ** 3)]),
            (6, [*range(2, 6 ** 3)]),
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

            if use_ranked_layer_enbsemble:
                indexer = samples
            else:
                indexer = (
                    num_ensemble ** 3 if samples == "full" else samples * num_ensemble
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
            summary_select_agent_params["true_layer_ensemble_einsum_cor"].append(params)


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
                "noise_scale": [1.0],
                "prior_scale": [1.0],
                "num_layers": [2],
                "hidden_size": [50],
                "num_ensemble": [num_ensemble],
                "indexer": [indexer],
            }
            summary_select_agent_params["subsample_ensemble"].append(params)


add_true_layer_ensemble_einsum_cor_summary_params()
add_subsample_ensemble_summary_params()

summary_input_dims = [
    # [1],
    [10],
    # [100],
    # [1000],
    # [10, 100],
    # [10, 100, 1000],
    # [1, 10, 100],
    # [1, 10, 100, 1000]
]


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
                + file.replace(".txt", "").replace("results/", "").replace("results1/", ""),
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


def parse_enn_experiment_parameters(file):

    param_string = file.split("_")[-1]
    input_dim, data_ratio, noise_std = re.findall(r"\d+(?:\.\d+|\d*)", param_string)

    input_dim = int(input_dim)
    data_ratio = float(data_ratio)
    noise_std = float(noise_std)

    return {
        "input_dim": input_dim,
        "data_ratio": data_ratio,
        "noise_std": noise_std,
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
            print("scipping file", file, "due to input dim filter")
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
            print(f"Skippng agent {agent} due to summary_select_agent_params filter")
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
                    raise ValueError("Empty frame after filtering")

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
    frame["agent"] = Categorical(
        frame["agent"],
        [
            "dropout",
            "bbb",
            "vnn",
            "hypermodel",
            # "ensemble\n3",
            "ensemble\n10",
            "ensemble\n30",
            # "layer_ensemble",
            # "layer_ensemble_cor",
            # "layer_ensemble_einsum_cor",
            # "true_layer_ensemble_einsum",
            "true_layer_ensemble_einsum_cor",
        ],
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
            aes(colour="agent", ymin="mean-std", ymax="mean+std"), width=0.8, size=1.5,
        )
        + theme(axis_title=element_text(size=15), axis_text=element_text(size=4))
        + scale_color_discrete(guide=False)
        # + scale_x_discrete(guide=guide_legend())
        + ylab("Mean KL")
        + xlab("Method")
    )
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
            print(f"Skippng agent {agent} due to summary_select_agent_params filter")
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
                    raise ValueError("Empty frame after filtering")

            mean = sum(sum(f[params["y"]]) for f in frames) / sum(
                len(f) for f in frames
            )
            std = sum(sum((f[params["y"]] - mean) ** 2) for f in frames) / sum(
                len(f) for f in frames
            )

            data["agent_full"].append((agent + agent_suffix).replace("_", "\n"))
            data["agent"].append(agent.replace("subsample_ensemble", "Deep Ensembles").replace("true_layer_ensemble_einsum_cor", "Layer Ensembles"))
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
        + facet_wrap(["num_ensemble"], nrow=2, labeller=labeller(cols=lambda x: str(x) + " ensembles"))
        + scale_y_continuous(trans="log10")
        + scale_x_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=2, stroke=0.1)
        + geom_errorbar(
            aes(colour="agent", ymin="mean-std", ymax="mean+std"), width=0.3, size=0.9,
        )
        + theme(axis_title=element_text(size=17), axis_text=element_text(size=10), figure_size=(6, 4))
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
            print(f"Skippng agent {agent} due to summary_select_agent_params filter")
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
                    raise ValueError("Empty frame after filtering")

            mean = sum(sum(f[params["y"]]) for f in frames) / sum(
                len(f) for f in frames
            )
            std = sum(sum((f[params["y"]] - mean) ** 2) for f in frames) / sum(
                len(f) for f in frames
            )

            data["agent_full"].append((agent + agent_suffix).replace("_", "\n"))
            data["agent"].append(agent)
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
        + facet_wrap(["num_ensemble"], ncol=3, labeller=labeller(cols=lambda x: str(x) + " ensembles"))
        + scale_y_continuous(trans="log10")
        + scale_x_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=3, stroke=0.1)
        + geom_errorbar(
            aes(colour="'#FFFF00'", ymin="mean-std", ymax="mean+std"),
            width=0.07,
            size=0.9,
        )
        + theme(axis_title=element_text(size=15), axis_text=element_text(size=8), figure_size=(12, 4))
        + scale_color_discrete(guide=False)
        # + scale_x_discrete(guide=guide_legend())
        + ylab("Mean KL")
        + xlab("Number of samples")
    )
    plot.save(
        "plots/summary_ranked_ensemble_enn_plot_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + ".png",
        dpi=600,
    )
    frame.to_csv(
        "plots/summary_ranked_ensemble_enn_id"
        + "_".join([str(a) for a in allowed_input_dims])
        + ".csv"
    )
    create_tex_table(
        frame,
        "all",
        "summary_ranked_ensemble_enn_plot_id" + "_".join([str(a) for a in allowed_input_dims]),
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

    for (experiment_param, all_agent_frames,) in all_experiment_agent_frames.items():
        for agent, frames in all_agent_frames.items():
            if len(frames) > 0:
                plot_multiple_frames(
                    frames,
                    agent,
                    "hyperexperiment_enn_plot_" + experiment_param + "_" + agent,
                )


def plot_optimized_layer_ensemble_speed():

    frame = DataFrame(
        {
            "Ensembles": [2, 3, 4, 5, 6, 7, 8, 9, 10,],
            "Speed Up": [
                196.02435110737994 / 111.8295512448305,
                57.80796495367296 / 32.03804974678484,
                24.843940387464368 / 5.64624682777498,
                16.562746831583034 / 1.5746171174634735,
                9.675929083702409 / 0.6817255399724734,
                5.729144740571466 / 0.3516358885652403,
                3.590836155147295 / 0.20132821390009117,
                2.3235564106985414 / 0.12423794040411039,
                1.466085111750885 / 0.08107119449007497,
            ],
            "Memory Save": [
                (1234 - 954) / (1162 - 954),
                (1352 - 976) / (1252 - 976),
                (1516 - 998) / (1342 - 998),
                (1762 - 1020) / (1434 - 1020),
                (2140 - 1044) / (1524 - 1044),
                (2702 - 1066) / (1614 - 1066),
                (3518 - 1088) / (1706 - 1088),
                (4664 - 1110) / (1798 - 1110),
                (6226 - 1132) / (1890 - 1132),
            ],
        }
    )

    def plot_frame(frame, output_file_name):

        plot1 = (
            ggplot(frame)
            + aes(x="Ensembles", y="Speed Up")
            + geom_point(aes(), size=4, stroke=0.1)
            + geom_line(size=1)
            + theme(axis_title=element_text(size=20), axis_text=element_text(size=16))
        )

        plot1.save("plots/" + output_file_name + "_speed.png", dpi=600)

        plot2 = (
            ggplot(frame)
            + aes(x="Ensembles", y="Memory Save")
            + geom_point(aes(), size=4, stroke=0.1)
            + geom_line(size=1)
            + theme(axis_title=element_text(size=20), axis_text=element_text(size=16))
        )

        plot2.save("plots/" + output_file_name + "_memory.png", dpi=600)

    plot_frame(frame, "optimized_layer_ensemble")


def plot_summary_from_csv(
    file,
):

    frame = read_csv(file)
    frame = frame.sort_values(by=['mean'])

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
        +
        # ylim(0, 2) +
        scale_y_continuous(trans="log10")
        + geom_point(aes(colour="agent"), size=4, stroke=0.2)
        + geom_errorbar(
            aes(colour="agent", ymin="mean-std", ymax="mean+std"), width=0.8, size=1.5,
        )
        + theme(axis_title=element_text(size=15), axis_text=element_text(size=9))
        + scale_color_discrete(guide=False)
        # + scale_x_discrete(guide=guide_legend())
        + ylab("Mean KL")
        + xlab("Method")
    )
    plot.save(
        file
        + ".png",
        dpi=600,
    )


# plot_summary_from_csv("plots_ranked/summary_all_enn_id10_100.csv")

# plot_optimized_layer_ensemble_speed()

# plot_summary(files, [100])
# plot_summary_vnn(files, [10, 100, 1000])

for ids in summary_input_dims:
    plot_ranked_ensemble_summary(files, ids)

# for ids in summary_input_dims:
#     plot_ensemble_summary(files, ids)

# plot_all_hyperexperiment_frames(files)
# plot_all_single_frames(files)
