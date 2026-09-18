#!/usr/bin/env python3
"""Find best parameters from result files and plot metric distribution.

This script scans a results folder recursively, parses text result files, and:
1) Finds best parameter sets for each agent using both `val_loss` and `val_kl`.
2) Computes the most influential parameters for a selected metric.
3) Creates a scatter plot of selected metric vs KL with color/shape mapped to
   top influential parameters.

Outputs are written into one subfolder per agent type.

Example:
  python tools/find_best_parameters.py \
    --results-folder ./results \
    --metric val_loss \
    --output-dir ./results/analysis
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fire
from tqdm import tqdm


EXCLUDED_INFLUENCE_FIELDS = {
    "agent_id",
    "kl",
    "kl_variance",
    "mean_error",
    "std_error",
    "best_epoch",
    "val_loss",
    "val_kl",
    "val_ll",
    "val_kl_mean",
    "val_kl_variance",
    "val_ll_mean",
    "val_ll_variance",
    "val_loss_mean",
    "val_loss_variance",
    "file_path",
    "line_no",
    "agent",
    "input_dim",
    "data_ratio",
    "noise_std",

    # Skipped by models, redundant
    "batch_norm_mode"
}


AGENT_EXCLUDED_INFLUENCE_FIELDS = {
    "ensemble": {"max_num_samples"},
}


DATASET_PATTERN = re.compile(
    r"_id(?P<input_dim>-?\d+)dr(?P<data_ratio>-?\d+(?:\.\d+)?)ns(?P<noise_std>-?\d+(?:\.\d+)?)"
)


def to_number(value: str) -> Any:
    low = value.lower()
    if low == "none":
        return None
    if low == "true":
        return True
    if low == "false":
        return False

    try:
        if "." not in value and "e" not in low:
            return int(value)
        return float(value)
    except ValueError:
        return value


def normalize_metric_fields(record: Dict[str, Any]) -> None:
    if "val_kl" not in record and "val_kl_mean" in record:
        record["val_kl"] = record["val_kl_mean"]
    if "val_loss" not in record and "val_loss_mean" in record:
        record["val_loss"] = record["val_loss_mean"]


def add_dataset_fields(record: Dict[str, Any], file_path: str) -> None:
    match = DATASET_PATTERN.search(file_path)
    if not match:
        return

    record["input_dim"] = int(match.group("input_dim"))
    record["data_ratio"] = float(match.group("data_ratio"))
    record["noise_std"] = float(match.group("noise_std"))


def parse_line(line: str, file_path: str, line_no: int) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split(" ")
    if len(parts) < 2:
        return None

    try:
        agent_id = int(parts[0])
        kl = float(parts[1])
    except ValueError:
        return None

    raw_params: List[str] = []
    for token in parts[2:]:
        if "=" in token:
            raw_params.append(token)
        elif raw_params:
            raw_params[-1] += " " + token

    record: Dict[str, Any] = {
        "agent_id": agent_id,
        "kl": kl,
        "file_path": file_path,
        "line_no": line_no,
    }

    for token in raw_params:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        record[key] = to_number(value)

    if "agent" not in record:
        try:
            rel_parts = os.path.relpath(file_path).split(os.sep)
            if len(rel_parts) >= 2:
                record["agent"] = rel_parts[-2]
            else:
                record["agent"] = "unknown"
        except Exception:
            record["agent"] = "unknown"

    normalize_metric_fields(record)
    add_dataset_fields(record, file_path)
    return record


def mean_and_variance(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance


def aggregate_agent_dataset_metric_stats(
    records: Sequence[Dict[str, Any]], metric: str
) -> Dict[str, Dict[str, float]]:
    values_by_agent_dataset: Dict[str, Dict[Tuple[int, float, float], List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in records:
        if (
            metric not in row
            or not is_number(row[metric])
            or row.get("input_dim") is None
            or row.get("data_ratio") is None
            or row.get("noise_std") is None
        ):
            continue

        agent = str(row.get("agent", "unknown"))
        dataset_key = (
            int(row["input_dim"]),
            float(row["data_ratio"]),
            float(row["noise_std"]),
        )
        values_by_agent_dataset[agent][dataset_key].append(float(row[metric]))

    stats: Dict[str, Dict[str, float]] = {}
    for agent, by_dataset in values_by_agent_dataset.items():
        dataset_means = [sum(vs) / len(vs) for vs in by_dataset.values() if vs]
        mean, variance = mean_and_variance(dataset_means)
        stats[agent] = {
            f"{metric}_dataset_mean": mean,
            f"{metric}_dataset_variance": variance,
            "num_datasets": float(len(dataset_means)),
        }

    return stats


def load_records(results_folder: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(results_folder, "*.txt")
    records: List[Dict[str, Any]] = []

    for file_path in tqdm(glob.glob(pattern, recursive=True), desc="Loading records"):
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    parsed = parse_line(line, file_path, line_no)
                    if parsed is not None:
                        records.append(parsed)
        except (OSError, UnicodeDecodeError):
            continue

    return records


def best_rows_by_metric(
    records: Sequence[Dict[str, Any]], metric: str
) -> List[Dict[str, Any]]:
    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in records:
        if metric in row and isinstance(row[metric], (int, float)) and row[metric] is not None:
            by_agent[str(row.get("agent", "unknown"))].append(row)

    best_rows: List[Dict[str, Any]] = []
    for agent, agent_rows in tqdm(by_agent.items(), desc="Finding best rows"):
        best = min(agent_rows, key=lambda r: float(r[metric]))
        best_copy = dict(best)
        best_copy["agent"] = agent
        best_rows.append(best_copy)

    best_rows.sort(key=lambda r: str(r.get("agent", "")))
    return best_rows


def top_rows_by_metric_per_agent(
    records: Sequence[Dict[str, Any]], metric: str, top_n: int = 5
) -> List[Dict[str, Any]]:
    aggregated_records = aggregate_rows_across_datasets(records, metric)
    by_agent: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in aggregated_records:
        if metric in row and isinstance(row[metric], (int, float)) and row[metric] is not None:
            by_agent[str(row.get("agent", "unknown"))].append(row)

    top_rows: List[Dict[str, Any]] = []
    for agent in sorted(by_agent.keys()):
        agent_rows = sorted(by_agent[agent], key=lambda row: float(row[metric]))[:top_n]
        for rank, row in enumerate(agent_rows, start=1):
            row_copy = dict(row)
            row_copy["agent"] = agent
            row_copy["rank_within_agent"] = rank
            top_rows.append(row_copy)

    return top_rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def prepare_values_for_grouping(values: Sequence[Any], bins: int = 5) -> List[str]:
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return ["none" for _ in values]

    if all(is_number(v) for v in clean_values):
        numeric_values = [float(v) for v in values if v is not None]
        unique_count = len(set(numeric_values))
        if unique_count > 10:
            sorted_vals = sorted(numeric_values)
            cut_points = []
            for i in range(1, bins):
                idx = int(i * len(sorted_vals) / bins)
                idx = min(max(idx, 0), len(sorted_vals) - 1)
                cut_points.append(sorted_vals[idx])

            grouped: List[str] = []
            for v in values:
                if v is None:
                    grouped.append("none")
                    continue
                fv = float(v)
                bucket = 0
                while bucket < len(cut_points) and fv > cut_points[bucket]:
                    bucket += 1
                grouped.append(f"bin_{bucket}")
            return grouped

    return ["none" if v is None else str(v) for v in values]


def influence_score(groups: Sequence[str], metric_values: Sequence[float]) -> float:
    if len(groups) != len(metric_values) or not metric_values:
        return 0.0

    overall_mean = sum(metric_values) / len(metric_values)
    total_var = sum((v - overall_mean) ** 2 for v in metric_values) / len(metric_values)
    if total_var <= 1e-12:
        return 0.0

    bucket_values: Dict[str, List[float]] = defaultdict(list)
    for group, metric_value in zip(groups, metric_values):
        bucket_values[group].append(metric_value)

    between = 0.0
    for values in bucket_values.values():
        mean_value = sum(values) / len(values)
        between += len(values) * (mean_value - overall_mean) ** 2

    between /= len(metric_values)
    return between / total_var


def compute_top_influential_parameters(
    rows: Sequence[Dict[str, Any]], metric: str, top_k: int = 2
) -> List[Tuple[str, float]]:
    filtered = [r for r in rows if metric in r and is_number(r[metric])]
    if not filtered:
        return []

    metric_values = [float(r[metric]) for r in filtered]
    all_keys = set().union(*(r.keys() for r in filtered))
    ignored_fields = set(EXCLUDED_INFLUENCE_FIELDS)
    for row in filtered:
        ignored_fields.update(
            AGENT_EXCLUDED_INFLUENCE_FIELDS.get(str(row.get("agent", "unknown")), set())
        )
    candidate_keys = [k for k in all_keys if k not in ignored_fields]

    scored: List[Tuple[str, float]] = []
    for key in tqdm(candidate_keys, desc="Computing influence scores"):
        values = [r.get(key, None) for r in filtered]
        grouped = prepare_values_for_grouping(values)
        if len(set(grouped)) < 2:
            continue
        score = influence_score(grouped, metric_values)
        if math.isfinite(score) and score > 0:
            scored.append((key, score))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:top_k]


def write_rows_csv(rows: Sequence[Dict[str, Any]], output_path: str) -> None:
    all_keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_influence_csv(influences: Sequence[Tuple[str, float]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "influence_score"])
        for key, score in influences:
            writer.writerow([key, score])


def extract_model_parameters(row: Dict[str, Any]) -> str:
    agent = str(row.get("agent", "unknown"))
    excluded_keys = {
        "agent_id",
        "agent",
        "file_path",
        "line_no",
        "input_dim",
        "data_ratio",
        "noise_std",
        "num_datasets",
        "rank_within_agent",
        "kl",
        "kl_mean",
        "kl_variance",
        "val_loss",
        "val_kl",
        "val_ll",
        "val_loss_mean",
        "val_loss_variance",
        "val_kl_mean",
        "val_kl_variance",
        "val_ll_mean",
        "val_ll_variance",
        "mean_error",
        "std_error",
        "best_epoch",
    }
    excluded_keys.update(AGENT_EXCLUDED_INFLUENCE_FIELDS.get(agent, set()))
    model_items = [
        (key, row[key])
        for key in sorted(row.keys())
        if key not in excluded_keys and row.get(key) is not None
    ]
    return "; ".join(f"{key}={value}" for key, value in model_items)


def add_model_parameters_column(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_copy = dict(row)
        row_copy["model_parameters"] = extract_model_parameters(row_copy)
        enriched_rows.append(row_copy)
    return enriched_rows


def safe_category(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


def aggregate_rows_across_datasets(
    rows: Sequence[Dict[str, Any]], metric: str
) -> List[Dict[str, Any]]:
    """Aggregate records by configuration and summarize metric across datasets."""
    exclude_group_fields = {
        "file_path",
        "line_no",
        "input_dim",
        "data_ratio",
        "noise_std",
        "kl",
        "val_loss",
        "val_kl",
        "val_ll",
        "val_kl_mean",
        "val_kl_variance",
        "val_ll_mean",
        "val_ll_variance",
        "val_loss_mean",
        "val_loss_variance",
        "mean_error",
        "std_error",
        "best_epoch",
    }
    exclude_group_fields.add(metric)

    grouped_metrics: Dict[Tuple[Tuple[str, str], ...], List[float]] = defaultdict(list)
    grouped_kls: Dict[Tuple[Tuple[str, str], ...], List[float]] = defaultdict(list)
    grouped_representative: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = {}
    grouped_datasets: Dict[Tuple[Tuple[str, str], ...], set] = defaultdict(set)

    for row in rows:
        if not is_number(row.get(metric)) or not is_number(row.get("kl")):
            continue

        group_items = tuple(
            sorted(
                (key, repr(value))
                for key, value in row.items()
                if key not in exclude_group_fields
            )
        )

        grouped_metrics[group_items].append(float(row[metric]))
        grouped_kls[group_items].append(float(row["kl"]))
        grouped_representative.setdefault(group_items, row)

        dataset_key = (
            row.get("input_dim"),
            row.get("data_ratio"),
            row.get("noise_std"),
        )
        grouped_datasets[group_items].add(dataset_key)

    aggregated_rows: List[Dict[str, Any]] = []
    for group_items, metric_values in grouped_metrics.items():
        metric_mean, metric_variance = mean_and_variance(metric_values)
        kl_mean, kl_variance = mean_and_variance(grouped_kls[group_items])

        row_copy = dict(grouped_representative[group_items])
        row_copy[metric] = metric_mean
        row_copy["kl"] = kl_mean
        row_copy[f"{metric}_mean"] = metric_mean
        row_copy[f"{metric}_variance"] = metric_variance
        row_copy["kl_mean"] = kl_mean
        row_copy["kl_variance"] = kl_variance
        row_copy["num_datasets"] = len(grouped_datasets[group_items])
        row_copy["metric_values"] = metric_values
        aggregated_rows.append(row_copy)

    return aggregated_rows


def make_distribution_plot(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    color_param: str,
    shape_param: str,
    output_path: str,
    metric_limit: float = 20.0,
) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    plot_rows = aggregate_rows_across_datasets(rows, metric)
    if not plot_rows:
        raise RuntimeError(f"No rows contain both '{metric}' and 'kl' for plotting.")

    plot_rows = [r for r in plot_rows if float(r[metric]) < metric_limit]

    if len(plot_rows) <= 0:
        print("No rows to plot.")
        return False

    color_categories = sorted({safe_category(r.get(color_param)) for r in plot_rows})
    shape_categories = sorted({safe_category(r.get(shape_param)) for r in plot_rows})

    distinct_colors = [
        "#d62728",  # red
        "#2ca02c",  # green
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
        "#bcbd22",  # olive
        "#7f7f7f",  # gray
    ]
    color_map = {
        category: distinct_colors[idx % len(distinct_colors)]
        for idx, category in enumerate(color_categories)
    }
    marker_list = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    marker_map = {
        category: marker_list[idx % len(marker_list)]
        for idx, category in enumerate(shape_categories)
    }

    mean_x = [float(r[metric]) for r in plot_rows]
    mean_y = [float(r["kl"]) for r in plot_rows]
    x_min, x_max = min(mean_x), max(mean_x)
    y_min, y_max = min(mean_y), max(mean_y)
    x_pad = (x_max - x_min) * 0.05 or x_max * 0.05 or 0.01
    y_pad = (y_max - y_min) * 0.05 or y_max * 0.05 or 0.01

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for row in tqdm(plot_rows, desc="Creating distribution plot"):
        color_category = safe_category(row.get(color_param))
        shape_category = safe_category(row.get(shape_param))
        metric_std = math.sqrt(max(float(row.get(f"{metric}_variance", 0.0)), 0.0))
        kl_std = math.sqrt(max(float(row.get("kl_variance", 0.0)), 0.0))
        ax.errorbar(
            float(row[metric]),
            float(row["kl"]),
            xerr=metric_std,
            yerr=kl_std,
            fmt=marker_map[shape_category],
            color=color_map[color_category],
            ecolor=color_map[color_category],
            elinewidth=1.3,
            capsize=2.8,
            alpha=0.8,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="none",
        )

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{color_param}={category}",
            markerfacecolor=color_map[category],
            markeredgecolor="black",
            markersize=8,
        )
        for category in color_categories
    ]

    shape_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[category],
            color="gray",
            label=f"{shape_param}={category}",
            linestyle="None",
            markersize=8,
        )
        for category in shape_categories
    ]

    legend1 = ax.legend(handles=color_handles, loc="upper right", title="Color")
    ax.add_artist(legend1)
    ax.legend(handles=shape_handles, loc="lower right", title="Shape")

    ax.set_xlabel(metric)
    ax.set_ylabel("kl")
    ax.set_title(
        f"Result distribution for {metric} and kl mean +/- std across datasets "
        + f"(color={color_param}, shape={shape_param})"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def default_plot_params(influences: Sequence[Tuple[str, float]]) -> Tuple[str, str]:
    if len(influences) >= 2:
        return influences[0][0], influences[1][0]
    if len(influences) == 1:
        return influences[0][0], "agent_id"
    return "agent", "agent_id"


def make_best_models_comparison_plot(
    rows: Sequence[Dict[str, Any]], metric: str, output_path: str
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    plot_rows = [row for row in rows if is_number(row.get(metric)) and is_number(row.get("kl"))]
    if not plot_rows:
        print(f"No rows to plot for best-model comparison with metric '{metric}'.")
        return False

    plot_rows = sorted(plot_rows, key=lambda row: str(row.get("agent", "unknown")))
    agents = [str(row.get("agent", "unknown")) for row in plot_rows]
    metric_means = [float(row[metric]) for row in plot_rows]
    metric_stds = [
        math.sqrt(max(float(row.get(f"{metric}_variance", 0.0)), 0.0)) for row in plot_rows
    ]
    kl_means = [float(row["kl"]) for row in plot_rows]
    kl_stds = [math.sqrt(max(float(row.get("kl_variance", 0.0)), 0.0)) for row in plot_rows]

    positive_indices = [
        idx
        for idx in range(len(plot_rows))
        if metric_means[idx] > 0.0 and kl_means[idx] > 0.0
    ]
    if not positive_indices:
        print(
            "No rows with strictly positive metric and kl values for log-scale best-model plot."
        )
        return False

    filtered_agents = [agents[idx] for idx in positive_indices]
    filtered_metric_means = [metric_means[idx] for idx in positive_indices]
    filtered_metric_stds = [metric_stds[idx] for idx in positive_indices]
    filtered_kl_means = [kl_means[idx] for idx in positive_indices]
    filtered_kl_stds = [kl_stds[idx] for idx in positive_indices]

    x_positions = list(range(len(filtered_agents)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    axes[0].errorbar(
        x_positions,
        filtered_metric_means,
        yerr=filtered_metric_stds,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.4,
        capsize=3,
        markersize=6,
        linestyle="none",
    )
    axes[0].set_yscale("log")
    axes[0].set_title(f"Best {metric} by agent")
    axes[0].set_ylabel(metric)
    axes[0].grid(axis="y", which="both", alpha=0.25)

    axes[1].errorbar(
        x_positions,
        filtered_kl_means,
        yerr=filtered_kl_stds,
        fmt="o",
        color="#ff7f0e",
        ecolor="#ff7f0e",
        elinewidth=1.4,
        capsize=3,
        markersize=6,
        linestyle="none",
    )
    axes[1].set_yscale("log")
    axes[1].set_title(f"KL of best {metric} model")
    axes[1].set_ylabel("kl")
    axes[1].grid(axis="y", which="both", alpha=0.25)

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(filtered_agents, rotation=30, ha="right")

    fig.suptitle(f"Comparison of best models across agents ({metric})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def write_cross_agent_best_model_outputs(
    records: Sequence[Dict[str, Any]],
    output_dir: str,
    top_n_best: int,
) -> None:
    for metric in ["val_loss", "val_kl"]:
        top_rows = top_rows_by_metric_per_agent(records, metric, top_n=top_n_best)
        if not top_rows:
            print(f"Skipping cross-agent outputs for '{metric}' because no rows were found.")
            continue

        best_rows = [row for row in top_rows if int(row.get("rank_within_agent", -1)) == 1]
        best_rows = add_model_parameters_column(best_rows)
        best_rows_csv = os.path.join(output_dir, f"best_models_across_agents_{metric}.csv")
        write_rows_csv(best_rows, best_rows_csv)

        comparison_plot = os.path.join(output_dir, f"best_models_across_agents_{metric}.png")
        was_plotted = make_best_models_comparison_plot(
            best_rows, metric=metric, output_path=comparison_plot
        )

        print(f"Saved: {best_rows_csv}")
        if was_plotted:
            print(f"Saved: {comparison_plot}")


def format_limit_tag(metric_limit: float) -> str:
    if float(metric_limit).is_integer():
        return str(int(metric_limit))
    return str(metric_limit).replace(".", "p")


def normalize_metric_limits(metric_limits: Any) -> List[float]:
    if isinstance(metric_limits, (int, float)):
        return [float(metric_limits)]
    if isinstance(metric_limits, (list, tuple)):
        return [float(limit) for limit in metric_limits]
    raise TypeError("metric_limits must be a float or a list/tuple of floats")


def normalize_input_dimensions(input_dimensions: Any) -> Optional[List[int]]:
    if input_dimensions is None:
        return None
    if isinstance(input_dimensions, (int, float)):
        return [int(input_dimensions)]
    if isinstance(input_dimensions, str):
        return [int(input_dimensions)]
    if isinstance(input_dimensions, (list, tuple)):
        return [int(v) for v in input_dimensions]
    raise TypeError("input_dimensions must be None, int, or a list/tuple of ints")


def sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "unknown"


def analyze_agent_records(
    agent: str,
    agent_records: Sequence[Dict[str, Any]],
    output_dir: str,
    top_k_influential: int,
    top_n_best: int,
    parsed_metric_limits: Sequence[float],
) -> None:
    agent_output_dir = os.path.join(output_dir, sanitize_path_component(agent))
    os.makedirs(agent_output_dir, exist_ok=True)

    print(f"\nAnalyzing agent '{agent}' -> {agent_output_dir}")

    grouped_records_loss = aggregate_rows_across_datasets(agent_records, "val_loss")
    grouped_records_kl = aggregate_rows_across_datasets(agent_records, "val_kl")

    best_val_loss = best_rows_by_metric(grouped_records_loss, "val_loss")
    best_val_kl = best_rows_by_metric(grouped_records_kl, "val_kl")

    if not best_val_loss:
        print(f"Warning: no rows with val_loss were found for agent '{agent}'.")
    if not best_val_kl:
        print(f"Warning: no rows with val_kl were found for agent '{agent}'.")

    best_val_loss = add_model_parameters_column(best_val_loss)
    best_val_kl = add_model_parameters_column(best_val_kl)
    best_val_loss_path = os.path.join(agent_output_dir, "best_by_val_loss.csv")
    best_val_kl_path = os.path.join(agent_output_dir, "best_by_val_kl.csv")
    write_rows_csv(best_val_loss, best_val_loss_path)
    write_rows_csv(best_val_kl, best_val_kl_path)

    top_val_loss = top_rows_by_metric_per_agent(
        grouped_records_loss, "val_loss", top_n=top_n_best
    )
    top_val_kl = top_rows_by_metric_per_agent(grouped_records_kl, "val_kl", top_n=top_n_best)
    top_val_loss = add_model_parameters_column(top_val_loss)
    top_val_kl = add_model_parameters_column(top_val_kl)
    top_val_loss_path = os.path.join(agent_output_dir, "top5_by_val_loss.csv")
    top_val_kl_path = os.path.join(agent_output_dir, "top5_by_val_kl.csv")
    write_rows_csv(top_val_loss, top_val_loss_path)
    write_rows_csv(top_val_kl, top_val_kl_path)

    print(f"Loaded {len(agent_records)} result rows for agent '{agent}'")

    for metric in ["val_kl", "val_loss"]:
        influences = compute_top_influential_parameters(
            agent_records, metric=metric, top_k=max(2, top_k_influential)
        )
        influence_path = os.path.join(agent_output_dir, f"influential_params_{metric}.csv")
        write_influence_csv(influences, influence_path)

        color_param, shape_param = default_plot_params(influences)
        plot_files: List[str] = []
        for limit in parsed_metric_limits:
            plot_file = os.path.join(
                agent_output_dir, f"distribution_{metric}_lt{format_limit_tag(limit)}.png"
            )
            was_plotted = make_distribution_plot(
                agent_records,
                metric=metric,
                color_param=color_param,
                shape_param=shape_param,
                output_path=plot_file,
                metric_limit=limit,
            )
            if was_plotted:
                plot_files.append(plot_file)

        print(f"Saved: {best_val_loss_path}")
        print(f"Saved: {best_val_kl_path}")
        print(f"Saved: {top_val_loss_path}")
        print(f"Saved: {top_val_kl_path}")
        print(f"Saved: {influence_path}")
        for plot_file in plot_files:
            print(f"Saved: {plot_file}")

    if influences:
        print("Top influential parameters:")
        for key, score in influences[:2]:
            print(f"  - {key}: {score:.4f}")


def main(
    results_folder: str = "results",
    output_dir: Optional[str] = None,
    top_k_influential: int = 2,
    top_n_best: int = 5,
    metric_limits: List[float] = [10000, 20.0, 1, 0.5, 0.1],
    input_dimensions: Optional[List[int]] = None,
) -> None:
    """Find best parameters per agent and create metric distribution plots.

    Args:
        results_folder: Folder containing result text files (searched recursively).
        metric: Metric used for influence analysis and plotting (val_loss or val_kl).
        output_dir: Output folder for CSV summaries and plot
            (default: <results_folder>/analysis).
        top_k_influential: Number of influential parameters to compute
            (plot uses top two).
        top_n_best: Number of best rows to show/save per agent and metric.
        metric_limits: List of x-axis cutoffs used to generate plots.
        input_dimensions: Optional input_dim filter. If provided, only rows with
            input_dim in this set are used.
    """
    parsed_metric_limits = normalize_metric_limits(metric_limits)
    parsed_input_dimensions = normalize_input_dimensions(input_dimensions)

    results_folder = os.path.abspath(results_folder)
    output_dir = (
        os.path.abspath(output_dir)
        if output_dir
        else os.path.join(results_folder, "analysis")
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning results in: {results_folder}")

    records = load_records(results_folder)
    if not records:
        raise RuntimeError(f"No parsable result rows found in: {results_folder}")

    if parsed_input_dimensions is not None:
        allowed_dims = set(parsed_input_dimensions)
        records = [
            row
            for row in records
            if row.get("input_dim") is not None and int(row["input_dim"]) in allowed_dims
        ]
        if not records:
            raise RuntimeError(
                "No records found after applying input_dimensions filter: "
                + f"{sorted(allowed_dims)}"
            )
        print(f"Filtered by input_dimensions={sorted(allowed_dims)}")

    print(f"Loaded {len(records)} records from result files.")

    agents = sorted({str(row.get("agent", "unknown")) for row in records})
    print(f"Found agent types: {', '.join(agents)}")

    for agent in agents:
        agent_records = [row for row in records if str(row.get("agent", "unknown")) == agent]
        if not agent_records:
            continue
        analyze_agent_records(
            agent=agent,
            agent_records=agent_records,
            output_dir=output_dir,
            top_k_influential=top_k_influential,
            top_n_best=top_n_best,
            parsed_metric_limits=parsed_metric_limits,
        )

    write_cross_agent_best_model_outputs(
        records=records,
        output_dir=output_dir,
        top_n_best=top_n_best,
    )


if __name__ == "__main__":
    fire.Fire(main)
