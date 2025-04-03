import argparse
from collections.abc import Iterable
import os
from os import path as osp
import re

import ipdb
from create_table import (
    create_tidy_table,
    removed_earlier_duplicates,
    ordered_columns,
    ordered_columns_all,
)

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy


from plotnine import ggplot, aes, geom_line, theme_minimal, labs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

argparser = argparse.ArgumentParser()
argparser.add_argument("--use_title", action="store_true", required=False, default=False)
# argparser.add_argument("--filter", type=str, required=False, default=None)
argparser.add_argument("--heuristic_num_beams", type=str, required=False, default=None)


def get_group_name(group):
    name = ""
    for g in group:
        if isinstance(g, str):
            name += "_" + g.replace("/", "") + "_"
        else:
            name += "_" + str(g) + "_"
    return name


def get_model_name(
    algorithm, ts_uct_alg, ts_uct_entropy_combining_mode, ts_uct_forward_k
):
    if algorithm == "sample":
        return "Sampling"
    if ts_uct_alg == "p_ucb":
        if ts_uct_entropy_combining_mode == "na" or ts_uct_forward_k == 0:
            return f"PG-TD"
        else:
            return f"ERP"
    elif ts_uct_alg == "ucb":
        if ts_uct_entropy_combining_mode == "na":
            return f"UCT"
        else:
            return f"UCB-ERP (ought not to be included)"
    raise NotImplementedError


def y_labels_repr(metric: str):
    if metric == "number_of_valid_unique_samples":
        return "Number"
    if "solvability" in metric:
        return "Solvability"
    if "druglikeness" in metric:
        return "Druglikeness"
    if ".sa" in metric:
        return "Synthetic Accessibility"
    if "docking_score" in metric:
        return "Minus Docking Score"
    if metric[-6:] == "reward":
        return "Normalized Reward"
    raise NotImplementedError


def title_repr(metric: str) -> str:
    titles_metric = {
        "top_10_molecules_reward_individual_unnormalized.solvability": "Average Solvability of top 10 molecules",
        "top_10_molecules_reward_individual_unnormalized.druglikeness": "Average Druglikeness of top 10 molecules",
        "top_10_molecules_reward_individual_unnormalized.sa": "Average Synthetic Accessibility of top 10 molecules",
        "top_10_molecules_reward_individual_unnormalized.docking_score": "Average Minus Docking Score of top 10 molecules",
        "best_reward_individual_unnormalized.solvability": "Best Solvability",
        "best_reward_individual_unnormalized.druglikeness": "Best Druglikeness",
        "best_reward_individual_unnormalized.sa": "Best Synthetic Accessibility",
        "best_reward_individual_unnormalized.docking_score": "Best Minus Docking Score",
        "best_reward": "Best Reward",
        "average_reward": "Average Reward",
        "number_of_valid_unique_samples": "Number of Valid Unique Samples",
        "best_molecule_reward_individual_unnormalized.solvability": "Best Molecule Solvability",
        "best_molecule_reward_individual_unnormalized.druglikeness": "Best Molecule Druglikeness",
        "best_molecule_reward_individual_unnormalized.sa": "Best Molecule Synthetic Accessibility",
        "best_molecule_reward_individual_unnormalized.docking_score": "Best Molecule Minus Docking Score",
    }
    if re.fullmatch(r"top\_[0-9]+\_average_reward", metric) is not None:
        top_number = int(metric.split("_")[1])
        return f"Average Reward of top {top_number} molecules"
    if (
        re.fullmatch(r"top\_[0-9]+[.][0-9]+\_fraction\_average\_reward", metric)
        is not None
    ):
        return (
            f"Average Reward of top {float(metric.split('_')[1])} fraction of molecules"
        )
    return titles_metric[metric]


def get_top_k_metrics(numerals: Iterable[int]) -> tuple[str]:
    return tuple(f"top_{numeral}_average_reward" for numeral in numerals)


def get_top_f_metrics(numerals: Iterable[float]) -> tuple[str]:
    return tuple(
        f"top_{round(numeral, 1)}_fraction_average_reward" for numeral in numerals
    )


def get_title_checkpoints(checkpoint: str) -> str:
    biased_checkpoints = (
        "jarod0411zinc10M_gpt2_SMILES_bpe_combined_step1_finetune",
        "jarod0411zinc10M_gpt2_SMILES_bpe_combined_step1_finetune_covid",
    )
    b16_checkpoints = (
        "jarod0411zinc10M_gpt2_SMILES_bpe_combined_step1_rl_finetune_cancer_b16",
        "jarod0411zinc10M_gpt2_SMILES_bpe_combined_step1_rl_finetune_covid_b16",
    )
    b64_checkpoints = (
        "jarod0411zinc10M_gpt2_SMILES_bpe_combined_step1_rl_finetune_cancer_b64",
        "jarod0411zinc10M_gpt2_SMILES_bpe_combined_step1_rl_finetune_covid_b64",
    )
    if checkpoint in biased_checkpoints:
        return "Biased Fine-tuned"
    if checkpoint in b16_checkpoints:
        return "RL Fine-tuned, b16"
    if checkpoint in b64_checkpoints:
        return "RL Fine-tuned, b64"
    return "Pre-trained"


def get_title(group_name, metric):
    docking_names = {"cancer": "Cancer", "covid": "COVID-19"}
    title_metric = title_repr(metric)
    docking_dataset = docking_names[group_name[0]]
    checkpoint = group_name[1]
    title_checkpoint = get_title_checkpoints(checkpoint)
    return f"{title_metric} ({docking_dataset}, {title_checkpoint})"


def get_max_row_index_filtered(group) -> Series:
    mask = (group["ts.uct.alg"] == "p_ucb") & (
        group["ts.uct.entropy_combining_mode"] != "na"
    )
    indices = group.loc[mask, metric].idxmax()
    return indices


def drop_rows(df: DataFrame) -> DataFrame:
    df = df[df["algorithm"] != "bs"]
    df = df[df["nn.checkpoint"].str.contains("jarod0411")]
    df = df[df["ts.uct.rollouts"] >= 32]
    df = df[df["ts.uct.rollouts"] != 265]
    df = df[
        ~((df["ts.uct.alg"] == "ucb") & (df["ts.uct.entropy_combining_mode"] != "na"))
    ]
    return df


def plot(df_maxed: DataFrame, group_name: tuple[str], metric: str, use_title: bool):
    # palette = sns.color_palette("deep", len(df_maxed["model"].unique()))
    deep_palette = sns.color_palette("deep")
    palette = (deep_palette[3], deep_palette[0], deep_palette[1], deep_palette[2])
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 17
    fig, ax = plt.subplots(figsize=(6, 5.3))

    sns.lineplot(
        ax=ax,
        data=df_maxed,
        x=df_maxed["ts.uct.rollouts"],
        y=df_maxed[metric],
        hue=df_maxed["model"],
        style=df_maxed["model"],
        markers=True,
        palette=palette,
        linewidth=3,
        markersize=13,
    )
    legend = ax.legend(title=None, loc="upper left")
    frame = legend.get_frame()
    frame.set_alpha(0.6)
    frame.set_edgecolor("none")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xscale("log", base=2)
    xticks = tuple(2**exponent for exponent in range(5, 10))
    plt.xticks(xticks, xticks)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x)))

    ax.set_xlabel("Rollouts", fontsize=24)
    ax.set_ylabel(y_labels_repr(metric), fontsize=24)
    if use_title:
        plt.title(get_title(group_name, metric))

    fig.tight_layout()
    plt.savefig(
        osp.join(dir_plots, f"{get_title(group_name, metric)}.pdf"), bbox_inches="tight"
    )
    plt.savefig(
        osp.join(dir_plots, f"{get_title(group_name, metric)}.png"), bbox_inches="tight"
    )
    # plt.show()
    plt.close()


def parse_filter(filter: str) -> dict:
    conditions = filter.split(",")
    parsed_conditions = {}
    for condition in conditions:
        field, value = condition.split("=")
        parsed_conditions[field] = value
    return parsed_conditions


def get_table_to_plot(gb_lm: DataFrame, group: tuple[str], metric: str) -> DataFrame:
    df = gb_lm.get_group(group).copy(deep=True)
    # if num_beams is not None:
    #     num_beams = int(num_beams)
    #     df = df[df["heuristic.num_beams"] == num_beams]

    df.loc[df[metric] == "nan", metric] = 0.0
    df[metric] = df[metric].fillna(0.0)
    df[metric] = pd.to_numeric(df[metric])
    group_columnes = [
        "algorithm",
        "ts.uct.rollouts",
        "ts.uct.alg",
        "ts.uct.entropy_combining_mode",
    ]
    gb = df.groupby(by=group_columnes, dropna=False)

    idx_max = gb[metric].idxmax()
    df_maxed = df.loc[idx_max]
    df_maxed["model"] = df_maxed.apply(
        lambda row: get_model_name(
            row["algorithm"],
            row["ts.uct.alg"],
            row["ts.uct.entropy_combining_mode"],
            row["ts.uct.entropy_forward_k"],
        ),
        axis=1,
    )
    # if metric == "number_of_valid_unique_samples":
    #     df_maxed = df_maxed[(df_maxed["model"] == "ERP") | (df_maxed["model"] == "PG-TD")]
    return df_maxed


def plot_forward_k(gb, group, metric):
    df = gb.get_group(group).copy(deep=True)
    df.loc[df[metric] == "nan", metric] = 0.0
    df[metric] = df[metric].fillna(0.0)
    df[metric] = pd.to_numeric(df[metric])
    df = df[
        (df["ts.uct.alg"] == "p_ucb")
        & (df["ts.uct.exploration_denominator_summand"] == 0.1)
    ]
    df.loc[df["ts.uct.entropy_forward_k"] == "na", "ts.uct.entropy_forward_k"] = 0
    df["ts.uct.entropy_forward_k"] = df["ts.uct.entropy_forward_k"].astype(int)
    df["model"] = df.apply(
        lambda row: get_model_name(
            row["algorithm"],
            row["ts.uct.alg"],
            row["ts.uct.entropy_combining_mode"],
            row["ts.uct.entropy_forward_k"],
        ),
        axis=1,
    )
    idx = df.groupby("ts.uct.entropy_forward_k")[metric].idxmax()
    df_max = df.loc[idx]

    deep_palette = sns.color_palette("deep")
    palette = [deep_palette[3], deep_palette[0]]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 17
    fig, ax = plt.subplots(figsize=(6, 5.3))

    sns.lineplot(
        ax=ax,
        data=df_max,
        x="ts.uct.entropy_forward_k",
        y=metric,
        hue="model",
        style="model",
        markers=True,
        palette=palette,
        linewidth=3,
        hue_order=["ERP", "PG-TD"],
        style_order=["ERP", "PG-TD"],
        markersize=13,
    )
    legend = ax.legend(title=None, loc="upper left")
    frame = legend.get_frame()
    frame.set_alpha(0.6)
    frame.set_edgecolor("none")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    xticks = tuple(range(0, 7))
    plt.xticks(xticks, xticks)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: int(x)))

    ax.set_xlabel("Forward Step e", fontsize=24)
    ax.set_ylabel(y_labels_repr(metric), fontsize=24)
    fig.tight_layout()
    plt.savefig(
        osp.join(dir_plots, "forward_k", f"{metric}_{get_group_name(group)}.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        osp.join(dir_plots, "forward_k", f"{metric}_{get_group_name(group)}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    args = argparser.parse_args()
    glob_patterns = (
        "outputs/2024-01-2[789]/**/summary_new.json",
        "outputs/2024-01-3[01]/**/summary_new.json",
        "outputs/2024-02-0[12]/**/summary_new.json",
        "/home/cctien/Documents/ggldrvuchicago/0zeroth/projects/llmmcts_project/outputs/512 rollout/2024-01-2[789]/**/summary_new.json",
        "/home/cctien/Documents/ggldrvuchicago/0zeroth/projects/llmmcts_project/outputs/512 rollout/2024-01-3[01]/**/summary_new.json",
    )

    dir_plots = "results/plots/2024-01-31/"
    # if args.filter is not None:
    #     dir_plots = osp.join(dir_plots, args.filter)
    if args.heuristic_num_beams is not None:
        dir_plots = osp.join(dir_plots, f"num_beams={args.heuristic_num_beams}")

    dir_tables = osp.join(dir_plots, "tables")
    dir_forward_k = osp.join(dir_plots, "forward_k")
    for dir_ in (dir_plots, dir_tables, dir_forward_k):
        if not osp.exists(dir_):
            os.makedirs(dir_)

    unordered_df = create_tidy_table(glob_patterns)
    df = unordered_df.reindex(columns=ordered_columns_all)
    df = removed_earlier_duplicates(df, ordered_columns=ordered_columns)
    df = drop_rows(df)

    metrics_all = (
        (
            "top_10_molecules_reward_individual_unnormalized.solvability",
            "top_10_molecules_reward_individual_unnormalized.druglikeness",
            "top_10_molecules_reward_individual_unnormalized.sa",
            "top_10_molecules_reward_individual_unnormalized.docking_score",
            "best_reward_individual_unnormalized.solvability",
            "best_reward_individual_unnormalized.druglikeness",
            "best_reward_individual_unnormalized.sa",
            "best_reward_individual_unnormalized.docking_score",
            "best_molecule_reward_individual_unnormalized.solvability",
            "best_molecule_reward_individual_unnormalized.druglikeness",
            "best_molecule_reward_individual_unnormalized.sa",
            "best_molecule_reward_individual_unnormalized.docking_score",
            "average_reward",
            "best_reward",
            "number_of_valid_unique_samples",
        )
        + get_top_k_metrics(np.arange(10, 110, 10))
        + get_top_f_metrics(np.arange(0.1, 1.1, 0.1))
    )
    gb_lm = df.groupby(by=["mol_metric.docking_dataset", "nn.checkpoint"])
    for group in gb_lm.groups:
        group_name = tuple(g.replace("/", "") for g in group)
        print(group_name)
        for metric in metrics_all:
            df_temp = df.copy(deep=True)

            df_maxed = get_table_to_plot(
                gb_lm, group=group, metric=metric, num_beams=args.heuristic_num_beams
            )
            plot(df_maxed, group_name=group_name, metric=metric, use_title=args.use_title)

    column_groups_k_forward = [
        "mol_metric.docking_dataset",
        "nn.checkpoint",
        "ts.uct.rollouts",
        "heuristic.num_beams",
        "ts.uct.ucb_constant",
    ]
    metrics_all_k_f = (
        (
            "average_reward",
            "best_reward",
            "number_of_valid_unique_samples",
        )
        + get_top_k_metrics(np.arange(10, 60, 20))
        + get_top_f_metrics(np.arange(0.1, 0.6, 0.2))
    )
    df_fk = df[
        (~df["nn.checkpoint"].str.contains("_b64"))
        & (~df["nn.checkpoint"].str.contains("_b16"))
        & (df["ts.uct.alg"] == "p_ucb")
        & (df["ts.uct.rollouts"] >= 256)
        & (df["ts.uct.exploration_denominator_summand"] == 0.1)
    ]
    gb_fk = df_fk.groupby(by=column_groups_k_forward, dropna=False)
    for group in gb_fk.groups:
        for metric in metrics_all_k_f:
            print(group, metric)
            plot_forward_k(gb_fk, group, metric)
