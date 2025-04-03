import argparse
from collections.abc import Iterable
from glob import glob, iglob
import os
import os.path as osp
from pathlib import Path
import json

from omegaconf import DictConfig, OmegaConf
import ipdb
import pandas as pd

from src.functional import pop_dict

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--glob_patterns", type=str, required=False, default=None, nargs="+"
)
argparser.add_argument("--results_dir", type=str, required=False, default=None)

argparser.add_argument("--recursive", type=bool, required=False, default=True)
argparser.add_argument("--include_hidden", type=bool, required=False, default=True)


def rename_columns(str) -> str:
    """remove configuration. and results. from column names"""
    return str.replace("configuration.", "").replace("results.", "")


def get_more_results_columns() -> tuple[str]:
    prefixes = (
        "average_reward_individual",
        "best_reward_individual",
        "top_10_average_reward_individual",
        "top_10_molecules_reward_individual",
        "best_molecule_reward_individual",
    )
    middle = ("", "_unnormalized")
    suffixes = (".solvability", ".druglikeness", ".sa", ".docking_score")

    columns = []
    for p in prefixes:
        for m in middle:
            for s in suffixes:
                columns.append(p + m + s)

    average_k = (20, 30, 40, 50, 60, 70, 80, 90, 100)  # ought not to include 10
    for k in average_k:
        columns.append(f"top_{k}_average_reward")

    fractions = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    for f in fractions:
        columns.append(f"top_{f}_fraction_average_reward")

    return tuple(columns)


# fmt: off
ordered_coumns1 = (
    "algorithm",
    "ts.uct.rollouts",
    "heuristic.num_beams",
    "ts.uct.alg",
    "ts.uct.ucb_constant",
    "ts.uct.exploration_denominator_summand",
    "ts.uct.entropy_forward_k",

    "heuristic.top_k_expansion",
    "heuristic.top_p_expansion",

    "ts.max_sample_times",
    "ts.max_num_sample_tokens",

    )
ordered_coumns2 = (
    "average_reward",
    "top_10_average_reward",
    "best_reward",
    "number_of_unique_samples",
    "number_of_valid_unique_samples",
    "ratio_of_valid_samples",
)
ordered_coumns3 = get_more_results_columns()
ordered_coumns4 = (
    "ts.uct.entropy_entering_mode",
    "ts.uct.entropy_combining_mode",
    "ts.uct.entropy_k_averaging",

    "nn.checkpoint",

    "ts.uct.exploitation_mode",
    "ts.uct.selection",

    "ts.uct.exploration_mode",

    "ts.uct.entropy_alpha",

    "ts.uct.ucb_base"

    "mol_metric.invalid_values",
    "mol_metric.normalization",
    "mol_metric.docking_dataset",
    "mol_metric.metric_name",

    "heuristic.test_all_beams",
    "heuristic.top_k_uniform",

    "ts.horizon",
    "heuristic.horizon",

    "sample.rollouts",
    "sample.temperature",
    "bs.horizon",
    "bs.num_beams",

    "seed_random",
)
# fmt: on

ordered_columns = ordered_coumns1 + ordered_coumns2 + ordered_coumns3 + ordered_coumns4
ordered_columns_all = ordered_columns + ("path_outputs",)

# removed_columns= (
#     "configuration.heuristic.value_model.debug","configuration.heuristic.debug","configuration.ts.debug","configuration.nn.use_cpu","configuration.ts.entropy_weighted_strategy""configuration.ts.time_limit","configuration.heuristic.top_k_uniform"
#     )


def light_preprocess(summary: dict):
    cnfgr = summary["configuration"]
    if cnfgr["algorithm"] == "bs":
        cnfgr.pop("ts")
        cnfgr.pop("sample", None)
        cnfgr["ts"] = {"uct": {"rollouts": 1}}
    elif cnfgr["algorithm"] == "mcts":
        cnfgr.pop("bs")
        cnfgr.pop("sample", None)
        cnfgr_ts_uct = cnfgr["ts"]["uct"]
        if cnfgr_ts_uct["entropy_combining_mode"] is None:
            pop_dict(
                cnfgr_ts_uct,
                "entropy_combining_mode",
                "entropy_alpha",
                "entropy_k_averaging",
                "entropy_forward_k",
                "entropy_entering_mode",
            )
        if cnfgr_ts_uct["reward_estimate_combining_mode"] is None:
            pop_dict(
                cnfgr_ts_uct, "reward_estimate_combining_mode", "reward_estimate_weight"
            )
    elif cnfgr["algorithm"] == "sample":
        cnfgr.pop("bs")
        cnfgr.pop("ts")
        cnfgr["ts"] = {"uct": {"rollouts": cnfgr["sample"]["rollouts"]}}
    return summary


def create_tidy_table(
    glob_patterns: Iterable[str],
    results_dir: str = None,
    recursive: bool = True,
    include_hidden: bool = True,
):
    filepaths = []
    results = []
    paths = []
    paths_set = set()
    for p in glob_patterns:
        filepaths.extend(glob(p, recursive=recursive, include_hidden=include_hidden))
    for filepath in filepaths:
        with open(filepath, "r") as f:
            summary_data = json.load(f)
            paths.append(filepath)
            paths_set.add(filepath)

        filepathparts = Path(filepath).parts
        filepath_partial = osp.join(*filepathparts[-3:])
        summary_data["path_outputs"] = filepath_partial
        # print("Processing ", filepath)
        summary_data = light_preprocess(summary_data)
        results.append(summary_data)
    assert len(paths) == len(paths_set)
    print("Number of files : ", len(paths))

    df = pd.json_normalize(results)
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(tuple)

    df = df.rename(columns=rename_columns)

    df["path_outputs"] = df["path_outputs"].str.rstrip("/")
    df["nn.checkpoint"] = df["nn.checkpoint"].str.rstrip("/")

    df["ts.uct.entropy_forward_k"] = df["ts.uct.entropy_forward_k"].fillna("na")
    df["ts.uct.entropy_combining_mode"] = df["ts.uct.entropy_combining_mode"].fillna("na")
    df["ts.uct.entropy_entering_mode"] = df["ts.uct.entropy_entering_mode"].fillna("na")

    # ipdb.set_trace()

    return df


def save_one_sorted_table(df: pd.DataFrame, results_dir: str) -> None:
    filepath = f"{results_dir}.tsv"
    df_sorted_rollouts_bestreward = df.sort_values(
        by=["ts.uct.rollouts", "heuristic.num_beams", "best_reward", "average_reward"],
        ascending=[True, True, False, False],
    )
    df_sorted_rollouts_bestreward.to_csv(filepath, sep="\t")
    print(f"Saved {filepath}")


def save_sorted_table(df: pd.DataFrame, results_dir: str) -> None:
    df_sorted_rollouts_bestreward = df.sort_values(
        by=["ts.uct.rollouts", "heuristic.num_beams", "best_reward", "average_reward"],
        ascending=[True, True, False, False],
    )
    df_sorted_rollouts_bestreward.to_csv(
        osp.join(results_dir, "sorted_rollouts_bestreward.tsv"), sep="\t"
    )
    df_sorted_rollouts_top10 = df.sort_values(
        by=[
            "ts.uct.rollouts",
            "top_10_average_reward",
            "average_reward",
        ],
        ascending=[True, False, False],
    )
    df_sorted_rollouts_top10.to_csv(
        osp.join(results_dir, "sorted_rollouts_top10.tsv"), sep="\t"
    )

    df_sorted_rollouts_numbeams_top10 = df.sort_values(
        by=[
            "ts.uct.rollouts",
            "heuristic.num_beams",
            "top_10_average_reward",
            "average_reward",
        ],
        ascending=[True, True, False, False],
    )
    df_sorted_rollouts_numbeams_top10.to_csv(
        osp.join(results_dir, "sorted_rollouts_numbeams_top10.tsv"), sep="\t"
    )

    df_sorted_average_reward = df.sort_values(by=["average_reward"], ascending=False)
    df_sorted_average_reward.to_csv(
        osp.join(results_dir, "sorted_average_reward.tsv"), sep="\t"
    )
    df_sorted_top10 = df.sort_values(by=["top_10_average_reward"], ascending=False)
    df_sorted_top10.to_csv(osp.join(results_dir, "sorted_top10.tsv"), sep="\t")
    df_sorted_best_reward = df.sort_values(by=["best_reward"], ascending=False)
    df_sorted_best_reward.to_csv(
        osp.join(results_dir, "sorted_best_reward.tsv"), sep="\t"
    )
    df_sorted_num_valid = df.sort_values(
        by=["number_of_valid_unique_samples"], ascending=False
    )
    df_sorted_num_valid.to_csv(osp.join(results_dir, "sorted_num_valid.tsv"), sep="\t")
    df_sorted_ratio_valid = df.sort_values(by=["ratio_of_valid_samples"], ascending=False)
    df_sorted_ratio_valid.to_csv(
        osp.join(results_dir, "sorted_ratio_valid.tsv"), sep="\t"
    )


def save_best_table(df: pd.DataFrame, results_dir: str) -> None:
    metric = "average_reward"

    # Specify the fixed configuration
    fixed_config = {"ts.uct.entropy_entering_mode": "mul"}

    # Filter the DataFrame based on the fixed configuration
    df_filtered = df[df[list(fixed_config)] == pd.Series(fixed_config)].dropna()

    # Find the index of the row with the highest value for the metric
    gb = df_filtered.groupby(by=["ts.uct.rollouts", "heuristic.num_beams"])
    best_config_idx = gb[metric].idxmax()
    # Get the best configuration
    best_config = df_filtered.loc[best_config_idx]

    # Save the best configuration to a CSV file
    best_config.to_csv(osp.join(results_dir, "best_config.csv"))
    ipdb.set_trace()


def removed_earlier_duplicates(
    df: pd.DataFrame, ordered_columns=ordered_coumns1 + ordered_coumns4
) -> pd.DataFrame:
    df = df.sort_values(by=["path_outputs"], ascending=True)
    return df.drop_duplicates(subset=ordered_columns, keep="last")


if __name__ == "__main__":
    args: argparse.Namespace = argparser.parse_args()
    if args.glob_patterns is None:
        args.glob_patterns = (
            "outputs/2024-01-2[789]/**/summary_new.json",
            "outputs/2024-01-3[01]/**/summary_new.json",
            "outputs/2024-02-0[12]/**/summary_new.json",
            "/home/cctien/Documents/ggldrvuchicago/0zeroth/projects/llmmcts_project/outputs/512 rollout/2024-01-2[789]/**/summary_new.json",
            "/home/cctien/Documents/ggldrvuchicago/0zeroth/projects/llmmcts_project/outputs/512 rollout/2024-01-3[01]/**/summary_new.json",
        )
        print("No glob_path_name provided, using default: ", args.glob_patterns)
    if args.results_dir is None:
        args.results_dir = "results/2024-01-31/"
        print("No results_dir provided, using default: ", args.results_dir)

    unordered_df = create_tidy_table(**vars(args))
    df = unordered_df.reindex(columns=ordered_columns_all)
    print("Number of rows : ", len(df))
    df = removed_earlier_duplicates(df)
    print("Number of rows after removing duplicates : ", len(df))
    df = df[df["ts.uct.rollouts"] != 265]

    # Partition by docking dataset only and have baselines only
    gb = df.groupby(by=["mol_metric.docking_dataset"])
    print("Number of partitions : ", len(gb.groups))
    for group in gb.groups:
        results_dir = osp.join(args.results_dir, f"baselines_dockingdata_{group}")
        df_pb = gb.get_group(group)
        df_baselines = df_pb[
            (df_pb["algorithm"] == "bs") | (df_pb["algorithm"] == "sample")
        ]
        os.makedirs(results_dir, exist_ok=True)
        save_one_sorted_table(df_baselines, results_dir)
        save_sorted_table(df_baselines, results_dir)

        results_dir = osp.join(
            args.results_dir, f"baselines_RL_finetuned_dockingdata_{group}"
        )
        df_pb = gb.get_group(group)
        df_baselines = df_pb[
            (df_pb["nn.checkpoint"].str.contains("_rl_"))
            & ((df_pb["algorithm"] == "bs") | (df_pb["algorithm"] == "sample"))
        ]
        os.makedirs(results_dir, exist_ok=True)
        save_one_sorted_table(df_baselines, results_dir)
        save_sorted_table(df_baselines, results_dir)

    # Partition by nn.checkpoint and docking dataset
    gb_lm = df.groupby(by=["mol_metric.docking_dataset", "nn.checkpoint"])
    print("Number of lm partitions : ", len(gb_lm.groups))
    for group in gb_lm.groups:
        group_names = tuple(g.replace("/", "") for g in group)
        results_lm_dir = osp.join(
            args.results_dir, f"dockingdata_{group_names[0]}_lm_{group_names[1]}"
        )
        df_pb_lm = gb_lm.get_group(group)
        os.makedirs(results_lm_dir, exist_ok=True)
        save_one_sorted_table(df_pb_lm, results_lm_dir)
        save_sorted_table(df_pb_lm, results_lm_dir)

        # save_best_table(df_pb_lm, results_lm_dir)
        # # ============ table partitions
        # gb_hyper = df_pb_lm.groupby(
        #     [
        #         "configuration.ts.uct.rollouts",
        #         "configuration.heuristic.num_beams",
        #         "configuration.ts.uct.exploration_denominator_summand",
        #         "configuration.heuristic.top_k_expansion",
        #         "configuration.heuristic.top_p_expansion",
        #     ]
        # )
        # tables = [gb_hyper.get_group(x) for x in gb_hyper.groups]
        # # save each tabel as tsv with file name being their respective group
        # for table in tables:
        #     filename = f"table_rollouts{table.iloc[0]['configuration.ts.uct.rollouts']}_beams{table.iloc[0]['configuration.heuristic.num_beams']}_exd{table.iloc[0]['configuration.ts.uct.exploration_denominator_summand']}_topk{table.iloc[0]['configuration.heuristic.top_k_expansion']}_topp{table.iloc[0]['configuration.heuristic.top_p_expansion']}.tsv"
        #     table.to_csv(
        #         osp.join(results_lm_dir, filename),
        #         sep="\t",
        #     )

        # # ============ best for each rollouts+beamsize for ucb;pucb;entropy
        # gb_partition = df_pb_lm.groupby(
        #     by=[
        #         "configuration.ts.uct.rollouts",
        #         "configuration.heuristic.num_beams",
        #     ]
        # )
        # # [
        # #     "configuration.ts.uct.entropy_forward_k",
        # #     "configuration.ts.uct.entropy_k_averaging",
        # #     "configuration.ts.uct.exploration_denominator_summand",
        # #     "configuration.heuristic.top_k_expansion",
        # #     "configuration.heuristic.top_p_expansion",
        # # ]
        # for group in gb_partition.groups:
        #     for metric in (
        #         "results.average_reward",
        #         "results.best_reward",
        #         "results.number_of_valid_unique_samples",
        #     ):
        #         df_partition = gb_partition.get_group(group)
        #         value_max = df_partition.groupby(
        #             by=[
        #                 "configuration.ts.uct.alg",
        #                 "configuration.ts.uct.entropy_combining_mode",
        #                 # "configuration.ts.uct.entropy_forward_k",
        #             ],
        #         )[metric].transform("max")
        #         df_best = df_partition[df_partition[metric] == value_max]
        #         df_best_sorted = df_best.sort_values(
        #             metric,
        #             ascending=False,
        #         )
        #         df_best_sorted.to_csv(
        #             osp.join(
        #                 results_lm_dir,
        #                 f"best_{metric}_each_rollouts{group[0]}_beams{group[1]}.tsv",
        #             ),
        #             sep="\t",
        #         )
