from collections.abc import Callable, Iterable
import logging
from typing import Any
from functools import partial as prt
import json
from os import path as osp
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from src.randomness import seed_all
from src.nn.peripheral import get_tokenizer_and_nnmodel
from src.generate.default_pi import PolicyHeuristic, MlcPolicyHeuristic
from src.generate.program_env import ProgramEnv, MolProgramEnv
from src.generate.uct import uct_exp, uct_multistep_exp
from src.configuration_tools import get_output_dir
from src.functional import gt0, tpfilter

logger = logging.getLogger(__name__)


def bs_exp(
    args, env: ProgramEnv, dp: MlcPolicyHeuristic
) -> tuple[list[list[int]], dict[str, int]]:
    """Run beam search. Potentially test all."""
    seq = dp.get_predict_sequence(env.state, horizon=args.horizon, all_sequences=False)
    reward = env.get_reward(seq)
    return env.get_complete_states(), {"sample_times": dp.num_beams}


def sample_exp(args, env: ProgramEnv, dp: MlcPolicyHeuristic) -> None:
    """ """
    for _ in tqdm(range(args.rollouts)):
        dp.sample(
            env.state,
            horizon=args.horizon,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    return env.get_complete_states(), None


# def uniform_random_sample(args, env: ProgramEnv, dp: MlcPolicyHeuristic) -> None:
#     """ """
#     for _ in tqdm(range(args.rollouts)):
#         dp.uniform_sample(env.state, horizon=args.horizon)
#     return env.get_complete_states(), None


def do_experiment(
    cnfgr: DictConfig, env: ProgramEnv, dp: PolicyHeuristic, time_started: float
) -> tuple:
    if cnfgr.algorithm == "sample":
        return sample_exp(cnfgr.sample, env, dp)
    if cnfgr.algorithm == "bs":
        return bs_exp(cnfgr.bs, env, dp)
    if cnfgr.algorithm == "mcts":
        return uct_exp(cnfgr.ts, env, dp, time_started)
    if cnfgr.algorithm == "mcts_multi":
        return uct_multistep_exp(cnfgr.ts, env, dp, time_started)
    raise Exception(f"Unknown alg {cnfgr.algorithm}.")


def top_10_average(rewards: Iterable[float]) -> float:
    try:
        return np.mean(np.partition(rewards, -10)[-10:])
    except ValueError:
        return float("nan")


def get_top_indices(rewards: Iterable[float], top_num: int) -> tuple[int]:
    try:
        return tuple(np.argpartition(rewards, -top_num)[-top_num:])
    except ValueError:
        return tuple()


def get_top_f_fraction_indices(rewards: Iterable[float], f: float) -> tuple[int]:
    try:
        top_num = int(f * len(rewards))
        return tuple(np.argpartition(rewards, -top_num)[-top_num:])
    except ValueError:
        return tuple()


get_top_10_indices = prt(get_top_indices, top_num=10)


def average_selected(indices: Iterable[int], rewards: Iterable[float]) -> float:
    rewards_selected = tuple(rewards[i] for i in indices)
    return np.mean(rewards_selected)


def write_all_samples_tsv(
    output_strs: Iterable[str],
    rewards: Iterable[float],
    rewards_individual: dict[str, Iterable[float]],
    reward_individual_unnormalized: dict[str, Iterable[float]],
):
    """Write all strings, rewards, and unnormalized rewards as tsv"""
    keys_normed = tuple(k + "normalized" for k in rewards_individual.keys())
    keys_unnormed = tuple(
        k + "_unnormalized" for k in reward_individual_unnormalized.keys()
    )
    with open(osp.join(get_output_dir(), "all_samples.tsv"), "w") as f:
        f.write("\t".join(("code", "reward") + keys_normed + keys_unnormed) + "\n")
        for code, reward, *rewards_individual_both in zip(
            output_strs,
            rewards,
            *rewards_individual.values(),
            *reward_individual_unnormalized.values(),
        ):
            f.write(
                "\t".join(
                    (code, str(reward)) + tuple(str(r) for r in rewards_individual_both)
                )
                + "\n"
            )


def search(cnfgr: DictConfig):
    logger.info(f"cnfgr: {OmegaConf.to_yaml(cnfgr)}")
    seed_all(cnfgr.seed_random)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not cnfgr.nn.use_cpu
        else torch.device("cpu")
    )
    tokenizer, model, prompt_string = get_tokenizer_and_nnmodel(
        model_dir=cnfgr.nn.checkpoint, device=device
    )
    logger.debug(f"terminating token of nn: {repr(tokenizer.eos_token)}")
    env = MolProgramEnv(
        tokenizer=tokenizer,
        horizon=cnfgr.horizon,
        prompt_string=prompt_string,
        **cnfgr.mol_metric,
    )
    dp = (
        MlcPolicyHeuristic(
            tokenizer=tokenizer, model=model, device=device, env=env, **cnfgr.heuristic
        )
        if cnfgr.heuristic is not None
        else None
    )
    time_started = time.time()
    states, info = do_experiment(cnfgr, env, dp, time_started)
    # total time if no time per sample
    # time_elapsed = info.get("times", time.time() - time_started)
    logger.info("Search done.")

    output_strs = env.convert_state_to_program_batch(states)
    logger.info("Getting batch rewards.")
    rewards = env.get_rewards_batch(states, mode="test")
    top_10_indices = get_top_10_indices(rewards)
    rewards_individual = env.individual_rewards_batch(output_strs)
    unnormed_rewards_individual = env.unnormalized_individual_rewards_batch(output_strs)
    logger.info("Got all batch rewards.")

    num_strings: int = len(output_strs)
    num_valid: int = sum(map(gt0, rewards))  # valid if reward > 0
    valid_rewards = tpfilter(gt0, rewards)
    results: dict[str, Any] = {
        "number_of_unique_samples": str(num_strings),
        "number_of_valid_unique_samples": str(num_valid),
        "ratio_of_valid_samples": str(num_valid / num_strings),
        "average_reward": str(np.mean(valid_rewards)),
        "top_10_average_reward": str(top_10_average(valid_rewards)),
        "best_reward": str(np.max(rewards)),
        "average_reward_individual": {
            k: str(np.mean(tpfilter(gt0, rs))) for k, rs in rewards_individual.items()
        },
        "best_reward_individual": {
            k: str(np.max(v)) for k, v in rewards_individual.items()
        },
        "average_reward_individual_unnormalized": {
            k: str(np.mean(tpfilter(gt0, v)))
            for k, v in unnormed_rewards_individual.items()
        },
        "best_reward_individual_unnormalized": {
            k: str(np.max(v)) for k, v in unnormed_rewards_individual.items()
        },  # TODO: SA is wrong. SA is better if lower when unnormalized.
        "top_10_average_reward_individual": {
            k: str(top_10_average(tpfilter(gt0, v)))
            for k, v in rewards_individual.items()
        },
        "top_10_average_reward_individual_unnormalized": {
            k: str(top_10_average(tpfilter(gt0, v)))
            for k, v in unnormed_rewards_individual.items()
        },
        "top_10_molecules_reward_individual_unnormalized": {
            k: str(average_selected(top_10_indices, v))
            for k, v in unnormed_rewards_individual.items()
        },
        # "time_elapsed": time_elapsed,
        # "sample_times": info["sample_times"],
    }
    logger.info("All averages are among valid samples only.")
    logger.info(f"summary results :\n{json.dumps(results, indent=2)}")
    results_top_f = {
        f"top_{f:.1f}_fraction_average_reward": str(
            average_selected(get_top_f_fraction_indices(rewards, f), rewards)
        )
        for f in np.arange(0.1, 1.1, 0.1)
    }  # TODO: fix here and additional_summary.py to average out of the portion of valid samples
    results |= results_top_f
    summary = {
        "results": results,
        "configuration": OmegaConf.to_container(cnfgr, resolve=True),
    }
    with open(osp.join(get_output_dir(), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_all_samples_tsv(
        output_strs, rewards, rewards_individual, unnormed_rewards_individual
    )
    logger.info(f"results saved to {osp.relpath(get_output_dir())}")


hydra_app = hydra.main(version_base=None, config_path="cnfgr", config_name="search")(
    search
)
if __name__ == "__main__":
    hydra_app()
