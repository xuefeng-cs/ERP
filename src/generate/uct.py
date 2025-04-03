import time
from os import path as osp
import traceback
import logging

from omegaconf import DictConfig, OmegaConf
from scipy.stats import entropy

from .program_env import ProgramEnv
from .default_pi import PolicyHeuristic
from ..dyna_gym.agents.mcts_common import update_root
from ..dyna_gym.agents.mcts_utils import plot_tree
from ..dyna_gym.agents.uct import UCT
from ..configuration_tools import get_output_dir


logger = logging.getLogger(__name__)


def uct_exp(args: DictConfig, env: ProgramEnv, dp: PolicyHeuristic, time_started: float):
    """
    Run TG-MCTS
    """
    agent = UCT(
        action_space=[],  # this will not be used as we have a default policy
        gamma=1.0,  # no discounting
        dp=dp,
        env=env,
        reuse_tree=True,  # TODO: what does it mean?
        **args.uct,
        # width=args.width,
        # ts_mode=args.ts_mode,
    )
    # agent.display()

    # tell the mcts to stop when this is satisfied
    if not hasattr(args, "max_num_sample_tokens"):
        OmegaConf.update(args, "max_num_sample_tokens", None, force_add=True)
    term_cond = (
        lambda: dp.sample_times > args.max_sample_times
        or time.time() - time_started > args.time_limit
        or (
            args.max_num_sample_tokens is not None
            and dp.num_sample_tokens > args.max_num_sample_tokens
        )
    )

    try:
        if len(env.state) >= args.horizon:
            logger.error(
                f"Cannot process programs longer than {args.horizon}. Stop here."
            )
            return None, None
        else:
            # run mcts. a bit hacky, but this will run mcts simulation. we do not need to take any action
            agent.act(env=env, done=False, term_cond=term_cond)
    except Exception as e:
        if args.debug:
            raise e
        else:
            logger.error("Unexpected exception in generating solution")
            logger.error(traceback.format_exc() + "\n")
            return None, None

    # these may not be assigned, set default values
    if args.debug:
        # print the mcts tree
        try:
            plot_tree(agent.root, env, osp.join(get_output_dir(), "tree"))
        except Exception as e:
            logger.error(f"Error plotting tree.\n{e}")
            logger.error(traceback.format_exc())

    states = env.get_complete_states()
    samples_times_time_stamps = {
        "sample_times": dp.sample_times,
        "times": [t - time_started for t in dp.time_stamps],
    }
    return states, samples_times_time_stamps


def uct_multistep_exp(args, env, dp, time_started):
    agent = UCT(
        action_space=[],  # this will not be used as we have a default policy
        gamma=1.0,  # no discounting
        ucb_constant=args.ucb_constant,
        ucb_base=args.ucb_base,
        horizon=args.horizon,
        rollouts=args.rollouts,
        dp=dp,
        env=env,
        width=args.width,
        reuse_tree=True,
        alg=args.alg,
        exploration=args.exploration,
        selection=args.selection,
    )

    agent.display()

    try:
        done = False
        s = env.state
        for t in range(args.horizon):
            if dp.sample_times > args.max_sample_times:
                print("Maximum number of samples reached.")
                break

            if time.time() - time_started > args.time_limit:
                print("Time exceeded.")
                break

            if len(s) >= args.horizon:
                print(f"Cannot process programs longer than {args.horizon}. Stop here.")
                break

            if done:
                break

            if t > 0:
                # tree is not built at time step 0 yet
                ent = entropy([child.prob for child in agent.root.children])
            else:
                ent = 1  # this wouldn't change the rollout number

            if args.entropy_weighted_strategy == "linear":
                rollout_weight = ent
            elif args.entropy_weighted_strategy == "linear_with_minimum":
                rollout_weight = 0.2 + 0.8 * ent
            elif args.entropy_weighted_strategy == "none":
                rollout_weight = 1  # does not change rollout number
            else:
                raise ValueError(
                    f"Unknown rollout strategy {args.entropy_rollout_strategy}"
                )

            act = agent.act(env, done, rollout_weight=rollout_weight)
            s, r, done, _ = env.step(act)

            if args.debug:
                # print the current tree
                logger.debug("tree:")
                plot_tree(agent.root, env, osp.join(get_output_dir(), "tree"))

                logger.debug("took action:")
                act_str = env.tokenizer.decode(act)
                logger.debug(repr(act_str))
                logger.debug("========== state (excluding prompt) ==========")
                logger.debug(env.convert_state_to_program(s))

                logger.debug("entropy at this step: ", ent)

            update_root(agent, act, s)
            dp.clean_up(s)
    except Exception as e:
        if args.debug:
            raise e
        else:
            logger.error("Unexpected exception in generating solution")
            logger.error(traceback.format_exc() + "\n")
            return None, None

    if len(s) >= args.horizon:
        logger.error("Exceeds horizon.\n")
        return None, None

    states = env.get_complete_states()

    time_stamps = dp.time_stamps
    times = [t - time_started for t in time_stamps]

    return states, {"sample_times": dp.sample_times, "times": times}
