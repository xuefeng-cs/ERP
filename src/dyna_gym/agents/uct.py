"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import partial as prt
import logging
from math import sqrt, log, prod
from operator import add, mul, truediv
import random

from gym import spaces
from scipy.special import softmax
from scipy.stats import entropy

from . import Agent
from .mcts_common import (
    chance_node_value,
    mcts_procedure,
    combinations,
    ChanceNode,
    DecisionNode,
    typ_tree_policy,
)
from ...generate.program_env import ProgramEnv
from ...generate.default_pi import PolicyHeuristic, MlcPolicyHeuristic
from ...functional import first, constant_1, constant_none, tpmap_p4_str

logger = logging.getLogger(__name__)

biop_repr = {add: "+", mul: "*", truediv: "/", None: ""}
biopstr_repr = {"add": "+", "mul": "*", "div": "/", None: ""}
bin_ops: dict[str, Callable] = {"add": add, "mul": mul, "div": truediv}


def combine(
    x: float,
    y: float | None,
    op: Callable[[float, float], float],
    weight: float | int = 1,
) -> float:
    if y is None:
        return x
    return op(x, weight * y)


def dynagym_exploration_term(
    node: ChanceNode, exploration_denominator_summand: int | float
) -> float:
    """The exploration term in the UCB formula, as in dyna gym
    https://github.com/SuReLI/dyna-gym/blob/master/dyna_gym/agents/uct.py#L65"""
    return sqrt(
        log(node.parent.visits)
        / (exploration_denominator_summand + len(node.sampled_returns))
    )


def pgtd_exploration_term(
    node: ChanceNode, exploration_denominator_summand: int | float
) -> float:
    """The exploration term in the P-UCB formula as in PG-TD
    https://github.com/shunzh/Code-AI-Tree-Search/blob/main/dyna_gym/agents/uct.py#L90
    (https://github.com/shunzh/Code-AI-Tree-Search/blob/d72aeac265a0db56cdafcf263d2f724717208b89/dyna_gym/agents/uct.py#L90C64-L90C64)
    """
    return sqrt(log(node.parent.visits)) / (
        exploration_denominator_summand + len(node.sampled_returns)
    )


def get_prob(node: ChanceNode, entropy_prob_smoother: float | None = None) -> float:
    if entropy_prob_smoother is None:
        return node.prob
    # return (1 - entropy_prob_smoother) * prob + entropy_prob_smoother / len(node.children)
    return node.prob + entropy_prob_smoother


def get_k_forward_entropy_term(
    node: ChanceNode,
    forward_k: int,
    averaging: bool,
    entering: Callable[[float, float], float],
    entropy_prob_smoother: float | None,
    # entropy_smoother: float,
    discounting: float | int = 1,
) -> float | None:
    def get_next_chancenodes(node: ChanceNode) -> tuple[ChanceNode]:
        if len(node.children) > 0:
            node: DecisionNode = node.children[-1]
            return tuple(node.children)
        else:
            return ()

    assert forward_k > 0, "`forward_k` must be positive"
    assert isinstance(node, ChanceNode), "computation must be performed on `ChanceNode`"
    logger.debug(f"state : {node.parent.state}")

    get_prob_smoothed = prt(get_prob, entropy_prob_smoother=entropy_prob_smoother)

    cumsum: int | float = 0
    entropy_present_level: list[int] = [0] * forward_k
    level: int = 1
    probs_ancestral: tuple[float] = (1.0,)
    queue: list[tuple[int, tuple[float], tuple[ChanceNode]]]
    queue = [(level, probs_ancestral, get_next_chancenodes(node))]
    while len(queue) > 0:
        # get entropy term of current level, if not leaf
        level, probs_ancestral, nodes_same_parent = queue.pop(0)
        if len(nodes_same_parent) == 0:
            continue
        probs = tuple(map(get_prob, nodes_same_parent))  # for debugging
        if len(nodes_same_parent) > 1:  # skip if having a singular child
            weighting = (discounting**level) * prod(probs_ancestral)
            probs_smoothed = tuple(map(get_prob_smoothed, nodes_same_parent))
            entropy_value = entropy(probs_smoothed)
            cumsum_prev = cumsum  # for debugging
            cumsum += combine(weighting, entropy_value, op=entering)
            logger.debug(
                "\nEntropy :\n"
                f"num_children : {len(probs)}\n"
                f"probs[:4] : {probs[:4]}    ;    probs_smoothed[:4] : {probs_smoothed[:4]}\n"
                # f"probs : {probs}\nprobs_smoothed : {probs_smoothed}\n"
                f"entropy_cumsum = cumsum_prev + weighting {biop_repr[entering]} entropy_value\n"
                f"{cumsum} = {cumsum_prev} + {weighting} {biop_repr[entering]} {entropy_value}"
            )
            entropy_present_level[level - 1] = 1

        # traverse all nodes for their children if level < forward_k
        if level == forward_k:
            continue
        # It is assumed that sum(each.prob for each in nodes_same_parent) == 1
        for each in nodes_same_parent:
            if len(each.children) == 0:
                continue
            queue.append(
                (level + 1, probs_ancestral + (each.prob,), get_next_chancenodes(each))
            )

    if cumsum == 0:
        return None
    if not averaging:
        return cumsum
    return cumsum / sum(entropy_present_level)


exploration_mode_variants: dict[str, Callable[[ChanceNode], float]] = {
    "dynagym": dynagym_exploration_term,
    "pgtd": pgtd_exploration_term,
}


class PUCBKScorer:
    def __init__(
        self,
        algo: str,
        exploitation_mode: str,
        exploration_mode: str,
        exploration_denominator_summand: int | float,
        ucb_constant: float,
        entropy_forward_k: int,
        entropy_k_averaging: bool,
        entropy_prob_smoother: float,
        entropy_entering_mode: str,
        entropy_combining_mode: str | None,
        alpha: int | float,
        reward_estimate_combining_mode: str | None,
        reward_estimate_weight: float,
        dp: MlcPolicyHeuristic,
        env: ProgramEnv | None = None,
    ) -> None:
        """ """
        use_entropy = entropy_combining_mode is not None
        if use_entropy:
            assert entropy_entering_mode is not None
            assert entropy_forward_k > 0
            assert entropy_k_averaging is not None
        use_reward_estimate = reward_estimate_combining_mode is not None

        self.dp = dp
        self.env = env

        self.get_q = prt(chance_node_value, mode=exploitation_mode)

        self.get_exploration_fraction = prt(
            exploration_mode_variants[exploration_mode],
            exploration_denominator_summand=exploration_denominator_summand,
        )
        if algo == "ucb":
            self.get_p = constant_1
        elif algo == "p_ucb":
            self.get_p = prt(get_prob, entropy_prob_smoother=None)
        else:
            raise NotImplementedError(f"algo {algo} not yet implemented")
        self.ucb_constant = ucb_constant

        if not use_entropy:
            self.get_entropy_term = constant_none
            self.combine_with_entropy = first
        else:
            # TODO: implement this method ; and caching results (only if not None)
            self.get_entropy_term = prt(
                get_k_forward_entropy_term,
                forward_k=entropy_forward_k,
                averaging=entropy_k_averaging,
                entering=bin_ops[entropy_entering_mode],
                entropy_prob_smoother=entropy_prob_smoother,
                # entropy_smoother=entropy_smoother,
            )
            self.combine_with_entropy = prt(
                combine, op=bin_ops[entropy_combining_mode], weight=alpha
            )

        if not use_reward_estimate:
            self.get_estimate_reward = constant_none
            self.combine_with_reward_estimate = first
        else:
            self.reward_cache: OrderedDict[tuple, float] = OrderedDict()
            self.get_estimate_reward = self.get_bs_reward_estimate
            self.combine_with_reward_estimate = prt(
                combine,
                op=bin_ops[reward_estimate_combining_mode],
                weight=reward_estimate_weight,
            )

        # for debugging messages
        self.entropy_opsy = biopstr_repr[entropy_combining_mode]
        self.reward_estimate_opsy = biopstr_repr[reward_estimate_combining_mode]

    def get_bs_reward_estimate(self, node: ChanceNode) -> float:
        """Get the reward estimate of a chance node using the heuristic"""
        state_next, reward, terminal = self.env.transition(node.parent.state, node.action)
        if terminal:
            return reward
        return self.dp.get_reward_estimate(state_next)

    def dbmsg(
        self, node, out_value, q_value, p_lm, u_value, entropy_term, reward_estimate
    ) -> str:
        # fmt: off
        records: tuple[float] = (out_value, q_value, self.ucb_constant, p_lm, u_value,
            self.entropy_opsy, entropy_term, self.reward_estimate_opsy, reward_estimate)
        # fmt: on
        len_returns = len(node.sampled_returns)
        return (
            "\n"
            + f"length of state : {len(node.parent.state)}    ;    state : {node.parent.state}    ;    state : {node.parent.dp.tokenizer.tokenizer.decode(node.parent.state)}\n"
            + f"action : {node.action}    ;     action : {node.parent.dp.tokenizer.tokenizer.decode(node.action)}\n"
            + f"number of returns : {len_returns}    ;     last 8 returns : {[round(s, 2) for s in node.sampled_returns[-8:]]}\n"
            + f"best returns : {max(node.sampled_returns) if len_returns > 0 else 0}    ;    average returns : {sum(node.sampled_returns) / len_returns if len_returns > 0 else 0}\n"
            + f"(p)_ucb_(k)_value = q_value + (ucb_constant * p_lm * u_value) {self.entropy_opsy} entropy_agg_term {self.reward_estimate_opsy} reward_estimate\n"
            + "{} = {} + ({} * {} * {}) {} {} {} {}".format(*tpmap_p4_str(records))
        )

    def __call__(self, node: ChanceNode) -> float:
        q_value = self.get_q(node)
        u_value = self.get_exploration_fraction(node)
        p_lm = self.get_p(node)
        entropy_term = self.get_entropy_term(node)
        combined_with_entropy = self.combine_with_entropy(
            self.ucb_constant * p_lm * u_value, entropy_term
        )
        reward_estimate = self.get_estimate_reward(node)
        combined_with_reward_estimate = self.combine_with_reward_estimate(
            combined_with_entropy, reward_estimate
        )
        out_value = q_value + combined_with_reward_estimate
        logger.debug(
            self.dbmsg(
                node, out_value, q_value, p_lm, u_value, entropy_term, reward_estimate
            )
        )
        return out_value


# def pb_ucb(
#     node: ChanceNode,
#     exploitation_term: Callable[[ChanceNode], float],
#     exploration_term: Callable[[ChanceNode], float],
#     ucb_constant: float,
# ) -> float:
#     """Upper Confidence Bound of a chance node, weighted by prior probability"""
#     return node.prob * (exploitation_term(node) + ucb_constant * exploration_term(node))


def var_p_ucb(
    node: ChanceNode,
    exploitation_term: Callable[[ChanceNode], float],
    exploration_term: Callable[[ChanceNode], float],
    ucb_base: float,
    ucb_constant: float,
) -> float:
    """Upper Confidence Bound of a chance node, the ucb exploration weight is a variable"""
    ucb_parameter = log((node.parent.visits + ucb_base + 1) / ucb_base) + ucb_constant
    return exploitation_term(node) + (ucb_parameter * node.prob * exploration_term(node))


def max_tree_policy(ag: Agent, children: Iterable[ChanceNode]) -> ChanceNode:
    selected = max(children, key=ag.scoring)
    logger.debug(
        f"\nselected action : {ag.dp.tokenizer.tokenizer.decode(selected.action)}"
    )
    return selected


def sampling_tree_policy(ag: Agent, children: Iterable[ChanceNode]) -> ChanceNode:
    selected = random.choices(children, softmax(tuple(map(ag.scoring, children))))[0]
    logger.debug(
        f"\nselected action : {ag.dp.tokenizer.tokenizer.decode(selected.action)}"
    )
    return selected


class UCT(Agent):
    """UCT agent"""

    def __init__(
        self,
        action_space,
        entropy_combining_mode: str | None,
        entropy_forward_k: int,
        entropy_k_averaging: bool,
        entropy_entering_mode: str,
        # entropy_smoother: float,
        entropy_prob_smoother: float,
        entropy_alpha: float,
        reward_estimate_combining_mode: bool,
        reward_estimate_weight: float,
        rollouts=100,
        horizon=100,
        gamma=0.9,
        ucb_constant=6.36396103068,
        ucb_base=50.0,  # for var_p_ucb
        is_model_dynamic=True,
        dp=None,
        env: ProgramEnv | None = None,  # for PUCBKScorer only
        reuse_tree=False,
        alg="ucb",
        exploration_mode="",
        selection="",
        exploitation_mode="",
        exploration_denominator_summand: int | float = 1,
        # width=None,
        # ts_mode="best",
    ) -> None:
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        # self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic
        # self.width = width or self.n_actions  # the number of children for each node, default is num of actions
        self.dp = dp
        # self.ts_mode = ts_mode
        self.reuse_tree = reuse_tree

        self.scoring = PUCBKScorer(
            algo=alg,
            exploitation_mode=exploitation_mode,
            exploration_mode=exploration_mode,
            exploration_denominator_summand=exploration_denominator_summand,
            ucb_constant=ucb_constant,
            entropy_combining_mode=entropy_combining_mode,
            entropy_forward_k=entropy_forward_k,
            entropy_entering_mode=entropy_entering_mode,
            entropy_k_averaging=entropy_k_averaging,
            entropy_prob_smoother=entropy_prob_smoother,
            # entropy_smoother=entropy_smoother,
            alpha=entropy_alpha,
            reward_estimate_combining_mode=reward_estimate_combining_mode,
            reward_estimate_weight=reward_estimate_weight,
            dp=dp,
            env=env,
        )
        self.tree_policy: typ_tree_policy
        if selection == "max":
            self.tree_policy = max_tree_policy
        elif selection == "random":
            self.tree_policy = sampling_tree_policy
        else:
            raise Exception(f"unknown selection {selection}")

        self.root = None

    def act(
        self,
        env: ProgramEnv,
        done: bool,
        rollout_weight=1,
        term_cond: Callable | None = None,
    ):
        root = self.root if self.reuse_tree else None
        opt_act, self.root = mcts_procedure(
            ag=self,
            tree_policy=self.tree_policy,
            env=env,
            done=done,
            root=root,
            rollout_weight=rollout_weight,
            term_cond=term_cond,
        )
        return opt_act
