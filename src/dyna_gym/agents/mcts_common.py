"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
from collections.abc import Callable, Iterable
import logging
import itertools
import random
from typing import Any
import warnings

import ipdb
from gym import spaces
import numpy as np
from tqdm import tqdm

from . import Agent
from ...generate.default_pi import PolicyHeuristic, MlcPolicyHeuristic
from ...generate.program_env import ProgramEnv

logger = logging.getLogger(__name__)


class ChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """

    def __init__(self, parent: "DecisionNode", action: int, score: float):
        self.parent: "DecisionNode" = parent
        self.depth: int = parent.depth
        self.action: int = action
        self.prob: float = score  # the probability that this action should be token, provided by default policy
        self.children: list[DecisionNode] = []
        self.sampled_returns = []

    # def expanded(self):
    #     return len(self.children) > 0


class DecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """

    def __init__(
        self,
        parent: ChanceNode | None,
        state,
        possible_actions=[],
        is_terminal=False,
        dp: PolicyHeuristic | None = None,
        id=None,
    ):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Non root node
            self.depth = parent.depth + 1
        if dp is None:
            self.possible_actions = possible_actions
            random.shuffle(self.possible_actions)
            # if no default policy is provided, assume selection probability is uniform
            self.action_scores = [1.0 / len(self.possible_actions)] * len(
                self.possible_actions
            )
        else:
            # get possible actions from dp
            # default policy suggests what children to consider
            top_k_predict, top_k_scores = dp.get_top_k_predict(self.state)
            logger.debug(
                "\nExpansion :\n"
                + f"state : {self.state}\n"
                + f"state : {dp.tokenizer.tokenizer.decode(self.state)}\n"
                + f"top_k_predict : {top_k_predict}\n"
                + f"top_k_predict : {dp.tokenizer.tokenizer.decode(top_k_predict)}\n"
                + f"top_k_scores : {[round(s, 2) for s in top_k_scores]}"
            )
            self.possible_actions = top_k_predict
            self.action_scores = top_k_scores

        # populate its children
        self.children: list[ChanceNode] = [
            ChanceNode(parent=self, action=act, score=score)
            for act, score in zip(self.possible_actions, self.action_scores)
        ]
        self.explored_children: int = 0
        # self.explored_children_ids: set[int] = set()
        # this decision node should be visited at least once, otherwise p-uct makes no sense for this node
        self.visits: int = 1
        # used to save any information of the state, we use this for saving complete programs generated from it
        self.info = {}
        self.dp = dp  # for logging and debugging

    # def is_fully_expanded(self):
    #     return all([child.expanded() for child in self.children])


def chance_node_value(node: ChanceNode, mode="best") -> float | int:
    """Value of a chance node"""
    q_value: int = 0.0
    if len(node.sampled_returns) > 0:
        if mode == "best":  # max return (reasonable because the model is deterministic?)
            q_value = max(node.sampled_returns)
        elif mode in ("sample", "average"):  # Use average return
            q_value = sum(node.sampled_returns) / len(node.sampled_returns)
        else:
            raise Exception(f"Unknown tree search mode {mode}")
    # logger.debug(
    #     f"node.sampled_returns[-3:] : {[round(r, 2) for r in node.sampled_returns[-3:]]}"
    # )
    # logger.debug(f"q_value: {q_value}")
    return q_value


typ_tree_policy = Callable[[Agent, Iterable[ChanceNode]], ChanceNode]


def mcts_procedure(
    ag: Agent,
    tree_policy: typ_tree_policy,
    env: ProgramEnv,
    done,
    root=None,
    rollout_weight=1.0,
    term_cond=None,
    # ts_mode="best",
):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.
    """
    decision_node_num: int = 0
    if root is not None:
        # if using existing tree, making sure the root is updated correctly
        assert root.state == env.state
    else:
        # create an empty tree
        root = DecisionNode(
            parent=None,
            state=env.state,
            possible_actions=ag.action_space.copy(),
            is_terminal=done,
            dp=ag.dp,
            id=decision_node_num,
        )
        decision_node_num += 1

    # make sure the rollout number is at least one, and is at most ag.rollouts
    if rollout_weight > 1:
        warnings.warn("How come rollout_weight > 1? Setting to 1.")
    rollouts = np.clip(int(ag.rollouts * rollout_weight), 1, ag.rollouts)

    print("Performing rollouts.")
    for _ in tqdm(range(rollouts)):
        if term_cond is not None and term_cond():
            break
        rewards = []  # Rewards collected along the tree for the current rollout
        node = root  # Current node
        terminal = done

        # Selection
        a_leaf_selected = False
        while not a_leaf_selected:
            if isinstance(node, DecisionNode):  # DecisionNode
                if node.is_terminal:
                    a_leaf_selected = True  # Selected a terminal DecisionNode
                else:
                    logger.debug(
                        "\nSelection :\n"
                        + f"state : {node.state}\n"
                        + f"state : {node.dp.tokenizer.tokenizer.decode(node.state)}"
                    )
                    node: ChanceNode = tree_policy(ag, node.children)
                    # move down to a ChanceNode
            else:  # ChanceNode
                state_p, reward, terminal = env.transition(
                    node.parent.state, node.action, ag.is_model_dynamic
                )
                rewards.append(reward)

                is_leaf: bool = True
                for i in range(len(node.children)):
                    # Note that for deterministic transitions, node.children contains at most one child
                    if env.equality_operator(node.children[i].state, state_p):
                        # state_p already in the tree, point node to the corresponding Decision Node
                        node: DecisionNode = node.children[i]
                        is_leaf = False
                        break
                if is_leaf:
                    a_leaf_selected = True  # Selected a ChanceNode

        # # Some testing
        # if isinstance(node, ChanceNode):
        #     chance_node = node
        #     decision_node = node.parent
        #     decision_node.explored_children_ids.add(id(chance_node))
        #     if len(decision_node.explored_children_ids) > 1:
        #         print(
        #             f"decision_node.explored_children_ids : {decision_node.explored_children_ids}"
        #         )

        # Expansion
        # If node is a decision node, then it must be a terminal node, do nothing here
        if isinstance(node, ChanceNode):
            node.children.append(
                DecisionNode(
                    node,
                    state_p,
                    ag.action_space.copy(),
                    terminal,
                    dp=ag.dp,
                    id=decision_node_num,
                )
            )
            # assert len(node.children) == 1
            decision_node_num += 1
            node: DecisionNode = node.children[-1]

        # Evaluation
        # now `rewards` collected all rewards in the ChanceNodes above this node
        assert isinstance(node, DecisionNode)
        state = node.state
        if ag.dp is None:
            t = 0
            estimate = 0
            while (not terminal) and (t < ag.horizon):
                action = env.action_space.sample()
                state, reward, terminal = env.transition(
                    state, action, ag.is_model_dynamic
                )
                estimate += reward * (ag.gamma**t)
                t += 1
        else:
            if not node.is_terminal:
                if ag.dp.use_value:
                    state = ag.dp.get_short_horizon_sequence(state)
                    estimate = ag.dp.get_value(state)  # using the pre-trained value model
                else:
                    # follow the default policy to get a terminal state
                    logger.debug(
                        "\nEvaluation :\n"
                        + f"state : {state}\n"
                        + f"state : {ag.dp.tokenizer.tokenizer.decode(state)}"
                    )
                    estimate = ag.dp.get_reward_estimate(state)
                    # node.info["complete_program"] = state  # save for demo
            else:
                # the rewards are defined on terminating actions, the terminal states have no rewards
                estimate = 0

        # Backpropagation
        node.visits += 1
        node: ChanceNode = node.parent
        assert isinstance(node, ChanceNode)
        while node is not None:
            if len(rewards) != 0:
                estimate = rewards.pop() + ag.gamma * estimate
            node.sampled_returns.append(estimate)
            node.parent.visits += 1
            node: ChanceNode = node.parent.parent
        assert len(rewards) == 0  # finishing backpropagating all the rewards

    return (
        # max(root.children, key=lambda n: chance_node_value(n, ts_mode=ts_mode)).action,
        max(root.children, key=lambda n: chance_node_value(n)).action,
        root,
    )


def update_root(ag, act, state_p):
    root_updated = False
    for chance_node in ag.root.children:
        if act == chance_node.action:
            for decision_node in chance_node.children:
                if decision_node.state == state_p:
                    ag.root = decision_node
                    root_updated = True
                    break

    if not root_updated:
        raise Exception(
            "root update fails, can't find the next state, action pair in tree."
        )


def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError
