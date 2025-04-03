"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
import random
import itertools
import warnings

from gym import spaces
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import Agent
from .mcts_common import mcts_procedure, combinations
from ..utils import utils


def mcts_tree_policy(ag, children):
    return random.choice(children)


class MCTS(Agent):
    """
    MCTS agent
    """

    def __init__(
        self, action_space, rollouts=100, horizon=100, gamma=0.9, is_model_dynamic=True
    ):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.is_model_dynamic = is_model_dynamic

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p, [spaces.discrete.Discrete, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])

    def display(self):
        """
        Display infos about the attributes.
        """
        print("Displaying MCTS agent:")
        print("Action space       :", self.action_space)
        print("Number of actions  :", self.n_actions)
        print("Rollouts           :", self.rollouts)
        print("Horizon            :", self.horizon)
        print("Gamma              :", self.gamma)
        print("Is model dynamic   :", self.is_model_dynamic)

    def act(self, env, done):
        (
            opt_act,
            _,
        ) = mcts_procedure(self, mcts_tree_policy, env, done)
        return opt_act
