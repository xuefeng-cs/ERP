import logging
from collections.abc import Iterable
from functools import partial as prt
import json
import os
import warnings
from abc import abstractmethod, ABC
from collections import OrderedDict

from ..evaluation.compute_reward import compute_reward_logp, SDADScorer
from ..scoring.logp import get_score
from ..mol_metric.mol_metric import verify_sequence
from ..nn.peripheral import SpecialTokenizer

logger = logging.getLogger(__name__)


class ProgramEnv(ABC):
    """
    Code generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: pass rate of the program (on the training set in training, and on the test set in testing).
    """

    def __init__(self, terminal_token, horizon):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward: OrderedDict[tuple, float] = OrderedDict()  # state -> reward

    def transition(self, s: list, a, is_model_dynamic=True) -> tuple:
        next_state = s + [a]
        if a == self.terminal_token or len(next_state) == self.horizon:
            done = True
            reward = self.get_reward(next_state)
        else:
            done = False
            reward = 0  # no intermediate reward
        return next_state, reward, done

    def step(self, action) -> tuple:
        return self.transition(self.state, action) + ({},)

    @abstractmethod
    def get_reward(self, s, mode="train"):
        """
        This needs to be defined for each dataset
        """
        pass

    def convert_state_to_program(self, s):
        """
        The state may be different from the program. This converts it back to executable program.
        """
        return s

    def equality_operator(self, s1, s2):
        return s1 == s2

    def get_complete_states(self) -> tuple[tuple]:
        """
        Return the tuple of complete programs reached so far.
        This can be found from the tuple of cached rewards.
        """
        return tuple(self.cached_reward.keys())


class MolProgramEnv(ProgramEnv):
    """
    Molecule generation environment.
    """

    def __init__(
        self,
        tokenizer: SpecialTokenizer,
        horizon: int,
        metric_name: str,
        normalization: bool,
        invalid_values: Iterable[float],
        prompt_string: str = "",
        docking_dataset: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.prompt_string = prompt_string
        self._bos_token_id = self.tokenizer.encode(self.tokenizer.bos_token)
        if (prompt_string is None) or (not prompt_string):
            self._state = self._bos_token_id
        else:
            self._state = self._bos_token_id + self.tokenizer.encode(prompt_string)

        # terminal_token = self.tokenizer.encode("<|endoftext|>")[0]
        super().__init__(terminal_token=self.tokenizer.eos_token_id, horizon=horizon)
        self.metric_name = metric_name
        if metric_name == "logp":
            self.scorer = compute_reward_logp
        elif metric_name == "sdad":
            self.scorer = SDADScorer(
                docking_dataset,
                normalization=normalization,
                invalid_values=invalid_values,
            )
        else:
            raise Exception(f"Unknown metric {metric_name}")

    @property
    def state(self) -> list[int]:
        return self._state

    def convert_state_to_program(self, s: Iterable[int]) -> str:
        program: str = self.tokenizer.decode(s)
        # logger.debug(f"convert_state_to_program took state   : {s}")
        # logger.debug(f"convert_state_to_program gave program : {program}")
        return program

    def convert_state_to_program_batch(self, s: Iterable[Iterable[int]]) -> tuple[str]:
        return tuple(self.tokenizer.batch_decode(s))

    def get_reward(self, s: Iterable, mode="train") -> float:
        # if tuple(s) in self.cached_reward.keys() and mode == "train":
        if tuple(s) in self.cached_reward.keys():
            return self.cached_reward[tuple(s)]  # cache rewards for training

        output_str = self.convert_state_to_program(s)
        reward = self.scorer(output_str)
        if mode == "train":
            self.cached_reward[tuple(s)] = reward
        logger.debug(
            "\n"
            + "Evaluation :\n"
            + f"generation : {s}\n"
            + f"generation : {self.tokenizer.tokenizer.decode(s)}\n"
            + f"molecule : {output_str}\n"
            + f"reward : {reward}"
        )
        return reward

    def get_rewards_batch(self, s, mode="test") -> tuple[float]:
        assert mode == "test"
        return tuple(map(prt(self.get_reward, mode=mode), s))

    def unnormalized_individual_rewards_batch(
        self, ss: Iterable[str]
    ) -> dict[str, tuple[float]]:
        if not hasattr(self.scorer, "unnormalized_individual_rewards_batch"):
            return {}
        return self.scorer.unnormalized_individual_rewards_batch(ss)

    def individual_rewards_batch(self, ss: Iterable[str]) -> dict[str, tuple[float]]:
        if not hasattr(self.scorer, "individual_rewards_batch"):
            return {}
        return self.scorer.individual_rewards_batch(ss)
