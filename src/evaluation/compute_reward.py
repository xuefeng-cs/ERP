from collections.abc import Iterable
from functools import partial as prt
import logging

from ..scoring.logp import reward_x_compound as reward_x_compound_logp
from ..mol_metric.mol_metric import (
    logP,
    druglikeness,
    SA_score,
    batch_dockingScore,
    docking_score,
)


logger = logging.getLogger(__name__)


def compute_reward_logp(output_str: str):
    return reward_x_compound_logp(output_str)


class SDADScorer:
    def __init__(
        self, dataset: str, normalization: bool, invalid_values: Iterable[float]
    ):
        self.dataset = dataset
        self.normalization = normalization
        self.logP = prt(logP, norm=normalization, invalid_value=invalid_values[0])
        self.druglikeness = prt(druglikeness, invalid_value=invalid_values[1])
        self.SA_score = prt(
            SA_score,
            norm=normalization,
            clip=normalization,
            invalid_value=invalid_values[2],
        )
        self.batch_dockingScore = prt(
            batch_dockingScore,
            dataset=dataset,
            norm=normalization,
            invalid_value=invalid_values[3],
        )
        self.docking_score = prt(
            docking_score,
            dataset=dataset,
            normalization=normalization,
            invalid_value=invalid_values[3],
        )

    def __call__(self, output_str: str):
        return (
            self.logP(output_str)
            + self.druglikeness(output_str)
            + self.SA_score(output_str)
            + self.docking_score(output_str)
        )

    def individual_rewards_batch(
        self, output_strs: Iterable[str]
    ) -> dict[str, tuple[float]]:
        return {
            "solvability": tuple(map(self.logP, output_strs)),
            "druglikeness": tuple(map(self.druglikeness, output_strs)),
            "sa": tuple(map(self.SA_score, output_strs)),
            "docking_score": self.batch_dockingScore(output_strs),
        }

    def unnormalized_individual_rewards_batch(
        self, output_strs: Iterable[str]
    ) -> dict[str, tuple[float]]:
        return {
            "solvability": tuple(map(logP, output_strs)),
            "druglikeness": tuple(map(druglikeness, output_strs)),
            "sa": tuple(map(SA_score, output_strs)),
            "docking_score": tuple(batch_dockingScore(output_strs, dataset=self.dataset)),
        }
