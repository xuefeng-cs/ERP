from collections.abc import Callable, Iterable

from .logp import reward_x_compound


class ValueModel:
    def __init__(self, scorer: Callable[[str], float]) -> None:
        self.scorer = scorer

    def __call__(self, compound: Iterable[str]) -> float:
        return self.scorer("".join(compound))
