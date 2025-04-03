from collections.abc import Callable
import logging

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from toolz import compose as cmp

from .sascorer import sascorer

logger = logging.getLogger(__name__)


def get_cycle_score(compound: str) -> float:
    cycle_list = nx.cycle_basis(
        nx.Graph(Chem.GetAdjacencyMatrix(Chem.MolFromSmiles(compound)))
    )
    cycle_length = 0 if len(cycle_list) == 0 else max([len(j) for j in cycle_list])
    return max(0, cycle_length - 6)


def get_score(compound: str) -> float:
    molecule = Chem.MolFromSmiles(compound)
    if molecule is None:
        return -1000
    try:
        logp = Descriptors.MolLogP(molecule)
    except BaseException as e:
        # logger.warning(f"MollLogP error: {e}")
        logp = -1000
    try:
        sa_score = sascorer.calculateScore(molecule)
    except BaseException as e:
        # logger.warning(f"SA score error: {e}")
        sa_score = 1000
    cycle_score = get_cycle_score(compound)
    score = logp - sa_score - cycle_score
    logger.debug(
        f"LogP: {logp}, SA score: {sa_score}, Cycle score: {cycle_score}, Score: {score}"
    )
    return logp - sa_score - cycle_score  # J score as in ChemTS (Yang et al. 2017)


def get_reward(score: float) -> float:
    return score / (1 + abs(score))


reward_x_compound: Callable[[str], float] = cmp(get_reward, get_score)


if __name__ == "__main__":
    examples: tuple[str] = (
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c2ccccc2c1OC(F)F)c1cccc2ccccc12",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)ccc1C1=CCCCC1)c1cc(F)cc(Cl)c1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2N=C(SC2CCCCC2)c2ccccc2)cc(Cl)c1Cl)c1ccc2ccccc2n1",
        "O=C(Nc1cc(Oc2ccc(Cl)cc2Cl)ccc1Nc1cc(Cl)ccc1Cl)c1ccc(Cl)cc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2Cl)c(Cl)cc1Br)N(c1ccccc1)c1ccc(Cl)cc1",
        "O=C(Nc1cc(Oc2c(Cl)cccc2Oc2ccc(-c3ccccc3)cc2)ccc1Cl)c1ccccc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2Cl)c(Cl)c(C(=O)N(Cc2ccccc2)c2ccccc2)c1Cl)c1ccccc1F",
        "O=C(Nc1cc(Oc2ccc(Cl)cc2Cl)cc(Cl)c1Cl)c1ncoc1-c1ccc(Sc2ccccc2)cc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c2ncccc2c1Cl)c1ccc(Cl)cc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2)c(Cl)cc1Cl)c1cc(F)ccc1Cl",
        "O=C(Nc1cc(Oc2c(Cl)cccc2Oc2ccccc2)nnc1-c1ccccc1)c1sc2ccccc2c1Cc1ccccc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c2ncccc2c1Cl)c1ccccc1Cl",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c(Cl)cc1Cl)c1ccc(F)cc1F",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccccc2)ccc1C(F)(F)F)c1ccc(Cl)c2ccccc12",
        "O=C(Nc1cc(Nc2c(Cl)cccc2Cl)c(Cl)cc1OC(F)F)N(Cc1ccccc1)c1ccccc1C(F)(F)F",
        "O=C(Nc1cc(Oc2c(Cl)cccc2Oc2ccccc2C2=CCCCC2)cc(Cl)c1)c1ccccc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2[N+](=O)[O-])cs1)c1sc2ccc(Br)cc2c1N(c1ccccc1)c1ccccc1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2Cl)c(C(=O)c2ccc(Cl)cc2F)c(Cl)c1)Nc1cccc(Cl)c1",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)cc(F)c1F)c1cccc2ccccc12",
        "O=C(Nc1cc(Nc2c(Cl)cccc2NCc2ccc(Cl)cc2Cl)c(Cl)cc1Cl)c1cccs1",
    )

    # reward_x_compound = cmp(get_reward, get_score)
    # rewards = map(reward_x_compound, examples)
    print("Examples and their scores:")
    for example, reward in zip(examples, map(get_score, examples)):
        print(f"{example} : {reward}")

    # compound = input("Provide a SMILES string:")
    # print(f"J score: {get_score(compound)}")
    # print(f"Reward: {get_reward(get_score(compound))}")
