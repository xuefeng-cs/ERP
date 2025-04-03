from rdkit.Chem.Descriptors import qed

# from rdkit import Chem

import numpy as np

import pickle
import gzip
import os

from rdkit.Chem import AllChem as Chem

import math

from rdkit import rdBase

rdBase.DisableLog("rdApp.error")

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras

# from .ST_inference.ST_funcs.clr_callback import *
from .smiles_pair_encoders_functions import SMILES_SPE_Tokenizer

from .utils import pad, r2


def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != "" and mol is not None and mol.GetNumAtoms() > 1


def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


# ---------- Solubility ----------
def logP(smile, norm=False, clip=False, invalid_value=0.0):
    # values/code from https://github.com/gablg1/ORGAN/blob/master/organ/mol_metrics.py#L607
    if not verify_sequence(smile):
        return invalid_value
    low_logp = -2.12178879609
    high_logp = 6.0429063424
    logp = Chem.Crippen.MolLogP(Chem.MolFromSmiles(smile))
    if norm:
        logp = remap(logp, low_logp, high_logp)
    if clip:
        logp = np.clip(logp, 0.0, 1.0)
    return logp


def batch_solubility(smiles, norm=False, clip=False):
    vals = [logP(s, norm=norm, clip=clip) for s in smiles]
    return vals


# ---------- Druglikeness ----------
def druglikeness(smile, invalid_value=0.0):
    if not verify_sequence(smile):
        return invalid_value
    qed_ = qed(Chem.MolFromSmiles(smile))
    return qed_


def batch_druglikeness(smiles):
    vals = [druglikeness(s) for s in smiles]
    return vals


# ---------- Synthetizability / Synthetic Accessibility ----------
def readSAModel(filename="./models/metricsmodels/SA_score.pkl.gz"):
    print("mol_metrics: reading SA model ...")
    if filename == "SA_score.pkl.gz":
        filename = os.path.join(os.path.dirname(__file__), filename)
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    return SA_model


SA_model = readSAModel()


def SA_score(smile, norm=False, clip=False, invalid_value=0.0):
    # values/code from https://github.com/gablg1/ORGAN/blob/master/organ/mol_metrics.py#L754
    if not verify_sequence(smile):
        return invalid_value
    mol = Chem.MolFromSmiles(smile)
    # fragment score
    fp = Chem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    if norm:
        sascore = remap(sascore, 5, 1.5)
    if clip:
        sascore = np.clip(sascore, 0.0, 1.0)
    return sascore


def batch_sa(smiles, norm=False, clip=False):
    vals = [SA_score(s, norm=norm, clip=clip) for s in smiles]
    return vals


# ---------- docking scores ----------
cancer_model = tf.keras.models.load_model(
    "./models/metricsmodels/ST_Model_rtcb/", custom_objects={"r2": r2}, compile=False
)
covid_model = tf.keras.models.load_model(
    "./models/metricsmodels/ST_Model/", custom_objects={"r2": r2}, compile=False
)
vocab_file = "./models/metricsmodels/VocabFiles/vocab_spe.txt"
spe_file = "./models/metricsmodels/VocabFiles/SPE_ChEMBL.txt"
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file=spe_file)
maxlen = 45


def batch_dockingScore(
    smiles, dataset="cencer", batch_size=64, norm: bool = False, invalid_value=0.0
):
    if dataset == "cancer":
        model = cancer_model
    elif dataset == "covid":
        model = covid_model
    else:
        raise ValueError("dataset must be 'cancer' or 'covid'")

    x_inference = np.array(
        [list(pad(tokenizer(smi)["input_ids"], maxlen, 0)) for smi in smiles]
    )
    Output = model.predict(x_inference, batch_size=batch_size, verbose=0)

    val = tuple(
        Output[i][0][0] if verify_sequence(smiles[i]) else invalid_value
        for i in range(len(smiles))
    )
    if not norm:
        return val

    min_docking_score = 0.0
    max_docking_score = 15.0
    return tuple(remap(v, min_docking_score, max_docking_score) for v in val)


def docking_score(
    smiles: str, dataset: str, normalization: bool, invalid_value=0.0
) -> float:
    return batch_dockingScore(
        [smiles],
        dataset=dataset,
        batch_size=1,
        norm=normalization,
        invalid_value=invalid_value,
    )[0]


if __name__ == "__main__":
    import os
    import warnings

    warnings.filterwarnings("ignore")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    test_list = [
        "CCNC(=O)[C@H](C)N(Cc1cccc(Cl)c1)C(=O)CN(c1cccc(C)c1)S(C)(=O)=O",
        "CO2",
        "CN(C)c1nc(-c2cccc(C(=O)NC[C@@]3(O)CC[NH2+]C3)c2)ncc1F",
        "SO2",
    ]

    print(batch_sa(test_list))
    print(batch_sa(test_list, norm=True))
    print(batch_sa(test_list, norm=True, clip=True))

    print(batch_solubility(test_list))
    print(batch_solubility(test_list, norm=True))
    print(batch_solubility(test_list, norm=True, clip=True))

    print(batch_druglikeness(test_list))

    print(batch_dockingScore(test_list, dataset="cancer"))
    print(batch_dockingScore(test_list, dataset="covid"))
