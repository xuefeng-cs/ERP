from collections.abc import Callable, Iterable, Mapping
from functools import partial as prt
import logging
from os import path as osp

from datasets import load_dataset, Dataset as HGDataset, DatasetDict
import ipdb
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from toolz import compose as cmp
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from ..configuration_tools import get_output_dir


logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(
        self, data: HGDataset, keys: tuple[str] = ("input_ids", "attention_mask")
    ):
        self._data = data
        self._keys = keys

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {k: torch.tensor(v) for k, v in self._data[idx].items() if k in self._keys}

    def __len__(self) -> int:
        return len(self._data)


def seperate_except_brackets(text: str) -> tuple[str]:
    """Seperates all characters in a text except brackets."""
    seperated: list[str] = []
    j: int = 0
    while j < len(text):
        if text[j] != "[":
            seperated.append(text[j])
            j += 1
        else:
            i = j
            while text[j] != "]":
                j += 1
            seperated.append(text[i : j + 1])
            j += 1
    return tuple(seperated)


# tokenize_with_whitespace: Callable = cmp(" ".join, seperate_except_brackets)


def tokenize_with_whitespace_dict(
    example: Mapping, key: str, tokenizer: PreTrainedTokenizerFast
) -> dict[str, str]:
    return {
        key
        + "_spaced": " ".join(
            (tokenizer.bos_token,)
            + seperate_except_brackets(example[key])
            + (tokenizer.eos_token,),
        )
    }


def tokenize_dict_single(
    example: Mapping, tokenizer: PreTrainedTokenizerFast, key: str
) -> dict[str, str]:
    return tokenizer(example[key]).convert_to_tensors("pt", prepend_batch_axis=False)


def get_smiles_data(datasets: DatasetDict, tokenizercls: type[PreTrainedTokenizerFast]):
    """
    Returns a list of SMILES strings for the ZINC data set.
    """
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    text_tp = tuple(map(seperate_except_brackets, datasets["train"]["smiles"]))
    text: tuple[str] = tuple(map(" ".join, text_tp))

    tokenizer = Tokenizer(models.WordLevel())
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk_trainer = trainers.WordLevelTrainer(
        special_tokens=[pad_token, bos_token, eos_token]
    )
    tokenizer.train_from_iterator(text, trainer=tk_trainer)
    tokenizer_fp = osp.join(get_output_dir(), "smiles_tokenizer.json")
    # os.makedirs(osp.dirname(tokenizer_fp))
    tokenizer.save(tokenizer_fp)
    pretrainedtokenizer: tokenizercls = tokenizercls(
        tokenizer_file=tokenizer_fp,
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
        add_bos_token=True,
        add_eos_token=True,
        model_max_length=max(map(len, text_tp)),
        truncation_side="right",
    )
    datasets = datasets.map(
        prt(tokenize_with_whitespace_dict, key="smiles", tokenizer=pretrainedtokenizer)
    )
    _pretrainedtokenizer = prt(
        pretrainedtokenizer,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    datasets = datasets.map(
        prt(tokenize_dict_single, tokenizer=_pretrainedtokenizer, key="smiles_spaced")
        # batched=True,
        # batch_size=1000,
    )
    # TODO: bos and eos tokens are not added to the dataset!
    datasets = datasets.with_format(
        type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True
    )
    # datasets = {s: CustomDataset(ds) for s, ds in datasets.items()}
    for i in range(3):
        logger.info(f"datasets_dict['train'][{i}]: {datasets['train'][i]}")

    return datasets, pretrainedtokenizer
