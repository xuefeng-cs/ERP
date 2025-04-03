from collections.abc import Iterable
import csv
from itertools import chain
from functools import partial as prt
from logging import getLogger
import os
import os.path as osp

from jaxtyping import Float, Int
from toolz import compose as cmp
from toolz.curried import map
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, BatchEncoding

from ..configuration_tools import get_output_dir

tokenizer_fp = osp.join(get_output_dir(), "smiles_tokenizer.json")
os.makedirs(osp.dirname(tokenizer_fp), exist_ok=True)

logger = getLogger(__name__)


def tokenize_one(tokenizer: PreTrainedTokenizerFast, text: str):
    batchencoding: BatchEncoding = tokenizer(
        text,
        is_split_into_words=True,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )
    return batchencoding.convert_to_tensors("pt", prepend_batch_axis=False)


class ZincSmilesDataset(Dataset):
    def __init__(
        self,
        data: Iterable[Iterable[str]],
        tokenizer: PreTrainedTokenizerFast,
        max_len: int,
    ):
        assert max_len == tokenizer.model_max_length
        assert tokenizer.eos_token_id is not None
        _tokenize_one = prt(tokenize_one, tokenizer)
        self.data: tuple[dict[str, Float[Tensor, "s"]]]
        self.data = cmp(tuple, map(_tokenize_one))(data)
        self.tokenizer: PreTrainedTokenizerFast = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def max_len(self) -> int:
        return self.tokenizer.model_max_length

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


def get_datasets_zinc250k(
    rawfilepath: str,
    tokenizercls: type[PreTrainedTokenizerBase],
    seed: int,
    val_ratio: float,
) -> tuple[ZincSmilesDataset]:
    pad_token = "<pad>"
    bos_token = "&"
    eos_token = "\n"
    zinc_processed = zinc_data_with_bracket_original(rawfilepath=rawfilepath)
    data = zinc_processed_with_bracket(zinc_processed)

    max_len = max(map(len, data))
    vocab = set(chain.from_iterable(data))
    logger.info(f"vocab size: {len(vocab)}")
    tokenizer = Tokenizer(models.WordLevel())
    tk_trainer = trainers.WordLevelTrainer(special_tokens=[pad_token])
    tokenizer.train_from_iterator(data, trainer=tk_trainer)
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(tokenizer_fp)
    pretrainedtokenizer = tokenizercls(
        tokenizer_file=tokenizer_fp,
        pad_token=pad_token,
        bos_token=bos_token,
        eos_token=eos_token,
        add_prefix_space=True,
        max_len=max_len,
    )
    # pretrainedtokenizer.save_pretrained(dir_generated_config)
    gnrtr = torch.Generator().manual_seed(seed)
    data_split = prt(random_split, lengths=(1 - val_ratio, val_ratio), generator=gnrtr)
    dts_cnstrtr = prt(ZincSmilesDataset, tokenizer=pretrainedtokenizer, max_len=max_len)
    return cmp(tuple, map(dts_cnstrtr), map(tuple), data_split)(data)


def zinc_data_with_bracket_original(rawfilepath: str):
    # adapted from https://github.com/tsudalab/ChemTS
    sen_space = []
    with open(rawfilepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            sen_space.append(row)

    word1 = sen_space[0]

    zinc_processed = []
    for i in range(len(sen_space)):
        word1 = sen_space[i]
        zinc_processed.append(word1[0])
    return zinc_processed


def zinc_processed_with_bracket(sen_space):
    # adapted from https://github.com/tsudalab/ChemTS
    all_smile = []
    length = []
    end = "\n"
    element_table = [
        "C",
        "N",
        "B",
        "O",
        "P",
        "S",
        "F",
        "Cl",
        "Br",
        "I",
        "(",
        ")",
        "=",
        "#",
    ]

    for i in range(len(sen_space)):
        word_space = sen_space[i]
        word = []
        j = 0
        while j < len(word_space):
            word_space1 = []
            if word_space[j] == "[":
                word_space1.append(word_space[j])
                j = j + 1
                while word_space[j] != "]":
                    word_space1.append(word_space[j])
                    j = j + 1
                word_space1.append(word_space[j])
                word_space2 = "".join(word_space1)
                word.append(word_space2)
                j = j + 1
            else:
                word_space1.append(word_space[j])

                if j + 1 < len(word_space):
                    word_space1.append(word_space[j + 1])
                    word_space2 = "".join(word_space1)
                else:
                    word_space1.insert(0, word_space[j - 1])
                    word_space2 = "".join(word_space1)

                if word_space2 not in element_table:
                    word.append(word_space[j])
                    j = j + 1
                else:
                    word.append(word_space2)
                    j = j + 2

        word.append(end)
        word.insert(0, "&")
        len1 = len(word)
        length.append(len1)
        all_smile.append(tuple(word))
    return all_smile


if __name__ == "__main__":
    import ipdb
    from transformers import GPT2TokenizerFast

    dataset_train, dataset_val = get_datasets_zinc250k(
        rawfilepath="data/chemts/raw/250k_rndm_zinc_drugs_clean.smi",
        tokenizercls=GPT2TokenizerFast,
        seed=42,
        val_ratio=0.1,
    )
    print(type(dataset_train))
    print(len(dataset_train))
    print(dataset_train[42])
