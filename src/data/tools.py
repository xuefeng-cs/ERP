import logging

from datasets import load_dataset_builder, load_dataset, Dataset, DatasetDict
from omegaconf import DictConfig
from transformers import GPT2TokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerFast

from .make_smile import get_datasets_zinc250k, ZincSmilesDataset
from .smiles import get_smiles_data

logger = logging.getLogger(__name__)
tokenizer_clss: dict[str, type] = {
    "gptneo": GPT2TokenizerFast,
    "gpt2": GPT2TokenizerFast,
    "llama": LlamaTokenizerFast,
    "llama2": LlamaTokenizerFast,
}


def get_datasets_zinc250k_from_config(
    cnfgr: DictConfig,
) -> tuple[dict[str, ZincSmilesDataset], PreTrainedTokenizerFast]:
    dataset_train, dataset_val = get_datasets_zinc250k(
        rawfilepath=cnfgr.data.rawfilepath,
        tokenizercls=tokenizer_clss[cnfgr.data.tokenizer],
        seed=cnfgr.trainer.seed,
        val_ratio=cnfgr.data.val_ratio,
    )
    return {"train": dataset_train, "validation": dataset_val}, dataset_train.tokenizer


def get_smiles_from_config(
    cnfgr: DictConfig,
) -> tuple[DatasetDict, PreTrainedTokenizerFast]:
    """ """
    datasets: dict = load_dataset(
        cnfgr.data.name, cache_dir=f"data/datasets/{cnfgr.data.name}"
    )
    datasets, pretrainedtokenizer = get_smiles_data(
        datasets, tokenizer_clss[cnfgr.lm_name]
    )
    return datasets, pretrainedtokenizer


dataset_factory: dict[str, type] = {
    "zinc250k": get_datasets_zinc250k_from_config,
    "jarod0411/zinc10M": get_smiles_from_config,
}
