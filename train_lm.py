import logging
import os.path as osp

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    TrainerState,
    TrainerControl,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    default_data_collator,
)
import ipdb

from src.configuration_tools import get_output_dir
from src.data.tools import dataset_factory
from src.nn.lms import lmconfig_clss, lmcausal_clss
from src.training.lm_training import LogLossCallback, CustomTrainer

logger = logging.getLogger(__name__)


def main(cnfgr: DictConfig = None):
    if cnfgr is None:
        cnfgr = OmegaConf.load("cnfgr/lm.yaml")
    logger.info(f"cnfgr:\n{cnfgr}")

    tokenizer: PreTrainedTokenizerFast
    datasets, tokenizer = dataset_factory[cnfgr.data.name](cnfgr)
    tokenizer.save_pretrained(osp.join(get_output_dir(), "tokenizer"))

    cnfgr_lm: dict = OmegaConf.to_container(cnfgr.lm) | {
        "vocab_size": tokenizer._tokenizer.get_vocab_size(),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "max_position_embeddings": tokenizer.model_max_length,
        "max_length": tokenizer.model_max_length,
    }
    lm_configuration: PretrainedConfig = lmconfig_clss[cnfgr.lm_name](**cnfgr_lm)
    lm: PreTrainedModel = lmcausal_clss[cnfgr.lm_name](lm_configuration)
    logger.info(f"lm configuration: {lm_configuration}")
    logger.info(f"lm:\n{lm}")

    trainerarguments = Seq2SeqTrainingArguments(
        output_dir=get_output_dir(), **OmegaConf.to_container(cnfgr.trainer)
    )
    trainer = CustomTrainer(
        model=lm,
        args=trainerarguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        callbacks=[LogLossCallback(cnfgr)],
        # data_collator=default_data_collator,
    )
    logger.info(f"trainerarguments:\n{trainerarguments}")
    logger.info(f"trainer:\n{trainer}")
    trainer.train(**OmegaConf.to_container(cnfgr.trainercall))


hydra_app = hydra.main(version_base=None, config_path="cnfgr", config_name="lm")(main)
if __name__ == "__main__":
    hydra_app()
