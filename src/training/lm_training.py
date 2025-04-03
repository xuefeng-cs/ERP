from functools import partial as prt
import logging

import ipdb
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
from transformers.integrations import TensorBoardCallback

logger = logging.getLogger(__name__)


class LogLossCallback(TrainerCallback):
    def __init__(self, cnfg: DictConfig):
        self.cnfg = cnfg

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        **kwargs,
    ):
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        **kwargs,
    ):
        logger.info(f"epoch: {state.epoch} | {state.log_history[-1]}")
        return control

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        **kwargs,
    ):
        logger.info("evaluation done")
        logger.info(f"epoch: {state.epoch} | {state.log_history[-1]}")
        # evaluate_on_txt_file(
        # cnfg=self.cnfg, model=model, tokenizer=tokenizer, device=args.device
        # )
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        # train_dataloader: DataLoader | None,
        # eval_dataloader: DataLoader | None,
        # metrics: dict[str, float],
        # logs: dict[str, float],
        **kwargs,
    ):
        logger.info("training ended")
        return control


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
