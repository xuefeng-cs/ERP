from collections.abc import Iterable
import logging
import os.path as osp

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    GenerationConfig,
    GPTNeoModel,
    GPTNeoForCausalLM,
)
from ..configuration_tools import get_output_dir

logger = logging.getLogger(__name__)


# TODO: not ideal to have custom class, but perhaps this is faster for developing for now.
class SpecialTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        skip_special_tokens_decoding: bool,  # omitting `"&"`, `"\n"`, and `"<pad>"`
        join_whitespaces_decoding: bool,
        prompt_string: str | None,
    ):
        self.tokenizer = tokenizer
        self.skip_special_tokens_decoding = skip_special_tokens_decoding
        self.join_whitespaces_decoding = join_whitespaces_decoding
        self.prompt_string = prompt_string
        tokenizer.save_pretrained(osp.join(get_output_dir(), "tokenizer"))

    def decode(self, *args, **kwargs) -> str:
        _string: str = self.tokenizer.decode(
            *args, **kwargs, skip_special_tokens=self.skip_special_tokens_decoding
        )
        if self.join_whitespaces_decoding:
            _string = _string.replace(" ", "")
        if self.prompt_string:
            _string = _string.replace(self.prompt_string, "")
        return _string

    def batch_decode(self, *args, **kwargs) -> list[str]:
        _liststring: list[str] = self.tokenizer.batch_decode(
            *args, **kwargs, skip_special_tokens=self.skip_special_tokens_decoding
        )
        if self.join_whitespaces_decoding:
            _liststring = [s.replace(" ", "") for s in _liststring]
        if self.prompt_string:
            _liststring = [s.replace(self.prompt_string, "") for s in _liststring]
        return _liststring

    def encode(self, *args, **kwargs):
        return self.tokenizer.encode(*args, **kwargs)

    @property
    def bos_token(self) -> str:
        return self.tokenizer.bos_token

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token(self) -> str:
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


def get_tokenizer_and_nnmodel(
    model_dir: str,
    device: torch.device,
) -> tuple[SpecialTokenizer, PreTrainedModel]:
    _tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if "jarod0411/zinc10M_gpt2_SMILES_bpe_combined" in model_dir:
        prompt_string = "<L>"
    else:
        prompt_string = None

    logger.info(f"prompt_string: {prompt_string}")
    tokenizer = SpecialTokenizer(
        _tokenizer,
        skip_special_tokens_decoding=True,
        join_whitespaces_decoding=True,
        prompt_string=prompt_string,
    )
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    model.eval()
    # TODO: I do not understand why get_model_by_name use eos in place of pad for causal lm

    logger.info(f"Loaded tokenizer of {type(_tokenizer).__name__}")
    logger.info(f"Loaded model of {type(model).__name__}")
    logger.debug(f"tokenizer: {tokenizer}")
    logger.debug(f"model: {model}")
    logger.debug(f"model.config: {model.config}")

    return tokenizer, model, prompt_string
