import logging

from omegaconf import DictConfig, OmegaConf
from transformers import (
    GPTNeoConfig,
    LlamaConfig,
    LlamaForCausalLM,
    GPTNeoForCausalLM,
)

logger = logging.getLogger(__name__)

lmconfig_clss: dict[str, type] = {
    "gptneo": GPTNeoConfig,
    "llama": LlamaConfig,
    "llama2": LlamaConfig,
}
lmcausal_clss: dict[str, type] = {
    "gptneo": GPTNeoForCausalLM,
    "llama": LlamaForCausalLM,
    "llama2": LlamaForCausalLM,
}
