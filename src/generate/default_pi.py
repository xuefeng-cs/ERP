# adapted from https://github.com/shunzh/Code-AI-Tree-Search/blob/main/generate/default_pi.py

from collections.abc import Callable, Iterable
import logging
from functools import partial as prt
from operator import getitem
import time
import warnings
from abc import abstractmethod

import torch
import numpy as np
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedModel,
    GenerationConfig,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import ipdb

from .transformer_cache_utils import GPTTopKCache, GPTSeqCache
from ..nn.peripheral import SpecialTokenizer
from ..functional import identity_mult

logger = logging.getLogger(__name__)


def plain_scores(*, scores, **kwargs):
    return scores


class PolicyHeuristic:
    @abstractmethod
    def get_predict_sequence(self, state, horizon=None):
        pass

    @abstractmethod
    def get_top_k_predict(self, state):
        pass

    def clean_up(self, new_state):
        # implement this if need to do anything after each token is generated
        pass


class MlcPolicyHeuristic(PolicyHeuristic):
    def __init__(
        self,
        tokenizer: SpecialTokenizer,
        # tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        generation_mode: str,
        top_k_expansion: int,
        top_p_expansion: float,
        # top_k_simulation: int,
        # top_p_simulation: float,
        num_beams: int,
        test_all_beams,
        horizon,
        device,
        env,
        generation_config: dict,
        use_reward_estimate_cache: bool,
        top_k_uniform: bool = False,
        uniform_expansion: bool = False,
        value_model: Callable | None = None,
        new_token_num=None,
        # use_seq_cache=False,  # disable all caching by default
        # top_k_cache_steps=0,
        debug=False,
        sampling_temperature=1.0,
    ):
        self.sample_times = 0
        self.num_sample_tokens = 0
        self.time_stamps = []  # time stamp when a new sample is generated

        self.tokenizer = tokenizer

        self.top_k_expansion = top_k_expansion  # TODO: remove
        self.top_p_expansion = top_p_expansion

        self.horizon = horizon
        self.generation_mode = generation_mode
        self.num_beams = num_beams
        self.test_all_beams = test_all_beams
        # self.top_k_simulation = top_k_simulation
        # self.top_p_simulation = top_p_simulation
        self.sampling_temperature = sampling_temperature
        if self.sampling_temperature != 1.0:
            assert self.generation_mode == "sample"

        self.device = device
        self.env = env

        self.generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # suppress_tokens=[tokenizer.pad_token_id],
            **generation_config,
        )
        # self.use_seq_cache = use_seq_cache
        # self.top_k_cache_steps = top_k_cache_steps

        self.new_token_num = new_token_num

        self.debug = debug

        self.model: PreTrainedModel = model
        self.value_model = value_model
        self.top_k_uniform = top_k_uniform
        self.uniform_expansion = uniform_expansion
        self.use_value = self.value_model is not None

        # if self.generation_mode == "sample" and self.use_seq_cache:
        #     warnings.warn("Cannot use sequence caching in sample mode, disabling it.")
        #     self.use_seq_cache = False

        if self.use_value and self.new_token_num is None:
            warnings.warn(
                "Using a value function but not setting a shorter planning horizon (args.new_token_num)."
                "Why using a value function?"
            )

        # TODO : fix this perhaps
        # if generation_mode == "sample" and use_reward_estimate_cache:
        #     raise ValueError("Ought not to use reward estimate cache in sample mode")

        self.model.to(device)
        if self.use_value:
            self.value_model.to(device)

        if device == torch.device("cuda"):
            if hasattr(self.model, "parallelize"):
                self.model.parallelize()
            if self.value_model is not None and hasattr(self.model, "parallelize"):
                self.value_model.parallelize()

        if use_reward_estimate_cache:
            self.reward_estimate_cache: dict[tuple[int], float] = {}
        else:
            raise NotImplementedError("Without reward estimate cache no implemented")

        # hard-coded k for caching
        # self.top_k_cache = GPTTopKCache(
        #     self.top_k_simulation, cache_steps=top_k_cache_steps, tokenizer=tokenizer
        # )
        # self.seq_cache = GPTSeqCache()
        self.prompt_key_values = None

        self.terminal_token = self.env.terminal_token

        if top_k_expansion > 0:
            self.top_k_warper = TopKLogitsWarper(top_k=top_k_expansion)
        else:
            self.top_k_warper = plain_scores

        if top_p_expansion > 0:
            self.top_p_warper = TopPLogitsWarper(top_p=top_p_expansion)
        else:
            self.top_p_warper = plain_scores

    def get_short_horizon_sequence(self, state):
        """
        Returns:
            predicted sequence, but only with up to self.new_token_num new tokens.
            This uses self.get_predict_sequence.
        """
        # add length of prompt and existing program
        horizon = len(state) + self.new_token_num
        # don't exceed the length of Transformer input
        horizon = min(horizon, self.horizon)

        return self.get_predict_sequence(state, horizon=horizon)

    def get_reward_estimate(self, state: Iterable[int]) -> float:
        state = tuple(state)
        reward_estimate = self.reward_estimate_cache.get(state)
        if reward_estimate is not None:
            logger.debug(
                f"reward_estimate_cache hit with state : {state} ; reward_estimate : {reward_estimate}"
            )
            return reward_estimate

        reward_estimate = self.env.get_reward(self.get_predict_sequence(state))
        self.reward_estimate_cache[state] = reward_estimate
        logger.debug(
            f"reward_estimate_cache miss with state : {state} ; reward_estimate : {reward_estimate}"
        )
        return reward_estimate

    def get_predict_sequence(
        self, state: Iterable[int], horizon=None, all_sequences=False
    ):
        """
        The main use being for the simulation/rollout part of MCTS.
        returning only one sequence.
        Args:
            horizon: return a new sequence with this extra length
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            # if self.use_seq_cache:
            #     output_ids = self.seq_cache.get(encoded_ids)
            #     if output_ids is not None:
            #         return output_ids

            if horizon is None:
                horizon = self.horizon

            start_time = time.time()

            model_output = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                # top_k=self.top_k_simulation,
                # top_p=self.top_p_simulation,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                do_sample=(self.generation_mode == "sample"),
                temperature=self.sampling_temperature,
                # early_stopping=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                max_length=horizon,
                use_cache=True,  # huggingface default cache is always enabled
            )

            # if self.top_k_cache_steps > 0:
            #     if hasattr(model_output, "beam_indices"):
            #         # beam search output
            #         self.top_k_cache.add(
            #             input_ids,
            #             model_output.sequences,
            #             model_output.scores,
            #             beam_indices=model_output.beam_indices,
            #         )
            #     else:
            #         self.top_k_cache.add(
            #             input_ids, model_output.sequences, model_output.scores
            #         )

            if self.debug:
                print("generate sequence time: " + str(time.time() - start_time))

            output_ids_list = model_output.sequences.tolist()
            if all_sequences:
                return output_ids_list

            if len(output_ids_list) > 1 and self.test_all_beams:
                # if got multiple output_ids using beam search, and going to test all beams (which takes more time)
                # pick the one that has the highest reward
                cand_rewards = [
                    self.env.get_reward(output_ids) for output_ids in output_ids_list
                ]
                output_ids = output_ids_list[np.argmax(cand_rewards)]
            else:
                output_ids = output_ids_list[0]

            # if self.use_seq_cache:
            #     self.seq_cache.add(encoded_ids, output_ids)

            self.sample_times += 1
            self.num_sample_tokens += len(output_ids)
            self.time_stamps.append(time.time())

            if self.debug:
                logger.debug("==== generated program ====")
                logger.debug(self.env.convert_state_to_program(output_ids))
                logger.debug("===========================")

            return output_ids

    def sample(
        self,
        state: Iterable[int],
        horizon: int,
        temperature: float,
        top_k,
        top_p,
        do_sample=True,
    ) -> tuple[int]:
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            if horizon is None:
                horizon = self.horizon
            model_output = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                top_k=top_k,
                top_p=top_p,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                do_sample=do_sample,
                # early_stopping=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                max_length=horizon,
                use_cache=True,  # huggingface default cache is always enabled
                temperature=temperature,
            )
            output_ids_list = model_output.sequences.tolist()

            if len(output_ids_list) > 1 and self.test_all_beams:
                # if got multiple output_ids using beam search, and going to test all beams (which takes more time)
                # pick the one that has the highest reward
                cand_rewards = [
                    self.env.get_reward(output_ids) for output_ids in output_ids_list
                ]
                output_ids = output_ids_list[np.argmax(cand_rewards)]
            else:
                output_ids = output_ids_list[0]

            self.sample_times += 1
            self.num_sample_tokens += len(output_ids)
            return output_ids

    def get_value(self, state):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            est_value = self.value_model(input_ids).logits.item()

            if self.debug:
                logger.debug(f"estimated value is {est_value}")

            return est_value

    def uniform_sample(state: Iterable[int], horizon: int) -> tuple[int]:
        raise NotImplementedError

    def get_top_k_predict(self, state: tuple[int]) -> tuple[tuple[int], tuple[float]]:
        """
        Returns:
            A list of k most likely tokens generate in state (descending in their scores)
            The probability of each action
        """
        if self.uniform_expansion:
            indices = torch.randperm(self.tokenizer.tokenizer.vocab_size)
            probs = [
                1 / self.tokenizer.tokenizer.vocab_size
            ] * self.tokenizer.tokenizer.vocab_size
            return indices.tolist(), probs

        if self.top_k_uniform:
            top_k_tokens = torch.randint(
                0, self.tokenizer.tokenizer.vocab_size, (self.top_k_expansion,)
            )
            top_k_scores = torch.ones(self.top_k_expansion) / self.top_k_expansion
            return top_k_tokens.tolist(), top_k_scores.tolist()

        # if self.top_k_cache_steps > 0:
        #     top_k_info = self.top_k_cache.get(state)
        #     if top_k_info is not None:
        #         logger.debug("top-k cache hit")
        #         return top_k_info

        # start_time = time.time()
        with torch.no_grad():
            model_output = self.model.generate(
                torch.LongTensor(state).unsqueeze(0).to(self.device),
                num_beams=1,
                max_new_tokens=1,
                return_dict_in_generate=True,
                do_sample=False,
                output_scores=True,
                use_cache=True,
            )
            logits = model_output.scores[0]
            logits = self.top_k_warper(input_ids=None, scores=logits)
            logits = self.top_p_warper(input_ids=None, scores=logits)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            indices = (probs > 0).nonzero().squeeze(1)
            indices = tuple(indices.tolist())
            indices_descending = sorted(indices, key=prt(getitem, probs), reverse=True)
            probs_descending = tuple(probs[indices_descending].tolist())

        return indices_descending, probs_descending

    def clean_up(self, new_state):
        pass
        # if self.use_seq_cache:
        #     # clear hashed sequences that are not consistent with new_state
        #     self.seq_cache.clear(new_state)

        # if self.top_k_cache_steps > 0:
        #     self.top_k_cache.clear(new_state)
