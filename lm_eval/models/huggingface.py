import math
import torch
import torch.nn.functional as F
import transformers
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from lm_eval.api import utils
from lm_eval.api.model import TokenLM, TokenSequence


class HuggingFaceAutoLM(TokenLM):

    AUTO_MODEL_CLASS: transformers.AutoModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[int] = 1,
        user_defined_max_generation_length: Optional[int] = 256,
        max_length: Optional[int] = None,
        device: Optional[Union[int, str]] = "cuda",
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.

        :param use_accelerate:
            If True, uses the `accelerate` library to load a large model across
            multiple devices.
        :param max_memory_per_gpu: Optional[Union[int, str]]
            The maximum memory available for each GPU in bytes as `int` or in
            the format f"{significand}{unit_symbol}" where {unit_symbol} is
            any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
            the "Parameters for big model inference" section of the following docs:
            https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/model#large-model-loading
        :param max_cpu_memory: Optional[Union[int, str]]
            The maximum available CPU RAM in bytes as `int` or in the format
            f"{significand}{unit_symbol}" where {unit_symbol} is any of
            ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
            "Parameters for big model inference" section of the following docs:
            https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/model#large-model-loading
        :param offload_folder: Optional[str]
            The folder to offload weights into if `device_map` contains any "disk" value.
        :param dtype: Optional[Union[str, torch.dtype]]
            Converts the model weights to `dtype`, if specified. Strings get
            converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
            Use `dtype="auto"` to derive the type from the model’s weights.
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        self._batch_size = batch_size  # TODO: Adaptive batch size
        self._user_defined_max_generation_length = user_defined_max_generation_length
        self._max_length = max_length
        self._config = transformers.AutoConfig.from_pretrained(pretrained)
        
        self.tokenizer = self._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
        )
        self.tokenizer.model_max_length = self.max_length

        self.model = self._create_auto_model(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
        )
        self.model.eval()
        torch.set_grad_enabled(False)

    def _create_auto_model(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
    ) -> transformers.AutoModel:
        """Returns a pre-trained pytorch model from a pre-trained model configuration."""
        model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        return model

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def user_defined_max_generation_length(self) -> int:
        return self._user_defined_max_generation_length

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.tokenizer(string, add_special_tokens=False)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings, padding=True, add_special_tokens=False, return_tensors="pt"
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, 
        context_inputs,
        stop_sequences,
        max_generation_length,
    ) -> List[str]:
        assert isinstance(max_generation_length, torch.Tensor)
        assert isinstance(stop_sequences, torch.Tensor) or stop_sequences is None
        responses = self._model_generate(
            inputs=context_inputs,
            max_tokens=max_generation_length,
            stop=stop_sequences,
        )
        return responses.contiguous()


class AutoCausalLM(HuggingFaceAutoLM):
    """Causal language modeling.
    You can find a set of supported models in the HF documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForCausalLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        revision: str,
        subfolder: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        tokenizer = super()._create_auto_tokenizer(
            pretrained=pretrained,
            revision=revision,
            subfolder=subfolder,
            tokenizer=tokenizer,
        )
        return tokenizer

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(**inputs)["logits"]

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> TokenSequence:
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop)
        generations = self.model.generate(
            **inputs,
            # GPT style models require the `generate` `max_length` arg to include the
            # context length, so we instead set `max_new_tokens` which is the number
            # of new tokens to generate, excluding the current number of tokens.
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            synced_gpus=True
        )
        return utils.select_continuation_from_batch_left_padding(
            generations, max_context_size=inputs["input_ids"].size(1)
        )


class AutoSeq2SeqLM(HuggingFaceAutoLM):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        TODO: Currently only works for relative position encoded Seq2Seq models.
        """
        if self._max_length is not None:
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    def loglikelihood(
        self,
        context_inputs,
        target_inputs,
        decoder_inputs,
    ) -> List[Tuple[float, bool]]:
        return self._loglikelihood_tokens(context_inputs, target_inputs, decoder_inputs)

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        loglikelihoods = []
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            contexts, conts = utils.split_and_pad_windows(
                rolling_token_windows,
                pad_token_id=self.eot_token_id,
                max_seq_len=self.max_length,
            )
            # Manually create BatchEncoding tensors with attention masks as
            # expected by `self._model_call` in `self._loglikelihood_tokens`.
            contexts_enc = torch.Tensor(contexts).long()
            contexts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": contexts_enc,
                    "attention_mask": (contexts_enc != self.eot_token_id).long(),
                }
            )
            conts_enc = torch.Tensor(conts).long()
            conts_enc = transformers.tokenization_utils_base.BatchEncoding(
                {
                    "input_ids": conts_enc,
                    "attention_mask": (conts_enc != self.eot_token_id).long(),
                }
            )
            # TODO: Extract out this call so it only gets called once and also
            # somehow figure out partial caching for.
            rolling_token_windows_request = [
                ((contexts, conts), contexts_enc, conts_enc)
            ]
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows_request, disable_tqdm=True
            )
            string_nll = [x[0] for x in string_nll]  # discard is_greedy
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self,
        context_inputs, # :List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        target_inputs,
        decoder_inputs,
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        """
        TODO:
        - Add check to see if the tokenizer applies a start of seq token.
        """
        context_tokens = context_inputs
        targets_tokens = target_inputs
        outputs = self._model_call(inputs=context_tokens, labels=targets_tokens)
        log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

        output_iterator = zip(
            log_softmaxes,
            targets_tokens["input_ids"],
            targets_tokens["attention_mask"],
        )
        logprobs_results = []
        exact_match_results = []
        for log_softmax, target_tokens, target_mask in output_iterator:
            length = target_mask.sum()
            log_softmax = log_softmax[:length]
            target_tokens = target_tokens[:length]
            greedy_tokens = log_softmax.argmax(dim=-1)
            exact_match = (greedy_tokens == target_tokens).all().unsqueeze(0).to(torch.bool)
            target_logits = torch.gather(
                log_softmax, 1, target_tokens.unsqueeze(-1)
            ).squeeze(-1)
            logprobs_results.append(target_logits.sum().unsqueeze(0))
            exact_match_results.append(exact_match)
            # if cache_key is not None:
            #     self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return torch.cat(logprobs_results, dim=0), torch.cat(exact_match_results, dim=0)

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(**inputs, labels=labels["input_ids"])

    def _model_generate(
        self, inputs: TokenSequence, max_tokens: int, stop: Optional[List[str]] = None
    ) -> Union[TokenSequence, List[str]]:
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop)
        generations = self.model.generate(
            **inputs,
            max_length=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence ids."""

    def __init__(self, sequence_ids: List[int], tokenizer: transformers.PreTrainedTokenizer):
        self.sequence_ids = sequence_ids
        self.sequence_ids_len = len(self.sequence_ids) + 1
        self.sequence = tokenizer.decode(sequence_ids)
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_token_id = input_ids[0, -self.sequence_ids_len :]
        last_tokens = self.tokenizer.decode(last_token_id)
        is_stopped = self.sequence in last_tokens
        return is_stopped


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences_ids: Optional[List[int]] = None,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(sequence_ids, tokenizer)
                for sequence_ids in stop_sequences_ids
            ],
        ]
    )
