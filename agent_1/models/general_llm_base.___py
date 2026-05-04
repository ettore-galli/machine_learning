# pyright: reportUnknownMemberType=none
# pyright: reportUnknownArgumentType=none
# pyright: reportPrivateImportUsage=none

from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, Iterable, List, cast

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass(frozen=True)
class Issue:
    message: str
    success: bool = False


def _unsafe_generate(model: Any, *args: Any, **kwargs: Any) -> Any:
    return model.generate(*args, **kwargs)


class GeneralLLMBase:
    def __init__(self, model_id: str, logger: Logger):
        self.model_id = model_id
        self.device = self.infer_device()
        self.logger = logger
        self.mandatory_envvars: List[str] = [
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "TRANSFORMERS_CACHE",
            "DIFFUSERS_CACHE",
        ]

        envvar_issues = self.verify_mandatory_envvars()

        self.log_issues(issues=envvar_issues)
        if self.is_failure_issues(issues=envvar_issues):
            message = "Previous errors happened. See above messages"
            self.logger.error(message)
            raise SystemExit(message)

    def infer_device(self) -> str:
        if torch.backends.mps.is_available():
            return torch.backends.mps.get_name()
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
        return "cpu"

    def prepare_input_for_device(
        self, device: str, tokenized_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return (
            {k: v.to("cuda") for k, v in tokenized_inputs.items()}
            if device == "cuda"
            else tokenized_inputs
        )

    def verify_mandatory_envvars(self) -> List[Issue]:
        return [
            Issue(
                message=f"No value for mandatory environment variable [{mandatory_envvar}] found",
                success=False,
            )
            for mandatory_envvar in self.mandatory_envvars
            if not mandatory_envvar
        ]

    def log_issues(self, issues: Iterable[Issue]) -> None:
        for issue in issues:
            if issue.success:
                self.logger.warning(issue.message)
            else:
                self.logger.error(issue.message)

    def is_failure_issues(self, issues: Iterable[Issue]) -> bool:
        return any(issue.success is False for issue in issues)

    def prepare_propmpt(self, prompt: str) -> str:
        return prompt.strip() + "</s>"

    def perform(self, prompt: str, **kwargs: Any) -> str:
        tokenizer: PreTrainedTokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(self.model_id),
        )
        model: PreTrainedModel = cast(
            PreTrainedModel,
            AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            ),
        )

        encoded: BatchEncoding = tokenizer(
            self.prepare_propmpt(prompt),
            return_tensors="pt",
        )

        inputs: torch.Tensor = encoded["input_ids"]

        generation_params: Any = {**dict(num_beams=4), **kwargs}

        raw_outputs = _unsafe_generate(model, inputs, **generation_params)

        outputs: torch.Tensor = cast(torch.Tensor, raw_outputs)

        decoded: str = cast(
            str,
            tokenizer.decode(outputs[0], skip_special_tokens=True),
        )

        return decoded
