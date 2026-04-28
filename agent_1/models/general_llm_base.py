# A

from dataclasses import dataclass
from logging import Logger
from typing import Iterable, List

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass(frozen=True)
class Issue:
    message: str
    success: bool = False


class GeneralLLMBase:
    def __init__(self, model_id: str, logger: Logger):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger

        load_dotenv()

        envvar_issues = self.verify_mandatory_envvars()

        self.log_issues(issues=envvar_issues)
        if self.is_failure_issues(issues=envvar_issues):
            message = "Previous errors happened. See above messages"
            self.logger.error(message)
            raise SystemExit(message)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)

    def verify_mandatory_envvars(self) -> List[Issue]:
        return [
            Issue(
                message=f"No value for mandatory environment variable [{mandatory_envvar}] found",
                success=False,
            )
            for mandatory_envvar in ["HF_HOME"]
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

    def perform(self, prompt: str, **kwargs) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        inputs = tokenizer(self.prepare_propmpt(prompt), return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generation_params = {**dict(num_beams=4), **kwargs}

        outputs = model.generate(**inputs, **generation_params)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
