from logging import Logger
from typing import Any, Dict, Iterable, List

import torch

from models.classifier_model_base import MANDATORY_ENVVARS, Issue
from models.classifier_nli_interface import (
    do_classifier_perform,
    instantiate_classifier_objects,
)


class ClassifierNLIBase:
    def __init__(self, model_id: str, logger: Logger):
        self.model_id = model_id
        self.device = self.infer_device()
        self.logger = logger
        self.mandatory_envvars: List[str] = MANDATORY_ENVVARS

        envvar_issues = self.verify_mandatory_envvars()

        self.log_issues(issues=envvar_issues)
        if self.is_failure_issues(issues=envvar_issues):
            message = "Previous errors happened. See above messages"
            self.logger.error(message)
            raise SystemExit(message)

    def infer_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
        return "cpu"

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

    def do_perform(self, prompt: str, **kwargs: Any) -> Dict[str, float]:
        tokenizer, model = instantiate_classifier_objects(
            model_id=self.model_id, device=self.device
        )

        hypothesis = "Given sentence requires to perform an arithmetic operation"

        result = do_classifier_perform(
            tokenizer=tokenizer,
            model=model,
            device=self.device,
            text=prompt,
            text_pair=hypothesis,
            **kwargs,
        )

        return result

    def perform(self, prompt: str, **kwargs: Any) -> Dict[str, float]:
        return self.do_perform(
            prompt,
            **{
                **dict(
                    max_new_tokens=512,
                    repetition_penalty=1.5,
                    do_sample=True,
                    top_k=50,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    truncation_strategy="only_first",
                ),
                **kwargs,
            },
        )
