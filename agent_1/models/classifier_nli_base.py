import os
from logging import Logger
from typing import Iterable, List

from models.classifier_model_base import (
    MANDATORY_ENVVARS,
    ClassifierResultType,
    Issue,
    KeywordArgsType,
    KeywordArgsValueType,
)
from models.classifier_nli_interface import (
    get_model_performer,
)


class ClassifierNLIBase:
    def __init__(self, model_id: str, logger: Logger):
        self.model_id = model_id
        self.logger = logger
        self.mandatory_envvars: List[str] = MANDATORY_ENVVARS

        envvar_issues = self.verify_mandatory_envvars()

        self.log_issues(issues=envvar_issues)

        if self.is_failure_issues(issues=envvar_issues):
            message = "Previous errors happened. See above messages"
            raise SystemExit(message)

        self.model_performer = get_model_performer(model_id=model_id)

    def verify_mandatory_envvars(self) -> List[Issue]:
        return [
            Issue(
                message=f"No value for mandatory environment variable [{mandatory_envvar}] found",
                success=False,
            )
            for mandatory_envvar in self.mandatory_envvars
            if not os.getenv(mandatory_envvar)
        ]

    def log_issues(self, issues: Iterable[Issue]) -> None:
        for issue in issues:
            if issue.success:
                self.logger.warning(issue.message)
            else:
                self.logger.error(issue.message)

    def is_failure_issues(self, issues: Iterable[Issue]) -> bool:
        return any(issue.success is False for issue in issues)

    def do_perform(
        self, prompt: str, hypothesis: str, **kwargs: KeywordArgsValueType
    ) -> ClassifierResultType:

        return self.model_performer(
            text=prompt,
            text_pair=hypothesis,
            **kwargs,
        )

    def perform(
        self, prompt: str, hypothesis: str, **kwargs: KeywordArgsValueType
    ) -> ClassifierResultType:
        options: KeywordArgsType = {
            **dict(
                max_length=512,
                truncation_strategy="only_first",
            ),
            **kwargs,
        }
        return self.do_perform(
            prompt,
            hypothesis,
            **options,
        )
