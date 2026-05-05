import os
from typing import List

from models.classifier_model_base import KNOWN_MANDATORY_ENVVARS, Issue


def verify_mandatory_envvars(mandatory_envvars: List[str]) -> List[Issue]:
    return [
        Issue(
            message=f"No value for mandatory environment variable [{mandatory_envvar}] found",
            success=False,
        )
        for mandatory_envvar in mandatory_envvars
        if not os.getenv(mandatory_envvar)
    ]


def verify_known_mandatory_envvars() -> List[Issue]:
    return verify_mandatory_envvars(mandatory_envvars=KNOWN_MANDATORY_ENVVARS)
