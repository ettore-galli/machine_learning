from logging import Logger

from models.classifier_nli_base import ClassifierNLIBase


class ClassifierNLILLM(ClassifierNLIBase):
    def __init__(self, logger: Logger):
        # super().__init__("facebook/bart-large-mnli", logger)
        super().__init__("facebook/bart-small", logger)
