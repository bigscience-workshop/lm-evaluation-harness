"""
MultiEURLEX
"""
from typing import Dict, List, Optional
from lm_eval.api.task import PromptSourceTask

_CITATION = """
"""

_LANGUAGES = [
    "en",
    "da",
    "de",
    "nl",
    "sv",
    "bg",
    "cs",
    "hr",
    "pl",
    "sk",
    "sl",
    "es",
    "fr",
    "it",
    "pt",
    "ro",
    "et",
    "fi",
    "hu",
    "lt",
    "lv",
    "el",
    "mt",
]

class MultiEURLEXMT(PromptSourceTask):
    DATASET_PATH = "multi_eurlex"
    DATASET_NAME = "all_languages"
    VERSION = 0

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self) -> Optional[int]:
        return 1024


