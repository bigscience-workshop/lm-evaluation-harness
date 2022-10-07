"""
XNLI is an evaluation corpus for language transfer and cross-lingual sentence classification in 15 languages.
https://arxiv.org/abs/1809.05053

Homepage: None, Repo: https://github.com/facebookresearch/XNLI
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{conneau2018xnli,
  title={XNLI: Evaluating Cross-lingual Sentence Representations},
  author={Conneau, Alexis and Rinott, Ruty and Lample, Guillaume and Williams, Adina and Bowman, Samuel and Schwenk, Holger and Stoyanov, Veselin},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={2475--2485},
  year={2018}
}
}"""


class XNLI(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "xnli"
    DATASET_NAME = None

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


class XNLIEn(XNLI):
    DATASET_NAME = "en"

class XNLIFr(XNLI):
    DATASET_NAME = "fr"

class XNLIEs(XNLI):
    DATASET_NAME = "es"

class XNLIDe(XNLI):
    DATASET_NAME = "de"

class XNLIEl(XNLI):
    DATASET_NAME = "el"

class XNLIBg(XNLI):
    DATASET_NAME = "bg"

class XNLIRu(XNLI):
    DATASET_NAME = "ru"

class XNLITr(XNLI):
    DATASET_NAME = "tr"

class XNLIAr(XNLI):
    DATASET_NAME = "ar"

class XNLIVi(XNLI):
    DATASET_NAME = "vi"

class XNLITh(XNLI):
    DATASET_NAME = "th"

class XNLIZh(XNLI):
    DATASET_NAME = "zh"

class XNLIHi(XNLI):
    DATASET_NAME = "hi"

class XNLISw(XNLI):
    DATASET_NAME = "sw"

class XNLIUr(XNLI):
    DATASET_NAME = "ur"


XNLI_TASKS = [
    XNLIEn,
    XNLIFr,
    XNLIEs,
    XNLIDe,
    XNLIEl,
    XNLIBg,
    XNLIRu,
    XNLITr,
    XNLIAr,
    XNLIVi,
    XNLITh,
    XNLIZh,
    XNLIHi,
    XNLISw,
    XNLIUr
]


def construct_tasks() -> typing.Dict[str, XNLI]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "GEM/wiki_lingua_ar"
    will dispatch to the GEM WikiLingua Arabic class.
    """
    tasks = {}
    for task_class in XNLI_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks
