"""
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
https://arxiv.org/pdf/2005.00333v1.pdf

Cross-lingual Choice of Plausible Alternatives (XCOPA) is a typologically diverse
multilingual dataset for causal commonsense reasoning in 11 languages.

Homepage: https://github.com/cambridgeltl/xcopa
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava�{s}, Olga Majewska, Qianchu Liu, Ivan Vuli'{c} and Anna Korhonen},
  journal={arXiv preprint},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
"""


class XCopaBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "xcopa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some XCOPA templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        try:
            text, target = self.prompt_template.apply(doc)
            return False
        except Exception:
            return True


class XCopaId(XCopaBase):
    DATASET_NAME = "id"


class XCopaIt(XCopaBase):
    DATASET_NAME = "it"


class XCopaSw(XCopaBase):
    DATASET_NAME = "sw"


class XCopaTa(XCopaBase):
    DATASET_NAME = "ta"


class XCopaVi(XCopaBase):
    DATASET_NAME = "vi"


class XCopaZh(XCopaBase):
    DATASET_NAME = "zh"


XCOPA_TASKS = [
    XCopaId,
    XCopaIt,
    XCopaSw,
    XCopaTa,
    XCopaVi,
    XCopaZh,
]


def construct_tasks() -> typing.Dict[str, XCopaBase]:
    """Returns a dictionary of tasks keyed by task name as: `"xcopa_{lang}": XCopaLang`"""
    tasks = {}
    for task_class in XCOPA_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks
