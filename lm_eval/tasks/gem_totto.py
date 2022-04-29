"""
ToTTo: A Controlled Table-To-Text Generation Dataset
https://aclanthology.org/2020.emnlp-main.89/

This is the ToTTo subset of the GEM benchmark. 
ToTTo is an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description. To obtain generated targets that are natural but also faithful to the source table, the authors introduce a dataset construction process where annotators directly revise existing candidate sentences from Wikipedia. The authors present systematic analyses of the dataset and annotation process as well as results achieved by several state-of-the-art baselines. While usually fluent, existing methods often hallucinate phrases that are not supported by the table, suggesting that this dataset can serve as a useful research benchmark for high-precision conditional text generation.
Homepage: https://github.com/google-research-datasets/totto
"""
from lm_eval.base import PromptSourceTask, Task, rf

_CITATION = """
@inproceedings{parikh-etal-2020-totto,
    title = "{ToTTo}: A Controlled Table-To-Text Generation Dataset",
    author = "Parikh, Ankur  and
      Wang, Xuezhi  and
      Gehrmann, Sebastian  and
      Faruqui, Manaal  and
      Dhingra, Bhuwan  and
      Yang, Diyi  and
      Das, Dipanjan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.89",
    doi = "10.18653/v1/2020.emnlp-main.89",
    pages = "1173--1186",
    abstract = "We present ToTTo, an open-domain English table-to-text dataset with over 120,000 training examples that proposes a controlled generation task: given a Wikipedia table and a set of highlighted table cells, produce a one-sentence description. To obtain generated targets that are natural but also faithful to the source table, we introduce a dataset construction process where annotators directly revise existing candidate sentences from Wikipedia. We present systematic analyses of our dataset and annotation process as well as results achieved by several state-of-the-art baselines. While usually fluent, existing methods often hallucinate phrases that are not supported by the table, suggesting that this dataset can serve as a useful research benchmark for high-precision conditional text generation.",
}
"""


class GEMTOTTOBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/totto"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class GEMTOTTO(GEMTOTTOBase):
    """this is for train/validation/test"""

    SPLIT = ""


class GEMTOTTOChallgeSample(GEMTOTTOBase):
    """this is for challenge_train_sample/challenge_validation_sample"""

    SPLIT = "challenge_sample"

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():

            if self._training_docs is None:
                self._training_docs = list(self.dataset["challenge_train_sample"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["challenge_validation_sample"]
