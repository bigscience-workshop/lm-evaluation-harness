"""
BIOSSES: 
NOTE: A stupid hack I am doing is forcing a "validation" set as this only has training data implemented.
"""
from lm_eval.base import BioTask

_CITATION = """
@article{souganciouglu2017biosses,
  title={BIOSSES: a semantic sentence similarity estimation system for the biomedical domain},
  author={Soğancıoğlu, Gizem, Hakime Öztürk, and Arzucan Özgür},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i49--i58},
  year={2017},
  publisher={Oxford University Press}
}
"""


class BiossesBase(BioTask):
    VERSION = 0
    DATASET_PATH = "/home/natasha/Projects/hfbiomed/full_prompting_pipeline/biomedical/bigbio/biodatasets/biosses"
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
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class BiossesPairs(BiossesBase):
    DATASET_NAME = "biosses_bigbio_pairs"