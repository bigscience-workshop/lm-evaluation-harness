"""
Gad: A corpus identifying associations between genes and diseases by a semi-automatic annotation procedure based on the Genetic Association Database

Homepage: "https://github.com/dmis-lab/biobert"
"""
from lm_eval.base import BioTask

_CITATION = """
@article{Bravo2015,
  doi = {10.1186/s12859-015-0472-9},
  url = {https://doi.org/10.1186/s12859-015-0472-9},
  year = {2015},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {16},
  number = {1},
  author = {{\`{A}}lex Bravo and Janet Pi{\~{n}}ero and N{\'{u}}ria Queralt-Rosinach and Michael Rautschka and Laura I Furlong},
  title = {Extraction of relations between genes and diseases from text and large-scale data analysis: implications for translational research},
  journal = {{BMC} Bioinformatics}
}
"""


class GadBase(BioTask):
    VERSION = 0
    DATASET_PATH = "lm_eval/datasets/biomedical/bigbio/biodatasets/gad"
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


class GadFold0Text(GadBase):
    DATASET_NAME = "gad_fold0_bigbio_text"
