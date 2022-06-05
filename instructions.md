## Setting up BioPrompting
**2022.06.03**

In order to get a fresh install, I did the following:

### Create a fresh conda environment

I created a new conda environment with the following commands:

```
conda create -n bbprompt python=3.9
conda activate bbprompt
```


### Install the bigbio fork of promptsource

I installed promptsource as such:

```
git clone https://github.com/OpenBioLink/promptsource
cd promptsource
pip install -e .
cd ../
```

You may want to fork your own version and install the remote fork.


### Install LM-Eval

Install specifically the bigbio version of LM-Eval. You can do so as follows:

```
git clone git@github.com:bigscience-workshop/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout bigbio
git pull origin bigbio
pip install -e .
cd ..
```

### Install the most recent BigBio dataloaders dataset

Install the main branch of bigbio:

```
git clone git@github.com:bigscience-workshop/biomedical.git
cd biomedical
pip install -e .
cd ..
```

### Creating a custom prompt

**Make sure that in your promptsource installation, a corresponding template exists!**

For this template to exist, you will find them here: `promptsource/promptsource/templates`. The file itself should be called `templates.yaml` and should be generated via the streamlit app (make sure protobuf <= 3.20.X). The folder the template should be in the following structure:

`promptsource/templates/your_dataset_name/your_dataset_name_bigbio_schema

where `your_dataset_name` and `schema` are replaced to the name of the dataset and the specific config you wish to use.

##### Create a new task

Create a new task with the following format filled in:

Note, you will not get results if your data does not have a validation + test set. A crappy hack is to return validation_docs and/or test_docs as the train set. 

Place a file `yourdataset.py` in `lm-evaluation-harness/lm_eval/tasks` that fills out the criteria below:

```python
from lm_eval.base import BioTask

_CITATION = """
PLACE_YOUR_CITATION_HERE
"""


class YourDatasetBase(BioTask):
    VERSION = 0
    DATASET_PATH = "path/to/dataloader/script/from/bigbio"
    DATASET_NAME = None
    SPLIT = None
    
    # Fill these out as T/F depending on your dataset
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
            return self.dataset["test"]  # you can replace with `train` to hack around


class YourDatasetSplit(YourDatasetBase):
    DATASET_NAME = "yourdataset_bigbio_<schema>"
```

Add this dataset task to `lm-evaluation-harness/lm_eval/tasks/__init__.py` by adding the following lines:

```python
from . import yourdataset  # Place this in the beginning import

# Within TASK_REGISTRY, add the following command
TASK_REGISTRY = {
    ...
    "your_dataset_name": yourdataset.Class_Corresponding_To_Schema
}
```

(For example, BIOSSES would look as such:)
```python
    "biosses": biosses.BiossesPairs
```

### Getting your task to run

In order to get lm-eval to run, I made the following changes to `lm-evaluation-harness/lm_eval/base.py`.

1) [L20](https://github.com/bigscience-workshop/lm-evaluation-harness/blob/ea1afe62423c4ff75d6579ccc6942ad8b5138298/lm_eval/base.py#L720): change `target = self.doc_to_target(doc)` to `target = [self.doc_to_target(doc)]` This seemed to give me an issue that returned on str as opposed to a List of str.
<br>
2) [L1055, L1056](https://github.com/bigscience-workshop/lm-evaluation-harness/blob/ea1afe62423c4ff75d6579ccc6942ad8b5138298/lm_eval/base.py#L1055): Edit `CONFIGURED_RANKED_CHOICE_PS_METRICS` or `CONFIGURED_GENERATION_PS_METRICS` to include your custom task metric (found in your `template.yaml` file prompts under `metrics`.) 

### Implementing a Custom Metric

In cases where you may need to implement a custom metric, you will need to write a custom function in `lm-evaluation-harness/lm_eval/metrics.py`. More advanced implementations can exist in `lm-evaluation-harness/lm_eval/metrics_impls`.

**NOTE** If you are working with numerical information (I.e. correlation etc) make sure you have answer choices. If your answer choices are `null` in your prompt, you will go into the "generation" part of the code which may not be useful.

Next, ensure your task has an output in `aggregation` from in `lm-evaluation-harness/lm_eval/base.py`.
In `lm-evaluation-harness/lm_eval/evaluator.py`, the actual eval code is executed. Changes can be made around 265 to change your logic.

# Running your Task

If you implemented the above successfully, your command should run as follows:
```
python main.py --model hf-seq2seq --model_args pretrained=t5-small --tasks yourdataset --device cpu
```

**If you want to run GAD's BLURB set, try:**<br>
```
python main.py --model hf-seq2seq --model_args pretrained=t5-small --tasks gad --device cpu
```

**If you want to run BIOSSES's BLURB set, try:**<br>
```
python main.py --model hf-seq2seq --model_args pretrained=t5-small --tasks biosses --device cpu
```
