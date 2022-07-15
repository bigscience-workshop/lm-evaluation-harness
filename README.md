# `lm-evaluation-harness` + `promptsource`

![](https://github.com/EleutherAI/lm-evaluation-harness/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/EleutherAI/lm-evaluation-harness/branch/master/graph/badge.svg?token=JSG3O2427J)](https://codecov.io/gh/EleutherAI/lm-evaluation-harness)


## Overview

This project provides a unified framework to test causal (GPT-2, GPT-3, GPTNeo, etc) and seq2seq (T5, T0) language models via prompt evaluation.

As of now, all prompts are provided via the `promptsource` [eval-hackathon branch](https://github.com/bigscience-workshop/promptsource/tree/eval-hackathon); all datasets are from huggingface `datasets`.

This fork is __not__ backwards compatible with the original evaluation harness.


## Installation

```bash
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
```

## CLI Usage 🖥️

To evaluate a model (e.g. GPT-2) on NLP tasks such as SuperGLUE WiC, you can run the following command:

```bash
python main.py \
    --model_api_name 'hf-causal' \
    --model_args pretrained='gpt2' \
    --task_name 'wic' \
    --template_names 'same_sense','polysemous' \
    --device cpu
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. For larger models supported by HuggingFace `transformers`, we provide parallelism and mixed-precision utilities through the [`accelerate`](https://github.com/huggingface/accelerate) package. It can be activated for `hf-causal`/`hf-seq2seq` by passing `use_accelerate=True` and `dtype=half` to the `--model_args` flag, respectively. For finer grained control over `accelerate` options, see the constructor docstrings for `HuggingFaceAutoLM` in `huggingface.py`.

```bash
python main.py \
    --model_api_name 'hf-causal' \
    --model_args use_accelerate=True,pretrained='facebook/opt-13b' \
    --task_name wnli
```

If you have access to the OpenAI API, you can also evaluate GPT-3 engines:


```bash
export OPENAI_API_SECRET_KEY={YOUR_KEY_HERE}
python main.py \
    --model_api_name 'openai' \
    --model_args engine='curie' \
    --task_name hans
```

 **When reporting results from eval harness, please include the task versions (shown in `results["versions"]`) for reproducibility.** This allows bug fixes to tasks while also ensuring that previously reported scores are reproducible.


## Library Usage 📖

You can also use `lm_eval` as a library:

```python
import lm_eval

model = lm_eval.get_model("hf-causal", pretrained="gpt2", device="cpu")
tasks = lm_eval.get_task_list(
    "superglue_rte",
    template_names=["does this imply", "must be true"])
results = lm_eval.evaluate(model=model, tasks=tasks)
```

The main user-facing functions are:

- [`lm_eval.get_model(model_api_name, **kwargs)`](./lm_eval/models/__init__.py) creates a model from a model API
- [`lm_eval.get_task(task_name, template_name, **kwargs)`](./lm_eval/tasks/__init__.py) creates a task with the prompt template
- [`lm_eval.get_task_list(task_name, template_names, **kwargs)`](./lm_eval/tasks/__init__.py) creates multiple instances of a task with different prompt templates
- [`lm_eval.evaluate(model, tasks, **kwargs)`](./lm_eval/evaluator.py) evaluates a model on a list of tasks

Some high-level convenience functions are also made available:
- [`lm_eval.list_model_apis()`](./lm_eval/models/__init__.py) lists all available model APIs you can evaluate from
- [`lm_eval.list_tasks()`](./lm_eval/tasks/__init__.py) lists all available tasks
- [`lm_eval.list_templates(task_name)`](./lm_eval/tasks/__init__.py) lists all available templates for a task


## Gotchas 🩹

- __You must pass templates to `PerplexityTask`s__  even though they will be ignored, as models will be scored from the raw text found in the task's dataset.

- __Multi-lingual ROUGE is unsupported__ as general token splitting is absent from [rouge-score](https://github.com/google-research/google-research/tree/master/rouge). For multi-lingual tasks, please ignore rouge metrics until this is resolved. _NOTE_: `English` works as intended.

- __Task versioning is not fully integrated__! If you're reporting your model's results, please include the package versions or commit IDs for this `lm-evaluation-harness` branch as well as the HuggingFace `datasets` and `promptsource` packages.

- __`promptsource` installation issue__: Some prompts may be excluded from the installed `promptsource` branch due to git-based pip installation issues. If the latest commit on the `promptsource/eval-hackathon` branch contains a prompt you're looking for but was not included in the installed version from our `setup.py`, you should run the following from within your environment:
    ```bash
    pip uninstall promptsource
    git clone --single-branch --branch eval-hackathon https://github.com/bigscience-workshop/promptsource
    cd promptsource
    pip install -e .
    ```

## Features

- Growing number of tasks integrated with `promptsource` (20+).

- Support for hugging face causal language models, HuggingFace Seq2Seq models, and the OpenAI completion API (gpt3), with flexible tokenization-agnostic interfaces.


## Implementing new tasks

To implement a new task in eval harness, follow the [`PromptSourceTask` template](./templates/new_prompt_source_task.py).
