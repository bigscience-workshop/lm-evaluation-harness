import lm_eval
import json
import accelerate
from accelerate import Accelerator 


def test_llh_tasks():
    tasks = lm_eval.get_task_list(
        'sst',
        template_names=['happy or mad', 'review', 'said']
    )
    tasks += lm_eval.get_task_list(
        'mnli',
        template_names=['GPT-3 style', 'MNLI crowdsource', 'always/sometimes/never']
    )
    tasks += lm_eval.get_task_list(
        'rte',
        template_names=['imply', 'imply separated', 'mean']
    )
    tasks += lm_eval.get_task_list(
        'mrpc',
        template_names=['want to know', 'equivalent', 'replace']
    )
    tasks += lm_eval.get_task_list(
        'wic',
        template_names=['GPT-3-prompt', 'GPT-3-prompt-with-label', 'affirmation_true_or_false']
    )
    tasks += lm_eval.get_task_list(
        'axg',
        template_names=['GPT-3 style', 'MNLI crowdsource', 'based on the previous passage']
    )
    return tasks

def test_greedy_tasks():
    tasks = lm_eval.get_task_list(
        'mrpc',
        template_names=['generate_sentence']
    )
    return tasks

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--branch', type=str, default='add-data-parallel')
args = argparser.parse_args()

if __name__ == "__main__":
    accelerator = Accelerator()
    model = lm_eval.get_model("hf-seq2seq", pretrained='google/t5-small-lm-adapt')
    # tasks = test_greedy_tasks()
    tasks = test_llh_tasks()

    if args.branch == 'master':
        results = lm_eval.evaluate(model=model, tasks=tasks)
        with open(f'results_batch_size={model.batch_size}-branch={args.branch}.json', 'w') as f:
            json.dump(results, f, indent=2)
    else:
        for batch_size in [80, 1]:
            results = lm_eval.evaluate(
                model=model, tasks=tasks, batch_size=batch_size, accelerator=accelerator)
            with open(f'results_batch_size={batch_size}-branch={args.branch}.json', 'w') as f:
                json.dump(results, f, indent=2)
