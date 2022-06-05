"""Hacky script to get pearson's correlation
2022.06.04
"""
import os
from pathlib import Path
from scipy.stats import pearsonr
import json

if __name__ == "__main__":

    # path to lm-eval and outputs
    dpath = "/home/natasha/Projects/hfbiomed/full_prompting_pipeline/lm-evaluation-harness/outputs"
    dataset = "biosses"

    # Get all files with a dataset
    fnames = Path(dpath).glob("*" + dataset + "*")
    fnames = [i for i in fnames if "examples-" in i.__str__()]

    # Get the latest result 
    latest_result = sorted(fnames, key=os.path.getmtime)[-1]

    with open(latest_result, 'r') as json_file:
        json_list = list(json_file)
    

    # Update only the tasks with float-type predictions
    tasks = {}
    for json_str in json_list:
        result = json.loads(json_str)

        task = result.get("prompt_name", None)

        if result.get("prompt_name", None) is not None:
            
            if task not in tasks:
                tasks.update(
                    {task: {"pred": [], "target": []} }
                    )
                
            pred = result.get("pred", None)
            target = result.get("target", None)

            if (pred is not None) and (target is not None):
                try:
                    tasks[task]["pred"].append(float(pred))
                    tasks[task]["target"].append(float(target))
                except ValueError:
                    pass
    
    # For each task with non-zero pred/targets, compute Pearson-R
    lines = []
    row = ["Task", "Correlation", "P-value"]
    lines.append(",".join(row) + "\n")
    for task in tasks:
        if len(tasks[task]["pred"]):
            corr, pval = pearsonr(tasks[task]["pred"], tasks[task]["target"])
            row = [task, str(round(corr, 3)), str(pval)[:5]]
            lines.append(" ".join(row).rjust(20, " ") + "\n")
            lines.append(",".join(row) + "\n")

    with open(dataset + "_results.txt", "w") as f:
        f.writelines(lines)
