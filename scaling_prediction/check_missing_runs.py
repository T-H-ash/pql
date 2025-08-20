import argparse
import math
import os
import re
from pathlib import Path
from pprint import pprint

import dotenv
import wandb
from tqdm import tqdm

dotenv.load_dotenv()

ELAPSED_TIME_THRESHOLD = 3400


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT"))
    parser.add_argument("--num_splits", type=int, default=1)

    parser.add_argument("--wandb_entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--out_dir", type=Path, default="./runs")
    return parser.parse_args()


def read_wandb_runs(entity, project):
    api = wandb.Api()
    if not project_exists(entity, project):
        print(f"Project '{project}' does not exist in entity '{entity}'.")
        return []

    runs = api.runs(f"{entity}/{project}")

    successful_runs, not_successful_runs = [], []
    for run in tqdm(runs):
        shell_script_name = run.config.get("logging", {}).get("shell_script_name", None)
        runtime = [step["_runtime"] for step in run.history(keys=["_runtime"], samples=100, pandas=False)]
        if runtime and max(runtime) < ELAPSED_TIME_THRESHOLD:
            not_successful_runs.append(shell_script_name)
            continue

        elapsed_time = [step["elapsed_time"] for step in run.history(keys=["elapsed_time"], samples=100, pandas=False)]
        if not elapsed_time or max(elapsed_time) < ELAPSED_TIME_THRESHOLD:
            not_successful_runs.append(shell_script_name)
            continue

        count, eval_returns, global_steps = 0, [], []
        for row in run.history(keys=["eval/return"], pandas=False):
            if row["eval/return"] is None:
                continue

            eval_returns.append(row["eval/return"])
            global_steps.append(row["_step"])
            count += 1

        if count == 0:
            print(f"Skipping {shell_script_name} due to no eval returns found.")
            continue

        successful_runs.append(shell_script_name)

    print(f"{len(successful_runs)} successful runs found. {len(not_successful_runs)} runs skipped due to short runtime.")
    return successful_runs


def project_exists(entity: str, project: str) -> bool:
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}", per_page=1)
        len(runs)  # Trigger an API call to check if the project exists
    except (wandb.errors.CommError, ValueError):
        return False
    else:
        return True


def fix_runall_script(runall_script_path: Path, wandb_project: str, not_done_scripts: list, num_splits: int):
    if not runall_script_path.exists():
        print(f"Rewrite file {runall_script_path} does not exist.")
        return

    with runall_script_path.open("r") as f:
        not_done_scripts_lines = [line for line in f if any(script in line for script in not_done_scripts)]

    new_runall_script_path = runall_script_path.parent / f"{runall_script_path.stem}-rerun.sh"
    with new_runall_script_path.open("w") as f:
        for _idx, line in enumerate(not_done_scripts_lines):
            idx = _idx // math.ceil(len(not_done_scripts_lines) / num_splits)
            f.write(line.replace(_extract_jnam(line), f"jnam={wandb_project}-{idx}"))


def _extract_jnam(text: str) -> str:
    return re.search(r'jnam=([^"\s]+)', text).group(0)


def removesuffix(s: str, suffix: str) -> str:
    return s[: -len(suffix)] if s.endswith(suffix) else s  # noqa: FURB188 (using python 3.8.20)


def main():
    args = get_args()
    successful_runs = read_wandb_runs(args.wandb_entity, args.wandb_project)

    phase = args.wandb_project.split("-")[-1]  # "step1" or "step2"
    wandb_project_key = removesuffix(args.wandb_project, f"-{phase}")
    base_dir = args.out_dir / wandb_project_key

    scripts = {path.name for path in (base_dir / f"scripts-{phase}").glob("*.sh")}
    not_done_scripts = scripts - set(successful_runs)
    print(f"{len(not_done_scripts)} scripts not done yet.")
    pprint(not_done_scripts)

    breakpoint()

    if not_done_scripts:
        runall_script_path = base_dir / f"runall-{phase}.sh"
        fix_runall_script(runall_script_path, args.wandb_project, not_done_scripts, args.num_splits)


if __name__ == "__main__":
    main()
