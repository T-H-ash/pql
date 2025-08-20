"""

Create shell scripts for grid search of batch size and learning rate. First set the GridParam and run this script.

Usage:
    - python grid_step2.py --wandb_project_key <your_wandb_project_key>
    - python grid_step2.py --additional
        - if you need additional runs with new grid parameters

"""

import argparse
import importlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

from grid_step1 import (
    DirectoryNotFoundError,
    create_shell_script,
    get_param_dict,
    get_shell_script_name,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project_key", type=str)
    parser.add_argument("--num_splits", type=int, default=1)

    parser.add_argument("--out_dir", type=Path, default="./runs")
    parser.add_argument("--step1_fit_filename", type=str, default="step1_fit_result")
    parser.add_argument("--template_path", type=Path, default="./template.sh")
    parser.add_argument("--additional", action="store_true")
    parser.add_argument("--group", type=str, default="gn53", choices=["gn53", "gk75"])
    return parser.parse_args()


@dataclass
class GridParam:
    wandb_project_key: str

    # x-axis
    UTD_RATIO_INVERSE = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # seed
    SEED = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    def __post_init__(self):
        """
        wandb_project_key: <prefix>-task=<task_name>-buffsize=<buffer_size>-num_envs=<num_envs>-per=<per_type>
        """
        pattern = r"task=([^-\s]+)-buffsize=([^-\s]+)-num_envs=([^-\s]+)-per=([^-\s]+)"
        match = re.search(pattern, self.wandb_project_key)
        if match:
            task, buffsize, num_envs, per = match.groups()

            # other configs
            self.TASK = task
            self.BUFFER_SIZE = int(float(buffsize.rstrip("M")) * 1_000_000)
            self.NUM_ENVS = int(num_envs)

            if per == "pal":
                self.USE_PAL = True
            elif per == "none":
                self.USE_PAL = False
            else:
                error_message = f"Invalid per type: {per}. Expected 'pal' or 'none'."
                raise ValueError(error_message)


def get_fit_functions(args):
    module_path = args.out_dir / args.wandb_project_key / args.step1_fit_filename
    return importlib.import_module(str(module_path).replace("/", "."))


def main():
    args = get_args()
    grid_param = GridParam(args.wandb_project_key)
    wandb_project = f"{args.wandb_project_key}-step2"

    step1_fit = get_fit_functions(args)

    # Generate combinations of parameters
    params = set()
    for seed in grid_param.SEED:
        for utd_inv in grid_param.UTD_RATIO_INVERSE:
            param_args = {"utd_inv": utd_inv, "seed": seed, "wandb_project": wandb_project}

            batch_size = int(step1_fit.batch_size_function(1 / utd_inv))
            lr = step1_fit.learning_rate_function(1 / utd_inv)

            param = get_param_dict(grid_param, batch_size=batch_size, learning_rate=lr, **param_args)
            params.add(frozenset(param.items()))

    # Create output directory if necessary
    out_root_dir = args.out_dir / args.wandb_project_key
    script_dir = out_root_dir / "scripts-step2"
    if not args.additional:
        script_dir.mkdir(parents=True, exist_ok=False)
    elif not out_root_dir.exists():
        raise DirectoryNotFoundError(out_root_dir)

    # Create shell scripts for each parameters
    sorted_params = []
    for _param in params:
        param = dict(_param)
        script_key = get_shell_script_name(
            utd_inv=param["UTD_RATIO_INVERSE"],
            batch_size=param["BATCH_SIZE"],
            learning_rate=param["LEARNING_RATE"],
            seed=param["SEED"],
        )
        sorted_params.append((param, script_key))

    sorted_params.sort(key=lambda _tuple: _tuple[1])

    for idx, (_param, script_key) in enumerate(sorted_params):
        script_path = script_dir / f"run-{idx:03d}-{script_key}"
        param = dict(sorted({**_param, "SCRIPT_NAME": script_path.name, "RUN_ID": idx}.items()))
        create_shell_script(param, args.template_path, script_path)

    # Write run commands to a file
    runall_script_path = out_root_dir / "runall-step2.sh"
    with runall_script_path.open("w") as f:
        for _idx, script_path in enumerate(sorted((script_dir).glob("*.sh"))):
            abs_script = script_path.resolve()
            idx = _idx // (math.ceil(len(sorted_params) / args.num_splits))
            f.write(f'pjsub -g {args.group} --step --sparam "jnam={wandb_project}-{idx}" "{abs_script}"\n')


if __name__ == "__main__":
    main()
