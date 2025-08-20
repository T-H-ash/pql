"""

Create shell scripts for grid search of batch size and learning rate. First set the GridParam and run this script.

Usage:
    - python grid_step1.py
    - python grid_step1.py --additional
        - if you need additional runs with new grid parameters

"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional", action="store_true")
    parser.add_argument("--group", type=str, default="gn53", choices=["gn53", "gk75"])
    parser.add_argument("--num_splits", type=int, default=1)

    parser.add_argument("--out_dir", type=Path, default="./runs")
    parser.add_argument("--template_path", type=Path, default="./template.sh")
    parser.add_argument("--wandb_project_prefix", type=str, default="scales_00")
    return parser.parse_args()


@dataclass
class GridParam:
    # x-axis
    UTD_RATIO_INVERSE = [128, 256, 512, 1024]

    # y-axis
    BATCH_SIZE = {
        "default": 16384,
        "choices": [4096, 8192, 16384, 32768, 65536],
    }
    LEARNING_RATE = {
        "default": 0.001,
        "choices": [0.001],
    }

    # seed
    SEED = [42, 43, 44, 45, 46]

    # other configs
    TASK = "Ant"
    NUM_ENVS = 4096
    BUFFER_SIZE = int(4e6)
    USE_PAL = False


def get_wandb_project_key(prefix, grid_param):
    task, buffer_size, use_pal = grid_param.TASK, grid_param.BUFFER_SIZE, grid_param.USE_PAL
    per_type = "pal" if use_pal else "none"
    return f"{prefix}-task={task}-buffsize={_to_millions(buffer_size)}-per={per_type}"


def get_shell_script_name(
    utd_inv: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> str:
    utd_inv_str = f"utd_inv={utd_inv:d}"
    batch_size_str = f"batch_size={batch_size:d}"
    learning_rate_str = f"lr={learning_rate}"
    seed_str = f"seed={seed}"
    return f"{utd_inv_str}-{batch_size_str}-{learning_rate_str}-{seed_str}.sh"


def _to_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}".rstrip("0").rstrip(".") + "M"


def get_param_dict(grid_param, *, utd_inv, batch_size, learning_rate, seed, wandb_project):  # noqa: PLR0913
    return {
        "UTD_RATIO_INVERSE": utd_inv,
        "BATCH_SIZE": batch_size,
        "LEARNING_RATE": learning_rate,
        "NUM_ENVS": grid_param.NUM_ENVS,
        "SEED": seed,
        "WANDB_PROJECT": wandb_project,
        "TASK": grid_param.TASK,
        "REPLAY_BUFFER_SIZE": grid_param.BUFFER_SIZE,
        "USE_PAL": grid_param.USE_PAL,
    }


def create_shell_script(param, template_path: Path, out_path: Path):
    # Read the template
    text = template_path.read_text()

    pattern = r"(# --- params settings --- #)(.*?)(# --- params settings --- #)"

    def format_value(val):
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, bool):
            return str(val)
        return str(val)

    param_lines = [f"{k}={format_value(v)}" for k, v in param.items()]
    param_block = "\n".join(param_lines)

    # Replace the block
    new_text = re.sub(
        pattern,
        lambda m: f"{m.group(1)}\n{param_block}\n{m.group(3)}",
        text,
        flags=re.DOTALL,
    )

    # Write to out path
    out_path.write_text(new_text)
    print(f"Wrote replaced script to {out_path}")


def main():
    args = get_args()
    grid_param = GridParam()
    wandb_project_key = get_wandb_project_key(args.wandb_project_prefix, grid_param)
    wandb_project = f"{wandb_project_key}-step1"

    # Generate combinations of parameters
    params = set()
    for seed in grid_param.SEED:
        for utd_inv in grid_param.UTD_RATIO_INVERSE:
            param_args = {"utd_inv": utd_inv, "seed": seed, "wandb_project": wandb_project}

            # grid search for utd vs batch-size
            for batch_size in grid_param.BATCH_SIZE["choices"]:
                param = get_param_dict(
                    grid_param,
                    batch_size=batch_size,
                    learning_rate=grid_param.LEARNING_RATE["default"],
                    **param_args,
                )
                params.add(frozenset(param.items()))

            # grid search for utd vs learning-rate
            for lr in grid_param.LEARNING_RATE["choices"]:
                param = get_param_dict(
                    grid_param,
                    batch_size=grid_param.BATCH_SIZE["default"],
                    learning_rate=lr,
                    **param_args,
                )
                params.add(frozenset(param.items()))

    # Create output directory if necessary
    out_root_dir = args.out_dir / wandb_project_key
    script_dir = out_root_dir / "scripts-step1"
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
    runall_script_path = out_root_dir / "runall-step1.sh"
    with runall_script_path.open("w") as f:
        for _idx, script_path in enumerate(sorted((script_dir).glob("*.sh"))):
            abs_script = script_path.resolve()
            idx = _idx // (math.ceil(len(sorted_params) / args.num_splits))
            f.write(f'pjsub -g {args.group} --step --sparam "jnam={wandb_project}-{idx}" "{abs_script}"\n')


class DirectoryNotFoundError(Exception):
    def __init__(self, dir_name):
        super().__init__(f"Output directory {dir_name} must exist with 'additional' flag.")


if __name__ == "__main__":
    main()
