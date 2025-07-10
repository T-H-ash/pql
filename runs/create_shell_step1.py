import re
from pathlib import Path
from typing import ClassVar, List

OUT = Path("./scripts")
TEMPLATE = Path("./template.sh")
BASE_WANDB_PROJECT = f"scales-{7:02}"


class KeyMaker:
    UTD_RATIO_INVERSE: ClassVar[List[int]] = [2048, 4096, 8192, 16384]
    BATCH_SIZE: ClassVar[List[int]] = [512, 1024, 2048, 4096, 8192]
    SEED: ClassVar[List[int]] = [42, 43, 44, 45, 46]
    USE_PAL: ClassVar[List[bool]] = [False, True]

    _DEFAULT_BATCH_SIZE: int = 2048
    _DEFAULT_LEARNING_RATE: float = 0.0002
    _DEFAULT_REPLAY_BUFFER_SIZE: int = int(1e5)

    def __init__(self, *, base_wandb_project: str):
        self.base_wandb_project = base_wandb_project
        self.keys = list()

    def create(
        self,
        *,
        seed: int,
        utd_ratio_inverse: int,
        use_pal: bool,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        replay_buffer_size: int = _DEFAULT_REPLAY_BUFFER_SIZE,
    ):
        key = {
            "SEED": seed,
            "UTD_RATIO_INVERSE": utd_ratio_inverse,
            "BATCH_SIZE": batch_size,
            "LEARNING_RATE": learning_rate,
            "USE_PAL": use_pal,
            "WANDB_PROJECT": self.get_wandb_project(use_pal),
            "REPLAY_BUFFER_SIZE": replay_buffer_size,
        }
        self.keys.append(key)

    def get_wandb_project(self, use_pal: bool) -> str:
        type = "pal" if use_pal else "nopal"
        return f"{self.base_wandb_project}-{type}-step1"


def create_script(keys, out_path: Path):
    # Read the template
    text = TEMPLATE.read_text()

    pattern = r"(# --- params settings --- #)(.*?)(# --- params settings --- #)"

    def format_value(val):
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, float):
            # Use scientific notation for large floats
            return f"{val:.0e}" if abs(val) >= 1e5 else str(val)
        return str(val)

    param_lines = [f"{k}={format_value(v)}" for k, v in keys.items()]
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
    key_maker = KeyMaker(base_wandb_project=BASE_WANDB_PROJECT)
    out_root = Path(f"./scripts-{BASE_WANDB_PROJECT}-step1")
    out_root.mkdir(parents=True, exist_ok=False)

    for use_pal in KeyMaker.USE_PAL:
        for seed in KeyMaker.SEED:
            for utd_ratio_inverse in KeyMaker.UTD_RATIO_INVERSE:
                args = {
                    "seed": seed,
                    "utd_ratio_inverse": utd_ratio_inverse,
                    "use_pal": use_pal,
                }

                key_maker.create(**args, learning_rate=0.0001)

                for batch_size in KeyMaker.BATCH_SIZE:
                    key_maker.create(**args, batch_size=batch_size)

                key_maker.create(**args, learning_rate=0.0003)

    scripts = []

    for run_id, key in enumerate(key_maker.keys):
        out_name = f"run-{run_id:03}-" + "-".join(
            [f"{k[:2]}={v}" for k, v in key.items()]
        )
        script_path = out_root / f"{out_name}.sh"
        create_script({"RUN_ID": run_id, **key.copy()}, script_path)
        scripts.append(script_path)

    # Write all script paths to runall file with pjsub command
    runall_path = out_root.parent / f"runall-{out_root.name}.sh"
    with runall_path.open("w") as f:
        for script in scripts:
            abs_script = script.resolve()
            f.write(f'pjsub --step --sparam "jnam={out_root.name}" "{abs_script}"\n')
    print(f"Wrote all pjsub commands to {runall_path}")


if __name__ == "__main__":
    main()
