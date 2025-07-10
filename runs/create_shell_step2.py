import re
from pathlib import Path
from typing import ClassVar, List

OUT = Path("./scripts")
TEMPLATE = Path("./template.sh")
BASE_WANDB_PROJECT = f"scales-{6:02}"


class KeyMaker:
    # UTD_RATIO_INVERSE: ClassVar[List[int]] = [
    #     1024.0,
    #     2048.0,
    #     4096.0,
    #     8192.0,
    #     16384.0,
    #     32768.0,
    #     65536.0,
    # ]
    # BATCH_SIZE: ClassVar[List[int]] = [
    #     17523.106534479004,
    #     8747.328294799401,
    #     4366.563208780558,
    #     2179.7368995070074,
    #     1088.0989748455502,
    #     543.1661863997047,
    #     271.14215973769893,
    # ]
    # LEARNING_RATE: ClassVar[List[int]] = [
    #     0.0004871808008772118,
    #     0.00035277266228317167,
    #     0.0002554463374383312,
    #     0.0001849713378818428,
    #     0.00013393966099067188,
    #     9.69870953615302e-05,
    #     7.022935997517315e-05,
    # ]
    # SEED: ClassVar[List[int]] = [42, 43, 44]
    # USE_PAL: bool = False

    UTD_RATIO_INVERSE: ClassVar[List[int]] = [
        1024.0,
        2048.0,
        4096.0,
        8192.0,
        16384.0,
        32768.0,
        65536.0,
    ]
    BATCH_SIZE: ClassVar[List[int]] = [
        10593.843174898848,
        7291.791239464436,
        5018.973624785415,
        3454.5827518975707,
        2377.805280899127,
        1636.6543689729554,
        1126.516769474657,
    ]
    LEARNING_RATE: ClassVar[List[int]] = [
        0.000350324524872686,
        0.00031417588025615056,
        0.00028175727568773186,
        0.0002526838226354214,
        0.00022661034773920896,
        0.0002032272947547479,
        0.0001822570493597414,
    ]
    SEED: ClassVar[List[int]] = [42, 43, 44]
    USE_PAL: bool = True

    _DEFAULT_REPLAY_BUFFER_SIZE: int = int(1e6)

    def __init__(self, *, base_wandb_project: str):
        self.base_wandb_project = base_wandb_project
        self.keys = list()

    def create(
        self,
        *,
        seed: int,
        utd_ratio_inverse: int,
        use_pal: bool,
        batch_size: int,
        learning_rate: float,
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
        return f"{self.base_wandb_project}-{type}-step2"


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
    out_root = Path(f"./scripts-{BASE_WANDB_PROJECT}-step2")
    out_root.mkdir(parents=True, exist_ok=False)

    for seed in KeyMaker.SEED:
        for utd_ratio_inverse, batch_size, lr in zip(
            KeyMaker.UTD_RATIO_INVERSE, KeyMaker.BATCH_SIZE, KeyMaker.LEARNING_RATE
        ):
            args = {
                "seed": seed,
                "utd_ratio_inverse": int(utd_ratio_inverse),
                "learning_rate": lr,
                "batch_size": int(batch_size),
                "use_pal": KeyMaker.USE_PAL,
            }
            key_maker.create(**args)

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
