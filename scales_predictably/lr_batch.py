import argparse
import json
import random
from pathlib import Path

import numpy as np
import wandb
from scipy.optimize import isotonic_regression


def extract_wandb_runs(project, entity=None):
    api = wandb.Api()
    runs = api.runs(f"{entity + '/' if entity else ''}{project}")
    runs_dict = {}

    for run in runs:
        config = run.config
        try:
            num_envs = config.get("num_envs", "N/A")
            batch_size = config.get("algo", {}).get("batch_size", "N/A")
            actor_lr = config.get("algo", {}).get("actor_lr", "N/A")
            seed = config.get("seed", "N/A")
            critic_ratio = config.get("algo", {}).get("critic_sample_ratio", "N/A")
            use_pal = config.get("algo", {}).get("pal", "N/A")
            utd_ratio_inverse = num_envs / critic_ratio

        except Exception as e:
            print(f"Error accessing config for run {run.id}: {e}")

        if run.state == "running":
            continue

        print(
            f"Run ID: {run.id}, Name: {run.name:<25}, State: {run.state:<10} | "
            f"batch_size: {batch_size:<5} | actor_lr: {actor_lr} | "
            f"seed={seed} | critic_sample_ratio={critic_ratio} | "
            f"utd_ratio_inverse={utd_ratio_inverse}"
        )
        name = f"SE={seed}-UT={int(utd_ratio_inverse)}-BA={batch_size}-LE={actor_lr}-US={use_pal}"

        count, eval_returns, global_steps = 0, [], []
        history = run.history(keys=["eval/return"])
        print(f"History length: {len(history)}")
        for row in history:
            if row["eval/return"] is None:
                continue

            eval_returns.append(row["eval/return"])
            global_steps.append(row["_step"])
            count += 1
        print(f"History length: {count}")

        if count == 0:
            continue

        ret = {
            "config": {
                "name": run.name,
                "script": name,
                "num_envs": num_envs,
                "batch_size": batch_size,
                "lr": actor_lr,
                "seed": seed,
                "critic_ratio": critic_ratio,
                "use_pal": use_pal,
                "utd_ratio_inverse": utd_ratio_inverse,
            },
            "eval_returns": eval_returns,
            "_step": global_steps,
        }

        runs_dict[run.id] = ret

    print(f"Successfully listed {len(runs_dict)} runs in project '{project}'.")

    with Path(f"{project}_runs.json").open("w") as f:
        json.dump(runs_dict, f, indent=4)


class Analyzer:
    UTD_INV = [2048, 4096, 8192, 16384]
    BATCH_SIZE = [
        (512, 0.0002),
        (1024, 0.0002),
        (2048, 0.0002),
        (4096, 0.0002),
        (8192, 0.0002),
    ]
    LR = [
        (2048, 0.0001),
        (2048, 0.0002),
        (2048, 0.0003),
    ]
    NUM_SEEDS = 3

    def __init__(self, project):
        self.project = project
        self.runs_dict = self.load_runs()

    def load_runs(self):
        with Path(f"{self.project}_runs.json").open("r") as f:
            return json.load(f)

    def extract(self, utd_ratio_inverse, batch_size, lr):
        ret = []
        for run_id, run_data in self.runs_dict.items():
            if (
                run_data["config"]["utd_ratio_inverse"] == utd_ratio_inverse
                and run_data["config"]["batch_size"] == batch_size
                and run_data["config"]["lr"] == lr
            ):
                ret.append(run_data)
        return ret

    def run(self):
        self.analyze_batch_size()
        self.analyze_lr()

    def analyze_batch_size(self):
        for utd_inv in self.UTD_INV:
            for batch_size, lr in self.BATCH_SIZE:
                runs = self.extract(utd_inv, batch_size, lr)
                if len(runs) < self.NUM_SEEDS:
                    print(
                        f"Not enough runs for UTD_INV={utd_inv}, Batch Size={batch_size}, LR={lr}: {len(runs)} runs found."
                    )
                    return

                sampled_runs = self.sample_run(runs)
                mean_series = self.interpolation_mean(sampled_runs)
                breakpoint()
                processed_series = self.isotonic_regression(mean_series)

                # processed_series: list[np.adarray] ... [return, data_steps]

            best_batch_size = self.choose_best(processed_series)

    def analyze_lr(self):
        for utd_inv in self.UTD_INV:
            for batch_size, lr in self.LR:
                runs = self.extract(utd_inv, batch_size, lr)
                if len(runs) < self.NUM_SEEDS:
                    print(
                        f"Not enough runs for UTD_INV={utd_inv}, Batch Size={batch_size}, LR={lr}: {len(runs)} runs found."
                    )
                    return

                sampled_runs = self.sample_run(runs)
                mean_series = self.interpolation_mean(sampled_runs)
                processed_series = self.isotonic_regression(mean_series)

            best_lr = self.choose_best(processed_series)

    @staticmethod
    def sample_run(runs):
        return Analyzer._simple_sample(runs)

    @staticmethod
    def _simple_sample(runs):
        return runs

    @staticmethod
    def _replacement_sample(runs):
        return random.sample(runs, len(runs))

    @staticmethod
    def interpolation_mean(runs):
        x_list = [np.array(run["_step"]) for run in runs]
        y_list = [np.array(run["eval_returns"]) for run in runs]
        return average_series(x_list, y_list)

    @staticmethod
    def isotonic_regression(y):
        return isotonic_regression(y)

    @staticmethod
    def choose_best(processed_series):
        # Implement logic to choose the best batch size based on processed_series
        pass


def average_series(x_list, y_list, num_points=None, method="linear"):
    """
    N個のデータ系列を共通のx軸に補間し、平均と標準偏差を計算する。

    Parameters
    ----------
    x_list : list of np.ndarray
        各系列のx座標 (長さはバラバラでOK)
    y_list : list of np.ndarray
        各系列のy座標 (x_list と対応)
    num_points : int, optional
        共通のx軸の分割数（デフォルト100）
    method : str, optional
        'linear'（np.interp）または将来の拡張用

    Returns
    -------
    x_common : np.ndarray
        共通のx座標
    y_mean : np.ndarray
        平均
    y_std : np.ndarray
        標準偏差
    """

    if len(x_list) != len(y_list):
        raise ValueError("x_list と y_list の長さが一致しません。")

    # 共通のx軸を決める
    x_min = min(x.min() for x in x_list)
    x_max = max(x.max() for x in x_list)

    lengths = [len(x) for x in x_list]
    if num_points is None:
        num_points = sum(lengths) / len(lengths)  # 平均的な長さを使用
        print(f"Using average number of points: {num_points}")
    x_common = np.linspace(x_min, x_max, num_points)

    # 各系列を補間
    y_interp_list = []
    for x, y in zip(x_list, y_list, strict=False):
        if method == "linear":
            y_interp = np.interp(x_common, x, y)
        else:
            raise NotImplementedError(f"補間方法 '{method}' は未サポートです。")
        y_interp_list.append(y_interp)

    # 平均・標準偏差を計算
    y_interp_array = np.array(y_interp_list)
    y_mean = np.mean(y_interp_array, axis=0)
    y_std = np.std(y_interp_array, axis=0)

    return x_common, y_mean, y_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=None, help="wandb entity (user or team)")
    args = parser.parse_args()

    # extract_wandb_runs(args.project, args.entity)

    analyzer = Analyzer(args.project)
    analyzer.run()
