import argparse
import json
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib import colors as mcolors
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

dotenv.load_dotenv()

ELAPSED_TIME_THRESHOLD = 3400


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_load", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT"))

    parser.add_argument("--out_dir", type=Path, default="./runs")
    parser.add_argument("--wandb_entity", type=str, default=os.getenv("WANDB_ENTITY"))
    return parser.parse_args()


def get_base_dir(wandb_project: str, out_dir: Path) -> Path:
    phase = wandb_project.split("-")[-1]  # "step1" or "step2"
    wandb_project_key = _removesuffix(wandb_project, f"-{phase}")
    return out_dir / wandb_project_key


def _removesuffix(s: str, suffix: str) -> str:
    return s[: -len(suffix)] if s.endswith(suffix) else s  # noqa: FURB188 (using python 3.8.20)


def read_wandb_runs(entity, project, save_path: Path):
    api = wandb.Api()
    if not _project_exists(entity, project):
        print(f"Project '{project}' does not exist in entity '{entity}'.")
        return

    runs = api.runs(f"{entity}/{project}")

    # ---------- tmp ---------- #
    runs = chain(runs, api.runs(f"{entity}/scales_00-task=Ant-buffsize=5M-per=pal"))
    # ---------- tmp ---------- #

    successful_runs_dict = {}
    for run in tqdm(runs):
        shell_script_name = run.config.get("logging", {}).get("shell_script_name", None)
        runtime = [step["_runtime"] for step in run.history(keys=["_runtime"], samples=100, pandas=False)]
        if runtime and max(runtime) < ELAPSED_TIME_THRESHOLD:
            continue

        try:
            config = _get_wandb_config(run.config)
        except ConfigKeyError as e:
            print(f"Skipping {shell_script_name} due to an error while retrieving config: {e}")
            continue

        count, eval_returns, global_steps = 0, [], []
        for row in run.history(keys=["eval/return"], pandas=False):
            if row["eval/return"] is None:
                continue

            eval_returns.append(row["eval/return"])
            global_steps.append(row["_step"])
            count += 1

        if count == 0:
            continue

        successful_runs_dict[run.id] = {**config, "eval_return": eval_returns, "step": global_steps}
    print(f"Successfully listed {len(successful_runs_dict)} runs in project '{project}'.")

    with save_path.open("w") as f:
        json.dump(successful_runs_dict, f, indent=4)
        print(f"Saved runs to {save_path}")


def _project_exists(entity: str, project: str) -> bool:
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project}", per_page=1)
        len(runs)  # Trigger an API call to check if the project exists
    except (wandb.errors.CommError, ValueError):
        return False
    else:
        return True


def _get_wandb_config(config):
    keys = ["task.name", "num_envs", "seed", "algo.batch_size", "algo.actor_lr", "algo.critic_sample_ratio", "algo.pal"]
    ret_dict = {}
    for key in keys:
        value = config
        for part in key.split("."):
            value = value.get(part, {})

        if value == {}:
            raise ConfigKeyError(key)

        ret_dict[key] = value

    ret_dict["utd_ratio_inverse"] = ret_dict["num_envs"] / ret_dict["algo.critic_sample_ratio"]
    return ret_dict


class ConfigKeyError(Exception):
    def __init__(self, key):
        error_message = f"key '{key}' is not found in the config."
        super().__init__(error_message)


def removesuffix(s: str, suffix: str) -> str:
    return s[: -len(suffix)] if s.endswith(suffix) else s  # noqa: FURB188 (using python 3.8.20)


class Analyzer:
    def __init__(self, json_path: Path, eval_threshold_range: tuple, num_eval_points: int = 20):
        self.all_runs = self._load_runs(json_path)
        self.utd_inv, self.lr_iter, self.bsize_iter = self._parse_runs(self.all_runs)

        self.eval_return_points = np.linspace(*eval_threshold_range, num=num_eval_points)

        self.fig = Figure(self.utd_inv)

    @staticmethod
    def _load_runs(json_path: Path):
        with json_path.open("r") as f:
            return json.load(f)

    @staticmethod
    def _parse_runs(all_runs):
        def _find_common_left_right(pairs):
            l2r, r2l = defaultdict(set), defaultdict(set)
            for left, right in pairs:
                l2r[left].add(right)
                r2l[right].add(left)
            all_r, all_l = set(r2l), set(l2r)
            set_r = {right for right, lefts in r2l.items() if all_l <= lefts}
            set_l = {left for left, rights in l2r.items() if all_r <= rights}
            return set_l, set_r

        utd_set, bsize_set, lr_set, bsize_lr_pair_set = set(), set(), set(), set()
        for run in all_runs.values():
            utd_set.add(run["utd_ratio_inverse"])
            bsize_set.add(run["algo.batch_size"])
            lr_set.add(run["algo.actor_lr"])
            bsize_lr_pair_set.add((run["algo.batch_size"], run["algo.actor_lr"]))

        default_batch_size, default_lr = _find_common_left_right(bsize_lr_pair_set)
        lr_iter = {(bsize, lr) for lr in lr_set for bsize in default_batch_size}  # lr: variable, bsize: fixed
        bsize_iter = {(bsize, lr) for bsize in bsize_set for lr in default_lr}  # bsize: variable, lr: fixed

        return utd_set, lr_iter, bsize_iter

    def run(self):
        for utd_idx, utd_inv in enumerate(sorted(self.utd_inv)):
            minimum_data_list = []
            for lr_idx, (bsize, lr) in enumerate(self.lr_iter):
                runs = self._extract_runs(utd_inv, bsize, lr)
                series, minimum_data, max_return = self._get_minimum_data(runs)
                self.fig.plot_learning_curve("lr", series, utd_idx=utd_idx, metric_idx=lr_idx)
                minimum_data_list.append(minimum_data)
                self.fig.log("lr", utd_inv, lr, max_return, len(runs))
            # best_lr = choose_best(minimum_data_list)

            minimum_data_list = []
            for bsize_idx, (bsize, lr) in enumerate(self.bsize_iter):
                runs = self._extract_runs(utd_inv, bsize, lr)
                series, minimum_data, max_return = self._get_minimum_data(runs)
                self.fig.plot_learning_curve("batch_size", series, utd_idx=utd_idx, metric_idx=bsize_idx)
                minimum_data_list.append(minimum_data)
                self.fig.log("batch_size", utd_inv, bsize, max_return, len(runs))
            # best_bsize = choose_best(minimum_data_list)

        self.fig.plot_fit()

    def _extract_runs(self, utd_inv: float, batch_size: int, lr: float):
        def _is_target(run: dict):
            return run["utd_ratio_inverse"] == utd_inv and run["algo.batch_size"] == batch_size and run["algo.actor_lr"] == lr

        return [run for run in self.all_runs.values() if _is_target(run)]

    def _get_minimum_data(self, runs):
        x_list, y_list = [np.array(run["step"]) for run in runs], [np.array(run["eval_return"]) for run in runs]
        x, y, y_std = Fit.linear_average_series(x_list, y_list)
        y_isotonic = Fit.isotonic_fit(x, y)
        x_minimum = [self.get_inverse(x, y_isotonic, y_threshold) for y_threshold in self.eval_return_points]
        y_maximum = max(y_isotonic)

        return {"data": x, "return": y, "return_std": y_std, "return_isotonic": y_isotonic}, x_minimum, y_maximum

    @staticmethod
    def get_inverse(x, y_isotonic, y_threshold):
        if max(y_isotonic) < y_threshold:
            return None

        idx = np.where(y_isotonic >= y_threshold)[0][0]
        return x[idx]


class Fit:
    @staticmethod
    def linear_average_series(x_list: list, y_list: list):
        """Computes the linear average series for the given x and y lists.

        Args:
            x_list (list): A list of x values (steps).
            y_list (list): A list of y values (returns).

        Raises:
            ValueError: If x_list and y_list have different lengths.

        Returns:
            tuple: A tuple containing the interpolated x values and the corresponding mean and std of y values.
        """
        if len(x_list) != len(y_list):
            error_message = f"x_list ({len(x_list)}) and y_list ({len(y_list)}) must have the same length."
            raise ValueError(error_message)

        x_min, x_max = min(x.min() for x in x_list), max(x.max() for x in x_list)
        mean_data_points = np.mean([len(x) for x in x_list])

        x_interp = np.linspace(x_min, x_max, num=int(mean_data_points))
        y_interp_list = np.array([np.interp(x_interp, _x, _y) for _x, _y in zip(x_list, y_list)])

        return x_interp, y_interp_list.mean(axis=0), y_interp_list.std(axis=0)

    @staticmethod
    def isotonic_fit(x, y):
        return IsotonicRegression(increasing=True, y_min=0).fit_transform(x, y)

    @staticmethod
    def power_law_fit(utd_inv, metrics):
        """
        Fit a power law to the data and return the parameters.
        y = ret[1] * (x ** ret[0])
        """
        x, y = 1 / np.array(utd_inv), np.array(metrics)
        log_x = np.log(x)
        log_y = np.log(y)
        coeffs = np.polyfit(log_x, log_y, 1)
        return lambda x: np.exp(coeffs[1]) * (x ** coeffs[0]), f"y = {np.exp(coeffs[1])} * (x ** {coeffs[0]})"

    @staticmethod
    def generalized_power_law_fit(utd_inv, metrics):
        """
        Fit y = a * (1 + (x/b)^c) to the data and return the parameters and formula string.
        Handles invalid values to avoid RuntimeWarning.
        """
        x = 1 / np.array(utd_inv)
        y = np.array(metrics)

        def func(x, a, b, c):
            # Ensure b > 0 and x/b > 0 to avoid invalid values
            x = np.asarray(x)
            b = np.abs(b) + 1e-8  # avoid division by zero or negative
            return a * (1 + np.power(np.clip(x / b, 1e-8, None), c))

        # Initial guess: a=max(y), b=0.01, c=1
        popt, _ = curve_fit(func, x, y, p0=[y.max(), 0.01, 1.0], maxfev=100000)
        a, b, c = popt
        formula = f"y = {a} * (1 + (x/{b})^{c})"
        return lambda x: a * (1 + np.power(np.clip(x / (b if b > 0 else 1e-8), 1e-8, None), c)), formula


class Figure:
    def __init__(self, utd_inv_list: list):
        self.batch_size = {
            "fit": self.init_fit_plot("Batch Size"),
            "learning_curve": self.init_learning_curve_plot(utd_inv_list),
            "log": {"x": [], "y": [], "max_return": [], "num_seeds": []},
        }
        self.lr = {
            "fit": self.init_fit_plot("Learning Rate"),
            "learning_curve": self.init_learning_curve_plot(utd_inv_list),
            "log": {"x": [], "y": [], "max_return": [], "num_seeds": []},
        }
        self.colors = list(mcolors.TABLEAU_COLORS.values())

    def init_fit_plot(self, y_label: str):
        # utd vs { batch_size, learning_rate }
        fig, ax = plt.subplots()
        ax.set_xlabel("UTD")
        ax.set_ylabel(y_label)
        ax.set_title(f"UTD vs {y_label}")
        return fig, ax

    def init_learning_curve_plot(self, utd_inv_list: list):
        # data_step vs returns
        fig, ax = plt.subplots(1, len(utd_inv_list))
        for i, utd_inv in enumerate(utd_inv_list):
            ax[i].set_xlabel("Data Step")
            if i == 0:
                ax[i].set_ylabel("Returns")
            ax[i].set_title(f"Learning Curve (UTD: 1/{int(utd_inv)})")
        return fig, ax

    def plot_learning_curve(self, plot_type, series, utd_idx, metric_idx):
        x, y, y_std, y_isotonic = series["data"], series["return"], series["return_std"], series["return_isotonic"]
        fig, ax = getattr(self, plot_type)["learning_curve"]
        ax[utd_idx].plot(x, y, alpha=0.7, color=self.colors[metric_idx], label="Mean", linestyle="--")
        ax[utd_idx].fill_between(x, y - y_std, y + y_std, alpha=0.2, color=self.colors[metric_idx])
        ax[utd_idx].plot(x, y_isotonic, color=self.colors[metric_idx])

    def log(self, plot_type, x, y, max_return, num_seeds):
        log = getattr(self, plot_type)["log"]
        for key, value in zip(["x", "y", "max_return", "num_seeds"], [x, y, max_return, num_seeds]):
            log[key].append(value)

    def plot_fit(self):
        for plot_type in ["batch_size", "lr"]:
            fig, ax = getattr(self, plot_type)["fit"]
            log = getattr(self, plot_type)["log"]
            ax.scatter(log["x"], log["y"], c=log["max_return"], cmap="viridis")

    def save(self, base_dir: Path):
        base_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size["fit"][0].savefig(base_dir / "batch_size_fit.png")
        self.batch_size["learning_curve"][0].savefig(base_dir / "batch_size_learning_curve.png")
        self.lr["fit"][0].savefig(base_dir / "learning_rate_fit.png")
        self.lr["learning_curve"][0].savefig(base_dir / "learning_rate_learning_curve.png")

        plt.close("all")


def main():
    args = get_args()

    base_dir = get_base_dir(args.wandb_project, args.out_dir)
    json_path = base_dir / "runs-step1.json"
    if not args.skip_load:
        read_wandb_runs(entity=args.wandb_entity, project=args.wandb_project, save_path=json_path)
    else:
        print(f"Use existing runs from {json_path}")

    # analyzer = Analyzer(json_path, eval_threshold_range=(0.0, 9000.0)).run()
    # analyzer.fig.save(base_dir)


if __name__ == "__main__":
    main()
