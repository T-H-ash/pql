import argparse
import json
import os
import random
from collections import Counter, defaultdict
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


def read_wandb_runs(entity, project, save_path: Path, elapsed_time_threshold: int):
    api = wandb.Api()
    if not _project_exists(entity, project):
        print(f"Project '{project}' does not exist in entity '{entity}'.")
        return

    runs = api.runs(f"{entity}/{project}")

    successful_runs_dict = {}
    for run in tqdm(runs):
        shell_script_name = run.config.get("logging", {}).get("shell_script_name", None)
        runtime = [step["_runtime"] for step in run.history(keys=["_runtime"], samples=100, pandas=False)]
        if runtime and max(runtime) < elapsed_time_threshold:
            continue

        elapsed_time = [step["elapsed_time"] for step in run.history(keys=["elapsed_time"], samples=100, pandas=False)]
        if not elapsed_time or max(elapsed_time) < ELAPSED_TIME_THRESHOLD:
            continue

        try:
            config = _get_wandb_config(run.config)
        except ConfigKeyError as e:
            print(
                f"Skipping {shell_script_name} due to an error while retrieving config: {e}",
            )
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

        successful_runs_dict[run.id] = {
            **config,
            "eval_return": eval_returns,
            "step": global_steps,
        }
    print(
        f"Successfully listed {len(successful_runs_dict)} runs in project '{project}'.",
    )

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
    keys = [
        "task.name",
        "num_envs",
        "seed",
        "algo.batch_size",
        "algo.actor_lr",
        "algo.critic_sample_ratio",
        "algo.pal",
    ]
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
    def __init__(
        self,
        json_path: Path,
        eval_threshold_range: tuple,
        num_eval_points: int = 100,
    ):
        self.all_runs = self._load_runs(json_path)
        self.utd_inv, self.lr_iter, self.bsize_iter = self._parse_runs(self.all_runs)

        self.eval_return_points = np.linspace(
            *eval_threshold_range,
            num=num_eval_points,
        )

        self.fig = Figure(self.utd_inv, self.eval_return_points)

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
            return (
                set_l.pop(),
                set_r.pop(),
            )  # FIXME: assumes only one common left and right

        utd_set, bsize_set, lr_set, bsize_lr_pair_set = set(), set(), set(), set()
        for run in all_runs.values():
            utd_set.add(run["utd_ratio_inverse"])
            bsize_set.add(run["algo.batch_size"])
            lr_set.add(run["algo.actor_lr"])
            bsize_lr_pair_set.add((run["algo.batch_size"], run["algo.actor_lr"]))

        default_batch_size, default_lr = _find_common_left_right(bsize_lr_pair_set)
        lr_iter = list(
            {(default_batch_size, lr) for lr in sorted(lr_set)},
        )  # lr: variable, bsize: fixed
        bsize_iter = list(
            {(bsize, default_lr) for bsize in sorted(bsize_set)},
        )  # bsize: variable, lr: fixed

        return sorted(utd_set, reverse=True), lr_iter, bsize_iter

    def run_point_estimate(self):
        for utd_idx, utd_inv in enumerate(self.utd_inv):
            minimum_data_list = []
            for lr_idx, (bsize, lr) in enumerate(self.lr_iter):
                runs = self._extract_runs(utd_inv, bsize, lr)
                if not runs:
                    breakpoint()
                series, minimum_data, max_return = self._get_minimum_data(runs)
                self.fig.plot_learning_curve(
                    "lr",
                    series,
                    utd_idx=utd_idx,
                    metric_idx=lr_idx,
                    metric=lr,
                )
                minimum_data_list.append(minimum_data)
                self.fig.log("lr", utd_inv, lr, max_return, len(runs))
            self.fig.log_best(
                "lr",
                (
                    1 / utd_inv,
                    self.choose_best(minimum_data_list, list(zip(*self.lr_iter))[1]),
                ),
            )

            minimum_data_list = []
            for bsize_idx, (bsize, lr) in enumerate(self.bsize_iter):
                runs = self._extract_runs(utd_inv, bsize, lr)
                if not runs:
                    breakpoint()
                series, minimum_data, max_return = self._get_minimum_data(runs)
                self.fig.plot_learning_curve(
                    "batch_size",
                    series,
                    utd_idx=utd_idx,
                    metric_idx=bsize_idx,
                    metric=bsize,
                )
                minimum_data_list.append(minimum_data)
                self.fig.log("batch_size", utd_inv, bsize, max_return, len(runs))
            self.fig.log_best(
                "batch_size",
                (
                    1 / utd_inv,
                    self.choose_best(minimum_data_list, list(zip(*self.bsize_iter))[0]),
                ),
            )

        self.fig.plot_fit_point()
        self.fig.plot_fit_best()

    def run_bootstrap(self, num_bootstrap):
        for utd_inv in self.utd_inv:
            best_lrs = []
            for _ in range(num_bootstrap):
                minimum_data_list = []
                for bsize, lr in self.lr_iter:
                    _runs = self._extract_runs(utd_inv, bsize, lr)
                    runs = random.choices(_runs, k=len(_runs))
                    _, minimum_data, _ = self._get_minimum_data(runs)
                    minimum_data_list.append(minimum_data)
                best_lrs.append(
                    self.choose_best(minimum_data_list, list(zip(*self.lr_iter))[1]),
                )
            self.fig.log_best("lr", (1 / utd_inv, np.mean(best_lrs)), bootstrap=True)

            best_bsizes = []
            for _ in range(num_bootstrap):
                minimum_data_list = []
                for bsize, lr in self.bsize_iter:
                    _runs = self._extract_runs(utd_inv, bsize, lr)
                    runs = random.choices(_runs, k=len(_runs))
                    _, minimum_data, _ = self._get_minimum_data(runs)
                    minimum_data_list.append(minimum_data)
                best_bsizes.append(
                    self.choose_best(minimum_data_list, list(zip(*self.bsize_iter))[0]),
                )
            self.fig.log_best(
                "batch_size",
                (1 / utd_inv, np.mean(best_bsizes)),
                bootstrap=True,
            )

        self.fig.plot_fit_best(bootstrap=True)

        bsize_func = Fit.power_law_fit(
            *zip(*self.fig.batch_size["log"]["best_bootstrap"]),
        )
        lr_func = Fit.power_law_fit(*zip(*self.fig.lr["log"]["best_bootstrap"]))

        func = {"lr": lr_func, "batch_size": bsize_func}
        self.fig.plot_fit_func(func)

        return func

    def _extract_runs(self, utd_inv: float, batch_size: int, lr: float):
        def _is_target(run: dict):
            return run["utd_ratio_inverse"] == utd_inv and run["algo.batch_size"] == batch_size and run["algo.actor_lr"] == lr

        return [run for run in self.all_runs.values() if _is_target(run)]

    def _get_minimum_data(self, runs):
        x_list, y_list = (
            [np.array(run["step"]) for run in runs],
            [np.array(run["eval_return"]) for run in runs],
        )
        x, y, y_std = Fit.linear_average_series(x_list, y_list)
        y_isotonic = Fit.isotonic_fit(x, y)
        x_minimum = [self.get_inverse(x, y_isotonic, y_threshold) for y_threshold in self.eval_return_points]
        y_maximum = max(y_isotonic)

        return (
            {
                "data": x,
                "return": y,
                "return_std": y_std,
                "return_isotonic": y_isotonic,
            },
            x_minimum,
            y_maximum,
        )

    @staticmethod
    def get_inverse(x, y_isotonic, y_threshold):
        if max(y_isotonic) < y_threshold:
            return None

        idx = np.where(y_isotonic >= y_threshold)[0][0]
        return x[idx]

    @staticmethod
    def choose_best(_minimum_data_list, metrics_list):
        minimum_data_list = np.array(_minimum_data_list, dtype=object)
        minimum_data_list = np.where(
            np.equal(minimum_data_list, None),
            np.inf,
            minimum_data_list,
        )
        counter = Counter(minimum_data_list.argmin(axis=0))
        best_idx = counter.most_common(1)[0][0]
        return metrics_list[best_idx]


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
        y_interp_list = np.array(
            [np.interp(x_interp, _x, _y) for _x, _y in zip(x_list, y_list)],
        )

        return x_interp, y_interp_list.mean(axis=0), y_interp_list.std(axis=0)

    @staticmethod
    def isotonic_fit(x, y):
        return IsotonicRegression(increasing=True, y_min=0).fit_transform(x, y)

    @staticmethod
    def power_law_fit(utd, metrics):
        """
        Fit a power law to the data and return the parameters.
        y = ret[1] * (x ** ret[0])
        """
        x, y = np.array(utd), np.array(metrics, dtype=float)
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]
        log_x, log_y = np.log(x), np.log(y)

        if len(log_x) <= 2:  # noqa: PLR2004
            return {"func": None, "func_str": None}

        coeffs = np.polyfit(log_x, log_y, 1)
        return {
            "func": lambda x: np.exp(coeffs[1]) * (x ** coeffs[0]),
            "func_str": f"y = {np.exp(coeffs[1])} * (x ** {coeffs[0]})",
        }

    @staticmethod
    def generalized_power_law_fit(utd, metrics):
        """
        Fit y = C + A * x^B to the data using non-linear least squares.

        Returns a dict with keys "func" (callable) and "func_str" (human string).
        If fitting fails or there is insufficient data, returns {"func": None, "func_str": None}.
        """
        x = np.array(utd, dtype=float)
        y = np.array(metrics, dtype=float)

        # remove nan entries
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 3:
            return {"func": None, "func_str": None}

        def model(xx, c, a, b):
            xx_pos = np.clip(xx, 1e-12, None)
            return c + a * np.power(xx_pos, b)

        # sensible initial guesses
        c0 = float(np.min(y))
        a0 = float(np.max(y) - c0)
        b0 = 1.0

        try:
            popt, _ = curve_fit(model, x, y, p0=[c0, a0, b0], maxfev=100000)
        except Exception:
            return {"func": None, "func_str": None}

        c, a, b = popt
        func_str = f"y = {c} + {a} * x**{b}"
        return {"func": lambda xx: c + a * np.power(np.clip(xx, 1e-12, None), b), "func_str": func_str}


class Figure:
    def __init__(self, utd_inv_list: list, eval_return_points: list):
        self.batch_size = {
            "fit": self.init_fit_plot("Batch Size"),
            "learning_curve": self.init_learning_curve_plot(utd_inv_list),
            "log": {
                "x_inv": [],
                "y": [],
                "max_return": [],
                "num_seeds": [],
                "best_point": [],
                "best_bootstrap": [],
            },
        }
        self.lr = {
            "fit": self.init_fit_plot("Learning Rate"),
            "learning_curve": self.init_learning_curve_plot(utd_inv_list),
            "log": {
                "x_inv": [],
                "y": [],
                "max_return": [],
                "num_seeds": [],
                "best_point": [],
                "best_bootstrap": [],
            },
        }
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.eval_range = min(eval_return_points), max(eval_return_points)

    def init_fit_plot(self, y_label: str):
        # utd vs { batch_size, learning_rate }
        fig, ax = plt.subplots()
        ax.set_xlabel("UTD")
        ax.set_ylabel(y_label)
        ax.set_title(f"UTD vs {y_label}")
        return fig, ax

    def init_learning_curve_plot(self, utd_inv_list: list):
        # data_step vs returns
        fig, ax = plt.subplots(
            1,
            len(utd_inv_list),
            figsize=(7 * len(utd_inv_list), 5),
            sharex=True,
            sharey=True,
        )
        for i, utd_inv in enumerate(utd_inv_list):
            ax[i].set_xlabel("Data Step")
            if i == 0:
                ax[i].set_ylabel("Returns")
            ax[i].set_title(f"Learning Curve (UTD: 1/{int(utd_inv)})")
        return fig, ax

    def plot_learning_curve(self, plot_type, series, utd_idx, metric_idx, metric):
        x, y, y_std, y_isotonic = (
            series["data"],
            series["return"],
            series["return_std"],
            series["return_isotonic"],
        )
        fig, ax = getattr(self, plot_type)["learning_curve"]
        ax[utd_idx].plot(x, y, alpha=0.5, color=self.colors[metric_idx], linestyle="--")
        ax[utd_idx].fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.1,
            color=self.colors[metric_idx],
        )
        ax[utd_idx].plot(
            x,
            y_isotonic,
            color=self.colors[metric_idx],
            label=f"{plot_type}: {metric}",
        )
        ax[utd_idx].yaxis.grid(visible=True, linestyle="--", color="gray", linewidth=1)
        ax[utd_idx].legend()

        if metric_idx == 0:
            ax[utd_idx].axhspan(*self.eval_range, alpha=0.1, color="gray", zorder=-1)

    def log(self, plot_type, x_inv, y, max_return, num_seeds):
        log = getattr(self, plot_type)["log"]
        for key, value in zip(
            ["x_inv", "y", "max_return", "num_seeds"],
            [x_inv, y, max_return, num_seeds],
        ):
            log[key].append(value)

    def log_best(self, plot_type, best_metrics_tuple, *, bootstrap: bool = False):
        bets_key = "best_bootstrap" if bootstrap else "best_point"
        getattr(self, plot_type)["log"][bets_key].append(best_metrics_tuple)

    def plot_fit_point(self):
        for plot_type in ["batch_size", "lr"]:
            fig, ax = getattr(self, plot_type)["fit"]
            log = getattr(self, plot_type)["log"]
            sc = ax.scatter(
                1 / np.array(log["x_inv"]),
                np.array(log["y"]),
                c=np.array(log["max_return"]),
                cmap="viridis",
                edgecolor="black",
                linewidths=1,
                s=144,
                zorder=3,
            )
            for x in set(log["x_inv"]):
                ax.axvline(
                    x=1 / x,
                    color="black",
                    linestyle="dotted",
                    linewidth=1,
                    zorder=0,
                )
            for y in set(log["y"]):
                ax.axhline(
                    y=y,
                    color="black",
                    linestyle="dotted",
                    linewidth=1,
                    zorder=0,
                )
            ax.set_xticks([1 / x for x in log["x_inv"]])
            ax.set_xticklabels([f"1/{int(x)}" for x in log["x_inv"]])
            fig.colorbar(sc, ax=ax, label="Max Return")

    def plot_fit_best(self, *, bootstrap: bool = False):
        for plot_type in ["batch_size", "lr"]:
            fig, ax = getattr(self, plot_type)["fit"]
            log = getattr(self, plot_type)["log"]
            if not bootstrap:
                ax.scatter(
                    *zip(*log["best_point"]),
                    c="blue",
                    s=432,
                    zorder=1,
                    alpha=0.5,
                    label="Point Estimate",
                )
            else:
                ax.scatter(
                    *zip(*log["best_bootstrap"]),
                    c="red",
                    s=432,
                    zorder=2,
                    alpha=0.5,
                    label="Bootstrap Estimate",
                )

    def plot_fit_func(self, func):
        for plot_type in ["batch_size", "lr"]:
            fig, ax = getattr(self, plot_type)["fit"]
            utd_inv = getattr(self, plot_type)["log"]["x_inv"]
            utd_min, utd_max = 1 / max(utd_inv), 1 / min(utd_inv)
            x = np.linspace(utd_min, utd_max, 100)
            ax.plot(
                x,
                func[plot_type]["func"](x),
                linestyle="--",
                zorder=4,
                color="orange",
                linewidth=2,
                label="Fit Function",
            )
            ax.set_title(ax.get_title() + f"\n{func[plot_type]['func_str']}")

    def save(self, base_dir: Path):
        base_dir.mkdir(parents=True, exist_ok=True)

        for plot_type in ["batch_size", "lr"]:
            fig, ax = getattr(self, plot_type)["learning_curve"]
            fig.savefig(base_dir / f"step1-{plot_type}_learning_curve.png")

            fig, ax = getattr(self, plot_type)["fit"]
            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
                frameon=False,
            )
            fig.tight_layout()
            fig.savefig(
                base_dir / f"step1-{plot_type}_fit.png",
                bbox_extra_artists=(legend,),
                bbox_inches="tight",
            )

        plt.close("all")


def write_function_to_python_file(func, file_path: Path):
    """
    Write the function to a Python file.
    """
    with file_path.open("w") as f:
        f.write("def batch_size_function(x):\n")
        f.write(f"    {func['batch_size']['func_str'].replace('y = ', 'return ')}\n")
        f.write("\n")
        f.write("def learning_rate_function(x):\n")
        f.write(f"    {func['lr']['func_str'].replace('y = ', 'return ')}\n")


def main():
    args = get_args()

    base_dir = get_base_dir(args.wandb_project, args.out_dir)
    json_path = base_dir / "runs-step1.json"
    if not args.skip_load:
        read_wandb_runs(
            entity=args.wandb_entity,
            project=args.wandb_project,
            save_path=json_path,
            elapsed_time_threshold=ELAPSED_TIME_THRESHOLD,
        )
    else:
        print(f"Use existing runs from {json_path}")

    analyzer = Analyzer(json_path, eval_threshold_range=(3000.0, 12000.0))
    analyzer.run_point_estimate()
    func = analyzer.run_bootstrap(num_bootstrap=100)
    analyzer.fig.save(base_dir)

    write_function_to_python_file(func, base_dir / "step1_fit_result.py")


if __name__ == "__main__":
    main()
