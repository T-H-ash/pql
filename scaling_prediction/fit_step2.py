import argparse
import importlib
import json
import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
from fit_step1 import Fit, get_base_dir, read_wandb_runs
from matplotlib import colors as mcolors

dotenv.load_dotenv()


ELAPSED_TIME_THRESHOLD = 3400


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_load", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT"))

    parser.add_argument("--out_dir", type=Path, default="./runs")
    parser.add_argument("--step1_fit_filename", type=str, default="step1_fit_result")
    parser.add_argument("--wandb_entity", type=str, default=os.getenv("WANDB_ENTITY"))
    return parser.parse_args()


def get_fit_functions(base_dir: Path, step1_fit_filename: str):
    module_path = base_dir / step1_fit_filename
    return importlib.import_module(str(module_path).replace("/", "."))


class Analyzer:
    def __init__(self, json_path: Path, step1_fit, eval_threshold_range: tuple, num_eval_points: int):
        self.all_runs = self._load_runs(json_path)
        self.utd_inv = self._parse_runs(self.all_runs)

        self.step1_fit = step1_fit
        self.eval_return_points = np.linspace(*eval_threshold_range, num=num_eval_points)

        self.fig = Figure(self.eval_return_points)

    @staticmethod
    def _load_runs(json_path: Path):
        with json_path.open("r") as f:
            return json.load(f)

    @staticmethod
    def _parse_runs(all_runs):
        utd_set = set()
        for run in all_runs.values():
            utd_set.add(run["utd_ratio_inverse"])

        return sorted(utd_set, reverse=True)

    def run_minimum_data_fit(self):
        for utd_idx, utd_inv in enumerate(self.utd_inv):
            runs = self._extract_runs(utd_inv)
            series, minimum_data, _ = self._get_minimum_data(runs)
            self.fig.plot_learning_curve(series, utd_inv, utd_idx)
            self.fig.log(utd_inv, minimum_data)

        utd = [1 / utd_inv for utd_inv in self.fig.data["log"]["utd_inv"]]
        minimum_data = self.fig.data["log"]["minimum_data"]
        func_dict = {}
        for eval_idx, reward in enumerate(self.eval_return_points):
            data_func = Fit.power_law_fit(utd=utd, metrics=[min_data[eval_idx] for min_data in minimum_data])
            print(f"Reward: {reward}, Function: {data_func['func_str']}")
            func_dict[reward] = data_func["func"]

        self.fig.plot_data_fit(func_dict)

        # Use the batch_size_function from step1_fit
        batch_size_function = self.step1_fit.batch_size_function
        self.fig.plot_compute_vs_data_fit(func_dict, batch_size_function)

    def _extract_runs(self, utd_inv: float):
        def _is_target(run: dict):
            return run["utd_ratio_inverse"] == utd_inv

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


class Figure:
    def __init__(self, eval_return_points: list):
        self.learning_curve = self.init_learning_curve_fit_plot()
        self.data = {
            "fit": self.init_data_fit_plot(),
            "log": {"utd_inv": [], "minimum_data": []},
        }
        self.compute_vs_data = {
            "fit": self.init_compute_vs_data_fit_plot(),
            "log": {},
        }
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.eval_return_points = eval_return_points

    def init_learning_curve_fit_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.set(xlabel="Data Step", ylabel="Returns", title="Learning Curve")
        return fig, ax

    def init_data_fit_plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].set(
            xlabel="UTD",
            ylabel="Minimum Data Step",
            title="UTD vs Minimum Data Step (Scale: Linear)",
        )
        ax[1].set(
            xlabel="UTD",
            ylabel="Minimum Data Step",
            title="UTD vs Minimum Data Step (Scale: LogLog)",
        )
        return fig, ax

    def init_compute_vs_data_fit_plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        ax[0].set(xlabel="Compute", ylabel="Data", title="Compute vs Data (Scale: Linear)")
        ax[1].set(xlabel="Compute", ylabel="Data", title="Compute vs Data (Scale: LogLog)")
        return fig, ax

    def log(self, utd_inv, minimum_data):
        log = self.data["log"]
        for key, value in zip(["utd_inv", "minimum_data"], [utd_inv, minimum_data]):
            log[key].append(value)

    def plot_learning_curve(self, series, utd_inv, utd_idx):
        x, y, y_std, y_isotonic = (series["data"], series["return"], series["return_std"], series["return_isotonic"])
        fig, ax = self.learning_curve
        ax.plot(x, y, alpha=0.5, linestyle="--", color=self.colors[utd_idx])
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.1, color=self.colors[utd_idx])
        ax.plot(x, y_isotonic, color=self.colors[utd_idx], label=f"UTD: 1 / {int(utd_inv)}")
        ax.yaxis.grid(visible=True, linestyle="--", color="gray", linewidth=1)
        ax.legend()

    def plot_data_fit(self, func_dict):
        fig, ax = self.data["fit"]
        log = self.data["log"]
        utd = np.repeat(1 / np.array(log["utd_inv"]), len(self.eval_return_points), axis=0)
        minimum_data = np.concatenate([np.array(log["minimum_data"][i]) for i in range(len(log["utd_inv"]))])
        return_threshold = np.tile(np.array(self.eval_return_points), len(log["utd_inv"]))
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(self.eval_return_points), vmax=max(self.eval_return_points))

        # Scatter plots and colorbars
        sc0 = ax[0].scatter(utd, minimum_data, c=return_threshold, cmap=cmap, norm=norm)
        sc1 = ax[1].scatter(utd, minimum_data, c=return_threshold, cmap=cmap, norm=norm)
        fig.colorbar(sc0, ax=ax[0], label="Return")
        fig.colorbar(sc1, ax=ax[1], label="Return")

        # Fit lines
        utd_min, utd_max = utd.min(), utd.max()
        x = np.linspace(utd_min, utd_max, 100)
        for reward, func in func_dict.items():
            if func is None:
                continue
            color = cmap(norm(reward))
            ax[0].plot(x, func(x), linestyle="--", zorder=4, color=color, linewidth=2)
            ax[1].plot(x, func(x), linestyle="--", zorder=4, color=color, linewidth=2)
        ax[1].set_xscale("log")
        ax[1].set_yscale("log")

    def plot_compute_vs_data_fit(self, func_dict, batch_size_function, model_size=179592):
        """
        Plots compute (CJ) vs data for each reward threshold.
        ax[0]: linear scale, ax[1]: log-log scale
        x-axis: minimum_data
        y-axis: compute (CJ)
        Uses continuous color mapping for reward thresholds and adds colorbar. Legends removed for clarity.
        """
        fig, ax = self.compute_vs_data["fit"]
        log = self.data["log"]
        utd_arr = 1 / np.array(log["utd_inv"])
        min_data_arr = np.array(log["minimum_data"])
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(self.eval_return_points), vmax=max(self.eval_return_points))
        print("CJ = 10 * MODEL_SIZE * batch_size_function(utd) * utd * minimum_data")

        # Prepare scatter and curve data
        all_x, all_y, all_c, curves = [], [], [], []
        utds = np.linspace(utd_arr.min(), utd_arr.max(), 100)
        for i, reward in enumerate(self.eval_return_points):
            func = func_dict.get(reward)
            if func is None:
                continue
            color = cmap(norm(reward))

            # Discrete points
            for utd, min_data in zip(utd_arr, min_data_arr[:, i]):
                if min_data is None:
                    continue
                cj = 10 * model_size * batch_size_function(utd) * utd * min_data
                all_x.append(min_data)
                all_y.append(cj)
                all_c.append(reward)

            # Fitted curve
            min_curve = func(utds)
            cj_curve = 10 * model_size * batch_size_function(utds) * utds * min_curve
            curves.append((min_curve, cj_curve, color))

        # Scatter plot and colorbar
        for j in [0, 1]:
            sc = ax[j].scatter(all_x, all_y, c=all_c, cmap=cmap, norm=norm)
            fig.colorbar(sc, ax=ax[j], label="Return Threshold")
            for min_curve, cj_curve, color in curves:
                ax[j].plot(min_curve, cj_curve, linestyle="-", color=color, alpha=0.7)
        ax[0].set(xlabel="Minimum Data Step", ylabel="Compute (CJ)", title="Compute vs Data (Linear)")
        ax[1].set(
            xlabel="Minimum Data Step",
            ylabel="Compute (CJ)",
            title="Compute vs Data (LogLog)",
            xscale="log",
            yscale="log",
        )

    def set_lims(self, xlim=None, ylim=None):
        fig, axes = self.compute_vs_data["fit"]
        for ax in axes:
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

    def save(self, base_dir: Path, *, filename_prefix=""):
        base_dir.mkdir(parents=True, exist_ok=True)

        self.learning_curve[0].savefig(base_dir / f"{filename_prefix}step2-learning_curve.png")
        self.data["fit"][0].savefig(base_dir / f"{filename_prefix}step2-data_fit.png")
        self.compute_vs_data["fit"][0].savefig(base_dir / f"{filename_prefix}step2-compute_vs_data_fit.png")

        plt.close("all")


def main():
    args = get_args()
    base_dir = get_base_dir(args.wandb_project, args.out_dir)
    json_path = base_dir / "runs-step2.json"
    if not args.skip_load:
        read_wandb_runs(
            entity=args.wandb_entity,
            project=args.wandb_project,
            save_path=json_path,
            elapsed_time_threshold=ELAPSED_TIME_THRESHOLD,
        )
    else:
        print(f"Use existing runs from {json_path}")

    step1_fit = get_fit_functions(base_dir, args.step1_fit_filename)
    analyzer = Analyzer(json_path, step1_fit, eval_threshold_range=(3000.0, 12000.0), num_eval_points=30)
    analyzer.run_minimum_data_fit()

    analyzer.fig.save(base_dir)


def batch_run():
    WANDB_PROJECTS = [
        "scales_01-task=Ant-buffsize=1M-num_envs=1024-per=none-step2",
        "scales_01-task=Ant-buffsize=1M-num_envs=4096-per=none-step2",
        "scales_01-task=Ant-buffsize=4M-num_envs=4096-per=none-step2",
        "scales_01-task=Ant-buffsize=4M-num_envs=16384-per=none-step2",
    ]

    args = get_args()

    for wandb_project in WANDB_PROJECTS:
        base_dir = get_base_dir(wandb_project, args.out_dir)
        json_path = base_dir / "runs-step2.json"
        if not args.skip_load:
            read_wandb_runs(
                entity=args.wandb_entity,
                project=wandb_project,
                save_path=json_path,
                elapsed_time_threshold=ELAPSED_TIME_THRESHOLD,
            )
        else:
            print(f"Use existing runs from {json_path}")

        step1_fit = get_fit_functions(base_dir, args.step1_fit_filename)
        analyzer = Analyzer(json_path, step1_fit, eval_threshold_range=(3000.0, 10000.0), num_eval_points=200)
        analyzer.run_minimum_data_fit()

        analyzer.fig.set_lims(xlim=(7e5, 8e8), ylim=(5e13, 1e17))
        # analyzer.fig.set_lims(xlim=(1e7, 1e8), ylim=(1e14, 1e16))
        analyzer.fig.save(base_dir, filename_prefix="lim_")


if __name__ == "__main__":
    # main()
    batch_run()
