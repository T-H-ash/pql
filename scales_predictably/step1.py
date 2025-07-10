import argparse
import json
import math
import secrets
from pathlib import Path
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression

import wandb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=None, help="wandb entity (user or team)")
    parser.add_argument("--use_bootstrapping", action="store_true", help="Use bootstrap sampling for runs (default: False)")
    return parser.parse_args()


def extract_wandb_runs(project, entity, save_path: Path):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    runs_dict = {}

    for run in runs:
        try:
            num_envs = run.config.get("num_envs", None)
            seed = run.config.get("seed", None)

            config_algo = run.config.get("algo", {})
            batch_size = config_algo.get("batch_size", None)
            actor_lr = config_algo.get("actor_lr", None)
            critic_ratio = config_algo.get("critic_sample_ratio", None)
            use_pal = config_algo.get("pal", None)

            utd_ratio_inverse = num_envs / critic_ratio

        except Exception as e:
            print(f"Error accessing config for run {run.id}: {e}")

        if run.state == "running":
            continue

        print(
            f"Run ID: {run.id}, Name: {run.name:<25}, State: {run.state:<10} | "
            f"batch_size: {batch_size:<5} | actor_lr: {actor_lr} | "
            f"seed={seed} | critic_sample_ratio={critic_ratio} | "
            f"utd_ratio_inverse={utd_ratio_inverse}",
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

    with save_path.open("w") as f:
        json.dump(runs_dict, f, indent=4)


class Analyzer:
    UTD_INV = [2048, 4096, 8192, 16384]
    BATCH_SIZE = [(512, 0.0002), (1024, 0.0002), (2048, 0.0002), (4096, 0.0002), (8192, 0.0002)]
    LR = [(2048, 0.0001), (2048, 0.0002), (2048, 0.0003)]

    NUM_REQUIRED_SEEDS = 4
    NUM_BOOSTING_ITERATIONS = 100
    REWARD_THRESHOLD = [1000 * i for i in range(1, 11)]

    def __init__(self, project):
        self.project = project
        self.runs_dict = self.load_runs()
        self.ir = IsotonicRegression(increasing=True, y_min=0)

    def load_runs(self):
        with Path(f"{self.project}_runs.json").open("r") as f:
            return json.load(f)

    def extract(self, utd_ratio_inverse: int, batch_size: int, lr: float, *, use_bootstrapping: bool):
        def _is_target(data):
            dc = data["config"]
            return dc["utd_ratio_inverse"] == utd_ratio_inverse and dc["batch_size"] == batch_size and dc["lr"] == lr

        extracted_runs = [run_data for run_data in self.runs_dict.values() if _is_target(run_data)]
        num_seeds = len(extracted_runs)

        if not use_bootstrapping:
            return extracted_runs, num_seeds

        bootstrap_sampled_runs = [secrets.choice(extracted_runs) for _ in range(num_seeds)]
        return bootstrap_sampled_runs, num_seeds

    def run(self, *, _use_bootstrapping: bool):
        best_bs_wo_bootstrapping = self.analyze(iterator_key="batch_size", use_bootstrapping=False)
        best_bs_w_bootstrapping = self.analyze(iterator_key="batch_size", use_bootstrapping=True)
        fit_func_bs = self.power_law_fit(
            list(best_bs_w_bootstrapping.keys()),
            list(best_bs_w_bootstrapping.values()),
        )
        self.plot_best_params(
            utd_inv=list(best_bs_wo_bootstrapping.keys()),
            best_metrics={
                "best batch size (point)": best_bs_wo_bootstrapping.values(),
                "best batch size (bootstrapping)": best_bs_w_bootstrapping.values(),
            },
            fit_func=fit_func_bs,
            metric_name="batch_size",
        )

        best_lr_wo_bootstrapping = self.analyze(iterator_key="lr", use_bootstrapping=False)
        best_lr_w_bootstrapping = self.analyze(iterator_key="lr", use_bootstrapping=True)
        fit_func_lr = self.power_law_fit(
            list(best_lr_w_bootstrapping.keys()),
            list(best_lr_w_bootstrapping.values()),
        )
        self.plot_best_params(
            utd_inv=list(best_lr_wo_bootstrapping.keys()),
            best_metrics={
                "best lr (point)": best_lr_wo_bootstrapping.values(),
                "best lr (bootstrapping)": best_lr_w_bootstrapping.values(),
            },
            fit_func=fit_func_lr,
            metric_name="lr",
        )

        self.save_best_params(fit_func_bs=fit_func_bs, fit_func_lr=fit_func_lr)

    def analyze(self, iterator_key: Literal["batch_size", "lr"], *, use_bootstrapping: bool):
        # Select the parameter iterator based on the key
        if iterator_key == "batch_size":
            iterator = self.BATCH_SIZE
        elif iterator_key == "lr":
            iterator = self.LR
        else:
            error_mesage = f"Unknown iterator key: {iterator_key}"
            raise ValueError(error_mesage)

        ret_dict = {}  # utd_inv: best_metric (batch_size or lr)
        for utd_inv in self.UTD_INV:
            data_requirements, best_metrics = {}, []
            for _ in range(self.NUM_BOOSTING_ITERATIONS if use_bootstrapping else 1):
                for batch_size, lr in iterator:
                    runs, num_seeds = self.extract(utd_inv, batch_size, lr, use_bootstrapping=use_bootstrapping)
                    key = batch_size if iterator_key == "batch_size" else lr
                    if not use_bootstrapping and num_seeds < self.NUM_REQUIRED_SEEDS:
                        print(
                            f"Not enough runs for UTD_INV={utd_inv}, BSize={batch_size}, LR={lr}: {num_seeds} runs found.",
                        )
                        continue

                    plot_path = None
                    if not use_bootstrapping:
                        plot_path = Path(self.project) / f"analysis_{utd_inv}_{lr}_{batch_size}.png"
                        plot_path.parent.mkdir(parents=True, exist_ok=True)

                    data_requirements[key] = self.get_data_requirement_with_plot(runs, save_path=plot_path)

                best_metric = self.choose_best(data_requirements)
                best_metrics.append(best_metric)
                print(f"Best {iterator_key} for UTD_INV={utd_inv}: {iterator_key}={best_metric}")

            # If multiple iterations (bootstrap), take the average if numeric, else majority
            if use_bootstrapping and isinstance(best_metrics[0], (int, float)):
                ret_dict[utd_inv] = sum(best_metrics) / len(best_metrics)
            else:
                # For non-numeric (e.g., categorical), take the most frequent
                ret_dict[utd_inv] = best_metrics[0]

        return ret_dict

    @staticmethod
    def interpolation_mean(runs):
        x_list = [np.array(run["_step"]) for run in runs]
        y_list = [np.array(run["eval_returns"]) for run in runs]
        return average_series(x_list, y_list)

    def get_data_requirement_with_plot(self, runs, save_path=None):
        """
        ret_dict: { th: min x where isotonic_series[x] >= th for th }
        If save_path is given, save the figure to that path instead of showing it.
        If save_path is None, do not plot at all.
        """
        x_common, y_mean, y_std = self.interpolation_mean(runs)
        isotonic_series = self.ir.fit_transform(x_common, y_mean)
        max_reward = isotonic_series.max()
        ret_dict = {}

        # Only plot if save_path is specified
        if save_path is not None:
            plt.figure(figsize=(10, 6))

            # Plot mean reward with std deviation as a band (like wandb style)
            plt.plot(x_common, y_mean, label="Mean Reward", color="tab:blue")
            plt.fill_between(
                x_common,
                y_mean - y_std,
                y_mean + y_std,
                color="tab:blue",
                alpha=0.2,
                label="Std Dev",
            )

            # Plot isotonic regression curve
            plt.plot(
                x_common,
                isotonic_series,
                label="Isotonic Regression",
                color="tab:orange",
            )

            # For each threshold, plot a black dot and dotted lines to axes
            for threshold in Analyzer.REWARD_THRESHOLD:
                if max_reward >= threshold:
                    idx = np.where(isotonic_series >= threshold)[0][0]
                    x_val = x_common[idx]
                    y_val = isotonic_series[idx]
                    ret_dict[threshold] = x_val

                    # Black dot at the threshold crossing
                    plt.scatter([x_val], [y_val], color="black", zorder=5)
                    # Dotted vertical line from x-axis to the point
                    plt.axvline(x=x_val, color="black", linestyle="dotted", linewidth=1)
                    # Dotted horizontal line from y-axis to the point
                    plt.axhline(y=y_val, color="black", linestyle="dotted", linewidth=1)

            plt.xlabel("Env steps (data samples)")
            plt.ylabel("Reward")
            plt.title("Learning Curve with Isotonic Regression and Thresholds")
            plt.legend()
            plt.tight_layout()
            # Set y-axis limits with 5% margin on both sides, but max is hardcoded to 12000
            y_min = 0
            y_max = 12000
            y_range = y_max - y_min
            margin = y_range * 0.05
            plt.ylim(y_min - margin, y_max + margin)
            plt.savefig(save_path)
            plt.close()
        else:
            # Still compute ret_dict for thresholds, but do not plot
            for threshold in Analyzer.REWARD_THRESHOLD:
                if max_reward >= threshold:
                    idx = np.where(isotonic_series >= threshold)[0][0]
                    ret_dict[threshold] = x_common[idx]

        return ret_dict

    def save_best_params(self, *, fit_func_bs, fit_func_lr):
        """
        Save the best parameters to a JSON file.
        """

        utd_inv = np.array([float(2**k) for k in range(10, 17)])

        save_path = Path(self.project) / "best_metrics.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            json.dump(
                {
                    "batch_size": {
                        "formula": fit_func_bs[1],
                        ""
                        "utd_inv": utd_inv.tolist(),
                        "best_batch_size": (fit_func_bs[0](1 / utd_inv)).tolist(),
                    },
                    "lr": {
                        "formula": fit_func_lr[1],
                        "utd_inv": utd_inv.tolist(),
                        "best_lr": (fit_func_lr[0](1 / utd_inv)).tolist(),
                    },
                },
                f,
                indent=4,
            )

    @staticmethod
    def choose_best(data_requirements):
        is_best = dict.fromkeys(data_requirements, 0)
        for threshold in Analyzer.REWARD_THRESHOLD:
            metrices = {
                metric: data_requirement.get(threshold, math.inf) for metric, data_requirement in data_requirements.items()
            }
            bast_metric = min(metrices, key=metrices.get)
            is_best[bast_metric] += 1

        return max(is_best, key=is_best.get)

    def plot_best_params(self, utd_inv, best_metrics, fit_func, metric_name: Literal["batch_size", "lr"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = [1 / k for k in utd_inv]
        if metric_name == "batch_size":
            y_ticks = [batch_size for batch_size, lr in self.BATCH_SIZE]
        else:
            y_ticks = [lr for batch_size, lr in self.LR]

        color_list = list(mcolors.TABLEAU_COLORS.values())
        color_iter = iter(color_list)

        for key, best_metric in best_metrics.items():
            color = next(color_iter)
            ax.plot(
                x_vals,
                best_metric,
                marker="o",
                linestyle="-",
                color=color,
                label=key,
            )

        # Fit a power law to the best metrics
        x_vals_fit = np.linspace(min(x_vals), max(x_vals), 100)
        color = next(color_iter)
        ax.plot(
            x_vals_fit,
            fit_func[0](x_vals_fit),
            marker="",
            linestyle="--",
            color=color,
            label="Power law fit",
        )

        # Set custom x-tick labels as '1/{utd_inv}'
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"1/{int(k)}" for k in utd_inv])
        ax.set_yticks(y_ticks)
        y_min, y_max = min(y_ticks), max(y_ticks)
        y_range = y_max - y_min
        margin = y_range * 0.05
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel("UTD Ratio")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Best {metric_name} vs UTD Ratio\n{fit_func[1]}")
        ax.grid(visible=True)
        ax.legend()
        plt.tight_layout()
        save_path = Path(self.project) / f"best_{metric_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)

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


def average_series(x_list, y_list, num_points=None, method="linear"):
    """
    Interpolate multiple data series to a common x-axis and calculate the mean and standard deviation.

    Parameters
    ----------
    x_list : list of np.ndarray
        x-coordinates of each series (lengths may vary)
    y_list : list of np.ndarray
        y-coordinates of each series (corresponds to x_list)
    num_points : int, optional
        Number of divisions for the common x-axis (default: average length)
    method : str, optional
        'linear' (np.interp) or for future extension

    Returns
    -------
    x_common : np.ndarray
        Common x-coordinates
    y_mean : np.ndarray
        Mean values
    y_std : np.ndarray
        Standard deviation values
    """

    if len(x_list) != len(y_list):
        error_message = "x_list and y_list must have the same length."
        raise ValueError(error_message)

    # Determine the common x-axis
    x_min = min(x.min() for x in x_list)
    x_max = max(x.max() for x in x_list)

    lengths = [len(x) for x in x_list]
    if num_points is None:
        num_points = int(sum(lengths) / len(lengths))  # Use average length
    x_common = np.linspace(x_min, x_max, num_points)

    # Interpolate each series
    y_interp_list = []
    for x, y in zip(x_list, y_list):
        if method == "linear":
            y_interp = np.interp(x_common, x, y)
        else:
            error_message = f"Interpolation method '{method}' is not supported."
            raise NotImplementedError(error_message)
        y_interp_list.append(y_interp)

    # Calculate mean and standard deviation
    y_interp_array = np.array(y_interp_list)
    y_mean = np.mean(y_interp_array, axis=0)
    y_std = np.std(y_interp_array, axis=0)

    return x_common, y_mean, y_std


if __name__ == "__main__":
    args = get_args()

    extracted_json_path = Path(f"{args.project}_runs.json")
    if not extracted_json_path.exists():
        extract_wandb_runs(args.project, args.entity, extracted_json_path)
    else:
        print(f"Using existing runs data from {extracted_json_path}")

    analyzer = Analyzer(args.project)
    analyzer.run(_use_bootstrapping=args.use_bootstrapping)
