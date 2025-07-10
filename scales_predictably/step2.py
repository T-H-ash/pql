import argparse
import json
import math
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.isotonic import IsotonicRegression
from step1 import average_series

import wandb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=None, help="wandb entity (user or team)")
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
        history = run.history(keys=["eval/return"], pandas=False)
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
    UTD_INV = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    NUM_REQUIRED_SEEDS = 3
    NUM_BOOSTING_ITERATIONS = 100
    REWARD_THRESHOLD = [1000 * i for i in range(1, 11)]

    MODEL_SIZE = 179_592

    def __init__(self, project):
        self.project = project
        self.runs_dict = self.load_runs()
        self.ir = IsotonicRegression(increasing=True, y_min=0)

    def load_runs(self):
        with Path(f"{self.project}_runs.json").open("r") as f:
            return json.load(f)

    def run(self):
        self.analyze()

    def optimal_batch_size(self, utd):
        return 0.012292567142873991 * (utd**0.4657185145312819)  # scales-05-nopal
        # return 0.001040992385405702 * (utd**0.15711956214095557)  # scales-06-pal

    def analyze(self):
        data_requirements = {}
        for utd_inv in self.UTD_INV:
            runs, num_seeds = self.exract(utd_inv)

            if num_seeds < self.NUM_REQUIRED_SEEDS:
                print(f"Not enough runs for utd_inv={utd_inv}. Found {num_seeds} runs.")
                continue

            save_path = Path(self.project) / f"analysis_{utd_inv}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            data_requirement = self.get_data_requirement(runs, save_path=save_path)
            if data_requirement is not None:
                data_requirements[utd_inv] = data_requirement
                print(f"Data requirement for utd_inv={utd_inv}: {data_requirement}")

        data_req_dict = {}
        for threshold in self.REWARD_THRESHOLD:
            utd_inv_list, data_req_list = [], []
            for utd_inv, data_req in data_requirements.items():
                if threshold in data_req:
                    utd_inv_list.append(utd_inv)
                    data_req_list.append(data_req[threshold])

            data_req_func = self.power_law_fit(utd_inv=utd_inv_list, metrics=data_req_list)
            data_req_dict[threshold] = {
                "function": data_req_func[0],
                "formula": data_req_func[1],
                "utd_inv": utd_inv_list,
                "data_req": data_req_list,
            }

        # plot the data efficiency curve
        fig, ax = plt.subplots(figsize=(10, 6))
        color_list = list(mcolors.TABLEAU_COLORS.values())
        color_iter = iter(color_list)
        for threshold, data_req in data_req_dict.items():
            color = next(color_iter)

            # Plot the data requirement points
            x_vals_discrete = 1 / np.array(data_req["utd_inv"])
            ax.plot(x_vals_discrete, data_req["data_req"], "o", color=color)

            # Plot fitted curve
            x_vals = np.linspace(1 / max(self.UTD_INV), 1 / min(self.UTD_INV), 100)
            y_vals = data_req["function"](x_vals)
            ax.plot(x_vals, y_vals, color=color, label=f"Threshold {threshold}")
        ax.set_xticks(1 / np.array(self.UTD_INV))
        ax.set_xticklabels([f"1/2^{int(math.log(k))}" for k in self.UTD_INV])
        ax.set_xlabel("UTD ratio")
        ax.set_ylabel("Minimum Data to Achieve Reward Threshold")
        ax.set_title("Data Efficiency Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(Path(self.project) / "data_efficiency_curve.png")
        plt.close(fig)

        # C_J
        def C_J(utd, threshold):
            return 10 * self.MODEL_SIZE * self.optimal_batch_size(utd) * utd * data_req_dict[threshold]["function"](utd)

        # plot the data efficiency curve
        fig, ax = plt.subplots(figsize=(10, 6))
        color_list = list(mcolors.TABLEAU_COLORS.values())
        color_iter = iter(color_list)
        for threshold, data_req in data_req_dict.items():
            color = next(color_iter)

            # Plot fitted curve
            utds = np.linspace(1 / max(self.UTD_INV), 1 / min(self.UTD_INV), 100)
            # utds = np.linspace(1 / 65536, 1 / 1, 100)

            x_vals = C_J(utds, threshold)
            y_vals = data_req["function"](x_vals)
            ax.plot(x_vals, y_vals, color=color, label=f"Threshold {threshold}")

        ax.set_xlabel("CJ: Compute until J")
        ax.set_ylabel("DJ: Data until J")
        ax.set_title("Scalability Curve")
        ax.legend()
        plt.tight_layout()
        plt.savefig(Path(self.project) / "scalability_curve.png")
        plt.close(fig)

    def exract(self, utd_inv):
        runs = [run for run in self.runs_dict.values() if run["config"]["utd_ratio_inverse"] == utd_inv]
        return runs, len(runs)

    def get_data_requirement(self, runs, save_path=None):
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
            y_min, y_max = 0, 12000
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

    @staticmethod
    def interpolation_mean(runs):
        x_list = [np.array(run["_step"]) for run in runs]
        y_list = [np.array(run["eval_returns"]) for run in runs]
        return average_series(x_list, y_list)

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


if __name__ == "__main__":
    args = get_args()

    extracted_json_path = Path(f"{args.project}_runs.json")
    if not extracted_json_path.exists():
        extract_wandb_runs(args.project, args.entity, extracted_json_path)
    else:
        print(f"Using existing runs data from {extracted_json_path}")

    analyzer = Analyzer(args.project)
    analyzer.run()
