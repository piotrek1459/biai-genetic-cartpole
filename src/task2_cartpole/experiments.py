"""
CartPole hyperparameter experiments.

Compares mutation strength (sigma), population size, and number of
evaluation episodes. Saves one comparison plot per group.

Run:
    python -m src.task2_cartpole.experiments
"""
import os

from src.task2_cartpole.train_ga import CONFIG, run_cartpole
from src.task2_cartpole.visualization import plot_experiment_comparison

RESULTS_DIR = "results/task2"

# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

SIGMA_EXPERIMENTS = [
    {"sigma": 0.05, "label": "sigma = 0.05 (weak)"},
    {"sigma": 0.30, "label": "sigma = 0.30 (default)"},
    {"sigma": 1.00, "label": "sigma = 1.00 (strong)"},
]

POPULATION_EXPERIMENTS = [
    {"pop_size": 20,  "label": "pop = 20"},
    {"pop_size": 50,  "label": "pop = 50  (default)"},
    {"pop_size": 100, "label": "pop = 100"},
]

EPISODES_EXPERIMENTS = [
    {"n_eval_episodes": 1, "label": "1 episode  (noisy)"},
    {"n_eval_episodes": 3, "label": "3 episodes"},
    {"n_eval_episodes": 5, "label": "5 episodes (default)"},
]


def run_group(experiments, base_config, group_name):
    """Run a list of config overrides and collect reward history curves."""
    histories = []
    labels = []
    for exp in experiments:
        label = exp.pop("label")
        cfg = {
            **base_config,
            **exp,
            "results_dir": os.path.join(RESULTS_DIR, "exp_" + label.replace(" ", "_").replace(".", "p")),
        }
        print(f"\n--- Experiment: {label} ---")
        result = run_cartpole(cfg)
        histories.append(result["history_best"])
        labels.append(label)
        exp["label"] = label  # restore

    save_path = os.path.join(RESULTS_DIR, f"comparison_{group_name}.png")
    plot_experiment_comparison(
        histories, labels, save_path,
        title=f"CartPole – {group_name.replace('_', ' ').title()} Comparison",
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base = {**CONFIG, "n_generations": 80, "log_every": 80}

    print("\n========== Sigma (mutation strength) Comparison ==========")
    run_group(SIGMA_EXPERIMENTS, base, "sigma")

    print("\n========== Population Size Comparison ==========")
    run_group(POPULATION_EXPERIMENTS, base, "population_size")

    print("\n========== Evaluation Episodes Comparison ==========")
    run_group(EPISODES_EXPERIMENTS, base, "eval_episodes")

    print("\nAll CartPole experiments done.")


if __name__ == "__main__":
    main()
