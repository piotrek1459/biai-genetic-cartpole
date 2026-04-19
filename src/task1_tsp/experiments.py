"""
TSP hyperparameter experiments.

Compares different operator combinations and population sizes, then
produces a single comparison plot per experiment group.

Run:
    python -m src.task1_tsp.experiments
"""
import os

from src.task1_tsp.tsp_ga import CONFIG, run_tsp
from src.task1_tsp.visualization import plot_experiment_comparison

RESULTS_DIR = "results/task1"

# ---------------------------------------------------------------------------
# Experiment groups
# ---------------------------------------------------------------------------

OPERATOR_EXPERIMENTS = [
    {"crossover": "ox",  "mutation": "swap",      "label": "swap + OX"},
    {"crossover": "ox",  "mutation": "inversion",  "label": "inversion + OX"},
    {"crossover": "ox",  "mutation": "scramble",   "label": "scramble + OX"},
    {"crossover": "pmx", "mutation": "inversion",  "label": "inversion + PMX"},
]

POPULATION_EXPERIMENTS = [
    {"pop_size": 40,  "label": "pop = 40"},
    {"pop_size": 80,  "label": "pop = 80  (default)"},
    {"pop_size": 160, "label": "pop = 160"},
]


def run_group(experiments, base_config, group_name):
    """Run a list of config overrides and collect history curves."""
    histories = []
    labels = []
    for exp in experiments:
        label = exp.pop("label")
        cfg = {**base_config, **exp, "results_dir": os.path.join(RESULTS_DIR, "exp_" + label.replace(" ", "_"))}
        print(f"\n--- Experiment: {label} ---")
        result, _ = run_tsp(cfg)
        histories.append(result["history_best"])
        labels.append(label)
        exp["label"] = label  # restore for potential reuse

    save_path = os.path.join(RESULTS_DIR, f"comparison_{group_name}.png")
    plot_experiment_comparison(
        histories, labels, save_path,
        title=f"TSP – {group_name.replace('_', ' ').title()} Comparison",
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base = {**CONFIG, "n_generations": 200, "log_every": 200}

    print("\n========== Operator Comparison ==========")
    run_group(OPERATOR_EXPERIMENTS, base, "operators")

    print("\n========== Population Size Comparison ==========")
    run_group(POPULATION_EXPERIMENTS, base, "population_size")

    print("\nAll TSP experiments done.")


if __name__ == "__main__":
    main()
