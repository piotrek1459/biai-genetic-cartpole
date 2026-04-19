import os
import numpy as np

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection, roulette_selection
from src.common.crossover import uniform_crossover, arithmetic_crossover
from src.common.mutation import gaussian_mutation
from src.task2_cartpole.cartpole_policy import evaluate_policy
from src.task2_cartpole.visualization import plot_reward_history

CONFIG = {
    "n_weights": 5,             # 4 observation weights + 1 bias
    "pop_size": 50,
    "n_generations": 100,
    "n_elite": 2,
    "mutation_rate": 0.10,
    "sigma": 0.3,               # Gaussian mutation std dev
    "crossover": "uniform",     # "uniform" | "arithmetic"
    "selection": "tournament",  # "tournament" | "roulette"
    "tournament_k": 4,
    "n_eval_episodes": 5,
    "weight_init_scale": 1.0,   # std dev for initial weight sampling N(0, scale)
    "seed": 42,
    "results_dir": "results/task2",
    "log_every": 10,
}

# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def init_population(pop_size, n_weights, scale, rng):
    """Return a list of random real-valued weight vectors."""
    return [rng.standard_normal(n_weights) * scale for _ in range(pop_size)]


def make_fitness_fn(n_episodes):
    """Return a fitness function that evaluates a CartPole policy."""
    def fitness_fn(weights):
        return evaluate_policy(weights, n_episodes=n_episodes)
    return fitness_fn

# ---------------------------------------------------------------------------
# Operator factories
# ---------------------------------------------------------------------------

_CROSSOVERS = {
    "uniform": uniform_crossover,
    "arithmetic": arithmetic_crossover,
}


def _make_crossover(name):
    return _CROSSOVERS[name]


def _make_mutation(mutation_rate, sigma):
    return lambda ind: gaussian_mutation(ind, mutation_rate, sigma=sigma)


def _make_selection(name, k):
    if name == "tournament":
        return lambda pop, fit, n: tournament_selection(pop, fit, n, k=k)
    return lambda pop, fit, n: roulette_selection(pop, fit, n)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cartpole(config=None):
    """Run the CartPole GA with the given config dict."""
    cfg = {**CONFIG, **(config or {})}

    os.makedirs(cfg["results_dir"], exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])

    init_fn = lambda c: init_population(c["pop_size"], c["n_weights"],
                                        c["weight_init_scale"], rng)
    fitness_fn = make_fitness_fn(cfg["n_eval_episodes"])
    crossover_fn = _make_crossover(cfg["crossover"])
    mutation_fn = _make_mutation(cfg["mutation_rate"], cfg["sigma"])
    selection_fn = _make_selection(cfg["selection"], cfg["tournament_k"])

    print(f"\n=== CartPole GA | pop={cfg['pop_size']} | gen={cfg['n_generations']} | "
          f"sigma={cfg['sigma']} | episodes={cfg['n_eval_episodes']} ===")

    ga = GeneticAlgorithm(
        config=cfg,
        init_fn=init_fn,
        fitness_fn=fitness_fn,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        selection_fn=selection_fn,
    )
    result = ga.run()

    print(f"\nBest mean reward: {result['best_fitness']:.1f} / 500")

    # Save best weights
    weights_path = os.path.join(cfg["results_dir"], "best_weights.npy")
    np.save(weights_path, result["best"])
    print(f"Weights saved to {weights_path}")

    # Save plot
    plot_path = os.path.join(cfg["results_dir"], "reward_history.png")
    plot_reward_history(result["history_best"], result["history_avg"], plot_path)

    return result


if __name__ == "__main__":
    run_cartpole()
