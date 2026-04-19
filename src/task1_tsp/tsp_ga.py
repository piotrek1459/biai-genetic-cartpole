import os
import numpy as np

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection, roulette_selection
from src.common.crossover import ox_crossover, pmx_crossover
from src.common.mutation import swap_mutation, inversion_mutation, scramble_mutation
from src.task1_tsp.visualization import plot_fitness_history, plot_best_route

CONFIG = {
    "n_cities": 20,
    "pop_size": 80,
    "n_generations": 300,
    "n_elite": 2,
    "mutation_rate": 0.10,
    "tournament_k": 4,
    "crossover": "ox",         # "ox" | "pmx"
    "mutation": "inversion",   # "swap" | "inversion" | "scramble"
    "selection": "tournament", # "tournament" | "roulette"
    "seed": 42,
    "results_dir": "results/task1",
    "epsilon": 1e-6,
    "log_every": 50,
}

# ---------------------------------------------------------------------------
# Problem helpers
# ---------------------------------------------------------------------------

def generate_cities(n, seed=None):
    """Return an (n, 2) array of random city coordinates in [0, 1]."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def route_distance(cities, permutation):
    """Compute the total round-trip distance for a given route permutation."""
    ordered = cities[permutation]
    diffs = np.diff(ordered, axis=0)
    dist = float(np.sum(np.linalg.norm(diffs, axis=1)))
    dist += float(np.linalg.norm(ordered[-1] - ordered[0]))
    return dist


def make_fitness_fn(cities, epsilon=1e-6):
    """Return a fitness function: f(permutation) = 1 / (distance + epsilon)."""
    def fitness_fn(permutation):
        return 1.0 / (route_distance(cities, permutation) + epsilon)
    return fitness_fn


def init_population(pop_size, n_cities, rng):
    """Return a list of random permutation individuals."""
    return [rng.permutation(n_cities) for _ in range(pop_size)]

# ---------------------------------------------------------------------------
# Operator factories
# ---------------------------------------------------------------------------

_CROSSOVERS = {
    "ox": ox_crossover,
    "pmx": pmx_crossover,
}

_MUTATIONS = {
    "swap": swap_mutation,
    "inversion": inversion_mutation,
    "scramble": scramble_mutation,
}


def _make_crossover(name):
    return _CROSSOVERS[name]


def _make_mutation(name, mutation_rate):
    fn = _MUTATIONS[name]
    return lambda ind: fn(ind, mutation_rate)


def _make_selection(name, k):
    if name == "tournament":
        return lambda pop, fit, n: tournament_selection(pop, fit, n, k=k)
    return lambda pop, fit, n: roulette_selection(pop, fit, n)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tsp(config=None):
    """Run the TSP genetic algorithm with the given config dict."""
    cfg = {**CONFIG, **(config or {})}

    os.makedirs(cfg["results_dir"], exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])  # also seed legacy numpy used by some operators

    cities = generate_cities(cfg["n_cities"], seed=cfg["seed"])

    init_fn = lambda c: init_population(c["pop_size"], c["n_cities"], rng)
    fitness_fn = make_fitness_fn(cities, cfg["epsilon"])
    crossover_fn = _make_crossover(cfg["crossover"])
    mutation_fn = _make_mutation(cfg["mutation"], cfg["mutation_rate"])
    selection_fn = _make_selection(cfg["selection"], cfg["tournament_k"])

    print(f"\n=== TSP GA | {cfg['n_cities']} cities | "
          f"pop={cfg['pop_size']} | gen={cfg['n_generations']} | "
          f"crossover={cfg['crossover']} | mutation={cfg['mutation']} ===")

    ga = GeneticAlgorithm(
        config=cfg,
        init_fn=init_fn,
        fitness_fn=fitness_fn,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        selection_fn=selection_fn,
    )
    result = ga.run()

    best_dist = route_distance(cities, result["best"])
    print(f"\nBest route distance: {best_dist:.4f}")

    # Save plots
    history_path = os.path.join(cfg["results_dir"], "fitness_history.png")
    route_path = os.path.join(cfg["results_dir"], "best_route.png")

    plot_fitness_history(result["history_best"], result["history_avg"], history_path)
    plot_best_route(cities, result["best"], best_dist, route_path)

    print(f"Plots saved to {cfg['results_dir']}/")

    # Save best permutation
    weights_path = os.path.join(cfg["results_dir"], "best_route.npy")
    np.save(weights_path, result["best"])

    return result, cities


if __name__ == "__main__":
    run_tsp()
