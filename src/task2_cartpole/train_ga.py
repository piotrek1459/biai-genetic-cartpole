import os
import numpy as np
import gymnasium

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection, roulette_selection
from src.common.crossover import uniform_crossover, arithmetic_crossover
from src.common.mutation import gaussian_mutation
from src.task2_cartpole.cartpole_policy import LinearPolicy, evaluate_policy_with_stats
from src.task2_cartpole.visualization import plot_reward_history, plot_control_performance

CONFIG = {
    "n_weights": 5,
    "pop_size": 50,
    "n_generations": 100,
    "n_elite": 2,
    "mutation_rate": 0.10,
    "sigma": 0.3,
    "crossover": "uniform",
    "selection": "tournament",
    "tournament_k": 4,
    "n_eval_episodes": 5,
    "stability_weight": 0.05,  # fitness = mean_reward - stability_weight * std_reward
    "weight_init_scale": 1.0,
    "seed": 42,
    "results_dir": "results/task2",
    "n_jobs": -1,              # parallel fitness evaluation
    "adaptive_mutation": True,
    "adaptive_patience": 15,
    "log_every": 10,
}

# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def init_population(pop_size, n_weights, scale, rng):
    """Return a list of random real-valued weight vectors."""
    return [rng.standard_normal(n_weights) * scale for _ in range(pop_size)]


def make_fitness_fn(n_episodes, stability_weight=0.0):
    """
    Return a combined fitness function:

        f = mean_reward - stability_weight * std_reward

    stability_weight=0 reduces to plain mean reward.
    Penalising reward variance rewards policies that are consistently
    good across different initial conditions, not just occasionally lucky.
    """
    def fitness_fn(weights):
        mean_r, std_r = evaluate_policy_with_stats(weights, n_episodes=n_episodes)
        return mean_r - stability_weight * std_r
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
    fitness_fn = make_fitness_fn(cfg["n_eval_episodes"], cfg.get("stability_weight", 0.0))
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
    plot_reward_history(result["history_best"], result["history_avg"], plot_path,
                        stagnation_events=result.get("stagnation_events"))

    # Control performance: record pole angle trace of the best policy
    ctrl_path = os.path.join(cfg["results_dir"], "control_performance.png")
    _save_control_performance(result["best"], ctrl_path)

    return result


def _save_control_performance(weights, save_path):
    """Run one episode with the best weights and save a pole-angle trace."""
    policy = LinearPolicy(weights)
    env = gymnasium.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    pole_angles = []
    done = False
    while not done:
        pole_angles.append(float(obs[2]))
        action = policy.act(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    env.close()
    plot_control_performance(pole_angles, save_path)


if __name__ == "__main__":
    run_cartpole()
