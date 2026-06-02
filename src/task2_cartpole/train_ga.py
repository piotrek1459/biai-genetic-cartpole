import os
import numpy as np
import gymnasium
import imageio

from src.common.ga_base import GeneticAlgorithm
from src.common.selection import tournament_selection, roulette_selection
from src.common.crossover import uniform_crossover, arithmetic_crossover
from src.common.mutation import gaussian_mutation
from src.task2_cartpole.cartpole_policy import LinearPolicy, evaluate_policy_with_stats
from src.task2_cartpole.visualization import (
    plot_reward_history,
    plot_control,
    plot_control_comparison,
)

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
    "stability_weight": 0.05,
    "weight_init_scale": 1.0,
    "seed": 42,
    "results_dir": "results/task2",
    "n_jobs": -1,
    "adaptive_mutation": True,
    "adaptive_patience": 15,
    "log_every": 10,
}

# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def init_population(pop_size, n_weights, scale, rng):
    return [rng.standard_normal(n_weights) * scale for _ in range(pop_size)]


def make_fitness_fn(n_episodes, stability_weight=0.0):
    """
    Combined fitness: mean_reward - stability_weight * std_reward.

    Penalising variance selects for policies that are consistently good
    across different initial conditions.
    """
    def fitness_fn(weights):
        mean_r, std_r = evaluate_policy_with_stats(weights, n_episodes=n_episodes)
        return mean_r - stability_weight * std_r
    return fitness_fn

# ---------------------------------------------------------------------------
# Operator factories
# ---------------------------------------------------------------------------

_CROSSOVERS = {"uniform": uniform_crossover, "arithmetic": arithmetic_crossover}


def _make_crossover(name):
    return _CROSSOVERS[name]


def _make_mutation(mutation_rate, sigma):
    return lambda ind: gaussian_mutation(ind, mutation_rate, sigma=sigma)


def _make_selection(name, k):
    if name == "tournament":
        return lambda pop, fit, n: tournament_selection(pop, fit, n, k=k)
    return lambda pop, fit, n: roulette_selection(pop, fit, n)

# ---------------------------------------------------------------------------
# Episode recording helpers
# ---------------------------------------------------------------------------

def _run_episode(policy, seed=None):
    """Run one episode, return (pole_angles, cart_positions, frames)."""
    env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    pole_angles, cart_positions, frames = [], [], []
    done = False
    while not done:
        pole_angles.append(float(obs[2]))
        cart_positions.append(float(obs[0]))
        frames.append(env.render())
        action = policy.act(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    env.close()
    return pole_angles, cart_positions, frames


def _save_video(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, frames, fps=30)
    print(f"  Saved: {path}")


def _make_comparison_video(frames_init, frames_trained, path):
    """
    Stitch each pair of frames side by side: [random | trained].

    Both frame sequences are zero-padded (black) to the same length.
    """
    import numpy as np
    n = max(len(frames_init), len(frames_trained))
    h = max(frames_init[0].shape[0], frames_trained[0].shape[0])
    w = frames_init[0].shape[1]

    def pad_frame(f):
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:f.shape[0], :f.shape[1]] = f
        return out

    def get_frame(seq, i):
        return pad_frame(seq[i]) if i < len(seq) else np.zeros((h, w, 3), dtype=np.uint8)

    combined = [
        np.concatenate([get_frame(frames_init, i), get_frame(frames_trained, i)], axis=1)
        for i in range(n)
    ]
    _save_video(combined, path)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_cartpole(config=None):
    """Run the CartPole GA — saves initial + trained outputs and comparisons."""
    cfg = {**CONFIG, **(config or {})}

    base_dir    = cfg["results_dir"]
    initial_dir = os.path.join(base_dir, "initial")
    trained_dir = os.path.join(base_dir, "trained")
    compare_dir = os.path.join(base_dir, "comparison")
    for d in (base_dir, initial_dir, trained_dir, compare_dir):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ------------------------------------------------------------------ #
    # 1. INITIAL STATE — random policy                                     #
    # ------------------------------------------------------------------ #
    print("\n--- Recording initial state (random policy) ---")
    random_weights = np.random.default_rng(cfg["seed"]).standard_normal(cfg["n_weights"])
    random_policy  = LinearPolicy(random_weights)
    init_angles, init_positions, init_frames = _run_episode(random_policy, seed=cfg["seed"])

    print(f"  Random policy survived {len(init_angles)} steps")
    plot_control(init_angles, init_positions,
                 os.path.join(initial_dir, "control.png"),
                 label="Random Policy", survived=len(init_angles))
    _save_video(init_frames, os.path.join(initial_dir, "agent.mp4"))

    # ------------------------------------------------------------------ #
    # 2. TRAIN                                                             #
    # ------------------------------------------------------------------ #
    print(f"\n=== CartPole GA | pop={cfg['pop_size']} | gen={cfg['n_generations']} | "
          f"sigma={cfg['sigma']} | episodes={cfg['n_eval_episodes']} ===")

    init_fn      = lambda c: init_population(c["pop_size"], c["n_weights"],
                                             c["weight_init_scale"], rng)
    fitness_fn   = make_fitness_fn(cfg["n_eval_episodes"], cfg.get("stability_weight", 0.0))
    crossover_fn = _make_crossover(cfg["crossover"])
    mutation_fn  = _make_mutation(cfg["mutation_rate"], cfg["sigma"])
    selection_fn = _make_selection(cfg["selection"], cfg["tournament_k"])

    ga = GeneticAlgorithm(
        config=cfg,
        init_fn=init_fn,
        fitness_fn=fitness_fn,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        selection_fn=selection_fn,
    )
    result = ga.run()
    best_weights = result["best"]

    print(f"\nBest mean reward: {result['best_fitness']:.1f} / 500")

    # Save weights and reward history to base dir
    np.save(os.path.join(base_dir, "best_weights.npy"), best_weights)
    plot_reward_history(result["history_best"], result["history_avg"],
                        os.path.join(base_dir, "reward_history.png"),
                        stagnation_events=result.get("stagnation_events"))

    # ------------------------------------------------------------------ #
    # 3. TRAINED STATE — best policy                                       #
    # ------------------------------------------------------------------ #
    print("\n--- Recording trained agent ---")
    trained_policy = LinearPolicy(best_weights)
    trained_angles, trained_positions, trained_frames = _run_episode(trained_policy, seed=0)

    print(f"  Trained policy survived {len(trained_angles)} steps")
    plot_control(trained_angles, trained_positions,
                 os.path.join(trained_dir, "control.png"),
                 label="Trained (GA) Policy", survived=len(trained_angles))
    _save_video(trained_frames, os.path.join(trained_dir, "agent.mp4"))

    # ------------------------------------------------------------------ #
    # 4. COMPARISON                                                        #
    # ------------------------------------------------------------------ #
    print("\n--- Building comparison outputs ---")
    plot_control_comparison(
        init_angles, init_positions,
        trained_angles, trained_positions,
        os.path.join(compare_dir, "control_comparison.png"),
    )
    _make_comparison_video(
        init_frames, trained_frames,
        os.path.join(compare_dir, "agent_comparison.mp4"),
    )

    print(f"\nAll outputs saved under {base_dir}/")
    print(f"  initial/  — random policy chart + video")
    print(f"  trained/  — GA policy chart + video")
    print(f"  comparison/ — side-by-side chart + video")

    return result


if __name__ == "__main__":
    run_cartpole()
