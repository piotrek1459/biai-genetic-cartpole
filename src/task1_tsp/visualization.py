import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_history(history_best, history_avg, save_path,
                         epsilon=1e-6, stagnation_events=None):
    """
    Plot best and average fitness over generations.

    A secondary y-axis on the right shows the corresponding route distance,
    making the chart interpretable without knowing the fitness formula.
    Stagnation-triggered adaptive boost events are marked as vertical lines.

    Parameters
    ----------
    history_best      : list[float]
    history_avg       : list[float]
    save_path         : str
    epsilon           : float  – the ε used in fitness = 1/(d+ε); needed to
                                 recover distance from fitness for the right axis
    stagnation_events : list[int] | None  – generation indices where boost fired
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    generations = range(len(history_best))

    ax1.plot(generations, history_best, label="Best fitness", linewidth=2, color="#2563eb")
    ax1.plot(generations, history_avg,  label="Avg fitness",  linewidth=1.5,
             color="#f59e0b", linestyle="--", alpha=0.85)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness  (1 / distance)", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    # right axis: route distance derived from best fitness
    ax2 = ax1.twinx()
    best_distances = [1.0 / max(f, 1e-12) - epsilon for f in history_best]
    ax2.plot(generations, best_distances, linewidth=0)  # invisible — sets scale only
    ax2.set_ylabel("Best route distance", color="#6b7280")
    ax2.tick_params(axis="y", labelcolor="#6b7280")
    # invert so that smaller distance = higher on right axis (matches fitness direction)
    ax2.set_ylim(max(best_distances) * 1.05, min(best_distances) * 0.95)

    # mark adaptive boost events
    if stagnation_events:
        for i, gen in enumerate(stagnation_events):
            ax1.axvline(x=gen, color="#ef4444", alpha=0.35, linewidth=1.2,
                        linestyle=":", label="Adaptive boost" if i == 0 else "")

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc="lower right", fontsize=9)
    ax1.set_title("TSP – Fitness & Route Distance over Generations")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_best_route(cities, permutation, total_distance, save_path, title=None):
    """
    Visualize a TSP route and save to file.

    Parameters
    ----------
    cities         : np.ndarray, shape (n, 2)
    permutation    : np.ndarray, shape (n,)
    total_distance : float
    save_path      : str
    title          : str | None  – custom plot title; defaults to "TSP – Best Route"
    """
    ordered = cities[permutation]
    loop = np.vstack([ordered, ordered[0]])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(loop[:, 0], loop[:, 1], "-o", color="#2563eb",
            markersize=8, linewidth=1.5, zorder=2)

    for idx, (x, y) in enumerate(cities):
        ax.annotate(str(permutation.tolist().index(idx)),
                    xy=(x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="#374151")

    ax.scatter(*cities[permutation[0]], color="#ef4444", s=120, zorder=3, label="Start")

    heading = title or "TSP – Best Route"
    ax.set_title(f"{heading}  (distance = {total_distance:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_experiment_comparison(results, labels, save_path, title="TSP – Experiment Comparison"):
    """
    Overlay multiple best-fitness curves on one plot.

    Parameters
    ----------
    results  : list of list[float]  – one history_best per experiment
    labels   : list of str
    save_path: str
    title    : str
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for history, label in zip(results, labels):
        ax.plot(history, label=label, linewidth=1.8)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness  (1 / distance)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
