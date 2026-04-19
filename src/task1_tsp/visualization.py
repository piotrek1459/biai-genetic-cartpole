import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness_history(history_best, history_avg, save_path):
    """
    Plot best and average fitness over generations and save to file.

    Parameters
    ----------
    history_best : list[float]
    history_avg  : list[float]
    save_path    : str
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    generations = range(len(history_best))

    ax.plot(generations, history_best, label="Best fitness", linewidth=2, color="#2563eb")
    ax.plot(generations, history_avg,  label="Avg fitness",  linewidth=1.5,
            color="#f59e0b", linestyle="--", alpha=0.85)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness  (1 / distance)")
    ax.set_title("TSP – Fitness over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_best_route(cities, permutation, total_distance, save_path):
    """
    Visualize the best TSP route found and save to file.

    Parameters
    ----------
    cities         : np.ndarray, shape (n, 2)
    permutation    : np.ndarray, shape (n,)  – city indices in visit order
    total_distance : float
    save_path      : str
    """
    ordered = cities[permutation]
    # close the loop for drawing
    loop = np.vstack([ordered, ordered[0]])

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(loop[:, 0], loop[:, 1], "-o", color="#2563eb",
            markersize=8, linewidth=1.5, zorder=2)

    # label cities with their index
    for idx, (x, y) in enumerate(cities):
        ax.annotate(str(permutation.tolist().index(idx)),
                    xy=(x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="#374151")

    # highlight the start city
    ax.scatter(*cities[permutation[0]], color="#ef4444", s=120, zorder=3, label="Start")

    ax.set_title(f"TSP – Best Route  (distance = {total_distance:.4f})")
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
