import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def plot_reward_history(history_best, history_avg, save_path):
    """
    Plot best and average reward over generations and save to file.

    Includes a dashed reference line at reward=500 (CartPole-v1 maximum).

    Parameters
    ----------
    history_best : list[float]
    history_avg  : list[float]
    save_path    : str
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    generations = range(len(history_best))

    ax.plot(generations, history_best, label="Best reward", linewidth=2, color="#16a34a")
    ax.plot(generations, history_avg,  label="Avg reward",  linewidth=1.5,
            color="#f59e0b", linestyle="--", alpha=0.85)

    ax.axhline(y=500, color="#dc2626", linestyle=":", linewidth=1.5,
               label="Max reward (500)", alpha=0.7)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Reward")
    ax.set_title("CartPole – Reward over Generations")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_experiment_comparison(results, labels, save_path,
                               title="CartPole – Experiment Comparison"):
    """
    Overlay multiple best-reward curves on one plot.

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

    ax.axhline(y=500, color="#dc2626", linestyle=":", linewidth=1.5,
               label="Max reward (500)", alpha=0.6)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Reward")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
