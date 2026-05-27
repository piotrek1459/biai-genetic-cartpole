import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_reward_history(history_best, history_avg, save_path, stagnation_events=None):
    """
    Plot best and average reward over generations.

    Includes a reference line at reward=500 and marks adaptive boost events.

    Parameters
    ----------
    history_best      : list[float]
    history_avg       : list[float]
    save_path         : str
    stagnation_events : list[int] | None
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    generations = range(len(history_best))

    ax.plot(generations, history_best, label="Best reward", linewidth=2, color="#16a34a")
    ax.plot(generations, history_avg,  label="Avg reward",  linewidth=1.5,
            color="#f59e0b", linestyle="--", alpha=0.85)
    ax.axhline(y=500, color="#dc2626", linestyle=":", linewidth=1.5,
               label="Max reward (500)", alpha=0.7)

    if stagnation_events:
        for i, gen in enumerate(stagnation_events):
            ax.axvline(x=gen, color="#7c3aed", alpha=0.35, linewidth=1.2,
                       linestyle=":", label="Adaptive boost" if i == 0 else "")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Reward")
    ax.set_title("CartPole – Reward over Generations")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_initial_cartpole(pole_angles, cart_positions, save_path):
    """
    Visualise the initial state of a random (untrained) CartPole agent.

    Shows how the pole falls without any learned policy: pole angle and
    cart position over the episode steps, with failure thresholds.

    Parameters
    ----------
    pole_angles    : list[float]  – pole angle (rad) at each step
    cart_positions : list[float]  – cart position at each step
    save_path      : str
    """
    steps      = np.arange(len(pole_angles))
    angles_deg = np.degrees(pole_angles)
    threshold  = 12.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(steps, angles_deg, linewidth=1.8, color="#dc2626", label="Pole angle")
    ax1.axhline(y= threshold, color="#9ca3af", linestyle=":", linewidth=1.2,
                alpha=0.8, label=f"Failure threshold (±{threshold}°)")
    ax1.axhline(y=-threshold, color="#9ca3af", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.axhline(y=0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Pole Angle (degrees)")
    ax1.set_ylim(-threshold * 2.5, threshold * 2.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, cart_positions, linewidth=1.8, color="#f59e0b", label="Cart position")
    ax2.axhline(y= 2.4, color="#9ca3af", linestyle=":", linewidth=1.2,
                alpha=0.8, label="Position limit (±2.4)")
    ax2.axhline(y=-2.4, color="#9ca3af", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cart Position")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    total = len(pole_angles)
    fig.suptitle(f"CartPole – Initial State: Random Policy  (survived {total} steps)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_control_performance(pole_angles, save_path):
    """
    Visualise the best agent's control behaviour during one episode.

    Shows pole angle over time on the top panel and cumulative reward on
    the bottom, with failure-threshold lines at ±12° (±0.2095 rad).

    Parameters
    ----------
    pole_angles : list[float]  – pole angle (rad) at each step
    save_path   : str
    """
    steps = np.arange(len(pole_angles))
    angles_deg = np.degrees(pole_angles)
    threshold  = 12.0  # CartPole failure threshold in degrees

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(steps, angles_deg, linewidth=1.8, color="#2563eb", label="Pole angle")
    ax1.axhline(y= threshold, color="#ef4444", linestyle=":", linewidth=1.4,
                alpha=0.8, label=f"Failure threshold (±{threshold}°)")
    ax1.axhline(y=-threshold, color="#ef4444", linestyle=":", linewidth=1.4, alpha=0.8)
    ax1.axhline(y=0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Pole Angle (degrees)")
    ax1.set_ylim(-threshold * 1.5, threshold * 1.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    cumulative = np.arange(1, len(pole_angles) + 1, dtype=float)
    ax2.plot(steps, cumulative, linewidth=1.8, color="#16a34a")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative Reward")
    ax2.grid(True, alpha=0.3)

    total = len(pole_angles)
    fig.suptitle(f"CartPole – Best Agent Control Performance  (total reward: {total})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_experiment_comparison(results, labels, save_path,
                               title="CartPole – Experiment Comparison"):
    """Overlay multiple best-reward curves on one plot."""
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
