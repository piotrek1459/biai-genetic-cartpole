import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_reward_history(history_best, history_avg, save_path, stagnation_events=None):
    """Plot best and average reward over generations with optional boost markers."""
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


def plot_control(pole_angles, cart_positions, save_path, label="", survived=None):
    """
    Two-panel control plot: pole angle (top) and cart position (bottom).

    Parameters
    ----------
    pole_angles    : list[float]  – rad
    cart_positions : list[float]
    save_path      : str
    label          : str  – shown in title, e.g. "Random Policy" or "Trained Policy"
    survived       : int | None  – steps survived; computed from data if None
    """
    steps      = np.arange(len(pole_angles))
    angles_deg = np.degrees(pole_angles)
    threshold  = 12.0
    n          = survived if survived is not None else len(pole_angles)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(steps, angles_deg, linewidth=1.8, color="#2563eb", label="Pole angle")
    ax1.axhline(y= threshold, color="#ef4444", linestyle=":", linewidth=1.4,
                alpha=0.8, label=f"Failure threshold (±{threshold}°)")
    ax1.axhline(y=-threshold, color="#ef4444", linestyle=":", linewidth=1.4, alpha=0.8)
    ax1.axhline(y=0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Pole Angle (degrees)")
    ax1.set_ylim(-threshold * 2, threshold * 2)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, cart_positions, linewidth=1.8, color="#16a34a", label="Cart position")
    ax2.axhline(y= 2.4, color="#9ca3af", linestyle=":", linewidth=1.2,
                alpha=0.8, label="Position limit (±2.4)")
    ax2.axhline(y=-2.4, color="#9ca3af", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cart Position")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"CartPole — {label}  (survived {n} steps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_control_comparison(init_angles, init_positions,
                             trained_angles, trained_positions,
                             save_path):
    """
    Side-by-side 2×2 comparison of random vs trained agent.

    Layout:
        [Pole angle — random]  |  [Pole angle — trained]
        [Cart pos  — random]   |  [Cart pos  — trained]
    """
    threshold = 12.0
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    datasets = [
        (axes[0, 0], np.degrees(init_angles),    "#dc2626",
         f"Pole Angle — Random Policy\n(survived {len(init_angles)} steps)"),
        (axes[0, 1], np.degrees(trained_angles),  "#2563eb",
         f"Pole Angle — Trained (GA) Policy\n(survived {len(trained_angles)} steps)"),
        (axes[1, 0], init_positions,              "#f59e0b",
         "Cart Position — Random Policy"),
        (axes[1, 1], trained_positions,           "#16a34a",
         "Cart Position — Trained (GA) Policy"),
    ]

    for ax, data, color, title in datasets:
        ax.plot(np.arange(len(data)), data, linewidth=1.8, color=color)
        is_angle = "Angle" in title
        limit    = threshold if is_angle else 2.4
        ax.axhline(y= limit, color="#9ca3af", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axhline(y=-limit, color="#9ca3af", linestyle=":", linewidth=1.2, alpha=0.8)
        if is_angle:
            ax.axhline(y=0, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylim(-threshold * 2, threshold * 2)
            ax.set_ylabel("Degrees")
        else:
            ax.set_ylabel("Position")
            ax.set_xlabel("Step")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("CartPole — Initial vs Trained Agent Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Legacy aliases kept for backward compatibility with experiments.py
# ---------------------------------------------------------------------------

def plot_initial_cartpole(pole_angles, cart_positions, save_path):
    plot_control(pole_angles, cart_positions, save_path, label="Random Policy")


def plot_control_performance(pole_angles, save_path):
    cart_positions = [0.0] * len(pole_angles)  # fallback if positions not available
    plot_control(pole_angles, cart_positions, save_path, label="Trained (GA) Policy")


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
