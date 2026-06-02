"""
Evaluate and record the best CartPole agent.

Usage:
    python -m src.task2_cartpole.evaluate
    python -m src.task2_cartpole.evaluate --weights results/task2/best_weights.npy
    python -m src.task2_cartpole.evaluate --weights results/task2/best_weights.npy --no-record
"""
import argparse
import os

import imageio
import numpy as np
import gymnasium

from src.task2_cartpole.cartpole_policy import LinearPolicy


def render_agent(weights_path, n_episodes=3, record=True, output_path=None):
    """Load weights, run the agent, optionally save a video."""
    weights = np.load(weights_path)
    policy  = LinearPolicy(weights)

    render_mode = "rgb_array" if record else "human"
    env = gymnasium.make("CartPole-v1", render_mode=render_mode)

    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            if record:
                frames.append(env.render())
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: reward = {ep_reward:.0f}")

    env.close()

    if record and frames:
        if output_path is None:
            output_path = "results/task2/trained/agent.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimwrite(output_path, frames, fps=30)
        print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best CartPole agent")
    parser.add_argument("--weights", default="results/task2/best_weights.npy")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    render_agent(
        weights_path=args.weights,
        n_episodes=args.episodes,
        record=not args.no_record,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

