import numpy as np
import gymnasium


class LinearPolicy:
    """
    Linear controller for CartPole-v1.

    The action is determined by the sign of a linear combination of the
    four observation values plus a bias term:

        action = 1  if  dot(obs, weights) + bias > 0  else  0

    Parameters
    ----------
    weights : np.ndarray, shape (5,)
        [w0, w1, w2, w3, bias]
        w0 – cart position weight
        w1 – cart velocity weight
        w2 – pole angle weight
        w3 – pole angular velocity weight
    """

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=float)
        self.w = weights[:4]
        self.b = weights[4]

    def act(self, obs):
        """Return action 0 or 1 based on the observation."""
        return int(np.dot(obs, self.w) + self.b > 0)


def evaluate_policy(weights, n_episodes=5, seed=None):
    """
    Run the CartPole-v1 environment for n_episodes and return mean reward.

    Creates and closes the environment internally so this function is
    stateless and safe to call repeatedly from the fitness function.

    Parameters
    ----------
    weights    : np.ndarray, shape (5,)
    n_episodes : int   – number of episodes to average over
    seed       : int | None  – passed to env.reset() for reproducibility

    Returns
    -------
    float – mean total reward across n_episodes
    """
    policy = LinearPolicy(weights)
    env = gymnasium.make("CartPole-v1")
    rewards = []

    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        total = 0.0
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)

    env.close()
    return float(np.mean(rewards))
