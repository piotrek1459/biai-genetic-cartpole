# Genetic Algorithm — TSP & CartPole

University project for the **BIAI** course (Biologically-Inspired Artificial Intelligence)
at Politechnika Śląska.

The project implements a genetic algorithm (GA) from scratch in Python to solve two problems:

| Task | Problem | Chromosome | Fitness |
|------|---------|------------|---------|
| 1 | Traveling Salesman Problem (TSP) | Permutation of city indices | `1 / (route_distance + ε)` |
| 2 | CartPole-v1 (Gymnasium) | Real-valued policy weights | Mean reward over 5 episodes |

---

## Project Structure

```
src/
  common/
    ga_base.py        # Generic GeneticAlgorithm class (evolution loop)
    selection.py      # tournament_selection, roulette_selection
    crossover.py      # ox_crossover, pmx_crossover, uniform_crossover, arithmetic_crossover
    mutation.py       # swap/inversion/scramble (TSP), gaussian (CartPole)
  task1_tsp/
    tsp_ga.py         # TSP main script
    visualization.py  # fitness history + route plots
    experiments.py    # operator & population size comparisons
  task2_cartpole/
    cartpole_policy.py  # LinearPolicy class + evaluate_policy()
    train_ga.py         # CartPole main script
    evaluate.py         # render & record best agent
    visualization.py    # reward history plots
    experiments.py      # sigma, pop_size, eval_episodes comparisons
results/
  task1/              # created automatically
  task2/
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

### Task 1 — TSP

```bash
# Train (saves fitness_history.png, best_route.png, best_route.npy)
python3 -m src.task1_tsp.tsp_ga

# Hyperparameter experiments (comparison plots)
python3 -m src.task1_tsp.experiments
```

### Task 2 — CartPole

```bash
# Train (saves reward_history.png, best_weights.npy)
python3 -m src.task2_cartpole.train_ga

# Record best agent video (saves agent.mp4)
python3 -m src.task2_cartpole.evaluate

# Hyperparameter experiments (comparison plots)
python3 -m src.task2_cartpole.experiments
```

---

## Genetic Algorithm Overview

The `GeneticAlgorithm` class in `src/common/ga_base.py` is fully generic.
Domain-specific logic is injected as callable parameters:

```
init_fn      → initial population
fitness_fn   → objective function (higher = better)
crossover_fn → recombination operator
mutation_fn  → perturbation operator
selection_fn → parent selection strategy
```

**One generation:**

```
evaluate fitness
    → record history (best, avg)
    → elitism: preserve top n_elite individuals
    → selection → crossover → mutation → new offspring
    → next population = elites + offspring
```

---

## Genetic Operators

### Selection

| Method | Description |
|--------|-------------|
| `tournament_selection(k)` | Draw k candidates, pick the best. Higher k = stronger pressure. |
| `roulette_selection` | Fitness-proportionate probability. Requires non-negative fitness. |

### Crossover

| Operator | Type | Description |
|----------|------|-------------|
| `ox_crossover` | Permutation | Order Crossover — preserves relative city order from parent 2 |
| `pmx_crossover` | Permutation | Partially Mapped Crossover — preserves absolute city positions |
| `uniform_crossover` | Real-valued | Each gene drawn from a random parent with probability p |
| `arithmetic_crossover` | Real-valued | Linear combination: `c = α·p1 + (1-α)·p2` |

### Mutation

| Operator | Type | Description |
|----------|------|-------------|
| `swap_mutation` | Permutation | Swap two random genes |
| `inversion_mutation` | Permutation | Reverse a random sub-sequence *(default for TSP)* |
| `scramble_mutation` | Permutation | Shuffle a random sub-sequence |
| `gaussian_mutation(σ)` | Real-valued | Add N(0, σ) noise to each gene independently |

---

## Task 1 — Traveling Salesman Problem

**Representation:** A chromosome is a permutation of city indices `[0, n-1]`.
The route visits cities in that order and returns to the start.

**Fitness:** `f = 1 / (total_distance + ε)`
Maximising fitness is equivalent to minimising total Euclidean route distance.

**Default config:**

| Parameter | Value |
|-----------|-------|
| Cities | 20 random points in [0,1]² |
| Population | 80 |
| Generations | 300 |
| Crossover | OX |
| Mutation | Inversion, rate=0.10 |
| Selection | Tournament, k=4 |
| Elitism | 2 individuals |

**Expected result:** Route distance ≈ 3.4 (theoretical optimum ≈ 3.1 for 20 random cities).

---

## Task 2 — CartPole-v1

**Environment:** Cart with a pole; agent must keep the pole upright by pushing left or right.
Observation: `[cart_pos, cart_vel, pole_angle, pole_angular_vel]`.
Maximum episode reward: **500**.

**Linear policy:**

```
action = 1  if  w0·pos + w1·vel + w2·angle + w3·ang_vel + bias > 0
action = 0  otherwise
```

Chromosome: 5 real numbers `[w0, w1, w2, w3, bias]`.

**Fitness:** Mean total reward over 5 independent episodes.

**Default config:**

| Parameter | Value |
|-----------|-------|
| Chromosome length | 5 |
| Population | 50 |
| Generations | 100 |
| Crossover | Uniform |
| Mutation | Gaussian, σ=0.3, rate=0.10 |
| Selection | Tournament, k=4 |
| Elitism | 2 individuals |
| Episodes per eval | 5 |

**Expected result:** Reward = 500 (maximum) reached within ~10–30 generations.

---

## Hyperparameter Experiments

### TSP

Experiments compare operators and population sizes across 200 generations:

| Group | Variants |
|-------|----------|
| Mutation operator | swap+OX, inversion+OX, scramble+OX, inversion+PMX |
| Population size | 40, 80, 160 |

### CartPole

| Group | Variants |
|-------|----------|
| Mutation sigma | 0.05, 0.30, 1.00 |
| Population size | 20, 50, 100 |
| Evaluation episodes | 1, 3, 5 |

Comparison plots are saved to `results/task1/comparison_*.png` and `results/task2/comparison_*.png`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Core array computations |
| `matplotlib` | Plotting fitness/reward history and routes |
| `gymnasium[classic-control]` | CartPole-v1 environment |
| `pygad` | Available for optional comparison (not used in main code) |
| `imageio[ffmpeg]` | Saving agent video as MP4 |

---

## Improvements (feedback 13.05.2026)

The following changes were made in response to tutor feedback requesting combined fitness functions, parallel evaluation, RL-inspired adaptive mutation, and richer visualisations.

### 1. Combined fitness functions

**TSP — path balance bonus**

The TSP fitness function was extended to reward routes with balanced segment lengths in addition to minimising total distance:

```
f = (1 / (distance + ε)) × (1 + smoothness_weight × smoothness)
```

`smoothness` is computed as `1 / (1 + CV)`, where CV is the coefficient of variation of segment lengths. A route where all segments are roughly equal in length scores close to 1.0; heavily zigzagging routes score lower. The bonus is controlled by `smoothness_weight` (default 0.2) and can be disabled by setting it to 0.

**CartPole — stability-penalised fitness**

The CartPole fitness now penalises inconsistent performance across evaluation episodes:

```
f = mean_reward - stability_weight × std_reward
```

A policy that scores 500 in all 5 episodes is preferred over one that scores 500 in some and 300 in others. `stability_weight` defaults to 0.05 and can be tuned via CONFIG. A new `evaluate_policy_with_stats()` function in `cartpole_policy.py` returns `(mean, std)` to support this.

---

### 2. Parallel population evaluation

Fitness evaluation is now parallelised using `concurrent.futures.ThreadPoolExecutor`. The number of worker threads is controlled by `n_jobs` in the config dict:

- `n_jobs = 1` — sequential evaluation (original behaviour)
- `n_jobs = -1` — use all available CPU threads (default for both tasks)

Threads work well for Gymnasium-based fitness functions because the environment's step loop releases the Python GIL. For TSP the fitness computation is pure NumPy which also benefits from threading. No changes to the operator API were needed.

---

### 3. RL-inspired adaptive mutation (task switching)

`GeneticAlgorithm` now tracks how many consecutive generations have passed without a fitness improvement (stagnation counter). When the counter exceeds `adaptive_patience` (default: 25 for TSP, 15 for CartPole), the mutation operator is applied **twice per offspring** instead of once for that generation, then the counter resets.

This mirrors the exploration–exploitation trade-off from reinforcement learning: when the reward signal (fitness improvement) plateaus, the agent (GA) increases exploration. Applying the mutation twice effectively doubles the perturbation strength without requiring any change to the mutation function signatures or the operator API.

Generations where the boost fired are recorded in `result["stagnation_events"]` and visualised as vertical dotted lines on the fitness/reward history plots.

Config keys added: `adaptive_mutation` (bool, default `True`), `adaptive_patience` (int).

---

### 4. Visualisation improvements

**TSP — dual y-axis fitness/distance plot**

`plot_fitness_history` now adds a secondary y-axis on the right showing the best route distance (recovered from fitness as `d = 1/f − ε`). The right axis is inverted so that lower distance values appear higher, matching the fitness direction. This makes the chart directly interpretable without knowing the fitness formula. Stagnation boost events are marked as red dotted vertical lines.

**CartPole — control performance plot**

After training, `train_ga.py` automatically runs one episode with the best weights and saves `control_performance.png` to the results directory. The plot shows two panels:

- **Top**: pole angle in degrees over each timestep, with ±12° failure-threshold lines
- **Bottom**: cumulative reward over the same episode

This visualises how stable and precise the learned policy is — not just the total reward, but the moment-by-moment control quality.

**Feedback**
- Provide the graphical representation of both tasks separately at the initial stage, and further work on the necessary improvements and refinements.