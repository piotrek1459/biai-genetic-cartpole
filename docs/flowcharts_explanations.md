# Flowcharts Explanations — Genetic Algorithm (TSP & CartPole)

This document explains the meaning of each diagram in [docs/flowcharts.md](docs/flowcharts.md). The descriptions are intentionally direct and detailed so each chart can be read as a faithful explanation of the code structure and execution flow.

---

## 1. Module Architecture

This diagram shows the project structure and the relationship between the shared genetic algorithm core and the two task-specific implementations.

The `src/common/` package contains the reusable parts of the project. `ga_base.py` defines the `GeneticAlgorithm` class, which implements the generic evolutionary loop. `selection.py` provides the parent selection strategies, `crossover.py` contains crossover operators, and `mutation.py` contains mutation operators.

The `src/task1_tsp/` package implements the Traveling Salesperson Problem variant. `tsp_ga.py` defines the task configuration, the population initializer, the fitness function, the city generator, and the route distance helpers. `visualization.py` produces the plots for fitness and routes. `experiments.py` runs grouped experiments and aggregates their results.

The `src/task2_cartpole/` package implements the CartPole control task. `cartpole_policy.py` defines the linear policy representation and evaluation helpers. `train_ga.py` defines the GA training entry point, the population initializer, the fitness function, the episode runner, and the comparison video helper. `visualization.py` creates the reward and control plots. `evaluate.py` renders a trained agent. `experiments.py` runs grouped experiments for this task.

The arrows in the diagram mean dependency flow. The task modules build `GeneticAlgorithm` objects from `src/common/`, and they pass task-specific selection, crossover, and mutation functions into the shared core. The label on each arrow describes the kind of data or operator being passed.

---

## 2. General GA Loop

This diagram describes the exact control flow of `GeneticAlgorithm.run()` and `_breed()` in `src/common/ga_base.py`.

The run starts by creating an initial population with `init_fn(config)`. It also initializes the tracking structures used during evolution: the best fitness history, the average fitness history, the current best fitness value, and the stagnation counter.

At every generation, the algorithm first evaluates the whole population. The evaluation happens either serially or in parallel, depending on `n_jobs`. In serial mode, each individual is passed to `fitness_fn` one by one. In parallel mode, a `ThreadPoolExecutor` maps `fitness_fn` over the population. The results are always converted into a NumPy float array.

After evaluation, the algorithm records the best fitness and the average fitness of the current generation. It then checks whether the current best individual improved over the previous global best by a small epsilon threshold. If the fitness improved, the stagnation counter is reset. If it did not improve, the stagnation counter is increased.

When adaptive mutation is enabled, and the stagnation counter reaches the configured patience limit, the algorithm activates the boost mode. This boost is recorded as a stagnation event and is used to increase exploration in the next breeding step.

Elitism is applied next. The best `n_elite` individuals are copied directly into the next population so that the strongest solutions are preserved unchanged.

The remaining individuals are produced by selection and breeding. The selection function chooses enough parents to fill the non-elite part of the population. The diagram shows two supported selection modes: tournament selection and roulette selection. Tournament selection repeatedly draws a fixed number of candidates and keeps the one with the highest fitness. Roulette selection assigns selection probability proportional to fitness and samples parents according to that distribution.

The `_breed()` step processes the parents in pairs. Each pair is passed to the crossover function, which produces two children. The crossover method depends on the task: order crossover and partially mapped crossover are used for permutations, while uniform crossover and arithmetic crossover are used for continuous vectors. After crossover, each child is mutated once. If boost mode is active, mutation is applied a second time to increase search pressure. The resulting offspring are appended until the new population is complete.

The next generation is built by combining elites and offspring. The population size stays constant across generations, and the loop repeats until the configured number of generations is reached. At the end, the algorithm evaluates the final population one last time and returns the best individual, its fitness, the recorded history, and the stagnation events.

---

## 3. TSP — Full Execution Flow

This diagram describes the complete execution of `run_tsp()` in `src/task1_tsp/tsp_ga.py`.

The run begins by creating the results directory and seeding the random number generators. The project then generates a fixed set of 20 cities in the unit square using `generate_cities(n=20, seed=42)`. Each city is represented as a two-dimensional coordinate.

Next, the code creates an initial random permutation of the cities. This permutation represents a full tour. The route distance of that initial tour is computed immediately so the starting point can be compared with the final solution.

The initial route is plotted and saved as a visual baseline. This gives a direct before-and-after comparison for the optimization process.

After that, the algorithm configures the genetic operators. For this task, the chromosome is a permutation of city indices, so the operators must preserve permutation validity. That is why the diagram lists order crossover or partially mapped crossover, inversion mutation, and tournament selection.

The fitness function is built with `make_fitness_fn(cities, ε=1e-6, w=0.2)`. The diagram breaks this computation into three steps. First, `route_distance()` computes the total closed-tour length, meaning the distance includes the return edge from the last city back to the first one. Second, the route smoothness term is computed from the coefficient of variation of the segment lengths. A lower coefficient of variation means the route segments are more balanced, so the smoothness score is higher. Third, the final fitness is computed as the inverse distance multiplied by a smoothness bonus. This means shorter and more balanced routes receive higher fitness.

The `GeneticAlgorithm` instance is then created with the task-specific configuration, population initializer, fitness function, crossover function, mutation function, and selection function. Parallel evaluation is enabled with `n_jobs=-1`, and adaptive mutation is enabled so that the algorithm can react to stagnation.

The main GA loop then runs using the shared logic shown in Diagram 2. In this task, each chromosome is a length-20 permutation, the population size is 80, the run lasts 300 generations, and elitism keeps 2 individuals unchanged each generation.

When the run finishes, the code extracts the best permutation found and recomputes its route distance. It then plots the full fitness history, including the distance conversion and the stagnation boost markers, and it plots the final best route. Finally, the best permutation is saved to `best_route.npy` so it can be reused later.

The output files shown at the end of the diagram are the final artifacts of the TSP run: the initial route image, the fitness history image, the best route image, and the NumPy file containing the best tour.

---

## 4. CartPole — Full Execution Flow

This diagram describes the complete execution of `run_cartpole()` in `src/task2_cartpole/train_ga.py`.

The run begins by creating the output directories for the CartPole task and seeding the random number generators. The code then creates a random linear policy by sampling five weights from a normal distribution and scaling them. These five values represent the four observation weights plus the bias term.

Before training starts, the code evaluates this random policy in one episode using `_run_episode()`. That helper opens a Gymnasium CartPole environment in RGB rendering mode, resets it with the requested seed, and repeatedly steps through the environment until the episode ends. On every step, it records the pole angle, the cart position, and the rendered frame. The policy action is chosen by the sign of the linear score computed from the observation.

The recorded data from the random policy is plotted and saved as the initial control visualization. The frames are also written to a video file so the untrained policy can be inspected visually.

After that, the fitness function is defined. Unlike TSP, CartPole fitness is based on episode performance over multiple runs. The diagram shows that `make_fitness_fn()` uses five episodes by default and applies a stability penalty. For each candidate weight vector, `evaluate_policy_with_stats()` is run several times. The mean reward is computed across those episodes, and the standard deviation is subtracted with a small penalty weight. This means the algorithm prefers policies that score well consistently, not just occasionally.

The `GeneticAlgorithm` instance is then created. The chromosome here is a five-element float vector, not a permutation. That is why the operator choices are different from the TSP task. Uniform crossover is used because the genes are continuous parameters, and Gaussian mutation is used to perturb the weights. Tournament selection is used again to choose parents. Adaptive mutation is enabled so that the algorithm can react to lack of progress.

The shared GA loop then runs with this CartPole-specific configuration. In this task, the population size is 50, the run lasts 100 generations, and elitism keeps 2 individuals unchanged.

Once training ends, the best weight vector is saved to `best_weights.npy`, and the reward history is plotted. The trained policy is then evaluated again with `_run_episode()` to generate a new episode trace and video.

The trained control plot shows how the angle and cart position behave after optimization. The comparison plot places the random policy traces and the trained policy traces side by side so the improvement is easy to inspect. Finally, the comparison video concatenates the random and trained agent videos frame by frame, producing a direct visual comparison.

The output files shown at the end of the diagram are the final artifacts of the CartPole run: the reward history plot, the saved best weights, the initial control plot and video, the trained control plot and video, the comparison plot, and the comparison video.