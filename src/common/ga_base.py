import numpy as np


class GeneticAlgorithm:
    """
    Generic genetic algorithm evolution loop.

    All domain logic is injected through callables; this class knows nothing
    about TSP or CartPole.

    Parameters
    ----------
    config : dict
        Must contain:
          pop_size      : int   – population size
          n_generations : int   – number of generations to run
          n_elite       : int   – number of best individuals preserved each generation
        Optional:
          log_every     : int   – print progress every N generations (default 10)

    init_fn : callable (config) -> list[np.ndarray]
        Returns the initial population.

    fitness_fn : callable (individual: np.ndarray) -> float
        Returns a scalar fitness value. Higher is always better.

    crossover_fn : callable (p1, p2) -> (c1, c2)
        Takes two parent arrays, returns two child arrays.

    mutation_fn : callable (individual: np.ndarray) -> np.ndarray
        Takes an individual, returns a (possibly mutated) copy.

    selection_fn : callable (population, fitness, n_select) -> list[np.ndarray]
        Returns n_select parent copies drawn from the population.
    """

    def __init__(self, config, init_fn, fitness_fn, crossover_fn, mutation_fn, selection_fn):
        self.config = config
        self.init_fn = init_fn
        self.fitness_fn = fitness_fn
        self.crossover_fn = crossover_fn
        self.mutation_fn = mutation_fn
        self.selection_fn = selection_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """
        Execute the full evolution loop.

        Returns
        -------
        dict with keys:
            best          : np.ndarray  – best individual found
            best_fitness  : float
            history_best  : list[float] – best fitness per generation
            history_avg   : list[float] – mean fitness per generation
        """
        pop_size = self.config["pop_size"]
        n_generations = self.config["n_generations"]
        n_elite = self.config.get("n_elite", 1)
        log_every = self.config.get("log_every", 10)

        population = self.init_fn(self.config)
        history_best = []
        history_avg = []

        for gen in range(n_generations):
            fitness = self._evaluate(population)

            history_best.append(float(np.max(fitness)))
            history_avg.append(float(np.mean(fitness)))

            if gen % log_every == 0 or gen == n_generations - 1:
                print(f"  Gen {gen:4d}/{n_generations} | "
                      f"best={history_best[-1]:.6f} | avg={history_avg[-1]:.6f}")

            # elitism: keep the top n_elite individuals unchanged
            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = [population[i].copy() for i in elite_idx]

            # selection + breeding to fill the rest of the population
            n_offspring = pop_size - n_elite
            parents = self.selection_fn(population, fitness, n_offspring)
            offspring = self._breed(parents)

            population = elites + offspring

        # final evaluation to find the best individual
        fitness = self._evaluate(population)
        best_idx = int(np.argmax(fitness))

        return {
            "best": population[best_idx],
            "best_fitness": float(fitness[best_idx]),
            "history_best": history_best,
            "history_avg": history_avg,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, population):
        """Return a numpy array of fitness values for every individual."""
        return np.array([self.fitness_fn(ind) for ind in population], dtype=float)

    def _breed(self, parents):
        """
        Produce len(parents) offspring from the parent list.

        Parents are consumed in pairs (with wrap-around). Each pair produces
        two children via crossover; each child is then independently mutated.
        """
        offspring = []
        n = len(parents)
        for i in range(0, n, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % n]
            c1, c2 = self.crossover_fn(p1, p2)
            offspring.append(self.mutation_fn(c1))
            offspring.append(self.mutation_fn(c2))
        return offspring[:n]  # trim to exactly n in case n is odd
