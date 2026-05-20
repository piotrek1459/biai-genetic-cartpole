import numpy as np
from concurrent.futures import ThreadPoolExecutor


class GeneticAlgorithm:
    """
    Generic genetic algorithm evolution loop.

    All domain logic is injected through callables; this class knows nothing
    about TSP or CartPole.

    Parameters
    ----------
    config : dict
        Must contain:
          pop_size          : int   – population size
          n_generations     : int   – number of generations to run
          n_elite           : int   – number of best individuals preserved each generation
        Optional:
          log_every         : int   – print progress every N generations (default 10)
          n_jobs            : int   – parallel fitness evaluations; 1=serial, -1=all cores
          adaptive_mutation : bool  – double-apply mutation when fitness stagnates
          adaptive_patience : int   – stagnation window before boost fires (default 20)

    init_fn      : (config) -> list[np.ndarray]
    fitness_fn   : (individual) -> float           higher is always better
    crossover_fn : (p1, p2) -> (c1, c2)
    mutation_fn  : (individual) -> np.ndarray
    selection_fn : (population, fitness, n_select) -> list[np.ndarray]
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
            best               : np.ndarray  – best individual found
            best_fitness       : float
            history_best       : list[float] – best fitness per generation
            history_avg        : list[float] – mean fitness per generation
            stagnation_events  : list[int]   – generations where adaptive boost fired
        """
        pop_size      = self.config["pop_size"]
        n_generations = self.config["n_generations"]
        n_elite       = self.config.get("n_elite", 1)
        log_every     = self.config.get("log_every", 10)
        adaptive      = self.config.get("adaptive_mutation", True)
        patience      = self.config.get("adaptive_patience", 20)

        population = self.init_fn(self.config)
        history_best      = []
        history_avg       = []
        stagnation_events = []

        best_so_far     = -np.inf
        stagnation_count = 0

        for gen in range(n_generations):
            fitness = self._evaluate(population)

            current_best = float(np.max(fitness))
            history_best.append(current_best)
            history_avg.append(float(np.mean(fitness)))

            # stagnation tracking
            if current_best > best_so_far + 1e-8:
                best_so_far = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1

            boost = adaptive and stagnation_count >= patience
            if boost:
                stagnation_events.append(gen)
                stagnation_count = 0  # reset after firing

            if gen % log_every == 0 or gen == n_generations - 1:
                tag = " [BOOST]" if boost else ""
                print(f"  Gen {gen:4d}/{n_generations} | "
                      f"best={history_best[-1]:.6f} | avg={history_avg[-1]:.6f}{tag}")

            # elitism
            elite_idx = np.argsort(fitness)[-n_elite:]
            elites = [population[i].copy() for i in elite_idx]

            # selection + breeding
            n_offspring = pop_size - n_elite
            parents  = self.selection_fn(population, fitness, n_offspring)
            offspring = self._breed(parents, boost=boost)

            population = elites + offspring

        # final evaluation
        fitness  = self._evaluate(population)
        best_idx = int(np.argmax(fitness))

        return {
            "best":              population[best_idx],
            "best_fitness":      float(fitness[best_idx]),
            "history_best":      history_best,
            "history_avg":       history_avg,
            "stagnation_events": stagnation_events,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, population):
        """
        Return a numpy array of fitness values.

        Uses ThreadPoolExecutor when n_jobs != 1.  Threads work well for
        fitness functions that release the GIL (e.g. Gymnasium step loops).
        """
        n_jobs = self.config.get("n_jobs", 1)
        if n_jobs == 1:
            return np.array([self.fitness_fn(ind) for ind in population], dtype=float)

        max_workers = None if n_jobs == -1 else n_jobs
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(self.fitness_fn, population))
        return np.array(results, dtype=float)

    def _breed(self, parents, boost=False):
        """
        Produce len(parents) offspring via crossover + mutation.

        When boost=True (stagnation detected), mutation is applied twice per
        offspring — increasing exploration without changing operator signatures.
        This is the RL-inspired adaptive exploration mechanism: when the reward
        signal (fitness improvement) flatlines, exploration is increased.
        """
        offspring = []
        n = len(parents)
        for i in range(0, n, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % n]
            c1, c2 = self.crossover_fn(p1, p2)
            c1 = self.mutation_fn(c1)
            c2 = self.mutation_fn(c2)
            if boost:
                c1 = self.mutation_fn(c1)
                c2 = self.mutation_fn(c2)
            offspring.append(c1)
            offspring.append(c2)
        return offspring[:n]
