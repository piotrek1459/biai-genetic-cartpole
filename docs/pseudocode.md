# Pseudocode — Genetic Algorithm

## The Plugin Interface

The `GeneticAlgorithm` class requires exactly **5 problem-specific functions**.
The core loop never changes — only these functions differ between TSP, CartPole, or any future task.

```
FUNCTION GeneticAlgorithm(init_fn, fitness_fn, crossover_fn, mutation_fn, selection_fn, config)
```

| Callable | Signature | Responsibility |
|----------|-----------|----------------|
| `init_fn` | `(config) → list[individual]` | Create the initial population |
| `fitness_fn` | `(individual) → float` | Score one individual (higher = better) |
| `crossover_fn` | `(p1, p2) → (c1, c2)` | Combine two parents into two children |
| `mutation_fn` | `(individual) → individual` | Randomly perturb one individual |
| `selection_fn` | `(pop, fitness, n) → list[individual]` | Pick n parents from the population |

---

## Full Pseudocode

```
ALGORITHM GeneticAlgorithm.run()

─── INITIALISE ──────────────────────────────────────────────────────────
  population      ← init_fn(config)        // pop_size random chromosomes
  best_so_far     ← -∞
  stagnation_count ← 0
  history_best    ← []
  history_avg     ← []

─── MAIN LOOP ───────────────────────────────────────────────────────────
  FOR gen = 0 TO n_generations - 1 DO:

    // 1. EVALUATE  (parallel via ThreadPoolExecutor if n_jobs ≠ 1)
    FOR each individual i in population DO:
      fitness[i] ← fitness_fn(population[i])

    // 2. RECORD HISTORY
    history_best.append( max(fitness) )
    history_avg.append(  mean(fitness) )

    // 3. STAGNATION CHECK  — RL-inspired adaptive mutation
    IF max(fitness) > best_so_far + 1e-8 THEN
      best_so_far      ← max(fitness)
      stagnation_count ← 0
      boost            ← FALSE
    ELSE
      stagnation_count ← stagnation_count + 1
      IF stagnation_count ≥ adaptive_patience THEN
        boost            ← TRUE
        stagnation_count ← 0        // reset; fire extra mutation this generation
      ELSE
        boost ← FALSE

    // 4. ELITISM  — preserve top n_elite unchanged
    elites ← individuals at top n_elite positions of fitness

    // 5. SELECTION
    n_offspring ← pop_size - n_elite
    parents ← selection_fn(population, fitness, n_offspring)
    // options: tournament(k=4)  or  roulette-wheel

    // 6. CROSSOVER + MUTATION
    offspring ← []
    FOR i = 0, 2, 4, ... TO n_offspring - 1 DO:
      p1 ← parents[i]
      p2 ← parents[(i + 1) mod n_offspring]

      (c1, c2) ← crossover_fn(p1, p2)
      // TSP:      OX or PMX  (permutation-safe)
      // CartPole: uniform or arithmetic blend

      c1 ← mutation_fn(c1)
      c2 ← mutation_fn(c2)
      // TSP:      inversion / swap / scramble
      // CartPole: gaussian noise N(0, sigma)

      IF boost THEN                 // double mutation breaks stagnation
        c1 ← mutation_fn(c1)
        c2 ← mutation_fn(c2)

      offspring.append(c1)
      offspring.append(c2)

    // 7. BUILD NEW POPULATION
    population ← elites  ∪  offspring[0 : n_offspring]

  END FOR

─── FINAL EVALUATION ────────────────────────────────────────────────────
  fitness ← evaluate entire population
  best    ← individual with highest fitness

─── RETURN ──────────────────────────────────────────────────────────────
  RETURN {
    best             : best individual (chromosome),
    best_fitness     : float,
    history_best     : list[float],   // one value per generation
    history_avg      : list[float],
    stagnation_events: list[int]      // generations where boost fired
  }
```

---

## Mapping: Pseudocode → Source Code

| Pseudocode step | File | Approx. lines |
|----------------|------|--------------|
| `population ← init_fn(config)` | `ga_base.py` | 64 |
| `fitness[i] ← fitness_fn(...)` | `ga_base.py` `_evaluate()` | 131–136 |
| Stagnation check + boost | `ga_base.py` | 77–94 |
| `elites ← top n_elite` | `ga_base.py` | 96–98 |
| `parents ← selection_fn(...)` | `ga_base.py` | 101–102 |
| `crossover_fn` + `mutation_fn` | `ga_base.py` `_breed()` | 150–161 |
| `population ← elites ∪ offspring` | `ga_base.py` | 105 |

---

## How to Add a New Problem (Task N)

1. Create `src/taskN_name/` with `__init__.py`
2. Implement the 5 functions:
   - `init_population(...)` → wrap as `init_fn`
   - `make_fitness_fn(...)` → return `fitness_fn` closure
   - Choose or implement `crossover_fn` (reuse from `common/` if suitable)
   - Choose or implement `mutation_fn`
   - Reuse `tournament_selection` or `roulette_selection` as `selection_fn`
3. Instantiate `GeneticAlgorithm(config, init_fn, fitness_fn, crossover_fn, mutation_fn, selection_fn)`
4. Call `.run()` — **zero changes to `ga_base.py`**
