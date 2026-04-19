import numpy as np


def tournament_selection(population, fitness, n_select, k=3):
    """
    Select n_select individuals via tournament selection.

    Each tournament draws k candidates at random; the one with the highest
    fitness wins and is added to the selection. Higher k = stronger
    selection pressure.

    Parameters
    ----------
    population : list of np.ndarray
    fitness    : np.ndarray, shape (len(population),)
    n_select   : int  – how many parents to return
    k          : int  – tournament size

    Returns
    -------
    list of np.ndarray (copies)
    """
    fitness = np.asarray(fitness, dtype=float)
    selected = []
    n = len(population)
    for _ in range(n_select):
        idx = np.random.choice(n, size=k, replace=False)
        winner = idx[np.argmax(fitness[idx])]
        selected.append(population[winner].copy())
    return selected


def roulette_selection(population, fitness, n_select):
    """
    Select n_select individuals proportionally to their fitness (roulette wheel).

    Requires non-negative fitness values.

    Parameters
    ----------
    population : list of np.ndarray
    fitness    : np.ndarray, shape (len(population),)  – must be >= 0
    n_select   : int

    Returns
    -------
    list of np.ndarray (copies)
    """
    fitness = np.asarray(fitness, dtype=float)
    total = fitness.sum()
    if total == 0:
        probs = np.ones(len(population)) / len(population)
    else:
        probs = fitness / total
    indices = np.random.choice(len(population), size=n_select, p=probs)
    return [population[i].copy() for i in indices]
