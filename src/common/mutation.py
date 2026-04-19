import numpy as np


# ---------------------------------------------------------------------------
# Permutation operators (TSP)
# ---------------------------------------------------------------------------

def swap_mutation(individual, mutation_rate):
    """
    With probability mutation_rate, swap two randomly chosen genes.

    Always produces a valid permutation.
    """
    if np.random.rand() < mutation_rate:
        ind = individual.copy()
        i, j = np.random.choice(len(ind), 2, replace=False)
        ind[i], ind[j] = ind[j], ind[i]
        return ind
    return individual.copy()


def inversion_mutation(individual, mutation_rate):
    """
    With probability mutation_rate, reverse a random sub-sequence.

    Stronger perturbation than swap; typically works best for TSP.
    """
    if np.random.rand() < mutation_rate:
        ind = individual.copy()
        i, j = sorted(np.random.choice(len(ind), 2, replace=False))
        ind[i:j + 1] = ind[i:j + 1][::-1]
        return ind
    return individual.copy()


def scramble_mutation(individual, mutation_rate):
    """
    With probability mutation_rate, shuffle a random sub-sequence.

    Useful when the population loses diversity; stronger than inversion.
    """
    if np.random.rand() < mutation_rate:
        ind = individual.copy()
        i, j = sorted(np.random.choice(len(ind), 2, replace=False))
        segment = ind[i:j + 1].copy()
        np.random.shuffle(segment)
        ind[i:j + 1] = segment
        return ind
    return individual.copy()


# ---------------------------------------------------------------------------
# Real-valued operators (CartPole)
# ---------------------------------------------------------------------------

def gaussian_mutation(individual, mutation_rate, sigma=0.1):
    """
    Independently perturb each gene with probability mutation_rate
    by adding Gaussian noise N(0, sigma).
    """
    ind = individual.copy()
    mask = np.random.rand(len(ind)) < mutation_rate
    ind[mask] += np.random.randn(int(mask.sum())) * sigma
    return ind
