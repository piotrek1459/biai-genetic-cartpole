import numpy as np


# ---------------------------------------------------------------------------
# Permutation operators (TSP)
# ---------------------------------------------------------------------------

def ox_crossover(parent1, parent2):
    """
    Order Crossover (OX) for permutation chromosomes.

    Copies a random segment from parent1 into the child, then fills the
    remaining positions with genes from parent2 in the order they appear
    (starting after the segment end, wrapping around), skipping genes
    already present in the child.

    Returns two children (c1, c2).
    """
    n = len(parent1)
    a, b = sorted(np.random.choice(n, 2, replace=False))

    def build_child(seg_parent, order_parent):
        child = np.full(n, -1, dtype=parent1.dtype)
        child[a:b] = seg_parent[a:b]
        segment_set = set(child[a:b].tolist())
        remaining = [x for x in np.concatenate([order_parent[b:], order_parent[:b]])
                     if x not in segment_set]
        fill_positions = list(range(b, n)) + list(range(0, a))
        for pos, val in zip(fill_positions, remaining):
            child[pos] = val
        return child

    return build_child(parent1, parent2), build_child(parent2, parent1)


def pmx_crossover(parent1, parent2):
    """
    Partially Mapped Crossover (PMX) for permutation chromosomes.

    Copies a random segment from parent1; fills the rest by copying from
    parent2, resolving conflicts by following the position-mapping defined
    by the segment.

    Returns two children (c1, c2).
    """
    n = len(parent1)
    a, b = sorted(np.random.choice(n, 2, replace=False))

    def build_child(p1, p2):
        child = np.full(n, -1, dtype=p1.dtype)
        child[a:b] = p1[a:b]

        # mapping: value in p1 segment → corresponding value in p2 segment
        for i in range(a, b):
            val = p2[i]
            if val not in child[a:b]:
                # find where to place it (follow mapping chain)
                pos = i
                while a <= pos < b:
                    mapped = p1[pos]
                    pos = np.where(p2 == mapped)[0][0]
                child[pos] = val

        # fill remaining -1 positions directly from p2
        for i in range(n):
            if child[i] == -1:
                child[i] = p2[i]
        return child

    return build_child(parent1, parent2), build_child(parent2, parent1)


# ---------------------------------------------------------------------------
# Real-valued operators (CartPole)
# ---------------------------------------------------------------------------

def uniform_crossover(parent1, parent2, prob=0.5):
    """
    Uniform crossover: each gene is drawn from a randomly chosen parent
    with probability `prob` (from parent1) or `1-prob` (from parent2).

    Returns two children (c1, c2).
    """
    mask = np.random.rand(len(parent1)) < prob
    c1 = np.where(mask, parent1, parent2)
    c2 = np.where(mask, parent2, parent1)
    return c1, c2


def arithmetic_crossover(parent1, parent2, alpha=None):
    """
    Arithmetic (blend) crossover for real-valued vectors.

    c1 = alpha * parent1 + (1-alpha) * parent2
    c2 = (1-alpha) * parent1 + alpha * parent2

    If alpha is None, a random value in [0, 1] is drawn each call.

    Returns two children (c1, c2).
    """
    if alpha is None:
        alpha = np.random.rand()
    c1 = alpha * parent1 + (1 - alpha) * parent2
    c2 = (1 - alpha) * parent1 + alpha * parent2
    return c1, c2
