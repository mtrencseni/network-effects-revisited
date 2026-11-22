"""Treatment assignment designs and graph partitions."""
import numpy as np


def bernoulli(n, p, rng):
    return rng.random(n) < p


def complete(n, N, rng):
    z = np.zeros(n, dtype=bool)
    z[rng.choice(n, N, replace=False)] = True
    return z


def cluster(labels, p, rng):
    """Graph cluster randomization: each cluster is treated with probability p."""
    k = labels.max() + 1
    return (rng.random(k) < p)[labels]


def metis(A, k, seed=0):
    """Balanced k-way partition of an undirected graph (METIS)."""
    import pymetis
    if k <= 1:
        return np.zeros(A.shape[0], dtype=np.int64)
    opts = pymetis.Options(seed=int(seed))
    _, parts = pymetis.part_graph(int(k), xadj=A.indptr, adjncy=A.indices, options=opts)
    return np.asarray(parts, dtype=np.int64)


def contiguous(n, size):
    """Consecutive node ranges, the natural clusters of a ring lattice."""
    return np.arange(n, dtype=np.int64) // size


def targeted(score, N):
    """The N nodes with the highest score (degree, centrality, ...)."""
    z = np.zeros(score.size, dtype=bool)
    z[np.argsort(-score, kind='stable')[:N]] = True
    return z


def exposure(A, z):
    """Fraction of each node's neighbours (in-neighbours for directed graphs) that are treated."""
    deg = np.diff(A.indptr).astype(np.float64)
    return np.divide(A @ z.astype(np.float64), deg, out=np.zeros(deg.size), where=deg > 0)
