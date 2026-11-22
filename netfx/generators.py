"""Fast random graph generators, returning symmetric CSR adjacency matrices.

These follow the networkx models but are vectorized (or compiled with numba)
and take an explicit seed, so they run in parallel worker processes. They
match networkx in distribution, not draw for draw; the exact 2023 replay in
scripts/reproduce_2023.py uses networkx itself.
"""
import numpy as np
from numba import njit

from . import graph


def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _keys(u, v, n):
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    return lo * n + hi


def unique(x):
    """Sorted unique values (numpy's hash-based np.unique is slow for large int arrays)."""
    x = np.sort(x)
    if x.size == 0:
        return x
    keep = np.empty(x.size, dtype=bool)
    keep[0] = True
    np.not_equal(x[1:], x[:-1], out=keep[1:])
    return x[keep]


def _member(sorted_keys, x):
    """Boolean mask: which x are in the sorted array sorted_keys."""
    if sorted_keys.size == 0:
        return np.zeros(x.size, dtype=bool)
    i = np.searchsorted(sorted_keys, x)
    i[i == sorted_keys.size] = 0
    return sorted_keys[i] == x


def is_connected(A):
    return A.components()[0] == 1


@njit(cache=True)
def _ws_edges(n, k, p, seed):
    """networkx.watts_strogatz_graph, step by step: edges (u, u+j) for j = 1..k/2, in that
    order, each rewired with probability p to (u, w), where w is drawn uniformly until it
    is neither u nor a current neighbour of u."""
    np.random.seed(seed)
    m = n * (k // 2)
    src = np.empty(m, dtype=np.int64)
    dst = np.empty(m, dtype=np.int64)
    present = set()
    e = 0
    for j in range(1, k // 2 + 1):
        for u in range(n):
            v = (u + j) % n
            src[e] = u
            dst[e] = v
            present.add(min(u, v) * n + max(u, v))
            e += 1
    deg = np.full(n, k, dtype=np.int64)
    e = 0
    for j in range(1, k // 2 + 1):
        for u in range(n):
            v = (u + j) % n
            if np.random.random() < p:
                if deg[u] >= n - 1:
                    e += 1
                    continue
                w = np.random.randint(0, n)
                while w == u or (min(u, w) * n + max(u, w)) in present:
                    w = np.random.randint(0, n)
                present.discard(min(u, v) * n + max(u, v))
                present.add(min(u, w) * n + max(u, w))
                deg[v] -= 1
                deg[w] += 1
                dst[e] = w
            e += 1
    return src, dst


def watts_strogatz(n, k, p, seed=None):
    """Watts-Strogatz small-world graph, same algorithm as networkx."""
    rng = _rng(seed)
    assert k % 2 == 0 and k < n
    u, v = _ws_edges(n, k, p, int(rng.integers(0, 2**31 - 1)))
    return graph.from_edges(n, u, v)


def connected_watts_strogatz(n, k, p, seed=None, tries=100):
    rng = _rng(seed)
    for _ in range(tries):
        A = watts_strogatz(n, k, p, rng)
        if is_connected(A):
            return A
    raise RuntimeError('no connected graph found')


def erdos_renyi(n, mean_degree, seed=None):
    """G(n, p) with p = mean_degree / (n - 1), for sparse graphs."""
    rng = _rng(seed)
    pairs = n * (n - 1) // 2
    m = rng.binomial(pairs, mean_degree / (n - 1))
    keys = np.empty(0, dtype=np.int64)
    while keys.size < m:
        need = int((m - keys.size) * 1.1) + 16
        u = rng.integers(0, n, need)
        v = rng.integers(0, n, need)
        ok = u != v
        keys = unique(np.concatenate([keys, _keys(u[ok], v[ok], n)]))
    keys = rng.permutation(keys)[:m]
    return graph.from_edges(n, keys // n, keys % n)


@njit(cache=True)
def _ba_edges(n, m, seed):
    np.random.seed(seed)
    src = np.empty((n - m) * m, dtype=np.int64)
    dst = np.empty((n - m) * m, dtype=np.int64)
    repeated = np.empty(2 * (n - m) * m + m, dtype=np.int64)
    r = 0
    targets = np.arange(m)
    e = 0
    chosen = np.empty(m, dtype=np.int64)
    for s in range(m, n):
        for i in range(m):
            src[e] = s
            dst[e] = targets[i]
            e += 1
            repeated[r] = targets[i]
            r += 1
            repeated[r] = s
            r += 1
        # pick m distinct targets proportional to degree
        c = 0
        while c < m:
            x = repeated[np.random.randint(0, r)]
            dup = False
            for q in range(c):
                if chosen[q] == x:
                    dup = True
                    break
            if not dup:
                chosen[c] = x
                c += 1
        targets = chosen.copy()
    return src, dst


def barabasi_albert(n, m, seed=None):
    """Preferential attachment, m edges per new node (networkx algorithm)."""
    rng = _rng(seed)
    u, v = _ba_edges(n, m, int(rng.integers(0, 2**31 - 1)))
    return graph.from_edges(n, u, v)


def configuration(degrees, seed=None):
    """Erased configuration model: random stub matching, self-loops and multi-edges removed."""
    rng = _rng(seed)
    degrees = np.asarray(degrees, dtype=np.int64)
    if degrees.sum() % 2:
        degrees = degrees.copy()
        degrees[rng.integers(0, degrees.size)] += 1
    stubs = rng.permutation(np.repeat(np.arange(degrees.size, dtype=np.int64), degrees))
    u, v = stubs[0::2], stubs[1::2]
    ok = u != v
    keys = unique(_keys(u[ok], v[ok], degrees.size))
    n = degrees.size
    return graph.from_edges(n, keys // n, keys % n)


def power_law_degrees(n, gamma, k_min, k_max=None, seed=None):
    """Integer degrees from a discrete power law P(k) ~ k^-gamma on [k_min, k_max]."""
    rng = _rng(seed)
    k_max = k_max or int(np.sqrt(n))
    ks = np.arange(k_min, k_max + 1)
    w = ks.astype(np.float64) ** (-gamma)
    return rng.choice(ks, size=n, p=w / w.sum())


def sbm(sizes, P, seed=None):
    """Stochastic block model with block sizes and symmetric edge-probability matrix P."""
    rng = _rng(seed)
    sizes = np.asarray(sizes, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(sizes)])
    n = int(off[-1])
    us, vs = [], []
    for a in range(sizes.size):
        for b in range(a, sizes.size):
            pab = P[a][b]
            if pab <= 0:
                continue
            if a == b:
                pairs = sizes[a] * (sizes[a] - 1) // 2
            else:
                pairs = sizes[a] * sizes[b]
            m = rng.binomial(pairs, pab)
            keys = np.empty(0, dtype=np.int64)
            while keys.size < m:
                need = int((m - keys.size) * 1.1) + 16
                x = rng.integers(0, sizes[a], need) + off[a]
                y = rng.integers(0, sizes[b], need) + off[b]
                ok = x != y
                keys = unique(np.concatenate([keys, _keys(x[ok], y[ok], n)]))
            keys = rng.permutation(keys)[:m]
            us.append(keys // n)
            vs.append(keys % n)
    return graph.from_edges(n, np.concatenate(us), np.concatenate(vs))
