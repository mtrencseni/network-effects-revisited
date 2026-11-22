"""Graph families used in the experiments, addressed by a small spec dict.

Every family takes n, a target mean degree kbar and a seed:

  ws    Watts-Strogatz ring with rewiring probability p (default 0.1)
  er    Erdos-Renyi G(n, p)
  ba    Barabasi-Albert preferential attachment (m = kbar / 2)
  cm    erased configuration model with power-law degrees, exponent gamma (default 2.5)
  gam   erased configuration model with gamma-distributed degrees, coefficient of variation cv
  sbm   stochastic block model with B equal blocks and mixing mu (fraction of edges
        between blocks)
  dws   directed: each node sees kbar nodes, a WS-like ring with rewiring p
  dba   directed follower graph: each new node follows m = kbar earlier nodes chosen by
        preferential attachment, and sees only the nodes it follows (acyclic)
"""
import numpy as np

from . import generators as G
from . import graph


def make(spec):
    f = spec['family']
    n, kbar, seed = int(spec['n']), float(spec['kbar']), spec.get('seed', 0)
    rng = np.random.default_rng(seed)
    if f == 'ws':
        return G.connected_watts_strogatz(n, int(kbar), spec.get('p', 0.1), rng)
    if f == 'er':
        return G.erdos_renyi(n, kbar, rng)
    if f == 'ba':
        return G.barabasi_albert(n, int(kbar) // 2, rng)
    if f == 'cm':
        gamma = spec.get('gamma', 2.5)
        k_min = _k_min_for_mean(gamma, kbar, n)
        return G.configuration(G.power_law_degrees(n, gamma, k_min, seed=rng), rng)
    if f == 'gam':
        cv = spec['cv']
        if cv == 0:
            deg = np.full(n, int(round(kbar)))
        else:
            shape = 1 / cv**2
            deg = np.maximum(1, np.round(rng.gamma(shape, kbar / shape, n))).astype(np.int64)
        return G.configuration(deg, rng)
    if f == 'sbm':
        B, mu = int(spec.get('B', 100)), spec.get('mu', 0.1)
        size = n // B
        p_in = kbar * (1 - mu) / (size - 1)
        p_out = kbar * mu / (n - size)
        P = np.full((B, B), p_out)
        np.fill_diagonal(P, p_in)
        return G.sbm([size] * B, P, rng)
    if f == 'dws':
        return _directed_ring(n, int(kbar), spec.get('p', 0.1), rng)
    if f == 'dba':
        return _directed_follow(n, int(kbar), rng)
    raise ValueError(f)


def _k_min_for_mean(gamma, kbar, n):
    best, err = 1, 1e9
    for k_min in range(1, 200):
        ks = np.arange(k_min, int(np.sqrt(n)) + 1)
        w = ks.astype(float) ** (-gamma)
        mean = (ks * w).sum() / w.sum()
        if abs(mean - kbar) < err:
            best, err = k_min, abs(mean - kbar)
    return best


def _directed_ring(n, k, p, rng):
    u = np.repeat(np.arange(n, dtype=np.int64), k)
    off = np.tile(np.concatenate([np.arange(1, k // 2 + 1), -np.arange(1, k - k // 2 + 1)]), n)
    v = (u + off) % n
    rewire = rng.random(u.size) < p
    v[rewire] = rng.integers(0, n, rewire.sum())
    return graph.from_edges(n, u, v, directed=True)


def _directed_follow(n, m, rng):
    """Node s >= m follows m distinct earlier nodes, chosen by preferential attachment
    on follower counts (+1); a node sees the nodes it follows."""
    u, v = G._ba_edges(n, m, int(rng.integers(0, 2**31 - 1)))
    # _ba_edges returns (new node, earlier node) pairs
    return graph.from_edges(n, u, v, directed=True)


def label(spec):
    f = spec['family']
    # the default rewiring (p = 0.1) and inter-block share (mu = 0.1) are not shown: in the paper
    # p is the treated fraction and mu a Poisson mean; non-default values are
    extra = {'ws': f"p={spec['p']}" if spec.get('p', 0.1) != 0.1 else '',
             'cm': f"γ={spec.get('gamma', 2.5)}", 'gam': f"cv={spec.get('cv')}",
             'sbm': f"μ={spec['mu']}" if spec.get('mu', 0.1) != 0.1 else ''}.get(f, '')
    names = {'ws': 'Watts-Strogatz', 'er': 'Erdős-Rényi', 'ba': 'Barabási-Albert',
             'cm': 'power-law', 'gam': 'gamma configuration', 'sbm': 'SBM',
             'dws': 'directed ring', 'dba': 'directed follower'}
    return names[f] + (f' ({extra})' if extra else '')
