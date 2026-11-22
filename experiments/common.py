"""Shared settings and helpers for the experiment scripts."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'

from netfx import families, solve  # noqa: E402

LAM = 1.0      # intrinsic production
DLAM = 0.05    # treatment boost (all results are linear in it, or reported relative to it)
KBAR = 20      # default mean degree
N = 100_000    # default graph size
SEEDS = 10     # default number of graph realizations per point

# the families compared throughout, at the default size and mean degree
FAMILIES = [
    dict(family='ws', p=0.1),
    dict(family='er'),
    dict(family='sbm', B=100, mu=0.1),
    dict(family='ba'),
    dict(family='cm', gamma=2.5),
]


def spec(fam, n=N, kbar=KBAR, seed=0):
    return dict(fam, n=n, kbar=kbar, seed=seed)


def graph_and_rho(sp, vector=False):
    A = families.make(sp)
    rho = solve.spectral_radius(A, return_vector=vector)
    return (A,) + (rho if vector else (rho,))


def localization(v, top=0.01):
    """Share of the squared principal eigenvector on the top `top` fraction of nodes, and
    the inverse participation ratio sum(v^4) (1/n for a uniform vector)."""
    w = np.sort(v * v)[::-1]
    return float(w[:max(1, int(top * w.size))].sum()), float((w * w).sum())


def observe(c, W, rng):
    """Observed mean production over a window of W periods: Poisson(W c) / W."""
    return rng.poisson(W * np.maximum(c, 0)) / W


def mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.nan, np.nan
    return xs.mean(), (xs.std(ddof=1) / np.sqrt(xs.size) if xs.size > 1 else 0.0)
