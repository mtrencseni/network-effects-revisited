"""Section 6: experiment duration.

The network starts in its untreated steady state c0 = lam (I - nu A)^{-1} 1. At t = 0
treatment starts, and c_t = lam + dlam z + nu A c_{t-1}. The measured lift DM(t) and
the full-rollout lift GATE(t) approach their limits geometrically, at rate nu rho(A)
for undirected graphs.

Parts:
  paths     DM(t) / DM(inf) and GATE(t) / GATE(inf) for each family and nu rho
  peeking   sequential testing on daily Poisson counts: stopping time and the estimate
            reported at stopping, relative to the steady-state DM and the GATE
"""
import sys

import numpy as np

from common import DLAM, FAMILIES, KBAR, LAM, N, RESULTS, SEEDS, graph_and_rho, spec
from netfx import designs, estimators, families, solve, sweep

T = 200
DIRECTED = [dict(family='dws', p=0.1), dict(family='dba')]


def _nu(cfg, A, rho):
    if cfg['fam']['family'] == 'dba':     # acyclic: rho = 0, parameterize by nu kbar
        return cfg['nu_kbar'] / (A.nnz / A.shape[0])
    return cfg['nu_rho'] / rho


def paths(cfg):
    sp = spec(cfg['fam'], n=cfg.get('n', N), seed=cfg['seed'])
    A = families.make(sp)
    rho = 0.0 if cfg['fam']['family'] == 'dba' else solve.spectral_radius(A)
    nu = _nu(cfg, A, rho)
    n = A.shape[0]
    rng = np.random.default_rng([cfg['seed'], 6])
    z = designs.bernoulli(n, cfg['p'], rng).astype(float)
    # lift dynamics are linear: L_t = dlam z + nu A L_{t-1}, L_0 = 0 (same for full rollout)
    L, G = np.zeros(n), np.zeros(n)
    dm, gate = [], []
    zb = z.astype(bool)
    for t in range(cfg.get('T', T)):
        L = DLAM * z + nu * (A @ L)
        G = DLAM + nu * (A @ G)
        dm.append(L[zb].mean() - L[~zb].mean())
        gate.append(G.mean())
    dm_inf = estimators.steady(A, DLAM * z, nu)
    dm_inf = dm_inf[zb].mean() - dm_inf[~zb].mean()
    gate_inf = estimators.gate(A, nu, DLAM)
    return dict(rho=rho, nu=nu, kbar=A.nnz / n, dm=dm, gate=gate, dm_inf=dm_inf,
                gate_inf=gate_inf)


def peeking(cfg):
    """R experiments on one graph, each with its own 50/50 assignment, daily Poisson counts.
    Sequential: one-sided z-test (z > 1.645) every day, stop at the first rejection.
    Fixed horizon: one test on day H. Both use the same outcomes (baseline included)."""
    sp = spec(cfg['fam'], n=cfg.get('n', N), seed=cfg['seed'])
    A, rho = graph_and_rho(sp)
    nu = cfg['nu_rho'] / rho
    n = A.shape[0]
    rng = np.random.default_rng([cfg['seed'], 7])
    dl = cfg['effect']
    c0 = LAM * estimators.steady(A, np.ones(n), nu)
    gate_inf = estimators.gate(A, nu, dl) if dl else 0.0
    out = []
    for r in range(cfg['R']):
        z = designs.bernoulli(n, 0.5, rng)
        L = np.zeros(n)
        sum_t = np.zeros(n)
        stop, est, fixed, fixed_rej = None, None, None, None
        for t in range(1, cfg['days'] + 1):
            if dl:
                L = dl * z + nu * (A @ L)
            sum_t += rng.poisson(np.maximum(c0 + L, 0))
            m = sum_t / t
            d = m[z].mean() - m[~z].mean()
            se = np.sqrt(m[z].var(ddof=1) / z.sum() + m[~z].var(ddof=1) / (~z).sum())
            if stop is None and d / se > 1.645:
                stop, est = t, d
            if t == cfg['H']:
                fixed, fixed_rej = d, bool(d / se > 1.645)
        out.append(dict(stop=stop, est=est, fixed=fixed, fixed_rej=fixed_rej))
    return dict(runs=out, gate_inf=gate_inf, rho=rho)


def main(parts):
    if 'paths' in parts:
        cfgs = [dict(fam=f, p=0.1, nu_rho=x, seed=s) for f in FAMILIES + DIRECTED[:1]
                for x in [0.5, 0.8, 0.9, 0.95, 0.99] for s in range(SEEDS // 2)]
        cfgs += [dict(fam=DIRECTED[1], p=0.1, nu_kbar=x, seed=s)
                 for x in [0.5, 0.9, 1.5, 3.0] for s in range(SEEDS // 2)]
        cfgs += [dict(fam=f, p=0.1, nu_rho=0.99, T=800, seed=s) for f in FAMILIES
                 for s in range(SEEDS // 2)]
        sweep.run(paths, cfgs, str(RESULTS / 's6_paths.jsonl'))
    if 'peeking' in parts:
        cfgs = [dict(fam=f, nu_rho=x, effect=e, days=60, H=14, R=40, seed=s)
                for f in FAMILIES
                for x in [0.5, 0.9, 0.97] for e in [0.0, 0.002] for s in range(SEEDS // 2)]
        sweep.run(peeking, cfgs, str(RESULTS / 's6_peeking2.jsonl'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['paths', 'peeking'])
