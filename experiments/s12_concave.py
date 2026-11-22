"""Concave peer response: c = lam + dlam z + nu g(A c), g(s) = K log(1 + s/K) or s/(1 + s/K).

The saturation scale K is set relative to the typical exposure kbar * lam (sigma = K / (kbar lam));
the peer effect by its nominal value nu rho(A), the stability parameter of the linear model
(g'(0) = 1). K -> infinity is the linear model.

Parts:
  grid      multiplier, local amplification rho(J), E[DM]/GATE (10% and 50% treated) for a small and
            a large effect, and the annealed closed form, per family, response, sigma and nu rho
  clusters  E[DM]/GATE under Bernoulli and METIS clusters of 1,000 (50% treated)
  extrap    the two-fraction slope (10% vs 90%) and the historical baseline (50%), which assume a mean
            outcome linear in the treated fraction, against the GATE
"""
import json
import sys

import numpy as np

from common import DLAM, FAMILIES, LAM, RESULTS, graph_and_rho, spec
from netfx import concave as cc
from netfx import designs, sweep

SIGMAS = [0.3, 1, 3, 10]
NU_RHO = [0.3, 0.5, 0.9, 1.5, 3, 10]
STATS = None


def _hist(sp):
    global STATS
    if STATS is None:
        STATS = json.load(open(RESULTS / 'graph_stats.json'))
    st = STATS.get(json.dumps(sp, sort_keys=True))
    return st['deg_hist'] if st else None


def _setup(cfg):
    sp = spec(cfg['fam'], seed=cfg['seed'])
    A, rho = graph_and_rho(sp)
    n = A.shape[0]
    kbar = A.nnz / n
    nu = cfg['nu_rho'] / rho
    K = cfg['sigma'] * kbar * LAM
    return sp, A, rho, n, kbar, nu, K


def _lift_dm(C, c0, Z):
    L = C - c0[:, None]
    return float(np.mean([L[Z[:, r], r].mean() - L[~Z[:, r], r].mean() for r in range(Z.shape[1])]))


def grid(cfg):
    sp, A, rho, n, kbar, nu, K = _setup(cfg)
    kind = cfg['g']
    rng = np.random.default_rng([cfg['seed'], 21])
    c0, it = cc.steady(A, np.full(n, LAM), nu, kind, K)
    out = dict(rho=rho, nu=nu, K=K, iters=it, c0_mean=float(c0.mean()), c0_max=float(c0.max()),
               rhoJ=cc.jacobian_radius(A, c0, nu, kind, K))
    for tag, dl in [('small', DLAM), ('large', 10 * DLAM)]:
        c1, _ = cc.steady(A, np.full(n, LAM + dl), nu, kind, K)
        gate = float(c1.mean() - c0.mean())
        res = dict(gate=gate)
        for p in [0.1, 0.5]:
            Z = np.column_stack([designs.bernoulli(n, p, rng) for _ in range(cfg['R'])])
            C, _ = cc.steady(A, LAM + dl * Z.astype(float), nu, kind, K)
            res[f'dm_p{p}'] = _lift_dm(C, c0, Z)
            res[f'mean_lift_p{p}'] = float((C - c0[:, None]).mean())
        out[tag] = res
    # first-order (linearized) launch effect: 1'(I - J)^{-1} 1 dlam / N
    x = cc.linear_response(A, c0, nu, kind, np.ones(n), K)
    out['m_lin'] = float(x.mean())
    hist = _hist(sp)
    if hist is not None:
        m, rJ, cbar = cc.annealed_multiplier(nu, kind, hist, LAM, K)
        out.update(m_ann=m, rhoJ_ann=rJ, cbar_ann=cbar)
    return out


def clusters(cfg):
    sp, A, rho, n, kbar, nu, K = _setup(cfg)
    kind = cfg['g']
    rng = np.random.default_rng([cfg['seed'], 22])
    c0, _ = cc.steady(A, np.full(n, LAM), nu, kind, K)
    c1, _ = cc.steady(A, np.full(n, LAM + DLAM), nu, kind, K)
    gate = float(c1.mean() - c0.mean())
    labels = np.arange(n) if cfg['design'] == 'bernoulli' else designs.metis(A, n // cfg['size'], seed=cfg['seed'])
    Z = np.column_stack([designs.cluster(labels, 0.5, rng) for _ in range(cfg['R'])])
    C, _ = cc.steady(A, LAM + DLAM * Z.astype(float), nu, kind, K)
    cut = float((labels[A.indices] != np.repeat(labels, A.degrees())).mean())
    return dict(gate=gate, dm=_lift_dm(C, c0, Z), cut=cut, rho=rho)


def extrap(cfg):
    sp, A, rho, n, kbar, nu, K = _setup(cfg)
    kind = cfg['g']
    dl = cfg['dlam']
    rng = np.random.default_rng([cfg['seed'], 23])
    c0, _ = cc.steady(A, np.full(n, LAM), nu, kind, K)
    c1, _ = cc.steady(A, np.full(n, LAM + dl), nu, kind, K)
    gate = float(c1.mean() - c0.mean())
    means = {}
    for p in [0.1, 0.5, 0.9]:
        Z = np.column_stack([designs.bernoulli(n, p, rng) for _ in range(cfg['R'])])
        C, _ = cc.steady(A, LAM + dl * Z.astype(float), nu, kind, K)
        means[p] = float(C.mean()) - float(c0.mean())
    return dict(gate=gate, two_fraction=(means[0.9] - means[0.1]) / 0.8, pre_post=means[0.5] / 0.5,
                mean_lift=means, rho=rho)


def main(parts):
    if 'grid' in parts:
        cfgs = [dict(fam=f, g=g, sigma=s, nu_rho=x, R=5, seed=sd) for f in FAMILIES
                for g in ['log', 'sat'] for s in SIGMAS for x in NU_RHO for sd in range(3)]
        sweep.run(grid, cfgs, str(RESULTS / 's12_grid.jsonl'))
    if 'clusters' in parts:
        cfgs = [dict(fam=f, g='log', sigma=s, nu_rho=x, design=d, size=sz, R=10, seed=sd)
                for f in FAMILIES for s in [1, 3] for x in [0.9, 3]
                for d, sz in [('bernoulli', 1), ('metis', 1000)] for sd in range(3)]
        sweep.run(clusters, cfgs, str(RESULTS / 's12_clusters.jsonl'))
    if 'extrap' in parts:
        cfgs = [dict(fam=f, g='log', sigma=s, nu_rho=x, dlam=dl, R=3, seed=sd)
                for f in FAMILIES for s in [1, 3] for x in [0.9, 3, 10]
                for dl in [DLAM, 10 * DLAM] for sd in range(3)]
        sweep.run(extrap, cfgs, str(RESULTS / 's12_extrap.jsonl'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['grid', 'clusters', 'extrap'])
