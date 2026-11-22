"""Section 8.1 (threshold contagion) and an earlier raised-peer-effect model: models outside the linear one.

Parts:
  nutreat    treatment raises a node's peer effect: nu_i = nu (1 + delta z_i), no change to
             lam. Full rollout has nu (1 + delta) rho(A), which can exceed 1 while every
             partial experiment is stable.
  threshold  fractional threshold contagion (Watts 2002): a node becomes active when it is
             treated or when at least a fraction theta of its neighbours are active;
             production is lam + h for active nodes. Reports activity and DM against the
             treated fraction.
"""
import sys

import numpy as np

from common import FAMILIES, LAM, N, RESULTS, SEEDS, graph_and_rho, spec
from netfx import designs, sweep

FAMS = FAMILIES


def _steady_scaled(A, scale, nu, maxit=20000, tol=1e-10):
    """c = lam + nu * scale * (A c) by iteration; None if it diverges."""
    c = np.full(A.shape[0], LAM)
    for it in range(maxit):
        cn = LAM + nu * scale * (A @ c)
        if not np.all(np.isfinite(cn)) or cn.max() > 1e8:
            return None, it
        if np.max(np.abs(cn - c)) < tol * cn.max():
            return cn, it
        c = cn
    return None, maxit


def nutreat(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], n=cfg.get('n', N), seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 13])
    c0, _ = _steady_scaled(A, np.ones(n), nu)
    c1, _ = _steady_scaled(A, np.full(n, 1 + cfg['delta']), nu)
    gate = np.inf if c1 is None else c1.mean() - c0.mean()
    rows = []
    for p in cfg['ps']:
        z = designs.bernoulli(n, p, rng)
        c, _ = _steady_scaled(A, 1 + cfg['delta'] * z, nu)
        if c is None:
            rows.append(dict(p=p, unstable=True))
            continue
        rows.append(dict(p=p, unstable=False, dm=float(c[z].mean() - c[~z].mean()),
                         mean_lift=float(c.mean() - c0.mean())))
    return dict(gate=gate, full_rollout_nu_rho=cfg['nu_rho'] * (1 + cfg['delta']), rows=rows)


def threshold(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], n=cfg.get('n', N), seed=cfg['seed']))
    n = A.shape[0]
    deg = A.degrees().astype(float)
    rng = np.random.default_rng([cfg['seed'], 14])
    rows = []
    for p in cfg['ps']:
        for _ in range(cfg.get('R', 1)):
            z = designs.bernoulli(n, p, rng)
            a = z.copy()
            while True:
                frac = np.divide(A @ a.astype(float), deg, out=np.zeros(n), where=deg > 0)
                an = z | (frac >= cfg['theta'])
                if np.array_equal(an, a):
                    break
                a = an
            c = LAM + cfg['h'] * a
            rows.append(dict(p=p, active=float(a.mean()), active_control=float(a[~z].mean()),
                             dm=float(c[z].mean() - c[~z].mean()) if 0 < z.sum() < n else np.nan,
                             mean_lift=float(c.mean() - LAM)))
    return dict(gate=cfg['h'], rows=rows)


def main(parts):
    ps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    if 'nutreat' in parts:
        cfgs = [dict(fam=f, nu_rho=0.8, delta=d, ps=ps, seed=s) for f in FAMS
                for d in [0, 0.1, 0.2, 0.25, 0.3, 0.5] for s in range(SEEDS // 2)]
        sweep.run(nutreat, cfgs, str(RESULTS / 's10_nutreat.jsonl'))
    if 'threshold' in parts:
        pt = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
        cfgs = [dict(fam=f, theta=th, h=0.5, ps=pt, R=20, seed=s) for f in FAMS
                for th in [0.2, 0.3, 0.4, 0.5] for s in range(SEEDS // 2)]
        sweep.run(threshold, cfgs, str(RESULTS / 's10_threshold2.jsonl'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['nutreat', 'threshold'])
