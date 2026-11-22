"""Section 4: the effects across graph families and sizes.

Parts (run all, or name them on the command line):
  multiplier   social multiplier mean((I - nu A)^{-1} 1) against nu rho(A) and nu kbar
  hetero       degree heterogeneity: multiplier against the degree coefficient of variation
  hops         spillover against hop distance from the treated set
  attenuation  E[DM] / GATE and its decomposition into treated lift and control spillover
  size         attenuation, rho / kbar and localization against graph size, 1e4 .. 1e8
  clustering   attenuation against exposure, for random and clustered assignment
  saturation   DM and ratio lift against the treated fraction; targeted treatment groups
"""
import sys

import numpy as np
from scipy.sparse.csgraph import shortest_path  # noqa: F401  (BFS below is our own)

from common import (DLAM, FAMILIES, KBAR, LAM, N, RESULTS, SEEDS, graph_and_rho,
                    localization, spec)
from netfx import designs, estimators, solve, sweep

NU_RHO = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


# ---------------------------------------------------------------------------

def multiplier(cfg):
    A, rho, v = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']), vector=True)
    kbar = A.nnz / A.shape[0]
    top, ipr = localization(v)
    out = dict(rho=rho, kbar=kbar, top1=top, ipr=ipr, dmax=int(A.degrees().max()), points=[])
    for x in NU_RHO:
        nu = x / rho
        m = estimators.steady(A, np.ones(A.shape[0]), nu).mean()
        out['points'].append(dict(nu_rho=x, nu_kbar=nu * kbar, multiplier=m,
                                  mean_field=1 / (1 - nu * kbar)))
    # the 2023 stability condition nu kbar < 1 at nu kbar = 0.5: is the model stable?
    nu = 0.5 / kbar
    out['half_kbar'] = dict(nu_rho=nu * rho, stable=bool(nu * rho < 1))
    return out


def hetero(cfg):
    sp = dict(family='gam', cv=cfg['cv'], n=N, kbar=KBAR, seed=cfg['seed'])
    A, rho = graph_and_rho(sp)
    d = A.degrees().astype(float)
    kbar, var = d.mean(), d.var()
    nu = cfg['nu_kbar'] / kbar
    if nu * rho >= 1:
        return dict(cv=cfg['cv'], rho=rho, kbar=kbar, var=var, stable=False)
    m = estimators.steady(A, np.ones(A.shape[0]), nu).mean()
    # walk-count expansion: mean(A^t 1) for t = 0..3
    x, walks = np.ones(A.shape[0]), []
    for t in range(4):
        walks.append(x.mean())
        x = A @ x
    second = 1 + nu * kbar + nu**2 * (kbar**2 + var)
    return dict(cv=cfg['cv'], rho=rho, kbar=kbar, var=var, stable=True, multiplier=m,
                mean_field=1 / (1 - nu * kbar), second_order=second, walks=walks)


def hetero_base(cfg):
    """A base family at the setting of the heterogeneity experiment: degree spread and the
    multiplier at fixed nu kbar."""
    A, rho = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']))
    d = A.degrees().astype(float)
    kbar = d.mean()
    out = dict(cv=float(d.std() / kbar), rho=rho, kbar=kbar, points=[])
    for nk in cfg['nu_kbars']:
        nu = nk / kbar
        if nu * rho >= 1:
            out['points'].append(dict(nu_kbar=nk, stable=False, nu_rho=nu * rho))
            continue
        m = estimators.steady(A, np.ones(A.shape[0]), nu).mean()
        out['points'].append(dict(nu_kbar=nk, stable=True, nu_rho=nu * rho, multiplier=m))
    return out


def _bfs_distance(A, sources, cap=6):
    dist = np.full(A.shape[0], cap, dtype=np.int64)
    frontier = np.flatnonzero(sources)
    dist[frontier] = 0
    for h in range(1, cap):
        reach = (A @ np.isin(np.arange(A.shape[0]), frontier).astype(float)) > 0
        new = np.flatnonzero(reach & (dist == cap))
        if new.size == 0:
            break
        dist[new] = h
        frontier = new
    return dist


def hops(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']))
    rng = np.random.default_rng([cfg['seed'], 1])
    z = designs.bernoulli(A.shape[0], cfg['p'], rng)
    nu = cfg['nu_rho'] / rho
    lift = DLAM * estimators.steady(A, z.astype(float), nu)   # c(z) - c(0)
    dist = _bfs_distance(A, z)
    gate = estimators.gate(A, nu, DLAM)
    return dict(gate=gate, by_hop=[dict(h=int(h), share=float((dist == h).mean()),
                                        lift=float(lift[dist == h].mean()))
                                   for h in np.unique(dist)])


def attenuation(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 2])
    Z = np.column_stack([designs.bernoulli(n, cfg['p'], rng) for _ in range(cfg['R'])])
    L = DLAM * solve.cg(A, Z.astype(float), nu)                # lift per node per assignment
    gate = estimators.gate(A, nu, DLAM)
    lt = np.array([L[Z[:, r], r].mean() for r in range(cfg['R'])])
    sc = np.array([L[~Z[:, r], r].mean() for r in range(cfg['R'])])
    kbar = A.nnz / n
    return dict(gate=gate, lift_treated=lt.mean(), spill_control=sc.mean(),
                dm=(lt - sc).mean(), ratio=(lt - sc).mean() / gate,
                rho=rho, kbar=kbar, nu=nu)


def attenuation_kbar(cfg):
    """E[DM]/GATE at a fixed nu kbar (the same peer effect per friend on every family); returns
    stable=False when nu rho >= 1."""
    A, rho = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']))
    n = A.shape[0]
    kbar = A.nnz / n
    nu = cfg['nu_kbar'] / kbar
    if nu * rho >= 1:
        return dict(stable=False, rho=rho, kbar=kbar, nu_rho=nu * rho)
    rng = np.random.default_rng([cfg['seed'], 2])
    Z = np.column_stack([designs.bernoulli(n, cfg['p'], rng) for _ in range(cfg['R'])])
    L = DLAM * solve.cg(A, Z.astype(float), nu)
    gate = estimators.gate(A, nu, DLAM)
    dm = np.array([L[Z[:, r], r].mean() - L[~Z[:, r], r].mean() for r in range(cfg['R'])])
    return dict(stable=True, gate=gate, dm=dm.mean(), ratio=dm.mean() / gate, rho=rho, kbar=kbar,
                nu=nu, nu_rho=nu * rho)


def size(cfg):
    sp = spec(dict(family=cfg['family'], **cfg.get('extra', {})), n=cfg['n'], kbar=cfg['kbar'],
              seed=cfg['seed'])
    A, rho, v = graph_and_rho(sp, vector=True)
    n = A.shape[0]
    kbar = A.nnz / n
    top, ipr = localization(v)
    out = dict(rho=rho, kbar=kbar, dmax=int(A.degrees().max()), top1=top, ipr=ipr,
               unstable_at_half_kbar=bool(0.5 / kbar * rho >= 1))
    if 'nu_kbar' in cfg:
        nu = cfg['nu_kbar'] / kbar
        out['nu_rho'] = nu * rho
        if nu * rho >= 1:
            out['stable'] = False
            return out
        out['stable'] = True
    else:
        nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 3])
    z = designs.bernoulli(n, 0.1, rng)
    lift = DLAM * estimators.steady(A, z.astype(float), nu)
    gate = estimators.gate(A, nu, DLAM)
    out.update(gate=gate, dm=lift[z].mean() - lift[~z].mean(),
               ratio=(lift[z].mean() - lift[~z].mean()) / gate,
               multiplier=gate / DLAM)
    return out


def clustering(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    gate = estimators.gate(A, nu, DLAM)
    rng = np.random.default_rng([cfg['seed'], 4])
    if cfg['design'] == 'bernoulli':
        labels = np.arange(n)
    elif cfg['design'] == 'metis':
        labels = designs.metis(A, n // cfg['size'], seed=cfg['seed'])
    elif cfg['design'] == 'contiguous':
        labels = designs.contiguous(n, cfg['size'])
    elif cfg['design'] == 'blocks':
        labels = designs.contiguous(n, n // cfg['fam'].get('B', 100))
    Z = np.column_stack([designs.cluster(labels, 0.5, rng) for _ in range(cfg['R'])])
    L = DLAM * solve.cg(A, Z.astype(float), nu)
    ratios, expo = [], []
    for r in range(cfg['R']):
        z = Z[:, r]
        ratios.append((L[z, r].mean() - L[~z, r].mean()) / gate)
        e = designs.exposure(A, z)
        expo.append(0.5 * (e[z].mean() + (1 - e[~z]).mean()))   # same-arm neighbour share
    cut = float((labels[A.indices] != np.repeat(labels, A.degrees())).mean())
    return dict(ratio=float(np.mean(ratios)), ratio_sd=float(np.std(ratios)),
                same_arm=float(np.mean(expo)), cut=cut)


def saturation(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    base = LAM * estimators.steady(A, np.ones(n), nu)          # untreated production
    gate = estimators.gate(A, nu, DLAM)
    rng = np.random.default_rng([cfg['seed'], 5])
    if cfg['design'] == 'random':
        z = designs.bernoulli(n, cfg['p'], rng)
    elif cfg['design'] == 'degree':
        z = designs.targeted(A.degrees().astype(float), int(cfg['p'] * n))
    elif cfg['design'] == 'centrality':
        z = designs.targeted(base, int(cfg['p'] * n))
    c = base + DLAM * estimators.steady(A, z.astype(float), nu)
    dm = c[z].mean() - c[~z].mean()
    ratio = c[z].mean() / c[~z].mean() - 1
    # what a beta test of z predicts for full rollout if the treated lift is taken at face
    # value: lift of the treated group over its own untreated level
    beta_lift = c[z].mean() - base[z].mean()
    return dict(gate=gate, dm=dm, ratio_lift=ratio, gate_rel=gate / base.mean(),
                beta_lift=beta_lift, centrality_share=base[z].sum() / base.sum(),
                treated=int(z.sum()))


# ---------------------------------------------------------------------------

def main(parts):
    R = 20
    if 'multiplier' in parts:
        cfgs = [dict(fam=f, seed=s) for f in FAMILIES for s in range(SEEDS)]
        sweep.run(multiplier, cfgs, str(RESULTS / 's5_multiplier.jsonl'))
    if 'hetero' in parts:
        cfgs = [dict(cv=cv, nu_kbar=nk, seed=s)
                for cv in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                for nk in [0.3, 0.6] for s in range(SEEDS)]
        sweep.run(hetero, cfgs, str(RESULTS / 's5_hetero.jsonl'))
    if 'hetero_base' in parts:
        cfgs = [dict(fam=f, nu_kbars=[0.3, 0.6], seed=s) for f in FAMILIES for s in range(5)]
        sweep.run(hetero_base, cfgs, str(RESULTS / 's5_hetero_base.jsonl'))
    if 'hops' in parts:
        cfgs = [dict(fam=f, p=0.02, nu_rho=x, seed=s) for f in FAMILIES
                for x in [0.5, 0.9] for s in range(SEEDS)]
        sweep.run(hops, cfgs, str(RESULTS / 's5_hops.jsonl'))
    if 'attenuation' in parts:
        cfgs = [dict(fam=f, p=0.1, nu_rho=x, R=R, seed=s) for f in FAMILIES
                for x in NU_RHO for s in range(SEEDS)]
        sweep.run(attenuation, cfgs, str(RESULTS / 's5_attenuation.jsonl'))
    if 'clustering' in parts:
        cfgs = []
        for f in FAMILIES:
            designs_f = [('bernoulli', 1)] + [('metis', sz) for sz in [10, 100, 1000, 10000]]
            if f['family'] == 'ws':
                designs_f += [('contiguous', sz) for sz in [10, 100, 1000, 10000]]
            if f['family'] == 'sbm':
                designs_f += [('blocks', 1000)]
            for d, sz in designs_f:
                for s in range(SEEDS // 2):
                    cfgs.append(dict(fam=f, design=d, size=sz, nu_rho=0.5, R=R, seed=s))
        sweep.run(clustering, cfgs, str(RESULTS / 's5_clustering.jsonl'))
    if 'clustering_mu' in parts:
        # SBM with more edges between blocks: fills the within-cluster share between the
        # graphs without communities (at most 0.31) and the base SBM (0.9)
        cfgs = [dict(fam=dict(family='sbm', B=100, mu=mu), design=d, size=sz, nu_rho=x, R=R, seed=s)
                for mu in [0.2, 0.3, 0.5, 0.7]
                for d, sz in [('bernoulli', 1), ('blocks', 1000), ('metis', 1000)]
                for x in [0.5, 0.9] for s in range(SEEDS // 2)]
        sweep.run(clustering, cfgs, str(RESULTS / 's5_clustering_mu.jsonl'))
    if 'saturation' in parts:
        cfgs = [dict(fam=f, design=d, p=p, nu_rho=x, seed=s) for f in FAMILIES
                for d in ['random', 'degree', 'centrality']
                for p in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
                for x in [0.2, 0.5, 0.8] for s in range(SEEDS // 2)]
        sweep.run(saturation, cfgs, str(RESULTS / 's5_saturation.jsonl'))
    if 'attenuation_p' in parts:
        # E[DM]/GATE against the treated fraction under random assignment, treatment-induced
        # DM only (no baseline difference between the arms)
        cfgs = [dict(fam=f, p=p, nu_rho=0.5, R=R, seed=s) for f in FAMILIES
                for p in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9] for s in range(3)]
        sweep.run(attenuation, cfgs, str(RESULTS / 's5_attenuation_p.jsonl'))
    if 'attenuation_kbar' in parts:
        cfgs = [dict(fam=f, p=0.1, nu_kbar=x, R=R, seed=s) for f in FAMILIES
                for x in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                for s in range(3)]
        sweep.run(attenuation_kbar, cfgs, str(RESULTS / 's5_attenuation_kbar.jsonl'))
    if 'size' in parts:
        cfgs = []
        for fam, kbar, extra in [('ws', 10, {'p': 0.1}), ('er', 10, {}),
                                 ('cm', 10, {'gamma': 2.5}), ('ba', 10, {}),
                                 ('sbm', 10, {'B': 100, 'mu': 0.1})]:
            for n, seeds in [(10**4, 10), (10**5, 10), (10**6, 5), (10**7, 2), (10**8, 1)]:
                if fam == 'sbm' and n > 10**7:
                    continue
                for s in range(seeds):
                    cfgs.append(dict(family=fam, extra=extra, kbar=kbar, n=n, nu_rho=0.5, seed=s))
        small = [c for c in cfgs if c['n'] <= 10**6]
        big = [c for c in cfgs if c['n'] > 10**6]
        sweep.run(size, small, str(RESULTS / 's5_size.jsonl'))
        sweep.run(size, big, str(RESULTS / 's5_size.jsonl'), inline=True)
    if 'size_fixednu' in parts:
        cfgs = [dict(family=fam, extra=extra, kbar=10, n=n, nu_kbar=0.3, seed=s)
                for fam, extra in [('ws', {'p': 0.1}), ('er', {}), ('cm', {'gamma': 2.5}), ('ba', {}),
                                   ('sbm', {'B': 100, 'mu': 0.1})]
                for n, seeds in [(10**4, 10), (10**5, 10), (10**6, 5), (10**7, 2)]
                for s in range(seeds)]
        sweep.run(size, [c for c in cfgs if c['n'] <= 10**6], str(RESULTS / 's5_size_fixednu.jsonl'))
        sweep.run(size, [c for c in cfgs if c['n'] > 10**6], str(RESULTS / 's5_size_fixednu.jsonl'),
                  inline=True)


if __name__ == '__main__':
    main(sys.argv[1:] or ['multiplier', 'hetero', 'hops', 'attenuation', 'clustering',
                          'saturation', 'size'])
