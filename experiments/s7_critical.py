"""Section 6.3: near the stability limit nu rho(A) -> 1.

Parts:
  power      GATE, E[DM], the spread of DM over random assignments, and the power of a
             one-sided z-test on observed (Poisson) outcomes, against nu rho
  noise      stationary variance of production under the uniform and Poisson noise
             models, against nu rho; the multiplicative uniform noise diverges first
"""
import sys

import numpy as np
from scipy.stats import kurtosis, skew

from common import DLAM, LAM, RESULTS, SEEDS, graph_and_rho, localization, observe, spec
from netfx import designs, estimators, solve, sweep

NU_RHO = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
FAMS = [dict(family='ws', p=0.1), dict(family='er'), dict(family='sbm', B=100, mu=0.1),
        dict(family='ba'), dict(family='cm', gamma=2.5)]


def power(cfg):
    A, rho, v = graph_and_rho(spec(cfg['fam'], n=cfg['n'], seed=cfg['seed']), vector=True)
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 8])
    R = cfg['R']
    Z = np.column_stack([designs.bernoulli(n, 0.5, rng) for _ in range(R)])
    base = LAM * estimators.steady(A, np.ones(n), nu)
    dl = cfg.get('effect', DLAM)
    L = dl * solve.cg(A, Z.astype(float), nu, tol=1e-9) if dl else np.zeros((n, R))
    gate = dl * base.mean() / LAM
    dms, zs, dms_obs, lifts, zc, dcs, zcd = [], [], [], [], [], [], []
    for r in range(R):
        z = Z[:, r]
        c = base + L[:, r]
        dms.append(c[z].mean() - c[~z].mean())
        lifts.append(L[z, r].mean() - L[~z, r].mean())   # DM without the baseline difference
        y = observe(c, cfg['W'], rng)
        d = y[z].mean() - y[~z].mean()
        se = np.sqrt(y[z].var(ddof=1) / z.sum() + y[~z].var(ddof=1) / (~z).sum())
        dms_obs.append(d)
        zs.append(d / se)
        # CUPED with a pre-period window of the same length
        x = observe(base, cfg['W'], rng)
        xc = x - x.mean()
        theta = (xc * (y - y.mean())).sum() / (xc * xc).sum()
        ya = y - theta * xc
        da = ya[z].mean() - ya[~z].mean()
        sa = np.sqrt(ya[z].var(ddof=1) / z.sum() + ya[~z].var(ddof=1) / (~z).sum())
        dcs.append(da)
        zc.append(da / sa)
        # CUPED when each node's baseline changed between periods (10% node-level drift)
        xd = observe(base * (1 + 0.1 * rng.standard_normal(n)), cfg['W'], rng)
        xdc = xd - xd.mean()
        th = (xdc * (y - y.mean())).sum() / (xdc * xdc).sum()
        yd = y - th * xdc
        dd = yd[z].mean() - yd[~z].mean()
        sd_ = np.sqrt(yd[z].var(ddof=1) / z.sum() + yd[~z].var(ddof=1) / (~z).sum())
        zcd.append(dd / sd_)
    dms, zs = np.array(dms), np.array(zs)
    top, ipr = localization(v)
    return dict(gate=gate, dm_mean=dms.mean(), dm_sd=dms.std(ddof=1),
                dm_lift_mean=float(np.mean(lifts)), dm_lift_sd=float(np.std(lifts, ddof=1)),
                dm_skew=float(skew(dms)), dm_kurt=float(kurtosis(dms)),
                obs_sd=float(np.std(dms_obs, ddof=1)), naive_se=float(np.mean(np.abs(np.array(dms_obs) / zs))),
                power=float((zs > 1.645).mean()), top1=top, ipr=ipr, rho=rho,
                power_cuped=float((np.array(zc) > 1.645).mean()),
                power_cuped_drift=float((np.array(zcd) > 1.645).mean()),
                cuped_sd=float(np.std(dcs, ddof=1)), cuped_mean=float(np.mean(dcs)),
                base_mean=base.mean())


def noise(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], n=cfg['n'], seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 9])
    c = LAM * estimators.steady(A, np.ones(n), nu)          # start at the mean steady state
    T, burn = cfg['T'], cfg['T'] // 4
    sums, sq, mx = np.zeros(n), np.zeros(n), 0.0
    for t in range(T):
        lam = LAM + nu * (A @ c)
        if cfg['sampler'] == 'uniform':
            c = rng.uniform(0.0, 1.0, n) * (2 * lam)
        else:
            c = rng.poisson(lam).astype(float)
        if not np.all(np.isfinite(c)) or c.max() > 1e12:
            return dict(diverged=True, t=t, rho=rho)
        if t >= burn:
            sums += c
            sq += c * c
            mx = max(mx, c.max())
    k = T - burn
    var = sq / k - (sums / k) ** 2
    deg = A.degrees()
    hub = deg >= np.quantile(deg, 0.999)
    return dict(diverged=False, mean=float((sums / k).mean()), var=float(var.mean()),
                var_hubs=float(var[hub].mean()), max=mx, rho=rho)


def main(parts):
    if 'power' in parts:
        cfgs = [dict(fam=f, n=100_000, nu_rho=x, R=100, W=7, effect=e, seed=s) for f in FAMS
                for x in NU_RHO for e in [0.0, DLAM] for s in range(SEEDS // 2)]
        sweep.run(power, cfgs, str(RESULTS / 's7_power3.jsonl'))
    if 'noise' in parts:
        cfgs = [dict(fam=f, n=20_000, nu_rho=x, sampler=smp, T=2000, seed=s) for f in FAMS
                for x in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99] for smp in ['uniform', 'poisson']
                for s in range(SEEDS // 2)]
        sweep.run(noise, cfgs, str(RESULTS / 's7_noise.jsonl'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['power', 'noise'])
