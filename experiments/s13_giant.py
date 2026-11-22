"""Size experiment of Section 4.4 at N = 10^8 and 10^9 with mean degree 2.

The same quantities as s5_effects.size (spectral radius, multiplier, E[DM]/GATE under a Bernoulli 10%
split at nu rho = 0.5) on Erdos-Renyi and Barabasi-Albert graphs with kbar = 2, one graph per point.
At 10^9 nodes eigsh (about ncv + 6 vectors of 8 GB) does not fit in memory, so the spectral radius
comes from solve.lanczos_radius (three vectors, degree vector as the starting vector, relative change
below 1e-10 over 10 steps); it agrees with eigsh to 2e-13 on six graphs up to 10^6 nodes and to ten
digits at 10^8. The two 10^8 rows of results/s13_giant.jsonl were computed with eigsh (ncv=4, tol=1e-8,
degree vector as v0), before the switch; they also have the eigenvector localization (top1, ipr).
Each configuration runs in its own process, so the peak memory is per run.

Usage: s13_giant.py [n ...]      (default: 10^8 then 10^9)
"""
import json
import resource
import subprocess
import sys
import time

import numpy as np

from common import DLAM, RESULTS
from netfx import designs, estimators, families, solve

PATH = RESULTS / 's13_giant.jsonl'
FAMS = [('er', {}), ('ba', {})]
KBAR = 2


def _peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # GB


def run(cfg):
    out, t0 = {}, time.time()
    A = families.make(dict(family=cfg['family'], n=cfg['n'], kbar=cfg['kbar'], seed=cfg['seed'],
                           **cfg['extra']))
    n = A.shape[0]
    out.update(kbar=A.nnz / n, gen_seconds=time.time() - t0, gen_peak_gb=_peak())
    deg = A.degrees()
    out['dmax'] = int(deg.max())
    t = time.time()
    v0 = deg.astype(np.float64)
    del deg
    rho, steps = solve.lanczos_radius(A, v0)
    del v0
    out.update(rho=rho, lanczos_steps=steps, rho_seconds=time.time() - t, rho_peak_gb=_peak())
    nu = cfg['nu_rho'] / rho
    t = time.time()
    rng = np.random.default_rng([cfg['seed'], 3])
    z = designs.bernoulli(n, 0.1, rng)
    lift = DLAM * estimators.steady(A, z.astype(np.float64), nu)
    dm = lift[z].mean() - lift[~z].mean()
    del lift
    gate = estimators.gate(A, nu, DLAM)
    out.update(gate=gate, dm=dm, ratio=dm / gate, multiplier=gate / DLAM,
               solve_seconds=time.time() - t, peak_gb=_peak(), seconds=time.time() - t0)
    return out


def main(sizes):
    done = set()
    if PATH.exists():
        done = {json.dumps(json.loads(l)['config'], sort_keys=True) for l in open(PATH)}
    for n in sizes:
        for fam, extra in FAMS:
            cfg = dict(family=fam, extra=extra, kbar=KBAR, n=n, nu_rho=0.5, seed=0)
            key = json.dumps(cfg, sort_keys=True)
            if key in done:
                continue
            print('running', key, flush=True)
            subprocess.run([sys.executable, __file__, '--one', key], check=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--one']:
        cfg = json.loads(sys.argv[2])
        res = run(cfg)
        print(json.dumps(res), flush=True)
        with open(PATH, 'a') as f:
            f.write(json.dumps(dict(config=cfg, result=res)) + '\n')
    else:
        main([int(float(x)) for x in sys.argv[1:]] or [10**8, 10**9])
