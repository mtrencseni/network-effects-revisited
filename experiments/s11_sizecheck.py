"""Size check for Sections 6 to 9: the headline quantity of each section at N = 10^5, 10^6, 10^7.

Each part calls the experiment function of its section with a reduced configuration (fewer
assignments, one or two parameter values) and an explicit n. Results go to s11_<part>.jsonl,
separate from the main results, so the main figures are unchanged.

Parts:
  paths      Section 6: DM(t) and GATE(t) after launch, nu rho = 0.9
  clusters   Section 7.1: E[DM]/GATE and RMSE, Bernoulli vs METIS clusters of 1,000, nu rho = 0.6
  power      Section 7.4: power and noise near the stability limit
  edges      Section 8.1: new edges, user-level and cluster split
  users      Section 8.2: new users
  nutreat    Section 9.1: treatment that raises the peer effect
  threshold  Section 9.2: threshold contagion
"""
import sys

from common import DLAM, FAMILIES, RESULTS
from netfx import sweep

import s6_duration
import s7_critical
import s8_designs
import s9_graph_change
import s10_nonlinear

SIZES = [(10**5, 3), (10**6, 3), (10**7, 1)]


def _cfgs(base):
    return [dict(b, n=n, seed=s) for b in base for n, seeds in SIZES for s in range(seeds)]


MODE = 'all'   # 'small' (process pool, n <= 10^6) or 'big' (inline, all threads); never both in
               # one process: forking after numba's OpenMP threads have started aborts the pool


def _run(fn, cfgs, name):
    path = str(RESULTS / f's11_{name}.jsonl')
    if MODE in ('small', 'all'):
        sweep.run(fn, [c for c in cfgs if c['n'] <= 10**6], path)
    if MODE in ('big', 'all'):
        sweep.run(fn, [c for c in cfgs if c['n'] > 10**6], path, inline=True)


def main(parts):
    if 'paths' in parts:
        _run(s6_duration.paths, _cfgs([dict(fam=f, p=0.1, nu_rho=0.9, T=150) for f in FAMILIES]), 'paths')
    if 'clusters' in parts:
        _run(s8_designs.clusters,
             _cfgs([dict(fam=f, design=d, size=sz, nu_rho=0.6, R=10, hajek=False)
                    for f in FAMILIES for d, sz in [('bernoulli', 1), ('metis', 1000)]]), 'clusters')
    if 'power' in parts:
        _run(s7_critical.power,
             _cfgs([dict(fam=f, nu_rho=x, R=10, W=7, effect=DLAM)
                    for f in FAMILIES for x in [0.9, 0.99]]), 'power')
    if 'edges' in parts:
        # at 10^7 only the two settings of the Appendix F table (r = 5: random edges with the
        # user-level split, friends of friends with METIS clusters)
        _run(s9_graph_change.edges,
             [c for c in _cfgs([dict(fam=f, r=r, rule=rule, boost=True, design=d, size=1000, nu_rho=0.5, R=5)
                                for f in FAMILIES for r in [1, 5] for rule in ['random', 'fof']
                                for d in ['bernoulli', 'metis']])
              if c['n'] <= 10**6 or (c['r'] == 5 and (c['rule'], c['design']) in
                                     [('random', 'bernoulli'), ('fof', 'metis')])], 'edges')
    if 'users' in parts:
        _run(s9_graph_change.users,
             _cfgs([dict(fam=f, r=r, q=2, a=0.5, nu_rho=0.5, R=5)
                    for f in FAMILIES for r in [0.5, 2]]), 'users')
    if 'nutreat' in parts:
        _run(s10_nonlinear.nutreat,
             _cfgs([dict(fam=f, nu_rho=0.8, delta=d, ps=[0.1, 0.5])
                    for f in FAMILIES for d in [0.2, 0.3]]), 'nutreat')
    if 'threshold' in parts:
        _run(s10_nonlinear.threshold,
             _cfgs([dict(fam=f, theta=0.3, h=0.5, ps=[0.05, 0.1, 0.15, 0.2], R=5)
                    for f in FAMILIES]), 'threshold')


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] in ('small', 'big'):
        MODE = args.pop(0)
    main(args or ['paths', 'clusters', 'power', 'edges', 'users', 'nutreat', 'threshold'])
