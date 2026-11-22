"""Graph statistics needed by the closed forms (degree moments, k_max, rho, hub eigenvalue,
triangles, tr(M)/N and the spread of the baseline M1), computed from generated graphs.

Run from the project root:  .venv/bin/python experiments/graph_stats.py
Writes results/graph_stats.json; figures.py reads it for the closed-form curves. Each entry is keyed by the
JSON of the family spec (as experiments/common.spec builds it).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from netfx import families, solve  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'graph_stats.json')
NU_RHO = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
FAMILIES = [dict(family='ws', p=0.1), dict(family='er'), dict(family='sbm', B=100, mu=0.1),
            dict(family='ba'), dict(family='cm', gamma=2.5)]


def key(sp):
    return json.dumps(sp, sort_keys=True)


def hub_eigenvalue(A):
    """Largest eigenvalue of the two-level tree around the maximum-degree node (hub with
    its neighbours and their other neighbours): solves 1 = sum_j 1/(lam^2 - k_j + 1) over
    the hub's neighbours j, with lam^2 > max_j k_j - 1."""
    deg = A.degrees()
    hub = int(np.argmax(deg))
    kj = deg[A.indices[A.indptr[hub]:A.indptr[hub + 1]]].astype(float)
    lo = np.sqrt(kj.max() - 1 + 1e-9)
    hi = np.sqrt(kj.max() + kj.size)
    f = lambda lam: np.sum(1.0 / (lam * lam - kj + 1)) - 1.0   # decreasing in lam
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def triangles_per_node(A):
    S = A.to_scipy()
    S2 = S @ S
    return float(S2.multiply(S).sum() / A.shape[0] / 2.0)   # tr(A^3) = 6 triangles; per node: 3 tri / N


def basic(A):
    d = A.degrees().astype(float)
    n = A.shape[0]
    rho, v = solve.spectral_radius(A, return_vector=True)
    hist = np.bincount(d.astype(int))
    return dict(n=n, kbar=float(d.mean()), k2=float((d * d).mean()), kmax=int(d.max()),
                cv=float(d.std() / d.mean()), rho=float(rho),
                perron_sum_sq=float(v.sum() ** 2 / n),           # (1'v)^2 / N
                hub_lambda=hub_eigenvalue(A),
                deg_hist=hist.tolist())


def resolvent_stats(A, nu_rhos, probes=3, seed=0):
    """For each nu rho: multiplier 1'M1/N, tr(M)/N (Hutchinson, Rademacher probes, each
    column solved on its own), sd/mean of M1 over nodes."""
    n = A.shape[0]
    rho = solve.spectral_radius(A)
    rng = np.random.default_rng(seed)
    Z = rng.choice([-1.0, 1.0], size=(n, probes))
    out = []
    for x in nu_rhos:
        nu = x / rho
        m1 = solve.cg(A, np.ones(n), nu, tol=1e-10)
        ests = []
        for j in range(probes):
            xj = solve.cg(A, Z[:, j], nu, tol=1e-10)
            ests.append((Z[:, j] * xj).sum() / n)
        out.append(dict(nu_rho=x, nu=nu, multiplier=float(m1.mean()), tr_over_n=float(np.mean(ests)),
                        tr_over_n_sd=float(np.std(ests)), base_cv=float(m1.std() / m1.mean()),
                        base_sd=float(m1.std())))
    return out


def main():
    stats = json.load(open(OUT)) if os.path.exists(OUT) else {}

    def save():
        json.dump(stats, open(OUT + '.tmp', 'w'))
        os.replace(OUT + '.tmp', OUT)

    jobs = []
    # the five families at the default size, with resolvent statistics
    for f in FAMILIES:
        for s in range(5):
            jobs.append((dict(f, n=100_000, kbar=20, seed=s), True, True))
    # gamma configuration model (Section 5.2)
    for cv in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for s in range(3):
            jobs.append((dict(family='gam', cv=cv, n=100_000, kbar=20, seed=s), False, False))
    # size series (Section 5.5), kbar = 10
    for fam, extra in [('ba', {}), ('cm', {'gamma': 2.5}), ('er', {}), ('ws', {'p': 0.1})]:
        for n in [10**4, 10**5, 10**6]:
            jobs.append((dict(family=fam, **extra, n=n, kbar=10, seed=0), False, False))
    # small Erdos-Renyi graphs of s5_kbar
    for k in [5, 10, 20, 40]:
        for s in range(3):
            jobs.append((dict(family='er', n=3000, kbar=k, seed=s), False, False))

    redo = '--resolvent' in sys.argv      # recompute the resolvent statistics only
    for sp, want_res, want_tri in jobs:
        k = key(sp)
        if redo:
            if not want_res:
                continue
            t = time.time()
            A = families.make(sp)
            stats[k]['resolvent'] = resolvent_stats(A, NU_RHO)
            save()
            print(f'{k}: resolvent redone ({time.time() - t:.0f}s)', flush=True)
            continue
        if k in stats and ('resolvent' in stats[k] or not want_res):
            continue
        t = time.time()
        A = families.make(sp)
        st = basic(A)
        if want_tri:
            st['triangles_per_node'] = triangles_per_node(A)
        if want_res:
            st['resolvent'] = resolvent_stats(A, NU_RHO)
        stats[k] = st
        save()
        print(f'{k}: rho {st["rho"]:.2f} kappa {st["k2"] / st["kbar"]:.2f} kmax {st["kmax"]} '
              f'hub {st["hub_lambda"]:.2f} ({time.time() - t:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
