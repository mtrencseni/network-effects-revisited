"""Section 7: treatment that changes the graph.

Parts:
  edges   each treated node forms Poisson(r) new edges, to uniformly random nodes or to
          friends of friends; production is the steady state on the new graph. Reports
          the GATE (everyone treated, so everyone forms edges) and E[DM] for Bernoulli
          and for cluster randomization with clusters computed on the pre-experiment graph.
  users   each treated node invites Poisson(r) new users; a new user connects to its
          inviter and to q of the inviter's neighbours and has intrinsic production
          a * lam. Reports total and per-user effects and the estimates under three rules
          for counting new users in the experiment.
"""
import sys

import numpy as np

from common import DLAM, FAMILIES, LAM, N, RESULTS, SEEDS, graph_and_rho, spec
from netfx import designs, estimators, graph, solve, sweep

FAMS = FAMILIES


def _random_neighbour(A, nodes, rng):
    deg = A.degrees()[nodes]
    off = (rng.random(nodes.size) * deg).astype(np.int64)
    return A.indices[A.indptr[nodes] + off].astype(np.int64)


def _new_edges(A, z, r, rule, rng):
    """Poisson(r) new edges per treated node; proposals that are self-loops, existing
    edges or duplicates are redrawn, so both rules add the same number of edges."""
    n = A.shape[0]
    src = np.flatnonzero(z)
    k = rng.poisson(r, src.size)
    want = np.repeat(src, k)
    S = A.to_scipy()
    taken = set()
    us, vs = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
    todo = want
    for _ in range(50):
        if todo.size == 0:
            break
        if rule == 'random':
            v = rng.integers(0, n, todo.size)
        else:                               # friend of a friend
            v = _random_neighbour(A, _random_neighbour(A, todo, rng), rng)
        ok = (v != todo) & (np.asarray(S[todo, v]).ravel() == 0)
        keys = np.minimum(todo, v) * n + np.maximum(todo, v)
        fresh = np.array([ok[i] and keys[i] not in taken for i in range(todo.size)])
        for i in np.flatnonzero(fresh):
            if keys[i] in taken:
                fresh[i] = False
            else:
                taken.add(keys[i])
        us.append(todo[fresh])
        vs.append(v[fresh])
        todo = todo[~fresh]
    return np.concatenate(us), np.concatenate(vs)


def _with_edges(A, u, v):
    S = A.to_scipy().tocoo()
    keep = S.row < S.col
    uu = np.concatenate([S.row[keep].astype(np.int64), u])
    vv = np.concatenate([S.col[keep].astype(np.int64), v])
    return graph.from_edges(A.shape[0], uu, vv)


def _solve(A, b, nu):
    try:
        return estimators.steady(A, b, nu)
    except solve.Unstable:
        return None


def edges(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], n=cfg.get('n', N), seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 11])
    dl = DLAM if cfg['boost'] else 0.0
    c0 = estimators.steady(A, np.full(n, LAM), nu)
    u, v = _new_edges(A, np.ones(n, bool), cfg['r'], cfg['rule'], rng)
    c1 = _solve(_with_edges(A, u, v), np.full(n, LAM + dl), nu)
    gate = np.inf if c1 is None else c1.mean() - c0.mean()
    labels = (np.arange(n) if cfg['design'] == 'bernoulli'
              else designs.metis(A, n // cfg['size'], seed=cfg['seed']))
    dms, cross = [], []
    for _ in range(cfg['R']):
        z = designs.cluster(labels, 0.5, rng)
        u, v = _new_edges(A, z, cfg['r'], cfg['rule'], rng)
        c = _solve(_with_edges(A, u, v), LAM + dl * z, nu)
        if c is None:
            dms.append(np.inf)
            continue
        dms.append(c[z].mean() - c[~z].mean())
        if u.size:
            cross.append(float((z[u] != z[v]).mean()))
    return dict(gate=gate, dm=float(np.mean(dms)), dm_sd=float(np.std(dms)),
                new_edges_cross_arm=float(np.mean(cross)) if cross else 0.0, nu=nu)


def users(cfg):
    A, rho = graph_and_rho(spec(cfg['fam'], n=cfg.get('n', N), seed=cfg['seed']))
    n = A.shape[0]
    nu = cfg['nu_rho'] / rho
    rng = np.random.default_rng([cfg['seed'], 12])
    S = A.to_scipy().tocoo()
    keep = S.row < S.col
    base_u, base_v = S.row[keep].astype(np.int64), S.col[keep].astype(np.int64)

    def world(z):
        """Grown graph and production when the nodes in z are treated."""
        inv = np.flatnonzero(z)
        k = rng.poisson(cfg['r'], inv.size)
        inviter = np.repeat(inv, k)
        m = inviter.size
        new = n + np.arange(m)
        u = [base_u, new]
        v = [base_v, inviter]
        for _ in range(cfg['q']):
            u.append(new)
            v.append(_random_neighbour(A, inviter, rng))
        G = graph.from_edges(n + m, np.concatenate(u), np.concatenate(v))
        lam = np.concatenate([LAM + DLAM * z, np.full(m, cfg['a'] * LAM)])
        c = _solve(G, lam, nu)
        return c, inviter

    c_none, _ = world(np.zeros(n, bool))
    c_all, _ = world(np.ones(n, bool))
    if c_all is None:
        return dict(unstable=True)
    gate_total = (c_all.sum() - c_none.sum()) / n          # per original user
    gate_per_user = c_all.mean() - c_none.mean()
    est = {k: [] for k in ['inherit_per_user', 'inherit_total', 'exclude']}
    for _ in range(cfg['R']):
        z = designs.bernoulli(n, 0.5, rng)
        c, inviter = world(z)
        orig_t, orig_c = c[:n][z], c[:n][~z]
        new_c = c[n:]
        # inherit: new users join their inviter's arm (all inviters are treated)
        t_all = np.concatenate([orig_t, new_c])
        est['inherit_per_user'].append(t_all.mean() - orig_c.mean())
        est['inherit_total'].append(t_all.sum() / z.sum() - orig_c.sum() / (~z).sum())
        est['exclude'].append(orig_t.mean() - orig_c.mean())
    out = dict(gate_total=gate_total, gate_per_user=gate_per_user, unstable=False,
               new_users_all=int(c_all.size - n))
    for k, v in est.items():
        out[k] = float(np.mean(v))
    return out


def main(parts):
    if 'edges' in parts:
        cfgs = [dict(fam=f, r=r, rule=rule, boost=b, design=d, size=1000, nu_rho=0.5, R=10, seed=s)
                for f in FAMS for r in [0, 0.5, 1, 2, 5] for rule in ['random', 'fof']
                for b in [True, False] for d in ['bernoulli', 'metis']
                for s in range(SEEDS // 2)]
        sweep.run(edges, cfgs, str(RESULTS / 's9_edges2.jsonl'))
    if 'users' in parts:
        cfgs = [dict(fam=f, r=r, q=q, a=a, nu_rho=0.5, R=10, seed=s)
                for f in FAMILIES for r in [0.1, 0.5, 1, 2] for q in [0, 2]
                for a in [0.25, 0.5, 1.0] for s in range(SEEDS // 2)]
        sweep.run(users, cfgs, str(RESULTS / 's9_users.jsonl'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['edges', 'users'])
