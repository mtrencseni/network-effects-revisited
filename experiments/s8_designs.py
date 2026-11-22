"""Section 8: designs and estimators.

Observed outcomes are y = Poisson(W c) / W over a W-day window, with node
heterogeneity lam_i ~ Gamma(2, 1/2) (mean 1) unless stated otherwise.

Parts:
  clusters    bias, standard deviation and RMSE of DM for the GATE, Bernoulli against
              METIS cluster randomization at several cluster sizes and nu rho
  wronggraph  cluster randomization with clusters from a perturbed copy of the graph
  extrap      estimators of the full-rollout effect: DM, neighbourhood regression,
              two-fraction slope, 2SLS plug-in; under the true model and misspecified
              (saturating response; perturbed graph)
  cuped       variance reduction from a pre-period covariate, Bernoulli and cluster designs
  bayes       one-sided p-value (naive and cluster-robust) against the Bayesian posterior
              probability P(effect > 0); false positive rates under the null
"""
import sys

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from common import DLAM, LAM, RESULTS, SEEDS, graph_and_rho, observe, spec
from netfx import designs, estimators, families, generators, graph, solve, sweep

FAMS = [dict(family='ws', p=0.1), dict(family='er'), dict(family='sbm', B=100, mu=0.1),
        dict(family='cm', gamma=2.5), dict(family='ba')]
W = 7
SAT_SCALE = 3.0   # cap on the peer term in the saturating model (the linear peer term is 1 to 9)


def _setup(cfg, hetero=True):
    A, rho = graph_and_rho(spec(cfg['fam'], n=cfg.get('n', 100_000), seed=cfg['seed']))
    n = A.shape[0]
    rng = np.random.default_rng([cfg['seed'], 10])
    lam = rng.gamma(2.0, 0.5, n) if hetero else np.full(n, LAM)
    nu = cfg['nu_rho'] / rho
    return A, rho, n, rng, lam, nu


def _labels(A, design, size, seed, B=None):
    n = A.shape[0]
    if design == 'bernoulli' or size <= 1:
        return np.arange(n)
    return designs.metis(A, max(2, n // size), seed=seed)


def clusters(cfg):
    A, rho, n, rng, lam, nu = _setup(cfg)
    base = estimators.steady(A, lam, nu)
    gate = estimators.gate(A, nu, DLAM)
    labels = _labels(A, cfg['design'], cfg['size'], cfg['seed'])
    R = cfg['R']
    Z = np.column_stack([designs.cluster(labels, 0.5, rng) for _ in range(R)])
    L = DLAM * solve.cg(A, Z.astype(float), nu)
    est, haj = [], []
    # exposure conditions: treated with >= 75% treated neighbours, control with <= 25%;
    # their probabilities under the design by Monte Carlo (Aronow and Samii 2017)
    if not cfg.get('hajek', True):
        P1, P0 = np.zeros(n), np.zeros(n)
    elif cfg['design'] == 'bernoulli' or cfg['size'] <= 1:
        # exact: own treatment and Binomial(d_i, 1/2) treated neighbours
        from scipy.stats import binom
        deg = A.degrees()
        P1 = 0.5 * binom.sf(np.ceil(0.75 * deg) - 1, deg, 0.5)
        P0 = 0.5 * binom.cdf(np.floor(0.25 * deg), deg, 0.5)
    else:
        P1, P0 = np.zeros(n), np.zeros(n)
        mc = cfg.get('mc', 2000)
        for _ in range(mc):
            zz = designs.cluster(labels, 0.5, rng)
            f = designs.exposure(A, zz)
            P1 += zz & (f >= 0.75)
            P0 += ~zz & (f <= 0.25)
        P1 /= mc
        P0 /= mc
    for r in range(R):
        z = Z[:, r]
        y = observe(base + L[:, r], W, rng)
        est.append(y[z].mean() - y[~z].mean())
        f = designs.exposure(A, z)
        e1 = z & (f >= 0.75) & (P1 > 0)
        e0 = ~z & (f <= 0.25) & (P0 > 0)
        if e1.sum() and e0.sum():
            w1, w0 = 1 / P1[e1], 1 / P0[e0]
            haj.append((w1 * y[e1]).sum() / w1.sum() - (w0 * y[e0]).sum() / w0.sum())
        else:
            haj.append(np.nan)
    est, haj = np.array(est), np.array(haj)
    hj = haj[np.isfinite(haj)]
    cut = float((labels[A.indices] != np.repeat(labels, A.degrees())).mean())
    # first-order bias prediction: -dlam nu (fraction of edge ends cut) kbar / 2 * ... (see paper)
    return dict(gate=gate, mean=est.mean(), sd=est.std(ddof=1),
                rmse=float(np.sqrt(((est - gate) ** 2).mean())), cut=cut, nu=nu,
                kbar=A.nnz / n,
                hajek_mean=float(hj.mean()) if hj.size else np.nan,
                hajek_sd=float(hj.std(ddof=1)) if hj.size > 1 else np.nan,
                hajek_rmse=float(np.sqrt(((hj - gate) ** 2).mean())) if hj.size else np.nan,
                hajek_defined=float(np.isfinite(haj).mean()),
                covered=float(((P1 > 0) & (P0 > 0)).mean()))


def _perturb(A, q, rng):
    """Copy of A with a fraction q of its edges replaced by uniformly random edges."""
    S = A.to_scipy().tocoo()
    keep = S.row < S.col
    u, v = S.row[keep].astype(np.int64), S.col[keep].astype(np.int64)
    swap = rng.random(u.size) < q
    u = u.copy()
    v = v.copy()
    u[swap] = rng.integers(0, A.shape[0], swap.sum())
    v[swap] = rng.integers(0, A.shape[0], swap.sum())
    return graph.from_edges(A.shape[0], u, v)


def wronggraph(cfg):
    A, rho, n, rng, lam, nu = _setup(cfg)
    base = estimators.steady(A, lam, nu)
    gate = estimators.gate(A, nu, DLAM)
    D = _perturb(A, cfg['q'], rng) if cfg['q'] > 0 else A
    labels = _labels(D, 'metis', cfg['size'], cfg['seed'])
    R = cfg['R']
    Z = np.column_stack([designs.cluster(labels, 0.5, rng) for _ in range(R)])
    L = DLAM * solve.cg(A, Z.astype(float), nu)
    est = []
    for r in range(R):
        z = Z[:, r]
        y = observe(base + L[:, r], W, rng)
        est.append(y[z].mean() - y[~z].mean())
    est = np.array(est)
    cut_true = float((labels[A.indices] != np.repeat(labels, A.degrees())).mean())
    return dict(gate=gate, mean=est.mean(), sd=est.std(ddof=1),
                rmse=float(np.sqrt(((est - gate) ** 2).mean())), cut_true=cut_true)


def _outcomes(A, lam, nu, z, model, Aobs=None):
    """Steady-state production under the true model; 'linear' or 'saturating'."""
    if model == 'linear':
        return estimators.steady(A, lam + DLAM * z, nu)
    # saturating: c = lam + dlam z + s * tanh(nu A c / s), solved by iteration
    s = SAT_SCALE
    c = estimators.steady(A, lam + DLAM * z, nu)
    for _ in range(500):
        cn = lam + DLAM * z + s * np.tanh(nu * (A @ c) / s)
        if np.max(np.abs(cn - c)) < 1e-11:
            break
        c = cn
    return cn


def extrap(cfg):
    A, rho, n, rng, lam, nu = _setup(cfg)
    ones = np.ones(n)
    gate = (_outcomes(A, lam, nu, ones, cfg['model']).mean()
            - _outcomes(A, lam, nu, 0 * ones, cfg['model']).mean())
    Aobs = _perturb(A, cfg['q'], rng) if cfg['q'] > 0 else A
    out = {k: [] for k in ['dm', 'regression', 'regression2', 'two_fraction', 'pre_post',
                           'plug_in', 'nu_hat']}
    c_pre = _outcomes(A, lam, nu, 0 * ones, cfg['model'])
    for r in range(cfg['R']):
        z = designs.bernoulli(n, 0.5, rng)
        y = observe(_outcomes(A, lam, nu, z.astype(float), cfg['model']), W, rng)
        # Yu et al. 2022: historical baseline, mean outcome change over the treated fraction;
        # the pre-period may differ from the experiment period by a common drift
        drift_pre = 1 + cfg.get('drift', 0.0) * rng.standard_normal()
        y_pre = observe(_outcomes(A, lam * drift_pre, nu, 0 * ones, cfg['model']), W, rng)
        out['pre_post'].append((y.mean() - y_pre.mean()) / z.mean())
        out['regression2'].append(estimators.neighbourhood_regression2(y, z, Aobs))
        out['dm'].append(y[z].mean() - y[~z].mean())
        out['regression'].append(estimators.neighbourhood_regression(y, z, Aobs))
        z1 = designs.bernoulli(n, 0.1, rng)
        y1 = observe(_outcomes(A, lam, nu, z1.astype(float), cfg['model']), W, rng)
        z2 = designs.bernoulli(n, 0.9, rng)
        drift = 1 + cfg.get('drift', 0.0) * rng.standard_normal()   # baseline shift between runs
        y2 = observe(_outcomes(A, lam * drift, nu, z2.astype(float), cfg['model']), W, rng)
        out['two_fraction'].append(estimators.two_fraction(y1, z1, y2, z2))
        g, nh = estimators.plug_in(y, z, Aobs)
        out['plug_in'].append(g)
        out['nu_hat'].append(nh)
    res = dict(gate=gate, nu=nu)
    for k, v in out.items():
        v = np.array(v)
        if k == 'nu_hat':
            res['nu_hat'] = float(np.mean(v))
            continue
        res[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                      rmse=float(np.sqrt(np.mean((v - gate) ** 2))))
    return res


def cuped(cfg):
    A, rho, n, rng, lam, nu = _setup(cfg)
    base = estimators.steady(A, lam, nu)
    gate = estimators.gate(A, nu, DLAM)
    labels = _labels(A, cfg['design'], cfg['size'], cfg['seed'])
    out = {k: [] for k in ['dm', 'cuped', 'cuped_exposure', 'cuped_exposure_contrast']}
    for r in range(cfg['R']):
        z = designs.cluster(labels, 0.5, rng)
        x = observe(base, W, rng)                                   # pre-period
        y = observe(base + DLAM * solve.cg(A, z.astype(float), nu), W, rng)
        out['dm'].append(y[z].mean() - y[~z].mean())
        out['cuped'].append(estimators.cuped(y, x, z))
        # adding treated-neighbour exposure as a second covariate (a post-treatment variable)
        e = designs.exposure(A, z)
        X = np.column_stack([np.ones(n), z.astype(float), x - x.mean(), e - e.mean()])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        out['cuped_exposure'].append(b[1])                    # treatment coefficient alone
        out['cuped_exposure_contrast'].append(b[1] + b[3])    # (z=1, exposure=1) vs (0, 0)
    res = dict(gate=gate)
    for k, v in out.items():
        v = np.array(v)
        res[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)))
    return res


def bayes(cfg):
    """Counts y_i ~ Poisson(W c_i); naive tests treat nodes as independent."""
    A, rho, n, rng, lam, nu = _setup(cfg)
    base = estimators.steady(A, lam, nu)
    dl = cfg['effect']
    labels = _labels(A, cfg['design'], cfg['size'], cfg['seed'])
    rows = []
    for r in range(cfg['R']):
        z = designs.cluster(labels, 0.5, rng)
        c = base + dl * solve.cg(A, z.astype(float), nu) if dl else base
        y = rng.poisson(W * c)
        # frequentist, naive (independent nodes) one-sided z-test on the rate difference
        m1, m0 = y[z].mean(), y[~z].mean()
        se = np.sqrt(y[z].var(ddof=1) / z.sum() + y[~z].var(ddof=1) / (~z).sum())
        p_naive = norm.sf((m1 - m0) / se)
        # frequentist, cluster-robust: cluster totals as the units
        k = labels.max() + 1
        tot = np.bincount(labels, weights=y, minlength=k)
        cnt = np.bincount(labels, minlength=k)
        zc = np.bincount(labels, weights=z, minlength=k) > 0
        mt, mc = tot[zc].sum() / cnt[zc].sum(), tot[~zc].sum() / cnt[~zc].sum()
        rt = (tot[zc] - mt * cnt[zc]) / cnt[zc].mean()
        rc = (tot[~zc] - mc * cnt[~zc]) / cnt[~zc].mean()
        se_c = np.sqrt(rt.var(ddof=1) / zc.sum() + rc.var(ddof=1) / (~zc).sum())
        p_robust = norm.sf((mt - mc) / se_c)
        # Bayesian: independent Gamma(1, 1) priors on the per-node rate in each arm,
        # Poisson likelihood with independent nodes; P(rate_T > rate_C | data)
        a1, b1 = 1 + y[z].sum(), 1 + z.sum()
        a0, b0 = 1 + y[~z].sum(), 1 + (~z).sum()
        s1 = rng.gamma(a1, 1 / b1, 20000)
        s0 = rng.gamma(a0, 1 / b0, 20000)
        p_bayes = float((s1 > s0).mean())
        # Bayesian normal model with unknown variances and flat priors, nodes as units:
        # the posterior of the mean difference is (approximately) normal around the
        # difference with the naive standard error
        p_bayes_node = float(norm.cdf((m1 - m0) / se))
        # the same model with clusters as units
        p_bayes_cluster = float(norm.cdf((mt - mc) / se_c))
        rows.append((p_naive, p_robust, p_bayes, p_bayes_node, p_bayes_cluster))
    rows = np.array(rows)
    return dict(p_naive=rows[:, 0].tolist(), p_robust=rows[:, 1].tolist(),
                p_bayes=rows[:, 2].tolist(), p_bayes_node=rows[:, 3].tolist(),
                p_bayes_cluster=rows[:, 4].tolist())


def main(parts):
    R = 50
    if 'clusters' in parts:
        cfgs = [dict(fam=f, design=d, size=sz, nu_rho=x, R=R, seed=s) for f in FAMS
                for x in [0.3, 0.6, 0.9]
                for d, sz in [('bernoulli', 1)] + [('metis', s_) for s_ in [10, 30, 100, 300, 1000, 3000, 10000]]
                for s in range(SEEDS // 2)]
        sweep.run(clusters, cfgs, str(RESULTS / 's8_clusters3.jsonl'))
    if 'wronggraph' in parts:
        cfgs = [dict(fam=f, q=q, size=1000, nu_rho=0.6, R=R, seed=s) for f in FAMS
                for q in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] for s in range(SEEDS // 2)]
        sweep.run(wronggraph, cfgs, str(RESULTS / 's8_wronggraph.jsonl'))
    if 'extrap' in parts:
        cfgs = [dict(fam=f, model=m, q=q, drift=dr, nu_rho=x, R=20, seed=s, v=3)
                for f in FAMS
                for m, q, dr in [('linear', 0.0, 0.0), ('linear', 0.2, 0.0), ('linear', 0.0, 0.01),
                                 ('saturating', 0.0, 0.0)]
                for x in [0.3, 0.6, 0.9] for s in range(SEEDS // 2)]
        sweep.run(extrap, cfgs, str(RESULTS / 's8_extrap3.jsonl'))
    if 'cuped' in parts:
        cfgs = [dict(fam=f, design=d, size=sz, nu_rho=0.6, R=R, seed=s) for f in FAMS
                for d, sz in [('bernoulli', 1), ('metis', 100), ('metis', 1000)]
                for s in range(SEEDS // 2)]
        sweep.run(cuped, cfgs, str(RESULTS / 's8_cuped2.jsonl'))
    if 'bayes' in parts:
        cfgs = [dict(fam=f, design=d, size=sz, effect=e, nu_rho=0.6, R=100, seed=s)
                for f in FAMS
                for d, sz in [('bernoulli', 1), ('metis', 100), ('metis', 1000)]
                for e in [0.0, 0.002] for s in range(SEEDS // 2)]
        sweep.run(bayes, cfgs, str(RESULTS / 's8_bayes.jsonl'))


if __name__ == '__main__':
    main(sys.argv[1:] or ['clusters', 'wronggraph', 'extrap', 'cuped', 'bayes'])
