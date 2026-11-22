"""Compare the closed forms of netfx/theory.py with the simulation results.

Run from the project root:
    PYTHONPATH=. .venv/bin/python experiments/theory_check.py [T1 T2 ...]

Reads results/*.jsonl and results/graph_stats.json (from experiments/graph_stats.py), prints
markdown tables, and writes results/theory_pairs.jsonl with one row per comparison:
    {"target", "quantity", "family", "setting", "predicted", "simulated", "simulated_se"}
where predicted and simulated are means over the graph seeds that have both a result and a
graph-statistics entry, and simulated_se is the standard error over those seeds (null when
there is a single seed). Each target uses the formula one would put in the paper: the annealed
multiplier with the measured rho (T1), the tree Green's function (T2 to T4), the annealed
multi-type models for heterogeneous degrees (T6 to T8), the Gleeson-Cahalane recursion with
the family's own degree histogram (T9).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from netfx import theory as cf  # noqa: E402

DLAM, LAM, N, KBAR = 0.05, 1.0, 100_000, 20
FAMS = {'ws': dict(family='ws', p=0.1), 'er': dict(family='er'),
        'sbm': dict(family='sbm', B=100, mu=0.1), 'ba': dict(family='ba'),
        'cm': dict(family='cm', gamma=2.5)}
NAMES = {'ws': 'Watts-Strogatz', 'er': 'Erdos-Renyi', 'sbm': 'SBM', 'ba': 'Barabasi-Albert',
         'cm': 'power-law CM', 'gam': 'gamma CM'}
ORDER = list(FAMS)
STATS = json.load(open(os.path.join(ROOT, 'results', 'graph_stats.json')))
PAIRS = []


def results(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, 'results', name + '.jsonl')) if l.strip()]


def stat(fam, seed=0, n=N, kbar=KBAR, **extra):
    sp = dict(FAMS[fam] if isinstance(fam, str) else fam, n=n, kbar=kbar, seed=seed)
    sp.update(extra)
    return STATS.get(json.dumps(sp, sort_keys=True))


def stats_for(fam, seeds=range(5)):
    return [s for s in (stat(fam, sd) for sd in seeds) if s is not None]


def resolvent(st, nu_rho):
    for r in st['resolvent']:
        if abs(r['nu_rho'] - nu_rho) < 1e-9:
            return r
    raise KeyError(nu_rho)


def tree_params(st):
    """(kbar, branching b = kappa - 1, degree histogram, triangles per node)."""
    kap = st['k2'] / st['kbar']
    return st['kbar'], kap - 1, np.array(st['deg_hist'], float), st.get('triangles_per_node', 0.0)


def table(title, headers, rows, fmt='%.3f'):
    print(f'\n### {title}\n')
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join('---' for _ in headers) + '|')
    for r in rows:
        cells = []
        for x in r:
            if isinstance(x, str):
                cells.append(x)
            elif x is None or (isinstance(x, float) and not np.isfinite(x)):
                cells.append('-' if x is None or np.isnan(x) else 'inf')
            elif isinstance(x, (int, np.integer)):
                cells.append(str(x))
            else:
                cells.append(fmt % x)
        print('| ' + ' | '.join(cells) + ' |')


def mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else np.nan


def pair(target, quantity, family, setting, predicted, simulated):
    """Record one comparison; predicted and simulated are lists over seeds (or scalars)."""
    pr = np.atleast_1d(np.asarray(predicted, float))
    sm = np.atleast_1d(np.asarray(simulated, float))
    sm_f = sm[np.isfinite(sm)]
    pr_f = pr[np.isfinite(pr)]
    PAIRS.append(dict(target=target, quantity=quantity, family=family, setting=setting,
                      predicted=float(pr_f.mean()) if pr_f.size else None,
                      simulated=float(sm_f.mean()) if sm_f.size else None,
                      simulated_se=float(sm_f.std(ddof=1) / np.sqrt(sm_f.size)) if sm_f.size > 1 else None))
    return PAIRS[-1]


def sort_key(kv):
    k = kv[0]
    return (ORDER.index(k[0]) if k[0] in ORDER else 9,) + tuple(str(x) if isinstance(x, (str, bool)) else x for x in k[1:])


# ----------------------------------------------------------------------------------

def t1():
    print('\n## T1  multiplier and stability')
    rows = []
    for f in FAMS:
        sts = stats_for(f)
        kap = mean([s['k2'] / s['kbar'] for s in sts])
        kmax = mean([s['kmax'] for s in sts])
        pred = [cf.rho_annealed(s['kbar'], s['k2'], s['kmax']) for s in sts]
        rows.append([NAMES[f], mean([s['rho'] for s in sts]), kap, np.sqrt(kmax), mean(pred),
                     mean([cf.rho_hub(s['kmax'], s['k2'] / s['kbar']) for s in sts]),
                     mean([s['hub_lambda'] for s in sts]), kmax])
        pair('T1', 'rho', f, dict(n=N, kbar=KBAR), pred, [s['rho'] for s in sts])
    table('T1a: spectral radius against degree statistics (N = 1e5, kbar = 20, 5 graphs)',
          ['family', 'rho (measured)', 'kappa = <k^2>/<k>', 'sqrt k_max', 'max(kappa, sqrt k_max)',
           'sqrt(k_max + kappa - 1)', 'hub two-level tree', 'k_max'], rows, '%.2f')
    rows = []
    for f, extra in [('ba', {}), ('cm', {'gamma': 2.5})]:
        for n in [10**4, 10**5, 10**6]:
            s = stat(dict(family=f, **extra), 0, n=n, kbar=10)
            if s is None:
                continue
            kap = s['k2'] / s['kbar']
            pred = cf.rho_annealed(s['kbar'], s['k2'], s['kmax'])
            rows.append([NAMES[f], n, s['rho'], kap, np.sqrt(s['kmax']), cf.rho_hub(s['kmax'], kap),
                         s['hub_lambda'], s['rho'] / np.sqrt(s['kmax'])])
            pair('T1', 'rho', f, dict(n=n, kbar=10), pred, s['rho'])
    table('T1b: spectral radius against size (kbar = 10, seed 0; s5_size has the same rho)',
          ['family', 'N', 'rho', 'kappa', 'sqrt k_max', 'sqrt(k_max + kappa - 1)',
           'hub two-level tree', 'rho / sqrt k_max'], rows, '%.2f')
    data = defaultdict(list)
    for r in results('s5_multiplier'):
        f, seed = r['config']['fam']['family'], r['config']['seed']
        st = stat(f, seed)
        if st is None:
            continue
        kap = st['k2'] / st['kbar']
        for p in r['result']['points']:
            nu = p['nu_rho'] / r['result']['rho']
            data[(f, p['nu_rho'])].append((p['multiplier'], p['mean_field'],
                                           cf.multiplier_annealed(nu, st['kbar'], kap),
                                           cf.multiplier_annealed(nu, st['kbar'], r['result']['rho'])))
    rows = []
    for (f, x), v in sorted(data.items(), key=sort_key):
        v = np.array(v)
        rows.append([NAMES[f], x] + list(v.mean(axis=0)) + [v[:, 3].mean() / v[:, 0].mean() - 1])
        pair('T1', 'multiplier', f, dict(nu_rho=x), v[:, 3], v[:, 0])
    table('T1c: multiplier 1\'M1/N (s5_multiplier, seeds 0-4)',
          ['family', 'nu rho', 'simulation', '1/(1 - nu kbar)', 'annealed 1 + nu kbar/(1 - nu kappa)',
           'spectral 1 + nu kbar/(1 - nu rho)', 'rel. error of spectral'], rows)
    byc = defaultdict(list)
    for r in results('s5_hetero'):
        c, res = r['config'], r['result']
        byc[(c['nu_kbar'], c['cv'])].append(res)
    rows = []
    for (nk, cv), v in sorted(byc.items()):
        sts = [s for s in (stat(dict(family='gam', cv=cv), s) for s in range(3)) if s]
        kap_real = mean([s['k2'] / s['kbar'] for s in sts])
        kap_nom = cf.gamma_kappa(KBAR, cv)
        kbar = mean([x['kbar'] for x in v])
        nu = nk / kbar
        rho_sim = [x['rho'] for x in v]
        m_sim = [x.get('multiplier', np.inf) for x in v]
        m_pred = [cf.multiplier_annealed(nk / x['kbar'], x['kbar'], x['rho']) for x in v]
        rows.append([nk, cv, mean(rho_sim), kap_nom, kap_real, mean([s['kmax'] for s in sts]) ** 0.5,
                     '%d/%d' % (sum(x['stable'] for x in v), len(v)),
                     'yes' if nu * kap_real < 1 else 'no', mean(m_sim) if all(x['stable'] for x in v) else np.inf,
                     cf.multiplier_annealed(nu, kbar, kap_real), mean(m_pred)])
        pair('T1', 'rho (gamma model)', 'gam', dict(cv=cv, nu_kbar=nk), [kap_nom] * len(v), rho_sim)
        pair('T1', 'stable fraction (gamma model)', 'gam', dict(cv=cv, nu_kbar=nk),
             float(nu * kap_real < 1), [float(x['stable']) for x in v])
        if all(x['stable'] for x in v):
            pair('T1', 'multiplier (gamma model)', 'gam', dict(cv=cv, nu_kbar=nk), m_pred, m_sim)
    table('T1d: gamma configuration model (s5_hetero); cv* = sqrt(1/(nu kbar) - 1) = '
          '%.3f at nu kbar = 0.3 and %.3f at 0.6' % (cf.cv_critical(0.3), cf.cv_critical(0.6)),
          ['nu kbar', 'cv', 'rho (sim)', 'kbar(1+cv^2)', 'kappa (realized degrees)', 'sqrt k_max',
           'stable (sim)', 'stable (nu kappa < 1)', 'multiplier (sim)', 'annealed (kappa)',
           'annealed (rho)'], rows)
    byf = defaultdict(list)
    for r in results('s5_hetero_base'):
        f = r['config']['fam']['family']
        st = stat(f, r['config']['seed'])
        if st is None:
            continue
        for p in r['result']['points']:
            nu = p['nu_kbar'] / r['result']['kbar']
            byf[(f, p['nu_kbar'])].append((p.get('multiplier', np.inf), p['stable'],
                                          cf.multiplier_annealed(nu, st['kbar'], st['k2'] / st['kbar']),
                                          cf.multiplier_annealed(nu, st['kbar'], r['result']['rho']),
                                          r['result']['cv']))
    rows = []
    for (f, nk), v in sorted(byf.items(), key=sort_key):
        rows.append([NAMES[f], nk, mean([x[4] for x in v]), '%d/%d' % (sum(x[1] for x in v), len(v)),
                     mean([x[0] for x in v]) if all(x[1] for x in v) else np.inf,
                     mean([x[2] for x in v]), mean([x[3] for x in v])])
        if all(x[1] for x in v):
            pair('T1', 'multiplier (fixed nu kbar)', f, dict(nu_kbar=nk), [x[3] for x in v], [x[0] for x in v])
    table('T1e: the five families at fixed nu kbar (s5_hetero_base, seeds 0-4)',
          ['family', 'nu kbar', 'degree cv', 'stable (sim)', 'multiplier (sim)', 'annealed (kappa)',
           'annealed (rho)'], rows)


def t2():
    print('\n## T2  spillover by hop distance (s5_hops, 2% treated)')
    acc = defaultdict(lambda: defaultdict(list))
    for r in results('s5_hops'):
        f, x, seed = r['config']['fam']['family'], r['config']['nu_rho'], r['config']['seed']
        st = stat(f, seed)
        if st is None:
            continue
        kbar, b, hist, _ = tree_params(st)
        lifts, shares = cf.hop_lift(x / st['rho'], kbar, b, 0.02,
                                    hops=(0, 1, 2, 3), deg_hist=hist)
        for bh in r['result']['by_hop']:
            h = bh['h']
            if h < 4:
                acc[(f, x)][h].append((bh['lift'] / DLAM, bh['share'], lifts[h], shares[h]))
    rows = []
    for (f, x), hops in sorted(acc.items(), key=sort_key):
        for h in sorted(hops):
            v = np.array(hops[h])
            rows.append([NAMES[f], x, h, v[:, 1].mean(), v[:, 3].mean(), v[:, 0].mean(), v[:, 2].mean(),
                         v[:, 2].mean() / v[:, 0].mean() - 1])
            pair('T2', f'lift hop {h}', f, dict(nu_rho=x, p=0.02), v[:, 2], v[:, 0])
            pair('T2', f'share hop {h}', f, dict(nu_rho=x, p=0.02), v[:, 3], v[:, 1])
    table('T2: lift / dlam by hop distance; tree Green\'s function with the measured degree histogram',
          ['family', 'nu rho', 'hop', 'share (sim)', 'share (tree)', 'lift/dlam (sim)',
           'lift/dlam (tree)', 'rel. error'], rows, '%.4f')


def t3():
    print('\n## T3  attenuation E[DM]/GATE')
    acc = defaultdict(list)
    for r in results('s5_attenuation'):
        c, res = r['config'], r['result']
        f, x, seed = c['fam']['family'], c['nu_rho'], c['seed']
        st = stat(f, seed)
        if st is None:
            continue
        kbar, b, hist, tri = tree_params(st)
        nu = x / res['rho']
        tau_tree = cf.tr_over_n_tree(nu, kbar, b, hist)
        tau_tri = cf.tr_over_n_tree(nu, kbar, b, hist, tri)
        m_spec = cf.multiplier_annealed(nu, kbar, res['rho'])
        rs = resolvent(st, x)
        acc[(f, x)].append((res['ratio'], (1 - x), cf.attenuation_exact_from_parts(tau_tree, m_spec, N),
                            cf.attenuation_exact_from_parts(tau_tri, m_spec, N),
                            cf.attenuation_exact_from_parts(rs['tr_over_n'], rs['multiplier'], N),
                            tau_tree, tau_tri, rs['tr_over_n'], m_spec, rs['multiplier']))
    rows, rows2 = [], []
    for (f, x), v in sorted(acc.items(), key=sort_key):
        a = np.array(v)
        v = a.mean(axis=0)
        rows.append([NAMES[f], x] + list(v[:5]) + [v[3] / v[0] - 1])
        rows2.append([NAMES[f], x, v[5], v[6], v[7], v[8], v[9]])
        pair('T3', 'E[DM]/GATE', f, dict(nu_rho=x, p=0.1), a[:, 3], a[:, 0])
        pair('T3', 'tr(M)/N', f, dict(nu_rho=x), a[:, 6], a[:, 7])
    table('T3a: E[DM]/GATE, 10% Bernoulli (s5_attenuation, seeds 0-4)',
          ['family', 'nu rho', 'simulation', '1 - nu rho', 'tree tau / spectral m',
           '+ triangles', 'measured tr(M)/N and m', 'rel. error (+triangles)'], rows)
    table('T3b: the parts: tr(M)/N (tree, tree + triangles, measured) and the multiplier (spectral, measured)',
          ['family', 'nu rho', 'tau tree', 'tau + triangles', 'tau measured', 'm spectral', 'm measured'],
          rows2, '%.4f')
    rows = []
    byk = defaultdict(list)
    for r in results('s5_kbar'):
        c = r['config']
        byk[(c['kbar'], c['nu_rho'])].append((r['result']['ratio'], r['result']['tr_over_n'], c['seed']))
    for (k, x), v in sorted(byk.items()):
        preds, sims = [], []
        for ratio, tau_ex, seed in v:
            st = stat(dict(family='er'), seed, n=3000, kbar=k)
            if st is None:
                continue
            kbar, b, hist, _ = tree_params(st)
            nu = x / st['rho']
            tau = cf.tr_over_n_tree(nu, kbar, b, hist)
            m = cf.multiplier_annealed(nu, kbar, st['rho'])
            preds.append((cf.attenuation_exact_from_parts(tau, m, 3000), tau, m, st['rho']))
            sims.append((ratio, tau_ex))
        if not preds:
            continue
        P, S = np.array(preds), np.array(sims)
        rows.append([k, x, S[:, 0].mean(), P[:, 0].mean(), S[:, 1].mean(), P[:, 1].mean(),
                     cf.attenuation_regular(x, k), cf.tr_over_n_kesten_mckay(x / k, k), P[:, 3].mean()])
        pair('T3', 'E[DM]/GATE (exact, N=3000)', 'er', dict(n=3000, kbar=k, nu_rho=x), P[:, 0], S[:, 0])
        pair('T3', 'tr(M)/N (exact, N=3000)', 'er', dict(n=3000, kbar=k, nu_rho=x), P[:, 1], S[:, 1])
    table('T3c: Erdos-Renyi N = 3000, exact dense M (s5_kbar) against the tree formula with the measured '
          'degrees, and the random k-regular graph (Kesten-McKay)',
          ['kbar', 'nu rho', 'ratio (exact)', 'ratio (tree, measured rho)', 'tr(M)/N (exact)',
           'tr(M)/N (tree)', 'ratio k-regular', 'tr(M)/N k-regular', 'rho (measured)'], rows, '%.4f')


def _phi_lookup():
    """phi = 1 - cut for the METIS clusters of s8_clusters3 by (family, size, seed); s9_edges2
    uses the same partitions (same graph, same METIS seed)."""
    out = {}
    for r in results('s8_clusters3'):
        c = r['config']
        out[(c['fam']['family'], c['size'], c['seed'])] = 1 - r['result']['cut']
    return out


def t4():
    print('\n## T4  cluster designs')
    acc = defaultdict(list)
    for r in results('s5_clustering'):
        c, res = r['config'], r['result']
        f, seed = c['fam']['family'], c['seed']
        st = stat(f, seed)
        if st is None:
            continue
        kbar, b, hist, _ = tree_params(st)
        nu = c['nu_rho'] / st['rho']
        phi = 1 - res['cut']
        size = c['size'] if c['design'] != 'blocks' else 1000
        C = N / size if c['design'] != 'bernoulli' else N
        acc[(f, c['design'], size)].append(
            (res['ratio'], phi, res['same_arm'], cf.cluster_ratio_mean_field(c['nu_rho'], phi, C),
             cf.cluster_ratio_tree(nu, st['rho'], kbar, b, phi, N, size, hist), res['ratio_sd']))
    rows = []
    for (f, d, size), v in sorted(acc.items(), key=sort_key):
        a = np.array(v)
        v = a.mean(axis=0)
        rows.append([NAMES[f], d, size, v[1], v[0], v[3], v[4], v[4] / v[0] - 1, v[5]])
        pair('T4', 'E[DM]/GATE (clusters)', f, dict(design=d, size=size, nu_rho=0.5, phi=float(v[1])), a[:, 4], a[:, 0])
    table('T4a: E[DM]/GATE under cluster assignment at nu rho = 0.5 (s5_clustering, seeds 0-4); '
          'phi = 1 - cut from the stored partitions',
          ['family', 'design', 'size', 'phi', 'ratio (sim)', '(1 - nu rho)/(1 - nu rho r)',
           'tree two-type', 'rel. error (tree)', 'sd of ratio over assignments (sim)'], rows)
    acc = defaultdict(list)
    for r in results('s8_clusters3'):
        c, res = r['config'], r['result']
        f, seed, x = c['fam']['family'], c['seed'], c['nu_rho']
        st = stat(f, seed)
        if st is None:
            continue
        kbar, b, hist, _ = tree_params(st)
        nu = x / st['rho']
        phi = 1 - res['cut']
        size = c['size']
        m = res['gate'] / DLAM
        rs = resolvent(st, x)
        var_k = st['k2'] - st['kbar'] ** 2
        ratio_f = cf.cluster_ratio_tree(nu, st['rho'], kbar, b, phi, N, size, hist)
        sd_f = cf.dm_sd(N, nu, kbar, b, phi, m, 7, var_lam=0.5, base_sd=rs['base_sd'], size=size,
                        deg_hist=hist, var_k=var_k, rho=st['rho'])
        sd_0 = cf.dm_sd(N, nu, kbar, b, phi, m, 7, var_lam=0.5, base_sd=rs['base_sd'], size=size, deg_hist=hist,
                        rho=st['rho'])
        acc[(f, x, size)].append((res['mean'] / res['gate'], ratio_f, cf.cluster_ratio_mean_field(x, phi, N / size),
                                  res['sd'] / res['gate'], sd_f / res['gate'], sd_0 / res['gate'], phi))
    rows = []
    for (f, x, size), v in sorted(acc.items(), key=sort_key):
        a = np.array(v)
        v = a.mean(axis=0)
        rows.append([NAMES[f], x, size, v[6], v[0], v[2], v[1], v[3], v[4], v[5]])
        pair('T4', 'E[DM]/GATE (clusters, noisy)', f, dict(size=size, nu_rho=x, phi=float(v[6])), a[:, 1], a[:, 0])
        pair('T4', 'sd(DM)/GATE', f, dict(size=size, nu_rho=x, phi=float(v[6])), a[:, 4], a[:, 3])
    table('T4b: s8_clusters3 (heterogeneous lam, weekly Poisson counts): mean and sd of DM over '
          'assignments, as fractions of the GATE',
          ['family', 'nu rho', 'size', 'phi', 'mean/GATE (sim)', 'mean field', 'tree two-type',
           'sd/GATE (sim)', 'sd/GATE (formula)', 'formula without cluster-degree term'], rows)
    acc = defaultdict(list)
    for r in results('s8_wronggraph'):
        c, res = r['config'], r['result']
        f, seed = c['fam']['family'], c['seed']
        st = stat(f, seed)
        if st is None:
            continue
        kbar, b, hist, _ = tree_params(st)
        nu = 0.6 / st['rho']
        phi = 1 - res['cut_true']
        m = res['gate'] / DLAM
        rs = resolvent(st, 0.6)
        acc[(f, c['q'])].append((phi, res['mean'] / res['gate'],
                                 cf.cluster_ratio_tree(nu, st['rho'], kbar, b, phi, N, 1000, hist),
                                 res['sd'] / res['gate'],
                                 cf.dm_sd(N, nu, kbar, b, phi, m, 7, 0.5, rs['base_sd'], size=1000, deg_hist=hist,
                                          var_k=st['k2'] - st['kbar'] ** 2, rho=st['rho']) / res['gate']))
    rows = []
    for (f, q), v in sorted(acc.items(), key=sort_key):
        a = np.array(v)
        v = a.mean(axis=0)
        rows.append([NAMES[f], q] + list(v))
        pair('T4', 'E[DM]/GATE (wrong graph)', f, dict(q=q, size=1000, nu_rho=0.6, phi=float(v[0])), a[:, 2], a[:, 1])
        pair('T4', 'sd(DM)/GATE (wrong graph)', f, dict(q=q, size=1000, nu_rho=0.6, phi=float(v[0])), a[:, 4], a[:, 3])
    table('T4c: clusters from a perturbed graph (s8_wronggraph, nu rho = 0.6, size 1000); phi is the '
          'within-cluster share on the true graph',
          ['family', 'q', 'phi (true graph)', 'mean/GATE (sim)', 'tree two-type', 'sd/GATE (sim)',
           'sd/GATE (formula)'], rows)


def t5():
    print('\n## T5  power near the stability limit (s7_power3, 50/50 Bernoulli, W = 7, N = 1e5)')
    acc = defaultdict(list)
    for r in results('s7_power3'):
        c, res = r['config'], r['result']
        if c['effect'] == 0.0:
            continue
        f, x, seed = c['fam']['family'], c['nu_rho'], c['seed']
        st = stat(f, seed)
        if st is None:
            continue
        kbar, b, hist, tri = tree_params(st)
        nu = x / res['rho']
        rs = resolvent(st, x)
        m = res['base_mean'] / LAM
        tau = cf.tr_over_n_tree(nu, kbar, b, hist, tri)
        z, pw, se = cf.power_z(N, m, tau, rs['base_cv'], 7)
        zc, pwc, sec = cf.power_z(N, m, tau, rs['base_cv'], 7, cuped=True)
        zd, pwd, _ = cf.power_z(N, m, tau, rs['base_cv'], 7, cuped=True, drift=0.1)
        acc[(f, x)].append((res['dm_lift_mean'] / DLAM, tau, res['dm_sd'], 2 * rs['base_sd'] / np.sqrt(N),
                            res['obs_sd'], se, res['cuped_sd'], sec, res['power'], pw,
                            res['power_cuped'], pwc, res['power_cuped_drift'], pwd, rs['base_cv'], m))
    rows, rows2 = [], []
    for (f, x), v in sorted(acc.items(), key=sort_key):
        a = np.array(v)
        v = a.mean(axis=0)
        rows.append([NAMES[f], x, v[15], v[14], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]])
        rows2.append([NAMES[f], x, v[8], v[9], v[10], v[11], v[12], v[13]])
        s = dict(nu_rho=x, W=7, n=N)
        pair('T5', 'E[DM]/dlam', f, s, a[:, 1], a[:, 0])
        pair('T5', 'sd(DM) no Poisson', f, s, a[:, 3], a[:, 2])
        pair('T5', 'se(DM)', f, s, a[:, 5], a[:, 4])
        pair('T5', 'se(DM) CUPED', f, s, a[:, 7], a[:, 6])
        pair('T5', 'power', f, s, a[:, 9], a[:, 8])
        pair('T5', 'power CUPED', f, s, a[:, 11], a[:, 10])
        pair('T5', 'power CUPED drift', f, s, a[:, 13], a[:, 12])
    table('T5a: signal and noise terms; the multiplier m and the baseline spread base_cv = sd(M1)/mean(M1) are '
          'graph statistics', ['family', 'nu rho', 'm', 'base_cv', 'E[DM_lift]/dlam (sim)', 'tau (tree)',
                               'sd DM, no Poisson (sim)', '2 base_sd/sqrt N', 'sd observed DM (sim)',
                               'se formula', 'sd CUPED DM (sim)', 'se CUPED formula'], rows, '%.4f')
    table('T5b: power of the one-sided test at 5%, 100 assignments on each of 5 graphs',
          ['family', 'nu rho', 'power (sim)', 'Phi(z - 1.645)', 'CUPED (sim)', 'CUPED formula',
           'CUPED, 10% drift (sim)', 'formula'], rows2, '%.3f')


def t6():
    print('\n## T6  treatment adds edges (s9_edges2, nu rho = 0.5 on the original graph)')
    phi = _phi_lookup()
    acc = defaultdict(list)
    for r in results('s9_edges2'):
        c, res = r['config'], r['result']
        f, seed = c['fam']['family'], c['seed']
        st = stat(f, seed)
        if st is None or (f, 1000, seed) not in phi:
            continue
        nu = res['nu']
        rho = 0.5 / nu
        ph = 0.0 if c['design'] == 'bernoulli' else phi[(f, 1000, seed)]
        psi = res['new_edges_cross_arm'] if c['r'] > 0 else 0.5
        ann = cf.edges_annealed_effects(nu, st['kbar'], rho, c['r'], c['rule'], ph, psi, boost=c['boost'])
        gate_mf = cf.edges_gate(nu, rho, st['kbar'], c['r'], boost=c['boost'])
        dm_mf = cf.edges_dm(nu, rho, c['r'], ph, psi, boost=c['boost'])[0]
        acc[(f, c['rule'], c['boost'], c['design'], c['r'])].append(
            (res['gate'], ann['gate'], res['dm'], ann['dm'], gate_mf, dm_mf, psi, ph))
    rows = []
    for (f, rule, boost, design, rr), v in sorted(acc.items(), key=sort_key):
        a = np.array(v)
        v = a.mean(axis=0)
        ratio = v[2] / v[0] if v[0] else np.nan
        ratio_f = v[3] / v[1] if v[1] else np.nan
        rows.append([NAMES[f], rule, 'yes' if boost else 'no', design, rr, v[7], v[6], v[0] / DLAM,
                     v[1] / DLAM, v[4] / DLAM, v[2] / DLAM, v[3] / DLAM, v[5] / DLAM, ratio, ratio_f])
        s = dict(rule=rule, boost=boost, design=design, r=rr, nu_rho=0.5, phi=float(v[7]), psi=float(v[6]))
        if rr > 0 or boost:
            pair('T6', 'GATE (edges)', f, s, a[:, 1], a[:, 0])
            pair('T6', 'DM (edges)', f, s, a[:, 3], a[:, 2])
            if v[0]:
                pair('T6', 'DM/GATE (edges)', f, s, a[:, 3] / a[:, 1], a[:, 2] / a[:, 0])
    table('T6: GATE and E[DM] in units of dlam; annealed two-type model (and the homogeneous mean field with '
          'k -> rho); psi = measured share of new edges between arms, phi = within-cluster share of the '
          'original edges',
          ['family', 'rule', 'boost', 'design', 'r', 'phi', 'psi', 'GATE (sim)', 'GATE (annealed)',
           'GATE (mf)', 'DM (sim)', 'DM (annealed)', 'DM (mf)', 'DM/GATE (sim)', 'DM/GATE (annealed)'],
          rows, '%.2f')


def t7():
    print('\n## T7  treatment adds users (s9_users, nu rho = 0.5)')
    acc = defaultdict(list)
    keys = ['gate_total', 'gate_per_user', 'inherit_total', 'inherit_per_user', 'exclude']
    for r in results('s9_users'):
        c, res = r['config'], r['result']
        if res.get('unstable'):
            continue
        f, seed = c['fam']['family'], c['seed']
        st = stat(f, seed)
        if st is None:
            continue
        nu = 0.5 / st['rho']
        pred = cf.users_effects(nu, st['rho'], c['r'], c['q'], c['a'], kbar=st['kbar'])
        acc[(f, c['r'], c['q'], c['a'])].append([res[k] for k in keys] + [pred[k] for k in keys])
    rows = []
    for (f, rr, q, a), v in sorted(acc.items(), key=sort_key):
        A = np.array(v)
        v = A.mean(axis=0)
        row = [NAMES[f], rr, q, a]
        for i in range(5):
            row += [v[i], v[5 + i]]
            pair('T7', keys[i], f, dict(r=rr, q=q, a=a, nu_rho=0.5), A[:, 5 + i], A[:, i])
        rows.append(row)
    table('T7: effects and estimates (production units; dlam = 0.05); annealed three-type model (kbar, rho)',
          ['family', 'r', 'q', 'a', 'GATE total (sim)', 'ann.', 'GATE per user (sim)', 'ann.',
           'inherit_total (sim)', 'ann.', 'inherit_per_user (sim)', 'ann.', 'exclude (sim)', 'ann.'], rows, '%.3f')
    for f in FAMS:
        e = [abs(np.array(v)[:, 5:10] - np.array(v)[:, :5]).max() for k, v in acc.items() if k[0] == f]
        if e:
            print(f'  {NAMES[f]}: largest absolute error over settings and quantities {max(e):.3f}')


def t8():
    print('\n## T8  treatment raises the peer effect (s10_nutreat, nu rho = 0.8)')
    ipr = defaultdict(list)
    for r in results('s5_multiplier'):
        ipr[r['config']['fam']['family']].append(r['result']['ipr'])
    acc = defaultdict(list)
    for r in results('s10_nutreat'):
        c, res = r['config'], r['result']
        f = c['fam']['family']
        st = stat(f, c['seed'])
        if st is None:
            continue
        kbar, rho = st['kbar'], st['rho']
        for row in res['rows']:
            pred = cf.nutreat(0.8, c['delta'], row['p'])
            pa = cf.nutreat_annealed(0.8, c['delta'], row['p'], kbar, rho)
            acc[(f, c['delta'], row['p'])].append(
                (row['unstable'], not pred['stable'], row.get('dm', np.inf), pred['dm'], pa['dm'],
                 row.get('mean_lift', np.inf), pred['mean_lift'], pa['mean_lift'], res['gate'], pred['gate'],
                 pa['gate'], pred['partial_nu_rho']))
    rows = []
    for (f, d, p), v in sorted(acc.items(), key=sort_key):
        if d == 0:
            continue
        a = np.array(v, dtype=float)
        s = dict(delta=d, p=p, nu_rho=0.8)
        pair('T8', 'unstable fraction', f, s, a[0, 1], a[:, 0])
        if not a[:, 0].any():
            pair('T8', 'DM (nutreat)', f, s, a[:, 4], a[:, 2])
            pair('T8', 'mean lift (nutreat)', f, s, a[:, 7], a[:, 5])
        if np.isfinite(a[0, 10]) and np.all(np.isfinite(a[:, 8])):
            pair('T8', 'GATE (nutreat)', f, s, a[:, 10], a[:, 8])
        if p not in (0.01, 0.1, 0.5):
            continue
        rows.append([NAMES[f], d, p, a[0, 11], '%d/%d' % (a[:, 0].sum(), len(a)), 'yes' if a[0, 1] else 'no',
                     a[:, 2].mean(), a[0, 3], a[0, 4], a[:, 5].mean(), a[0, 6], a[0, 7], a[:, 8].mean(),
                     a[0, 9], a[0, 10]])
    table('T8: stability and DM of the partial experiment, GATE of the full rollout (inf = no steady state); '
          'p = 0.01, 0.1, 0.5 shown',
          ['family', 'delta', 'p', 'nu rho (1 + delta p)', 'unstable (sim)', 'unstable (mf)', 'DM (sim)',
           'DM (mf)', 'DM (annealed, kbar/rho)', 'mean lift (sim)', 'mf', 'annealed', 'GATE (sim)', 'mf',
           'annealed'], rows, '%.3f')
    print('\nPerron-weight fluctuation of the treated half, sd(w_T) = sqrt(p(1-p) sum v_i^4) at p = 1/2, and '
          'the resulting spread of nu rho (1 + delta w_T) at delta = 0.5:')
    for f in FAMS:
        if f in ipr:
            sd = np.sqrt(0.25 * mean(ipr[f]))
            print(f'  {NAMES[f]}: ipr = {mean(ipr[f]):.2e}, sd(w_T) = {sd:.4f}, nu rho_eff = 1.000 +- {0.8 * 0.5 * sd:.3f}')


def t9():
    print('\n## T9  threshold contagion (s10_threshold2, h = 0.5, 20 assignments on each of 5 graphs)')
    acc = defaultdict(list)
    for r in results('s10_threshold2'):
        c, res = r['config'], r['result']
        for row in res['rows']:
            acc[(c['fam']['family'], c['theta'], row['p'])].append(
                (row['active'], row['active_control'], row['dm'] / res['gate'] if np.isfinite(row['dm']) else np.nan))
    hists = {}
    for f in FAMS:
        sts = stats_for(f)
        if sts:
            L = max(len(s['deg_hist']) for s in sts)
            hists[f] = np.mean([np.pad(np.array(s['deg_hist'], float), (0, L - len(s['deg_hist']))) for s in sts], axis=0)
    ps = sorted({p for (_, _, p) in acc})
    thetas = sorted({th for (_, th, _) in acc})
    fams = [f for f in FAMS if any(k[0] == f for k in acc)]
    rows = []
    for f in fams:
        for th in thetas:
            act = lambda p: np.mean([v[0] for v in acc[(f, th, p)]])
            below = [p for p in ps if (f, th, p) in acc and act(p) < 0.5]
            above = [p for p in ps if (f, th, p) in acc and act(p) >= 0.5]
            first_any = [p for p in ps if (f, th, p) in acc and any(v[0] > 0.5 for v in acc[(f, th, p)])]
            pstar = cf.threshold_critical_seed(th, hists[f]) if f in hists else np.nan
            rows.append([NAMES[f], th, max(below) if below else np.nan, min(first_any) if first_any else np.nan,
                         min(above) if above else np.nan, pstar, cf.threshold_critical_seed(th, hists['er'])])
            # the simulated threshold as the midpoint of the bracketing grid points
            if below and above:
                pair('T9', 'cascade threshold p*', f, dict(theta=th), pstar, 0.5 * (max(below) + min(above)))
    table('T9a: cascade threshold p*: simulation (largest p with mean activity below 1/2, smallest p with any '
          'cascade, smallest p with mean activity above 1/2) against the Gleeson-Cahalane recursion with the '
          'family\'s own degree histogram and with the Erdos-Renyi histogram',
          ['family', 'theta', 'p: no cascade (sim)', 'p: first cascade (sim)', 'p: cascades (sim)',
           'p* (G-C, own degrees)', 'p* (G-C, Poisson)'], rows, '%.4f')
    rows = []
    for f in fams:
        for th in thetas:
            for p in ps:
                if (f, th, p) not in acc:
                    continue
                v = np.array(acc[(f, th, p)])
                pred = cf.threshold_cascade(p, th, hists.get(f, hists['er']))
                s = dict(theta=th, p=p)
                pair('T9', 'active fraction', f, s, pred['active'], v[:, 0])
                pair('T9', 'DM/GATE (threshold)', f, s, pred['dm_over_gate'], v[:, 2])
                if p < 0.01 or (v[:, 0].mean() >= 0.999 and pred['active'] >= 0.999):
                    continue
                rows.append([NAMES[f], th, p, v[:, 0].mean(), float((v[:, 0] > 0.5).mean()), pred['active'],
                             np.nanmean(v[:, 2]), pred['dm_over_gate']])
    table('T9b: final active fraction and DM/GATE = 1 - P(control active), rows with p >= 1% where not both '
          'simulation and recursion cascade (recursion with the family\'s own degree histogram)',
          ['family', 'theta', 'p', 'active (sim)', 'cascade freq (sim)', 'active (G-C)', 'DM/GATE (sim)',
           'DM/GATE (G-C)'], rows, '%.4f')


if __name__ == '__main__':
    parts = sys.argv[1:] or ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
    for p in parts:
        globals()[p.lower()]()
    out = os.path.join(ROOT, 'results', 'theory_pairs.jsonl')
    if len(parts) == 9:
        with open(out, 'w') as fh:
            for row in PAIRS:
                fh.write(json.dumps(row) + '\n')
        print(f'\n{len(PAIRS)} comparison rows written to {out}')
