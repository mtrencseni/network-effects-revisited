"""Figures for all sections, drawn from results/*.jsonl. Missing results are skipped.

Usage: figures.py [section ...]   (default: all)
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import DLAM, FIGURES, RESULTS, mean_se
from netfx import theory
from netfx.families import label

plt.rcParams.update({'figure.figsize': (6.4, 4.0), 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'legend.fontsize': 8, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight'})
FAM_COLORS = {'ws': 'C0', 'er': 'C1', 'sbm': 'C2', 'ba': 'C3', 'cm': 'C4', 'gam': 'C5',
              'dws': 'C6', 'dba': 'C7'}


def load(name):
    path = RESULTS / f'{name}.jsonl'
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    good = [(r['config'], r['result']) for r in rows if 'error' not in r['result']]
    bad = len(rows) - len(good)
    if bad:
        print(f'  {name}: {bad} failed runs skipped')
    return good


def fam_key(cfg):
    f = cfg.get('fam') or {k: cfg[k] for k in ('family',) if k in cfg}
    return json.dumps(f, sort_keys=True)


def fam_of(key):
    return json.loads(key)


FAM_ORDER = ['ws', 'er', 'sbm', 'ba', 'cm', 'gam', 'dws', 'dba']


def fam_sorted(keys):
    """Family keys in the paper's order: Watts-Strogatz, Erdős-Rényi, SBM, Barabási-Albert,
    power-law configuration."""
    return sorted(keys, key=lambda k: (FAM_ORDER.index(fam_of(k)['family']), k))


def save(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / f'{name}.png')
    plt.savefig(FIGURES / f'{name}.pdf')
    plt.close('all')
    print('  wrote', name)


def pct_axis(ax):
    """Label a ratio-to-GATE axis in percent (0.1 -> 10%)."""
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def better(ax, up=True, sooner=False):
    """Small arrow just right of the axes, pointing in the direction that is better; with sooner=True,
    a leftward arrow just above the top right corner (earlier on a time axis is better)."""
    if sooner:
        ax.annotate('', xy=(0.72, 1.035), xytext=(0.98, 1.035), xycoords='axes fraction', annotation_clip=False,
                    arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8))
        ax.text(0.85, 1.05, 'better (sooner)', transform=ax.transAxes, fontsize=6, color='0.4',
                ha='center', va='bottom', clip_on=False)
        return
    y0, y1 = (0.62, 0.9) if up else (0.9, 0.62)
    ax.annotate('', xy=(1.025, y1), xytext=(1.025, y0), xycoords='axes fraction', annotation_clip=False,
                arrowprops=dict(arrowstyle='->', color='0.4', lw=0.8))
    ax.text(1.045, (y0 + y1) / 2, 'better', transform=ax.transAxes, rotation=90, fontsize=6,
            color='0.4', ha='left', va='center', clip_on=False)


def errline(ax, x, ys, **kw):
    m = [mean_se(y)[0] for y in ys]
    e = [mean_se(y)[1] for y in ys]
    return ax.errorbar(x, m, yerr=e, capsize=2, marker='o', ms=3, **kw)


_GSTATS = None


def gstat(fam, n=100_000, kbar=20, seeds=range(5)):
    """Graph statistics for the closed forms (results/graph_stats.json, written by
    graph_stats.py), averaged over the stored seeds: kbar, kappa = <k^2>/<k>, b = kappa - 1,
    rho, the pooled degree histogram, triangles per node."""
    global _GSTATS
    if _GSTATS is None:
        path = RESULTS / 'graph_stats.json'
        _GSTATS = json.load(open(path)) if path.exists() else {}
    sts = [_GSTATS.get(json.dumps(dict(fam, n=n, kbar=kbar, seed=s), sort_keys=True)) for s in seeds]
    sts = [s for s in sts if s]
    if not sts:
        return None
    L = max(len(s['deg_hist']) for s in sts)
    hist = sum(np.pad(np.array(s['deg_hist'], float), (0, L - len(s['deg_hist']))) for s in sts)
    kb = np.mean([s['kbar'] for s in sts])
    kap = np.mean([s['k2'] / s['kbar'] for s in sts])
    return dict(kbar=kb, kappa=kap, b=kap - 1, rho=np.mean([s['rho'] for s in sts]), hist=hist,
                tri=np.mean([s.get('triangles_per_node', 0.0) for s in sts]), n=n,
                sqrt_kmax=np.mean([np.sqrt(s['kmax']) for s in sts]))



# ---------------------------------------------------------------------------
# Section 4: what the graph families look like
# ---------------------------------------------------------------------------

def _draw_graph(ax, A, n, title, layout, colors, B=4):
    import networkx as nx
    S = A.to_scipy().tocoo()
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((int(a), int(b)) for a, b in zip(S.row, S.col) if a < b)
    deg_all = np.array([g.degree(v) for v in range(n)])
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    if layout == 'circle':
        pos = nx.circular_layout(sorted(g.nodes))
    elif layout == 'blocks':
        size = n // B
        centers = {0: (-1, 1), 1: (1, 1), 2: (-1, -1), 3: (1, -1)}
        init = {v: np.array(centers[v // size]) + 0.3 * np.random.default_rng(v).standard_normal(2)
                for v in g}
        pos = nx.spring_layout(g, pos=init, seed=1, k=0.25, iterations=60)
    else:
        pos = nx.spring_layout(g, seed=1, k=1.5 / np.sqrt(n), iterations=200)
    deg = np.array([g.degree(v) for v in g])
    if callable(colors):
        colors = [colors(v) for v in g]
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.4, alpha=0.35)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2 + 3 * deg, node_color=colors, linewidths=0)
    ax.set_title(f'{title}\nmax degree {deg_all.max()}', fontsize=9)
    ax.set_axis_off()
    return deg_all



def thumbs():
    """Small example graphs (N = 300, mean degree 4, as in Figure B.1) for Table [families]."""
    from netfx import families
    n, kbar, B = 300, 4, 4
    specs = [(dict(family='ws', p=0.1), 'circle'), (dict(family='er'), 'spring'),
             (dict(family='sbm', B=B, mu=0.05), 'blocks'), (dict(family='ba'), 'spring'),
             (dict(family='cm', gamma=2.5), 'spring')]
    for fam, layout in specs:
        fig, ax = plt.subplots(figsize=(1.6, 1.6))
        A = families.make(dict(fam, n=n, kbar=kbar, seed=3))
        col = (lambda v: f'C{v // (n // B)}') if fam['family'] == 'sbm' else FAM_COLORS[fam['family']]
        _draw_graph(ax, A, n, '', layout, col, B)
        ax.set_title('')
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(FIGURES / f"thumb_{fam['family']}.png", dpi=300)
        plt.close(fig)

def s4():
    from netfx import families
    n, kbar, B = 300, 4, 4
    specs = [(dict(family='ws', p=0.1), 'Watts-Strogatz (p = 0.1)', 'circle'),
             (dict(family='er'), 'Erdős-Rényi', 'spring'),
             (dict(family='sbm', B=B, mu=0.05), 'SBM (4 communities)', 'blocks'),
             (dict(family='ba'), 'Barabási-Albert', 'spring'),
             (dict(family='cm', gamma=2.5), 'power-law (γ = 2.5)', 'spring')]
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.2))
    for ax, (fam, title, layout) in zip(axes[0], specs):
        A = families.make(dict(fam, n=n, kbar=kbar, seed=3))
        col = (lambda v: f'C{v // (n // B)}') if fam['family'] == 'sbm' else FAM_COLORS[fam['family']]
        _draw_graph(ax, A, n, title, layout, col, B)
    # gamma configuration model: same mean degree, increasing spread
    cvs = [0.0, 0.5, 1.0, 1.5]
    degs = {}
    for ax, cv in zip(axes[1][:4], cvs):
        A = families.make(dict(family='gam', cv=cv, n=n, kbar=kbar, seed=3))
        degs[cv] = _draw_graph(ax, A, n, f'gamma configuration, spread {cv}', 'spring',
                               FAM_COLORS['gam'])
    ax = axes[1][4]
    for i, cv in enumerate(cvs):
        A = families.make(dict(family='gam', cv=cv, n=20000, kbar=20, seed=3))
        d = A.degrees()
        bins = np.arange(0, d.max() + 2)
        h, e = np.histogram(d, bins=bins, density=True)
        ax.plot(e[:-1], h, color=plt.cm.viridis(i / 3), label=f'spread {cv}')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('number of friends')
    ax.set_ylabel('fraction of users')
    ax.set_title('gamma configuration: friend counts\n(N = 20,000, average 20)', fontsize=9)
    ax.legend(fontsize=7)
    save('fig4_1_families')


# ---------------------------------------------------------------------------
# Section 5
# ---------------------------------------------------------------------------

def s5():
    rows = load('s5_multiplier')
    if rows:
        by = defaultdict(list)
        for c, r in rows:
            fam = c['fam']['family']
            if fam != 'sbm':   # within 1% of Erdős-Rényi at every point
                by[fam].append(r)
        names = {'ws': ('Watts-Strogatz', FAM_COLORS['ws']),
                 'er': ('Erdős-Rényi (SBM within 1%)', FAM_COLORS['er']),
                 'ba': ('Barabási-Albert', FAM_COLORS['ba']),
                 'cm': ('power-law (γ=2.5)', FAM_COLORS['cm'])}
        fig, ax = plt.subplots()
        for k in [f for f in FAM_ORDER if f in by]:
            rs = by[k]
            lab, col = names[k]
            xs = [p['nu_rho'] for p in rs[0]['points']]
            ys = [[r['points'][i]['multiplier'] for r in rs] for i in range(len(xs))]
            errline(ax, xs, ys, color=col, label=lab)
            mf = [np.mean([r['points'][i]['mean_field'] for r in rs]) for i in range(len(xs))]
            ax.plot(xs, mf, '--', color=col, lw=0.9)
        ax.plot([], [], 'k-', label='simulated (exact)')
        ax.plot([], [], 'k--', lw=0.9, label='predicted from average friend count, 1/(1 − ν k̄)')
        ax.set_yscale('log')
        ax.set_yticks([1, 10, 100])
        ax.set_yticklabels(['1', '10', '100'])
        ax.set_xlim(0.05, 1.1)
        ax.set_ylim(0.85, 160)
        ax.axvline(1.0, color='k', lw=1.0, ls=':')
        ax.text(1.012, 1.0, 'true stability limit: content explodes', rotation=90,
                ha='left', va='bottom', fontsize=7)
        ax.annotate('on graphs with hubs, the average-friend\nestimate stays near 1.5 and misses\nthe explosion',
                    xy=(0.94, 1.5), xytext=(0.47, 22), fontsize=7.5,
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='0.25'))
        ax.set_xlabel('peer-effect strength, νρ(A)  (0 = none, 1 = stability limit)')
        ax.set_ylabel('social multiplier, GATE / Δλ')
        ax.legend(fontsize=7, loc='upper left')
        save('fig5_1_multiplier')
        # Table 5.1: rho and its degree approximation (graph_stats.json, seeds 0 to 4), Perron
        # weight on the top 1% and stability at nu kbar = 0.5 (s5_multiplier, the same seeds)
        byf = defaultdict(list)
        for c, r in rows:
            if c['seed'] <= 4:
                byf[fam_key(c)].append(r)
        with open(FIGURES / 'tab5_1_spectra.md', 'w') as f:
            f.write('| family | ρ(A) | ρ/k̄ | κ = ⟨k²⟩/⟨k⟩ | √k_max | max(κ, √k_max) | '
                    'top 1% Perron weight | stable at ν k̄ = 0.5 |\n|---|---|---|---|---|---|---|---|\n')
            for k in fam_sorted(byf):
                rs = byf[k]
                st = gstat(fam_of(k))
                if st is None:
                    continue
                sq = st['sqrt_kmax']
                f.write(f"| {label(fam_of(k))} | {st['rho']:.2f} | {st['rho'] / st['kbar']:.2f} | "
                        f"{st['kappa']:.2f} | {sq:.2f} | {max(st['kappa'], sq):.2f} | "
                        f"{np.mean([r['top1'] for r in rs]):.3f} | "
                        f"{'yes' if all(r['half_kbar']['stable'] for r in rs) else 'no'} |\n")

    rows = load('s5_hetero')
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        nks = sorted({c['nu_kbar'] for c, _ in rows})
        cvs = sorted({c['cv'] for c, _ in rows})
        for i, nk in enumerate(nks):
            xs, ys, unst = [], [], []
            for cv in cvs:
                rs = [r for c, r in rows if c['cv'] == cv and c['nu_kbar'] == nk]
                st = [r['multiplier'] for r in rs if r['stable']]
                if len(st) == len(rs):
                    xs.append(cv)
                    ys.append(st)
                else:
                    unst.append(cv)
            errline(ax, xs, ys, color=f'C{i}', label=f'ν k̄ = {nk}')
            ax.axhline(1 / (1 - nk), color=f'C{i}', ls='--', lw=0.8)
            # closed form with gamma degrees: 1 + nu kbar / (1 - nu kbar (1 + cv^2)), up to cv*
            cstar = theory.cv_critical(nk)
            cg = np.linspace(0, min(cstar - 0.02, max(cvs)), 80)
            ax.plot(cg, 1 + nk / (1 - nk * (1 + cg ** 2)), '-.', color=f'C{i}', lw=0.9)
            if unst:
                # shade where nu rho(A) >= 1: interpolate rho / kbar against the spread
                rk = [np.mean([r['rho'] / r['kbar'] for c, r in rows
                               if c['cv'] == cv and c['nu_kbar'] == nk]) for cv in cvs]
                start = float(np.interp(1 / nk, rk, cvs))
                ax.axvspan(start, max(cvs) + 0.1, color=f'C{i}', alpha=0.10, lw=0)
                ax.axvline(start, color=f'C{i}', lw=0.9, ls=':')
                ax.text(start + 0.03, 0.97 - 0.09 * i, f'ν k̄ = {nk}: νρ(A) ≥ 1,\nunstable from here on',
                        color=f'C{i}', fontsize=7, va='top', transform=ax.get_xaxis_transform())
        base = load('s5_hetero_base')
        if base:
            marks = {'ws': ('s', 'WS'), 'er': ('o', 'ER'), 'sbm': ('D', 'SBM'), 'ba': ('^', 'BA'),
                     'cm': ('v', 'PL')}
            byb = defaultdict(list)
            for c, r in base:
                byb[c['fam']['family']].append(r)
            for fam, rs in byb.items():
                m, short = marks[fam]
                cv = np.mean([r['cv'] for r in rs])
                for i, nk in enumerate(nks):
                    pts = [p for r in rs for p in r['points'] if p['nu_kbar'] == nk]
                    if all(p['stable'] for p in pts):
                        y = np.mean([p['multiplier'] for p in pts])
                        ax.plot(cv, y, m, ms=7, mfc='white', mec='k', mew=1.0, zorder=5)
                        dx = {'er': 0.03, 'sbm': 0.03, 'ws': -0.03}.get(fam, 0.04)
                        dy = {'er': 1.06, 'sbm': 0.94}.get(fam, 1.0)
                        ax.text(cv + dx, y * dy, short, fontsize=7,
                                ha='right' if fam == 'ws' else 'left', va='center', zorder=6)
            ax.plot([], [], 'o', mfc='white', mec='k', label='the five base families at their own spread')
        ax.set_yscale('log')
        ax.set_yticks([1, 1.5, 2, 3, 5, 10, 20])
        ax.set_yticklabels(['1', '1.5', '2', '3', '5', '10', '20'])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(-0.08, max(cvs) + 0.1)
        ax.set_xlabel('spread of friend counts (coefficient of variation; 0 = everyone has k̄ friends)')
        ax.set_ylabel('social multiplier, GATE / Δλ')
        ax.plot([], [], 'k--', lw=0.8, label='predicted from average friend count, 1/(1 − ν k̄) (ignores spread)')
        ax.plot([], [], 'k-.', lw=0.9, label='closed form with the spread, 1 + ν k̄ / (1 − ν k̄ (1 + cv²))')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
        save('fig5_2_heterogeneity')

    rows = load('s5_hops')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8), sharey=True)
        for ax, x in zip(axes, sorted({c['nu_rho'] for c, _ in rows})):
            by = defaultdict(list)
            for c, r in rows:
                if c['nu_rho'] == x:
                    by[fam_key(c)].append(r)
            for k, rs in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
                f = fam_of(k)
                hs = sorted({b['h'] for r in rs for b in r['by_hop'] if b['h'] < 6})
                ys = [[b['lift'] / DLAM for r in rs for b in r['by_hop'] if b['h'] == h] for h in hs]
                errline(ax, hs, ys, color=FAM_COLORS[f['family']], label=label(f))
                st = gstat(f)
                if st:
                    hp = [h for h in hs if h <= 2]   # beyond hop 2 the tree is outside its range (Section 5.3)
                    pred, _ = theory.hop_lift(x / st['rho'], st['kbar'], st['b'], 0.02, hops=hp,
                                              deg_hist=st['hist'])
                    ax.plot(hp, pred, '--', color=FAM_COLORS[f['family']], lw=0.8)
            ax.set_yscale('log')
            ax.set_xlabel('hops from the nearest treated node')
            ax.set_title(f'$\\nu\\rho$ = {x}')
        axes[0].set_ylabel('lift of control users, lift / Δλ')
        axes[0].legend(handles=axes[0].get_legend_handles_labels()[0]
                       + _styles(('simulated', '-', ''), ('tree closed form', '--', '', 0.8)))
        save('fig5_3_spillover_hops')

    rows = load('s5_attenuation')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            by[fam_key(c)][c['nu_rho']].append(r)
        for k, d in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
            f = fam_of(k)
            xs = sorted(d)
            errline(axes[0], xs, [[r['ratio'] for r in d[x]] for x in xs],
                    color=FAM_COLORS[f['family']], label=label(f))
            st = gstat(f)
            if st:
                xg = np.linspace(min(xs), max(xs), 60)
                pred = [theory.attenuation(x / st['rho'], st['rho'], st['kbar'], st['b'], st['n'],
                                           deg_hist=st['hist'], triangles_per_node=st['tri']) for x in xg]
                axes[0].plot(xg, pred, '--', color=FAM_COLORS[f['family']], lw=0.8)
            axes[1].plot(xs, [np.mean([r['lift_treated'] / r['gate'] for r in d[x]]) for x in xs],
                         '-o', ms=3, color=FAM_COLORS[f['family']])
            axes[1].plot(xs, [np.mean([r['spill_control'] / r['gate'] for r in d[x]]) for x in xs],
                         ':s', ms=3, color=FAM_COLORS[f['family']])
        axes[0].set_xlabel(r'$\nu\rho(A)$')
        axes[0].set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
        axes[0].legend(handles=axes[0].get_legend_handles_labels()[0]
                       + _styles(('simulated', '-', ''), ('closed form', '--', '', 0.8)))
        axes[1].set_xlabel(r'$\nu\rho(A)$')
        axes[1].set_ylabel('share of GATE')
        axes[1].legend(handles=_styles(('lift of treated', '-', 'o'), ('spillover to control', ':', 's')))
        save('fig5_4_attenuation')

    rows = load('s5_size')
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            by[c['family']][c['n']].append(r)
        for fam, d in by.items():
            ns = sorted(d)
            col = FAM_COLORS[fam]
            errline(axes[0], ns, [[r['ratio'] for r in d[n]] for n in ns], color=col,
                    label=label({'family': fam}))
            errline(axes[1], ns, [[r['rho'] / r['kbar'] for r in d[n]] for n in ns], color=col)
            errline(axes[2], ns, [[r['top1'] for r in d[n]] for n in ns], color=col)
        for ax, yl in zip(axes, ['attenuation ratio (νρ = 0.5)', 'ρ(A) / k̄',
                                 'eigenvector weight on top 1% of nodes']):
            ax.set_xscale('log')
            ax.set_xlabel('number of nodes n')
            ax.set_ylabel(yl)
        axes[1].set_yscale('log')
        axes[0].legend()
        save('fig5_5_size')
    rows = load('s5_size_fixednu')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            by[c['family']][c['n']].append(r)
        for fam, d in by.items():
            ns = sorted(d)
            col = FAM_COLORS[fam]
            errline(axes[0], ns, [[r['nu_rho'] for r in d[n]] for n in ns], color=col,
                    label=label({'family': fam}))
            st = [n for n in ns if all(r['stable'] for r in d[n])]
            errline(axes[1], st, [[r['ratio'] for r in d[n]] for n in st], color=col)
        axes[0].axhline(1, color='k', ls='--', lw=0.8)
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlabel('number of nodes n')
        axes[0].set_ylabel('peer-effect strength, νρ(A)')
        axes[1].set_ylabel('E[DM] / GATE (stable graphs only)')
        axes[0].legend()
        save('fig5_5b_size_fixed_nu')

    rows = load('s5_clustering')
    if rows:
        fig, ax = plt.subplots()
        marks = {'bernoulli': 'o', 'metis': 's', 'contiguous': '^', 'blocks': 'D'}
        mark_name = {'bernoulli': 'user-level split', 'metis': 'cluster split (METIS)',
                     'contiguous': 'cluster split (contiguous on the ring)', 'blocks': 'cluster split (true SBM blocks)'}
        for c, r in rows:
            f = c['fam']
            ax.plot(r['same_arm'], r['ratio'], marks[c['design']], ms=4,
                    color=FAM_COLORS[f['family']], alpha=0.7)
        for fam_name in {c['fam']['family'] for c, _ in rows}:
            ax.plot([], [], 'o', color=FAM_COLORS[fam_name], label=label({'family': fam_name}))
        for d, m in marks.items():
            ax.plot([], [], m, color='gray', label=mark_name[d])
        # SBM sweep over the between-block share mu (results/s5_clustering_mu.jsonl), nu rho = 0.5
        mu_rows = load('s5_clustering_mu')
        for c, r in mu_rows:
            if c['nu_rho'] == 0.5 and c['design'] in ('blocks', 'metis'):
                ax.plot(r['same_arm'], r['ratio'], 'x' if c['design'] == 'blocks' else '+', ms=6,
                        color=FAM_COLORS['sbm'], alpha=0.9)
        if mu_rows:
            ax.plot([], [], 'x', color=FAM_COLORS['sbm'], label='SBM, μ = 0.2 to 0.7, true blocks')
            ax.plot([], [], '+', color=FAM_COLORS['sbm'], label='SBM, μ = 0.2 to 0.7, METIS 1,000')
        ax.set_xlabel('share of neighbors in the same arm')
        ax.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
        # the points fill the band from (0.5, 0.5) up to (1, 0.9); the lower right is empty
        ax.legend(ncol=2, fontsize=7, loc='lower right', framealpha=0.9)
        save('fig5_6_clustering')

    rows = load('s5_saturation')
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 3.8))
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            if c['nu_rho'] == 0.5:
                by[(fam_key(c), c['design'])][c['p']].append(r)
        for (k, d), dd in sorted(by.items()):
            f = fam_of(k)
            ps = sorted(dd)
            col = FAM_COLORS[f['family']]
            if d == 'random':
                errline(axes[0], ps, [[r['dm'] / r['gate'] for r in dd[p]] for p in ps], color=col,
                        label=label(f))
                errline(axes[0], ps, [[r['ratio_lift'] / r['gate_rel'] for r in dd[p]] for p in ps],
                        color=col, ls=':')
            else:
                ls = '--' if d == 'degree' else ':'
                axes[1].plot(ps, [np.mean([r['dm'] / r['gate'] for r in dd[p]]) for p in ps], ls,
                             marker='o', ms=3, color=col)
                axes[2].plot([np.mean([r['centrality_share'] for r in dd[p]]) for p in ps],
                             [np.mean([r['beta_lift'] / r['gate'] for r in dd[p]]) for p in ps], ls,
                             marker='o', ms=3, color=col)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('treated fraction (random assignment)')
        axes[0].set_ylabel('measured launch effect ratio,\nDM / GATE')
        axes[0].legend(handles=_styles(('DM of one assignment', '-', ''), ('relative lift', ':', '')), fontsize=7)
        # the left panel is full of curves; the family legend goes into the empty upper right
        # of the middle panel
        axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc='upper right')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('treated fraction (targeted)')
        axes[1].set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
        axes[2].legend(handles=_styles(('top degree', '--', 'o'), ('top centrality', ':', 'o')),
                       fontsize=7)
        axes[2].set_xlabel('treated share of untreated production')
        axes[2].set_ylabel('lift of the targeted group, lift / GATE')
        fig.tight_layout()
        save('fig5_7_saturation')


# ---------------------------------------------------------------------------
# Section 6
# ---------------------------------------------------------------------------

def s6():
    rows = load('s6_paths')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        seen = set()
        for c, r in sorted(rows, key=lambda cr: cr[0].get('nu_rho', 0)):
            if c['seed'] != 0 or c['fam']['family'] not in ('ws', 'ba'):
                continue
            t = np.arange(1, len(r['gate']) + 1)
            col = {0.5: 'C0', 0.8: 'C1', 0.9: 'C2', 0.95: 'C3', 0.99: 'C4'}[c['nu_rho']]
            ls = '-' if c['fam']['family'] == 'ws' else '--'
            lab = None
            if ls == '-' and c['nu_rho'] not in seen:   # one legend entry per value
                seen.add(c['nu_rho'])
                lab = f"νρ={c['nu_rho']}"
            axes[0].plot(t, np.array(r['gate']) / r['gate_inf'], ls, color=col, label=lab)
            axes[1].plot(t, np.array(r['dm']) / r['dm_inf'], ls, color=col)
        for ax, yl in zip(axes, ['GATE(t) / GATE(∞)', 'DM(t) / DM(∞)']):
            ax.set_xscale('log')
            ax.set_xlabel('periods since treatment started')
            ax.set_ylabel(yl)
        axes[0].legend(title='solid: WS; dashed: BA')
        save('fig6_1_paths')
        # time to 95% of the GATE against the prediction log(0.05) / log(nu rho)
        fig, ax = plt.subplots()
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            if 'nu_rho' not in c:
                continue
            g = np.array(r['gate']) / r['gate_inf']
            hit = np.flatnonzero(g >= 0.95)
            by[fam_key(c)][c['nu_rho']].append(hit[0] + 1 if hit.size else np.nan)
        for k, d in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
            f = fam_of(k)
            xs = sorted(d)
            errline(ax, xs, [d[x] for x in xs], color=FAM_COLORS[f['family']], label=label(f))
        xs = np.linspace(0.5, 0.99, 50)
        ax.plot(xs, np.log(0.05) / np.log(xs), 'k--', lw=0.8, label='log 0.05 / log νρ')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\nu\rho(A)$')
        ax.set_ylabel('periods until GATE(t) reaches 95%')
        ax.legend()
        save('fig6_2_time_to_95')
        # directed follower graph: finite multiplier for any nu, convergence in finite steps
        fig, ax = plt.subplots()
        for c, r in rows:
            if c['fam']['family'] != 'dba' or c['seed'] != 0:
                continue
            t = np.arange(1, len(r['gate']) + 1)
            ax.plot(t, np.array(r['gate']) / DLAM, label=f"νk̄ = {c['nu_kbar']}")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('periods since treatment started')
        ax.set_ylabel('launch effect, GATE(t) / Δλ')
        ax.set_title('directed follower graph (ρ(A) = 0)')
        ax.legend()
        save('fig6_3_directed')

    rows = load('s6_peeking2')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        by = defaultdict(list)
        for c, r in rows:
            by[(fam_key(c), c['nu_rho'], c['effect'])].append(r)
        fams = fam_sorted({k[0] for k in by})
        for k in fams:
            f = fam_of(k)
            col = FAM_COLORS[f['family']]
            xs = sorted({x for kk, x, e in by if kk == k})
            for e, ax in ((0.0, axes[0]), (0.002, axes[1])):
                seq = [np.mean([run['stop'] is not None for r in by[(k, x, e)] for run in r['runs']])
                       for x in xs]
                fix = [np.mean([bool(run['fixed_rej']) for r in by[(k, x, e)] for run in r['runs']])
                       for x in xs]
                ax.plot(xs, seq, '-o', ms=3, color=col, label=label(f) if e == 0.0 else None)
                ax.plot(xs, fix, ':s', ms=3, color=col)
        axes[0].axhline(0.05, color='k', ls='--', lw=0.8)
        axes[0].set_title('no effect: rejection rate')
        axes[1].set_title('small effect (Δλ = 0.002): rejection rate')
        for ax in axes:
            ax.set_xlabel(r'$\nu\rho(A)$')
        axes[0].set_ylabel('rejection rate')
        axes[1].legend(handles=axes[0].get_legend_handles_labels()[0]
                       + _styles(('daily peeking, 60 days', '-', 'o'), ('one test on day 14', ':', 's')),
                       fontsize=7, loc='upper right')
        save('fig6_4_peeking')


# ---------------------------------------------------------------------------
# Section 7
# ---------------------------------------------------------------------------

def s7():
    rows = load('s7_power3') or load('s7_power')
    if rows:
        rows = [(c, r) for c, r in rows if c.get('effect', DLAM) > 0] + \
               [(c, r) for c, r in rows if c.get('effect', DLAM) == 0]
        nulls = [(c, r) for c, r in rows if c.get('effect', DLAM) == 0]
        rows = [(c, r) for c, r in rows if c.get('effect', DLAM) > 0]
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            by[fam_key(c)][c['nu_rho']].append(r)
        for k, d in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
            f = fam_of(k)
            xs = sorted(d)
            col = FAM_COLORS[f['family']]
            inv = [1 / (1 - x) for x in xs]
            errline(axes[0], inv, [[r['gate'] / DLAM for r in d[x]] for x in xs], color=col,
                    label=label(f))
            errline(axes[0], inv, [[r.get('dm_lift_mean', r['dm_mean']) / DLAM for r in d[x]]
                                   for x in xs], color=col, ls=':')
            errline(axes[1], inv, [[r['power'] for r in d[x]] for x in xs], color=col)
            if 'power_cuped' in d[xs[0]][0]:
                errline(axes[1], inv, [[r['power_cuped'] for r in d[x]] for x in xs], color=col,
                        ls=':')
            if 'power_cuped_drift' in d[xs[0]][0]:
                errline(axes[1], inv, [[r['power_cuped_drift'] for r in d[x]] for x in xs],
                        color=col, ls='--')
            errline(axes[2], inv, [[r['obs_sd'] / r['naive_se'] for r in d[x]] for x in xs], color=col)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('effect / Δλ')
        axes[0].legend(handles=axes[0].get_legend_handles_labels()[0]
                       + _styles(('GATE', '-', ''), ('E[DM]', ':', '')))
        axes[1].set_ylabel('power')
        axes[1].legend(handles=_styles(('plain DM', '-', ''), ('CUPED', ':', ''),
                                       ('CUPED, drifting baseline', '--', '')))
        axes[2].set_ylabel('true SD of DM / naive standard error')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlabel(r'$1/(1-\nu\rho)$')
        save('fig7_1_critical')
        if nulls:
            fig, ax = plt.subplots()
            by = defaultdict(lambda: defaultdict(list))
            for c, r in nulls:
                by[fam_key(c)][c['nu_rho']].append(r)
            for k, d in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
                f = fam_of(k)
                xs = sorted(d)
                inv = [1 / (1 - x) for x in xs]
                col = FAM_COLORS[f['family']]
                errline(ax, inv, [[r['power'] for r in d[x]] for x in xs], color=col, label=label(f))
                errline(ax, inv, [[r['power_cuped'] for r in d[x]] for x in xs], color=col, ls=':')
            ax.axhline(0.05, color='k', ls='--', lw=0.8)
            ax.set_xscale('log')
            ax.set_xlabel(r'$1/(1-\nu\rho)$')
            ax.set_ylabel('rejection rate with no effect')
            ax.legend(handles=ax.get_legend_handles_labels()[0]
                      + _styles(('plain DM', '-', ''), ('CUPED', ':', '')))
            save('fig7_1b_null_rejection')

    rows = load('s7_noise')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, smp in zip(axes, ['uniform', 'poisson']):
            by = defaultdict(lambda: defaultdict(list))
            for c, r in rows:
                if c['sampler'] == smp:
                    by[fam_key(c)][c['nu_rho']].append(r)
            any_bad = False
            for k, d in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
                f = fam_of(k)
                xs = sorted(d)
                ys = [[r['var_hubs'] if not r['diverged'] else np.inf for r in d[x]] for x in xs]
                ok = [x for x, y in zip(xs, ys) if np.all(np.isfinite(y))]
                errline(ax, ok, [y for x, y in zip(xs, ys) if x in ok],
                        color=FAM_COLORS[f['family']], label=label(f))
                bad = [x for x in xs if x not in ok]
                if bad:
                    any_bad = True
                    ax.plot(bad, [1e6] * len(bad), 'x', color=FAM_COLORS[f['family']])
            ax.set_yscale('log')
            ax.set_xlabel(r'$\nu\rho(A)$')
            # the sweep stops at nu rho = 0.99, where every run converges; the marker note
            # appears only if a run diverged
            ax.set_title(f'{smp} noise' + (' (x: diverged)' if any_bad else ''))
        axes[0].set_ylabel('stationary variance of production, top 0.1% degree nodes')
        axes[0].legend()
        save('fig7_2_noise')


# ---------------------------------------------------------------------------
# Section 8
# ---------------------------------------------------------------------------

def s8():
    rows = load('s8_clusters3') or load('s8_clusters2')
    if rows:
        fams = fam_sorted({fam_key(c) for c, _ in rows})
        fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.8), sharey=True)
        for ax, k in zip(np.atleast_1d(axes), fams):
            for i, x in enumerate(sorted({c['nu_rho'] for c, _ in rows})):
                d = defaultdict(list)
                for c, r in rows:
                    if fam_key(c) == k and c['nu_rho'] == x:
                        d[c['size']].append(r)
                sz = sorted(d)
                ax.plot(sz, [np.mean([r['rmse'] / r['gate'] for r in d[s]]) for s in sz],
                        '-o', ms=3, color=f'C{i}', label=f'RMSE DM, νρ={x}')
                ax.plot(sz, [np.mean([abs(r['mean'] - r['gate']) / r['gate'] for r in d[s]]) for s in sz],
                        ':s', ms=3, color=f'C{i}', label=f'|bias| DM, νρ={x}')
                if 'hajek_rmse' in d[sz[0]][0]:
                    hz = [s_ for s_ in sz if np.isfinite(np.mean([r['hajek_rmse'] for r in d[s_]]))]
                    ax.plot(hz, [np.nanmean([r['hajek_rmse'] / r['gate'] for r in d[s_]]) for s_ in hz],
                            '--^', ms=3, color=f'C{i}', alpha=0.6,
                            label=f'RMSE Hájek exposure, νρ={x}')
                    ax.plot(hz, [np.nanmean([abs(r['hajek_mean'] - r['gate']) / r['gate'] for r in d[s_]])
                                 for s_ in hz], '-.v', ms=3, color=f'C{i}', alpha=0.6,
                            label=f'|bias| Hájek exposure, νρ={x}')
            ax.set_xscale('log')
            ax.set_xlabel('cluster size (1 = user-level split)')
            ax.set_title(label(fam_of(k)))
        np.atleast_1d(axes)[0].set_ylabel('error and bias,\n% of the launch effect')
        pct_axis(np.atleast_1d(axes)[0])
        better(np.atleast_1d(axes).ravel()[-1], up=False)
        np.atleast_1d(axes)[0].set_ylim(0, 1.2)
        np.atleast_1d(axes)[0].legend(fontsize=6)
        save('fig8_1_clusters')

    rows = load('s8_wronggraph')
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            by[fam_key(c)][c['q']].append(r)
        for k, d in sorted(by.items(), key=lambda kv: fam_sorted(by).index(kv[0])):
            f = fam_of(k)
            qs = sorted(d)
            errline(ax, qs, [[r['rmse'] / r['gate'] for r in d[q]] for q in qs],
                    color=FAM_COLORS[f['family']], label=label(f))
        bern = load('s8_clusters3') or load('s8_clusters2')
        if bern:
            for k in by:
                v = [r['rmse'] / r['gate'] for c, r in bern
                     if fam_key(c) == k and c['size'] == 1 and c['nu_rho'] == 0.6]
                if v:
                    ax.axhline(np.mean(v), color=FAM_COLORS[fam_of(k)['family']], ls='--', lw=0.8)
        ax.set_xlabel('share of design-graph edges that are wrong, ε')
        ax.set_ylabel('error of the estimate, % of the launch effect')
        pct_axis(ax)
        better(ax, up=False)
        ax.legend(handles=ax.get_legend_handles_labels()[0]
                  + _styles(('wrong design graph', '-', ''), ('user-level split', '--', '', 0.8)))
        save('fig8_2_wrong_graph')

    rows = load('s8_extrap3')
    if rows:
        ests = ['dm', 'regression', 'regression2', 'pre_post', 'two_fraction', 'plug_in']
        enames = {'dm': 'DM', 'regression': 'neighborhood regression',
                  'regression2': 'neighborhood regression, two hops', 'pre_post': 'historical baseline',
                  'two_fraction': 'two-fraction slope', 'plug_in': 'plug-in'}
        cases = [('linear', 0.0, 0.0), ('linear', 0.2, 0.0), ('linear', 0.0, 0.01)]
        titles = ['nothing changes', '20% of friendships carry no influence', 'seasonal 1% drift']
        fams = fam_sorted({fam_key(c) for c, _ in rows})
        fig, axes = plt.subplots(len(fams), 3, figsize=(13, 3.2 * len(fams)), sharey=True)
        for row, k in zip(axes, fams):
            for ax, (m, q, dr), t in zip(row, cases, titles):
                for i, e in enumerate(ests):
                    d = defaultdict(list)
                    for c, r in rows:
                        if (fam_key(c) == k and (c['model'], c['q'], c['drift']) == (m, q, dr)
                                and np.isfinite(r[e]['rmse'])):
                            d[c['nu_rho']].append(r[e]['rmse'] / r['gate'])
                    xs = sorted(d)
                    errline(ax, xs, [d[x] for x in xs], color=f'C{i}', label=enames[e])
                ax.set_yscale('log')
                ax.set_title(f'{label(fam_of(k))}: {t}', fontsize=8)
            row[0].set_ylabel('error of the estimate,\n% of the launch effect')
            pct_axis(row[0])
            better(row[-1], up=False)
        for ax in axes[-1]:
            ax.set_xlabel('peer-effect strength, νρ(A)')
        axes[0][0].legend(fontsize=7)
        save('fig8_3_extrapolation')

    rows = load('s8_cuped2') or load('s8_cuped')
    if rows:
        fig, ax = plt.subplots()
        lab = []
        xs = []
        for i, (c, r) in enumerate(sorted(rows, key=lambda cr: (fam_key(cr[0]), cr[0]['size']))):
            pass
        by = defaultdict(list)
        for c, r in rows:
            by[(label(c['fam']), c['size'])].append(r)
        keys = sorted(by)
        x = np.arange(len(keys))
        fig.set_size_inches(10, 4)
        ax.bar(x - 0.27, [np.mean([r['dm']['mean'] / r['gate'] for r in by[k]]) for k in keys],
               0.18, label='DM: mean / GATE', color='C7')
        ax.bar(x - 0.09, [np.mean([r['cuped']['sd'] / r['dm']['sd'] for r in by[k]]) for k in keys],
               0.18, label='CUPED: SD / SD(DM)', color='C0')
        ax.bar(x + 0.09, [np.mean([r['cuped_exposure']['mean'] / r['gate'] for r in by[k]]) for k in keys],
               0.18, label='+ exposure, treatment coefficient: mean / GATE', color='C1')
        if 'cuped_exposure_contrast' in by[keys[0]][0]:
            ax.bar(x + 0.27, [np.mean([r['cuped_exposure_contrast']['mean'] / r['gate'] for r in by[k]])
                              for k in keys], 0.18,
                   label='+ exposure, all-treated contrast: mean / GATE', color='C3')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k[0]}\n{"user-level split" if k[1] == 1 else f"clusters of {k[1]}"}' for k in keys],
                           rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('ratio')
        ax.legend()
        save('fig8_4_cuped')

    rows = load('s8_bayes')
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.8))
        for c, r in rows:
            if c['effect'] and c['seed'] == 0 and c['size'] == 1000:
                axes[0].plot(r['p_naive'], r['p_bayes_node'], '.', ms=3, color='C0', alpha=0.6)
                axes[0].plot(r['p_robust'], r['p_bayes_cluster'], '.', ms=3, color='C1', alpha=0.6)
                axes[0].plot(r['p_naive'], r['p_bayes'], '.', ms=3, color='C2', alpha=0.6)
        axes[0].plot([0, 1], [1, 0], 'k--', lw=0.8)
        axes[0].plot([], [], 'o', color='C0', label='nodes as units (normal model)')
        axes[0].plot([], [], 'o', color='C1', label='clusters as units (normal model)')
        axes[0].plot([], [], 'o', color='C2', label='Gamma-Poisson posterior vs naive p')
        axes[0].set_xlabel('one-sided p-value')
        axes[0].set_ylabel('posterior P(effect > 0)')
        axes[0].set_title('clusters of 1000, small effect')
        axes[0].legend()
        by = defaultdict(list)
        for c, r in rows:
            if not c['effect']:
                by[(c['size'], 'naive p (nodes)')] += r['p_naive']
                by[(c['size'], 'robust p (clusters)')] += r['p_robust']
                by[(c['size'], 'Bayes, normal (nodes)')] += [1 - p for p in r['p_bayes_node']]
                by[(c['size'], 'Bayes, normal (clusters)')] += [1 - p for p in r['p_bayes_cluster']]
                by[(c['size'], 'Bayes, Gamma-Poisson')] += [1 - p for p in r['p_bayes']]
        sizes = sorted({k[0] for k in by})
        for i, m in enumerate(['naive p (nodes)', 'robust p (clusters)', 'Bayes, normal (nodes)',
                               'Bayes, normal (clusters)', 'Bayes, Gamma-Poisson']):
            axes[1].plot(sizes, [np.mean(np.array(by[(s_, m)]) < 0.05) for s_ in sizes],
                         '-o' if i % 2 == 0 else '--s', color=f'C{i}', label=m, ms=4)
        axes[1].axhline(0.05, color='k', ls='--', lw=0.8)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('cluster size (1 = user-level split)')
        axes[1].set_ylabel('false positive rate, one-sided, no effect')
        axes[1].legend()
        save('fig8_5_bayes')


# ---------------------------------------------------------------------------
# Sections 9 and 10
# ---------------------------------------------------------------------------

def s9():
    rows = load('s9_edges2') or load('s9_edges')
    if rows:
        fams = fam_sorted({fam_key(c) for c, _ in rows})
        rule_name = {'fof': 'new friends: friends of friends', 'random': 'new friends: random users'}
        design_name = {'bernoulli': 'user-level split', 'metis': 'cluster split (METIS)'}
        style = {('fof', 'metis'): ('C2', '-'), ('random', 'metis'): ('C3', '-'),
                 ('fof', 'bernoulli'): ('C2', ':'), ('random', 'bernoulli'): ('C3', ':')}
        fig, axes = plt.subplots(2, len(fams), figsize=(4.5 * len(fams), 7.4), sharey=True)
        for row, boost in zip(axes, [True, False]):
            for ax, k in zip(row, fams):
                by = defaultdict(lambda: defaultdict(list))
                for c, r in rows:
                    if fam_key(c) == k and c['boost'] == boost and np.isfinite(r['gate']) and r['gate'] > 0:
                        by[(c['rule'], c['design'])][c['r']].append(r['dm'] / r['gate'])
                for kk in [('fof', 'metis'), ('random', 'metis'), ('fof', 'bernoulli'), ('random', 'bernoulli')]:
                    if kk not in by:
                        continue
                    d = by[kk]
                    rs = sorted(d)
                    col, ls = style[kk]
                    errline(ax, rs, [d[x] for x in rs], color=col, ls=ls,
                            label=f'{design_name[kk[1]]}, {rule_name[kk[0]]}')
                ax.set_title(label(fam_of(k)) + (': boost and new friends' if boost else ': new friends only'),
                             fontsize=9)
                if not boost:
                    ax.set_xlabel('new friends per treated user')
            row[0].set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
        fig.subplots_adjust(hspace=0.25)
        axes[0][0].legend(fontsize=7)
        save('fig9_1_new_edges')

    rows = load('s9_users')
    if rows:
        est_name = {'inherit_total': ("new users' posts added to the treated arm's total per original user", 'C0'),
                    'exclude': ('new users left out', 'C7')}
        q_name = {2: ('-', 'new user befriends inviter + 2 of its friends'),
                  0: ('--', 'new user befriends inviter only')}
        A_SHOWN = 0.5
        fams = fam_sorted({fam_key(c) for c, _ in rows})
        fig, grid = plt.subplots(2, 3, figsize=(19.2, 7.6), sharey=True)
        axes = list(grid.flat)
        for ax, fk in zip(axes, fams):
            f = fam_of(fk)
            st = gstat(f)
            for q, (ls, qlab) in q_name.items():
                for k, (elab, col) in est_name.items():
                    d = defaultdict(list)
                    for c, r in rows:
                        if (fam_key(c) == fk and c['a'] == A_SHOWN and c['q'] == q
                                and not r.get('unstable')):
                            d[c['r']].append(r[k] / r['gate_total'])
                    rs = sorted(d)
                    errline(ax, rs, [d[x] for x in rs], color=col, ls=ls)
                    if st:
                        nu = 0.5 / st['rho']
                        pred = []
                        for x in rs:
                            e = theory.users_effects(nu, st['rho'], x, q, A_SHOWN, kbar=st['kbar'])
                            pred.append(e[k] / e['gate_total'])
                        ax.plot(rs, pred, 'x', color='k', ms=5, mew=0.9, zorder=5)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel('new users invited per treated user')
            ax.set_title(label(f), fontsize=9)
        for row in grid:
            row[0].set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
        better(axes[2])
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=col, label=elab) for elab, col in est_name.values()]
        handles += [Line2D([], [], color='k', ls=ls, label=qlab) for ls, qlab in q_name.values()]
        handles += [Line2D([], [], color='k', ls='', marker='x', label='closed form (three-type model)')]
        axes[-1].axis('off')
        axes[-1].legend(handles=handles, fontsize=9, loc='center',
                        title=f'new users post {A_SHOWN:g}× as much as existing users; νρ = 0.5', title_fontsize=9)
        save('fig9_2_new_users')


def s10():
    rows = load('s10_nutreat')
    if rows:
        fams = fam_sorted({fam_key(c) for c, _ in rows})
        fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.8), sharey=True)
        any_partial = False
        for ax, k in zip(np.atleast_1d(axes), fams):
            f = fam_of(k)
            st = gstat(f)
            by = defaultdict(lambda: defaultdict(list))
            for c, r in rows:
                if fam_key(c) != k:
                    continue
                for row in r['rows']:
                    by[c['delta']][row['p']].append(np.nan if row['unstable'] else row['dm'])
            for i, (dlt, d) in enumerate(sorted(by.items())):
                if dlt == 0:
                    continue
                ps = [p for p in sorted(d) if np.all(np.isfinite(d[p]))]
                # settings at which some, but not all, graphs diverge: the mean over the graphs
                # that stay stable, with a distinct marker (delta = 0.5, p = 0.5 on BA and power-law)
                part = [p for p in sorted(d) if np.any(np.isfinite(d[p])) and not np.all(np.isfinite(d[p]))]
                unstable = 0.8 * (1 + dlt) >= 1
                errline(ax, ps, [d[p] for p in ps], color=f'C{i}',
                        label=f'δ = {dlt}' + (' (launch has no steady state)' if unstable else ''))
                if part:
                    ax.plot(part, [np.nanmean(d[p]) for p in part], 'x', ms=7, mew=1.5, color=f'C{i}')
                    any_partial = True
                if st:
                    pg = np.logspace(np.log10(min(d)), np.log10(max(d)), 40)
                    pred = [theory.nutreat_annealed(0.8, dlt, p, st['kbar'], st['rho'])['dm'] for p in pg]
                    ax.plot(pg, pred, '--', color=f'C{i}', lw=0.8)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('treated fraction')
            ax.set_title(label(f), fontsize=9)
        axes[0].set_ylabel('A/B difference, E[DM]')
        if any_partial:
            axes[0].plot([], [], 'x', ms=7, mew=1.5, color='k',
                         label='some graphs diverge: mean over the stable ones')
        axes[0].legend(fontsize=6.5, loc='upper left')
        fig.suptitle('treatment raises ν by a factor 1 + δ; νρ = 0.8 before treatment', fontsize=10)
        save('fig10_1_nu_treatment')

    rows = load('s10_threshold2')
    if rows:
        fams = fam_sorted({fam_key(c) for c, _ in rows})
        fig, axes = plt.subplots(1, len(fams), figsize=(4.5 * len(fams), 3.8), sharey=True)
        for ax, k in zip(np.atleast_1d(axes), fams):
            for i, th in enumerate(sorted({c['theta'] for c, _ in rows})):
                d = defaultdict(lambda: defaultdict(list))
                for c, r in rows:
                    if fam_key(c) == k and c['theta'] == th:
                        for row in r['rows']:
                            d['cascade'][row['p']].append(row['active'] > 0.99)
                            d['dm'][row['p']].append(row['dm'] / r['gate'])
                ps = sorted(d['cascade'])
                ax.plot(ps, [np.mean(d['cascade'][p]) for p in ps], '-o', ms=3, color=f'C{i}',
                        label=f'θ = {th}')
                ax.plot(ps, [np.nanmean(d['dm'][p]) for p in ps], ':', color=f'C{i}')
                st = gstat(fam_of(k))
                if st:
                    pg = np.logspace(np.log10(min(ps)), np.log10(max(ps)), 60)
                    pred = [theory.threshold_cascade(p, th, st['hist'])['dm_over_gate'] for p in pg]
                    ax.plot(pg, pred, '--', color=f'C{i}', lw=0.8, alpha=0.8)
            ax.set_xscale('log')
            ax.set_xlabel('treated fraction')
            ax.set_title(label(fam_of(k)))
        np.atleast_1d(axes)[0].set_ylabel('solid: P(full cascade); dotted: E[DM] / GATE\n'
                                          'dashed: E[DM] / GATE, tree recursion')
        np.atleast_1d(axes)[0].legend()
        save('fig10_2_threshold')


# ---------------------------------------------------------------------------
# Concave peer response (s12_concave.py)
# ---------------------------------------------------------------------------

def s12():
    rows = load('s12_grid')
    if not rows:
        return
    lin = load('s5_multiplier')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    styles = {1: ':', 3: '--', 10: '-'}
    for fam in ['er', 'ba']:
        col = FAM_COLORS[fam]
        for sig, ls in styles.items():
            d = defaultdict(list)
            for c, r in rows:
                if c['fam']['family'] == fam and c['g'] == 'log' and c['sigma'] == sig:
                    d[c['nu_rho']].append(r)
            xs = sorted(d)
            axes[0].plot(xs, [np.mean([r['small']['gate'] / DLAM for r in d[x]]) for x in xs], ls, marker='o',
                         ms=3, color=col, label=f"{label({'family': fam})}, K = {sig} k̄λ")
            axes[1].plot(xs, [np.mean([r['small']['dm_p0.1'] / r['small']['gate'] for r in d[x]]) for x in xs],
                         ls, marker='o', ms=3, color=col)
            axes[1].plot(xs, [np.mean([1 / r['m_ann'] for r in d[x]]) for x in xs], 'x', color='k', ms=4)
        if lin:
            rs = [r for c, r in lin if c['fam']['family'] == fam]
            xs = [p['nu_rho'] for p in rs[0]['points']]
            ys = [np.mean([r['points'][i]['multiplier'] for r in rs]) for i in range(len(xs))]
            axes[0].plot(xs, ys, '-', color=col, lw=2.5, alpha=0.35,
                         label=f"{label({'family': fam})}, linear model")
    axes[0].axvline(1, color='k', lw=0.6, ls=':')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('nominal peer-effect strength, νρ(A) (linear model unstable at 1)')
    axes[0].set_ylabel('social multiplier, GATE / Δλ')
    axes[0].legend(fontsize=6.5)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('nominal peer-effect strength, νρ(A)')
    axes[1].set_ylabel('E[DM] / GATE  (x: annealed closed form)')
    axes[1].set_ylim(0, 1.02)
    for c, r in rows:
        f = c['fam']['family']
        axes[2].plot(r['rhoJ'], r['small']['dm_p0.1'] / r['small']['gate'], 'o', ms=2.5, alpha=0.5,
                     color=FAM_COLORS[f], mec='none')
    xg = np.linspace(0, 1, 50)
    axes[2].plot(xg, 1 - xg, 'k-', lw=0.8, label='linear model: 1 − νρ')
    for fam in FAM_ORDER[:5]:
        axes[2].plot([], [], 'o', color=FAM_COLORS[fam], label=label({'family': fam}))
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel('local amplification at the untreated state, ρ(J)')
    axes[2].set_ylabel('E[DM] / GATE')
    axes[2].legend(fontsize=6.5)
    fig.tight_layout()
    save('fig11_1_concave')


# ---------------------------------------------------------------------------
# Appendix A: closed forms against the simulation (results/theory_pairs.jsonl,
# written by theory_check.py)
# ---------------------------------------------------------------------------

def sA():
    path = RESULTS / 'theory_pairs.jsonl'
    if not path.exists():
        return
    pairs = [json.loads(l) for l in open(path) if l.strip()]
    panels = [
        ('T1', ['multiplier', 'multiplier (fixed nu kbar)', 'multiplier (gamma model)'], 'B.1 multiplier', True),
        ('T2', ['lift hop 1', 'lift hop 2', 'lift hop 3'], 'B.2 spillover by hop (lift / Δλ)', True),
        ('T3', ['E[DM]/GATE', 'E[DM]/GATE (exact, N=3000)'], 'B.3 attenuation E[DM]/GATE', False),
        ('T4', ['E[DM]/GATE (clusters)', 'E[DM]/GATE (clusters, noisy)', 'E[DM]/GATE (wrong graph)'],
         'B.4 cluster designs E[DM]/GATE', False),
        ('T7', ['gate_total', 'inherit_total', 'exclude'], 'B.6 new users (production units)', False),
        ('T9', ['active fraction'], 'B.7 threshold contagion, active fraction', False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (t, qs, title, logs) in zip(axes.flat, panels):
        pts = [p for p in pairs if p['target'] == t and p['quantity'] in qs
               and np.isfinite(p['predicted']) and np.isfinite(p['simulated'])]
        if logs:
            pts = [p for p in pts if p['predicted'] > 0 and p['simulated'] > 0]
        for fam in FAM_ORDER:
            fp = [p for p in pts if p['family'] == fam]
            if fp:
                ax.plot([p['predicted'] for p in fp], [p['simulated'] for p in fp], 'o', ms=3,
                        alpha=0.6, color=FAM_COLORS[fam], mec='none')
        if pts:
            v = [p['predicted'] for p in pts] + [p['simulated'] for p in pts]
            lo, hi = min(v), max(v)
            ax.plot([lo, hi], [lo, hi], 'k-', lw=0.6)
        if logs:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('closed form', fontsize=8)
        ax.set_ylabel('simulation', fontsize=8)
    for fam in ['ws', 'er', 'sbm', 'ba', 'cm', 'gam']:
        axes[0][0].plot([], [], 'o', color=FAM_COLORS[fam],
                        label='gamma configuration' if fam == 'gam' else label({'family': fam}))
    axes[0][0].legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis('off')
    fig.tight_layout()
    save('figA_1_closed_forms')


# ---------------------------------------------------------------------------
# Main-text figures of the paper (paper/paper.md): compact re-plots of panels drawn in full
# above. Column width (aspect at most 1.9) except M5.
# ---------------------------------------------------------------------------

def _styles(*items, color='k'):
    """Legend handles that show a line style or marker: (label, ls, marker[, lw])."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=color, ls=it[1], marker=it[2], ms=4,
                   lw=(it[3] if len(it) > 3 else 1.5), label=it[0]) for it in items]


def _stack(n, h=3.0):
    return plt.subplots(n, 1, figsize=(6.4, h * n))


def _fam_lines(ax, by, key='ratio', ls='-', **kw):
    """One line per family (keys of `by`, nested dict x -> list of rows), in the paper's order."""
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        xs = sorted(d)
        errline(ax, xs, [[r[key] for r in d[x]] for x in xs], color=FAM_COLORS[f['family']],
                ls=ls, label=label(f), **kw)


def figM_1_multiplier_attenuation():
    """Top: multiplier against nu rho (fig5_1). Bottom: E[DM]/GATE against nu rho with the closed
    form of A.3 dashed (left panel of fig5_4)."""
    fig, (a0, a1) = _stack(2, 3.1)
    rows = load('s5_multiplier')
    # SBM is left out: its exact and mean-field curves are within 1% of Erdős-Rényi's at every
    # point; Watts-Strogatz is plotted on its own (its mean-field value is much closer to the
    # exact one). Error bars are within one family.
    by = defaultdict(list)
    for c, r in rows:
        fam = c['fam']['family']
        if fam != 'sbm':
            by[fam].append(r)
    names = {'ws': ('Watts-Strogatz', FAM_COLORS['ws']),
             'er': ('Erdős-Rényi (SBM within 1%)', FAM_COLORS['er']),
             'ba': ('Barabási-Albert', FAM_COLORS['ba']),
             'cm': ('power-law (γ=2.5)', FAM_COLORS['cm'])}
    for k in [f for f in FAM_ORDER if f in by]:
        rs = by[k]
        lab, col = names[k]
        xs = [p['nu_rho'] for p in rs[0]['points']]
        ys = [[r['points'][i]['multiplier'] for r in rs] for i in range(len(xs))]
        errline(a0, xs, ys, color=col, label=lab)
        mf = [np.mean([r['points'][i]['mean_field'] for r in rs]) for i in range(len(xs))]
        a0.plot(xs, mf, '--', color=col, lw=0.9)
    a0.plot([], [], 'k-', label='simulated (exact)')
    a0.plot([], [], 'k--', lw=0.9, label='from the average friend count, 1/(1 − ν k̄)')
    a0.set_yscale('log')
    a0.set_yticks([1, 10, 100])
    a0.set_yticklabels(['1', '10', '100'])
    a0.set_xlim(0.05, 1.1)
    a0.set_ylim(0.85, 160)
    a0.axvline(1.0, color='k', lw=1.0, ls=':')
    a0.text(1.012, 1.0, 'stability limit', rotation=90, ha='left', va='bottom', fontsize=7)
    a0.set_xlabel('peer-effect strength, νρ(A)  (0 = none, 1 = stability limit)')
    a0.set_ylabel('social multiplier, GATE / Δλ')
    a0.legend(fontsize=7, loc='upper left')
    rows = load('s5_attenuation')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[fam_key(c)][c['nu_rho']].append(r)
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        xs = sorted(d)
        errline(a1, xs, [[r['ratio'] for r in d[x]] for x in xs], color=FAM_COLORS[f['family']],
                label=label(f))
        st = gstat(f)
        if st:
            xg = np.linspace(min(xs), max(xs), 60)
            pred = [theory.attenuation(x / st['rho'], st['rho'], st['kbar'], st['b'], st['n'],
                                       deg_hist=st['hist'], triangles_per_node=st['tri']) for x in xg]
            a1.plot(xg, pred, '--', color=FAM_COLORS[f['family']], lw=0.8)
    a1.plot([], [], 'k--', lw=0.8, label='closed form (A.3)')
    a1.set_xlabel('peer-effect strength, νρ(A)')
    a1.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    a1.set_ylim(0, 1.02)
    a1.legend(fontsize=7)
    fig.tight_layout()
    save('figM_1_multiplier_attenuation')


def figM_2_size():
    """Top: attenuation against N at fixed nu rho = 0.5 (fig5_5, left). Bottom: nu rho against N at
    fixed nu kbar = 0.3 (fig5_5b, left)."""
    fig, (a0, a1) = _stack(2, 2.9)
    rows = load('s5_size')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[c['family']][c['n']].append(r)
    for fam in [f for f in FAM_ORDER if f in by]:
        d = by[fam]
        ns = sorted(d)
        errline(a0, ns, [[r['ratio'] for r in d[n]] for n in ns], color=FAM_COLORS[fam],
                label=label({'family': fam}))
    a0.set_xscale('log')
    a0.set_xlabel('number of users, N')
    a0.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    a0.set_ylim(0.4, 1.0)
    rows = load('s5_size_fixednu')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[c['family']][c['n']].append(r)
    for fam in [f for f in FAM_ORDER if f in by]:
        d = by[fam]
        ns = sorted(d)
        errline(a1, ns, [[r['nu_rho'] for r in d[n]] for n in ns], color=FAM_COLORS[fam],
                label=label({'family': fam}))
    a1.axhline(1, color='k', ls='--', lw=0.8)
    a1.text(1.5e6, 1.08, 'unstable above this line', fontsize=7)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.set_yticks([0.3, 0.5, 1, 2, 3])
    a1.set_yticklabels(['0.3', '0.5', '1', '2', '3'])
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel('number of users, N')
    a1.set_ylabel('peer-effect strength, νρ(A)')
    a1.legend(*a0.get_legend_handles_labels(), fontsize=6.5, loc='lower right',
              bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()
    save('figM_2_size')


def figM_3_paths():
    """fig6_1 stacked: GATE(t) and DM(t) relative to their limits, Watts-Strogatz (solid) and
    Barabási-Albert (dashed)."""
    rows = load('s6_paths')
    fig, axes = _stack(2, 2.7)
    seen = set()
    for c, r in sorted(rows, key=lambda cr: cr[0].get('nu_rho', 0)):
        if c['seed'] != 0 or c['fam']['family'] not in ('ws', 'ba'):
            continue
        t = np.arange(1, len(r['gate']) + 1)
        col = {0.5: 'C0', 0.8: 'C1', 0.9: 'C2', 0.95: 'C3', 0.99: 'C4'}[c['nu_rho']]
        ls = '-' if c['fam']['family'] == 'ws' else '--'
        lab = None
        if ls == '-' and c['nu_rho'] not in seen:
            seen.add(c['nu_rho'])
            lab = f"νρ = {c['nu_rho']}"
        axes[0].plot(t, np.array(r['gate']) / r['gate_inf'], ls, color=col, label=lab)
        axes[1].plot(t[1:], (np.array(r['dm']) / r['dm_inf'])[1:], ls, color=col)   # from period 2: period 1 equals period 2 (the first hop cancels)
    for ax, yl in zip(axes, ['launch effect, GATE(t) / GATE(∞)', 'A/B difference, DM(t) / DM(∞)']):
        ax.set_xscale('log')
        ax.set_ylabel(yl)
        better(ax, sooner=True)
    for ax in axes:
        ax.set_xlabel('periods since treatment started')
    h, l = axes[0].get_legend_handles_labels()
    h = _styles(('Watts-Strogatz', '-', ''), ('Barabási-Albert', '--', '')) + h
    axes[1].set_ylim(0.75, 1.01)
    axes[1].legend(handles=h, fontsize=7, loc='lower right')
    fig.tight_layout()
    save('figM_3_paths')


def figM_4_clusters():
    """Top: E[DM]/GATE against the share of edges inside clusters at nu rho = 0.5, with the SBM
    mu sweep and the mean-field curve of A.4 (fig5_6). Bottom: RMSE/GATE against cluster size at
    nu rho = 0.6 for the five families (from fig8_1)."""
    fig, (a0, a1) = _stack(2, 3.0)
    rows = load('s5_clustering')
    marks = {'bernoulli': 'o', 'metis': 's', 'contiguous': '^', 'blocks': 'D'}
    mark_name = {'bernoulli': 'user-level split', 'metis': 'cluster split (METIS)',
                 'contiguous': 'cluster split (contiguous on the ring)',
                 'blocks': 'cluster split (true SBM blocks)'}
    for c, r in rows:
        a0.plot(1 - r['cut'], r['ratio'], marks[c['design']], ms=4,
                color=FAM_COLORS[c['fam']['family']], alpha=0.7)
    mu_rows = load('s5_clustering_mu') or []
    for c, r in mu_rows:
        if c['nu_rho'] == 0.5 and c['design'] in ('blocks', 'metis'):
            a0.plot(1 - r['cut'], r['ratio'], 'x' if c['design'] == 'blocks' else '+', ms=6,
                    color=FAM_COLORS['sbm'], alpha=0.9)
    phi = np.linspace(0, 1, 60)
    a0.plot(phi, (1 - 0.5) / (1 - 0.5 * phi), 'k--', lw=0.8, label='mean field (A.4), (1 − νρ)/(1 − νρϕ)')
    for d, m in marks.items():
        a0.plot([], [], m, color='gray', label=mark_name[d])
    if mu_rows:
        a0.plot([], [], 'x', color=FAM_COLORS['sbm'], label='SBM, μ = 0.2 to 0.7, true blocks')
        a0.plot([], [], '+', color=FAM_COLORS['sbm'], label='SBM, μ = 0.2 to 0.7, METIS 1,000')
    a0.set_xlabel('share of edges inside clusters, ϕ  (νρ = 0.5)')
    a0.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(a0)
    a0.set_ylim(0.45, 1.0)
    leg = a0.legend(fontsize=6, loc='lower right', framealpha=0.9)
    fams = sorted({fam_key(c) for c, _ in rows}, key=lambda k: FAM_ORDER.index(fam_of(k)['family']))
    a0.legend(handles=[plt.Line2D([], [], ls='', marker='o', ms=4, color=FAM_COLORS[fam_of(k)['family']],
                                  label=label(fam_of(k))) for k in fams],
              fontsize=6, loc='upper left', framealpha=0.9)
    a0.add_artist(leg)
    rows = load('s8_clusters3') or load('s8_clusters2')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        if c['nu_rho'] == 0.6:
            by[fam_key(c)][c['size']].append(r)
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        sz = sorted(d)
        col = FAM_COLORS[f['family']]
        a1.plot(sz, [np.mean([r['rmse'] / r['gate'] for r in d[s]]) for s in sz], '-o', ms=3,
                color=col, label=label(f))
        a1.plot(sz, [np.mean([abs(r['mean'] - r['gate']) / r['gate'] for r in d[s]]) for s in sz],
                ':', color=col, lw=0.9)
    a1.plot([], [], 'k-', label='error (RMSE)')
    a1.plot([], [], 'k:', lw=0.9, label='bias (absolute)')
    a1.set_xscale('log')
    a1.set_xlabel('cluster size (1 = user-level split); νρ = 0.6')
    a1.set_ylabel('error of the A/B estimate,\n% of the launch effect')
    pct_axis(a1)
    better(a1, up=False)
    a1.set_ylim(0, 1.0)
    a1.legend(fontsize=6.5, ncol=2, loc='upper left')
    fig.tight_layout()
    save('figM_4_clusters')


def figM_5_extrapolation():
    """One row: RMSE/GATE of the DM, the one-feature regression, the historical baseline, the
    two-fraction slope and the plug-in under the four conditions of fig8_3. Lines: mean over
    the five families; markers: the families."""
    rows = load('s8_extrap3')
    ests = [('dm', 'A/B difference (DM)', 'C0', '-'),
            ('pre_post', 'slope: window before and during the test', 'C2', '--'),
            ('two_fraction', 'slope: two A/B tests, 10% and 90% treated', 'C2', '-'),
            ('plug_in', 'plug-in (uses the graph)', 'C3', '-')]
    cases = [('linear', 0.0, 0.0), ('linear', 0.2, 0.0), ('linear', 0.0, 0.01)]
    titles = ['(a) nothing changes between observation windows', '(b) 20% of friendships carry no influence',
              '(c) seasonal 1% drift in baseline content production']
    fmarks = {'ws': 's', 'er': 'o', 'sbm': 'D', 'ba': '^', 'cm': 'v'}
    fams = fam_sorted({fam_key(c) for c, _ in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (m, q, dr), t in zip(axes, cases, titles):
        for e, ename, col, lsty in ests:
            per_fam = {}
            for k in fams:
                d = defaultdict(list)
                for c, r in rows:
                    if (fam_key(c) == k and (c['model'], c['q'], c['drift']) == (m, q, dr)
                            and np.isfinite(r[e]['rmse'])):
                        d[c['nu_rho']].append(r[e]['rmse'] / r['gate'])
                per_fam[k] = {x: np.mean(v) for x, v in d.items()}
            xs = sorted({x for v in per_fam.values() for x in v})
            ax.plot(xs, [np.mean([per_fam[k][x] for k in fams if x in per_fam[k]]) for x in xs], lsty,
                    color=col, lw=1.0, marker='o', ms=3, label=ename)
        ax.set_yscale('log')
        ax.set_title(t, fontsize=9)
        ax.set_xlabel('peer-effect strength, νρ(A)')
        ax.set_xticks([0.3, 0.6, 0.9])
    axes[0].set_ylabel('typical error of the estimate,\n% of the launch effect')
    pct_axis(axes[0])
    better(axes[-1], up=False)
    axes[0].set_ylim(0.01, 4)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    save('figM_5_extrapolation')


def figM_6_critical():
    """Top: GATE and treatment-induced E[DM] against 1/(1 - nu rho). Bottom: power with and
    without CUPED (from fig7_1)."""
    rows = load('s7_power3') or load('s7_power')
    rows = [(c, r) for c, r in rows if c.get('effect', DLAM) > 0]
    fig, (a0, a1) = _stack(2, 2.9)
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[fam_key(c)][c['nu_rho']].append(r)
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        xs = sorted(d)
        col = FAM_COLORS[f['family']]
        inv = [1 / (1 - x) for x in xs]
        errline(a0, inv, [[r['gate'] / DLAM for r in d[x]] for x in xs], color=col)
        a1.plot([], [], '-', color=col, label=label(f))
        a0.plot([], [], '-', color=col, label=label(f))
        errline(a0, inv, [[r.get('dm_lift_mean', r['dm_mean']) / DLAM for r in d[x]] for x in xs],
                color=col, ls=':')
        errline(a1, inv, [[r['power'] for r in d[x]] for x in xs], color=col)
        if 'power_cuped' in d[xs[0]][0]:
            errline(a1, inv, [[r['power_cuped'] for r in d[x]] for x in xs], color=col, ls=':')
    a0.plot([], [], 'k-', label='launch effect, GATE / Δλ')
    a0.plot([], [], 'k:', label='A/B difference, E[DM] / Δλ')
    a0.set_yscale('log')
    a0.set_ylabel('size of the effect, GATE / Δλ and E[DM] / Δλ')
    a0.legend(fontsize=6.5, loc='upper left')
    a1.plot([], [], 'k-', label='plain DM')
    a1.plot([], [], 'k:', label='CUPED')
    a1.set_ylabel('power')
    better(a1)
    a1.legend(fontsize=6, loc='lower left', handlelength=1.6)
    from matplotlib.ticker import NullLocator
    for ax in (a0, a1):
        ax.set_xscale('log')
        ax.set_xticks([2, 10, 100, 1000])
        ax.set_xticklabels(['0.5', '0.9', '0.99', '0.999'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('peer-effect strength, νρ(A)')
    fig.tight_layout()
    save('figM_6_critical')


def figM_7_graph_change():
    """Top: E[DM]/GATE when treatment adds friends (boost and new friends), SBM and
    Barabási-Albert, both split designs and both edge rules (from fig9_1). Bottom: share of the
    launch effect the test sees when treatment invites new users, a = 0.5, q = 2, all families
    (from fig9_2)."""
    fig, (a0, a1) = _stack(2, 3.0)
    rows = load('s9_edges2') or load('s9_edges')
    rule_name = {'fof': 'friends of friends', 'random': 'random users'}
    design_name = {'bernoulli': 'user-level split', 'metis': 'cluster split'}
    for fam in ['sbm', 'ba']:
        col = FAM_COLORS[fam]
        by = defaultdict(lambda: defaultdict(list))
        for c, r in rows:
            if (c['fam']['family'] == fam and c['boost'] and np.isfinite(r['gate']) and r['gate'] > 0):
                by[(c['rule'], c['design'])][c['r']].append(r['dm'] / r['gate'])
        for (rule, design), d in sorted(by.items()):
            rs = sorted(d)
            ls = '-' if design == 'metis' else ':'
            mk = 'o' if rule == 'fof' else 's'
            a0.errorbar(rs, [mean_se(d[x])[0] for x in rs], yerr=[mean_se(d[x])[1] for x in rs],
                        capsize=2, marker=mk, ms=3, color=col, ls=ls,
                        label=None)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color='k', ls=':', label='user-level split'),
                Line2D([], [], color='k', ls='-', label='cluster split'),
                Line2D([], [], color='k', ls='', marker='o', label='new friends: friends of friends'),
                Line2D([], [], color='k', ls='', marker='s', label='new friends: random users'),
                Line2D([], [], color=FAM_COLORS['sbm'], label='SBM (μ=0.1)'),
                Line2D([], [], color=FAM_COLORS['ba'], label='Barabási-Albert')]
    a0.set_xlabel('new friends per treated user  (νρ = 0.5 before treatment)')
    a0.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(a0)
    a0.set_ylim(0, 1.02)
    a0.legend(handles=handles, fontsize=6, loc='upper right', ncol=3)
    rows = load('s9_users')
    est_name = {'inherit_total': ("new users' posts added to the treated arm", '-'),
                'exclude': ('new users left out', '--')}
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for c, r in rows:
        if c['a'] == 0.5 and c['q'] == 2 and not r.get('unstable'):
            for k in est_name:
                by[fam_key(c)][k][c['r']].append(r[k] / r['gate_total'])
    for fk in fam_sorted(by):
        f = fam_of(fk)
        for k, (_, ls) in est_name.items():
            d = by[fk][k]
            rs = sorted(d)
            errline(a1, rs, [d[x] for x in rs], color=FAM_COLORS[f['family']], ls=ls,
                    label=label(f) if k == 'inherit_total' else None)
    for k, (elab, ls) in est_name.items():
        a1.plot([], [], 'k', ls=ls, label=elab)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel('new users invited per treated user  (new users post half as much; νρ = 0.5)')
    a1.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(a1)
    a1.legend(fontsize=6, ncol=2, loc='center right')
    fig.tight_layout()
    save('figM_7_graph_change')


def figM_8_concave():
    """Top: multiplier against the nominal nu rho for the logarithmic response on Erdős-Rényi and
    Barabási-Albert graphs, sigma = 1, 3, 10, against the linear model (left panel of fig11_1).
    Bottom: E[DM]/GATE against rho(J) for every configuration of the grid (right panel)."""
    rows = load('s12_grid')
    lin = load('s5_multiplier')
    fig, (a0, a1) = _stack(2, 3.0)
    styles = {1: ':', 3: '--', 10: '-'}
    for fam in ['er', 'ba']:
        col = FAM_COLORS[fam]
        for sig, ls in styles.items():
            d = defaultdict(list)
            for c, r in rows:
                if c['fam']['family'] == fam and c['g'] == 'log' and c['sigma'] == sig:
                    d[c['nu_rho']].append(r)
            xs = sorted(d)
            a0.plot(xs, [np.mean([r['small']['gate'] / DLAM for r in d[x]]) for x in xs], ls, marker='o',
                    ms=3, color=col)
        if lin:
            rs = [r for c, r in lin if c['fam']['family'] == fam]
            xs = [p['nu_rho'] for p in rs[0]['points']]
            ys = [np.mean([r['points'][i]['multiplier'] for r in rs]) for i in range(len(xs))]
            a0.plot(xs, ys, '-', color=col, lw=2.5, alpha=0.35)
    a0.axvline(1, color='k', lw=0.6, ls=':')
    a0.set_xscale('log')
    a0.set_yscale('log')
    a0.set_xlabel('nominal peer-effect strength, νρ(A) (linear model unstable at 1)')
    a0.set_ylabel('social multiplier, GATE / Δλ')
    a0.legend(handles=_styles(*[(f'logarithmic, K = {sig} k̄λ', ls, 'o') for sig, ls in styles.items()])
              + [matplotlib.lines.Line2D([], [], color='gray', lw=2.5, alpha=0.35, label='linear model')],
              fontsize=6.5, loc='upper left')
    for c, r in rows:
        a1.plot(r['rhoJ'], r['small']['dm_p0.1'] / r['small']['gate'], 'o', ms=2.5, alpha=0.5,
                color=FAM_COLORS[c['fam']['family']], mec='none')
    xg = np.linspace(0, 1, 50)
    a1.plot(xg, 1 - xg, 'k-', lw=0.8, label='linear-model relation, 1 − νρ')
    for fam in FAM_ORDER[:5]:
        a1.plot([], [], 'o', color=FAM_COLORS[fam], label=label({'family': fam}))
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel('local amplification at the untreated state, ρ(J)')
    a1.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(a1)
    a1.legend(fontsize=6.5)
    fig.tight_layout()
    save('figM_8_concave')


# Section 4 as four single-column figures, one panel each, so that each sits next to its paragraph.

def _m1_multiplier(ax):
    rows = load('s5_multiplier')
    by = defaultdict(list)
    for c, r in rows:
        fam = c['fam']['family']
        if fam != 'sbm':
            by[fam].append(r)
    names = {'ws': ('Watts-Strogatz', FAM_COLORS['ws']),
             'er': ('Erdős-Rényi (SBM within 1%)', FAM_COLORS['er']),
             'ba': ('Barabási-Albert', FAM_COLORS['ba']),
             'cm': ('power-law (γ=2.5)', FAM_COLORS['cm'])}
    for k in [f for f in FAM_ORDER if f in by]:
        rs = by[k]
        lab, col = names[k]
        xs = [p['nu_rho'] for p in rs[0]['points']]
        ys = [[r['points'][i]['multiplier'] for r in rs] for i in range(len(xs))]
        errline(ax, xs, ys, color=col, label=lab)
        mf = [np.mean([r['points'][i]['mean_field'] for r in rs]) for i in range(len(xs))]
        ax.plot(xs, mf, '--', color=col, lw=0.9)
    ax.plot([], [], 'k-', label='simulated on the graph')
    ax.plot([], [], 'k--', lw=0.9, label='1/(1 − ν k̄): every user has k̄ friends')
    ax.set_yscale('log')
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(['1', '10', '100'])
    ax.set_xlim(0.05, 1.1)
    ax.set_ylim(0.85, 160)
    ax.axvline(1.0, color='k', lw=1.0, ls=':')
    ax.text(1.012, 1.0, 'stability limit', rotation=90, ha='left', va='bottom', fontsize=7)
    ax.set_xlabel('peer-effect strength, νρ(A)  (0 = none, 1 = stability limit)')
    ax.set_ylabel('social multiplier, GATE / Δλ')
    ax.legend(fontsize=7, loc='upper left')


def figM_1_multiplier():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _m1_multiplier(ax)
    fig.tight_layout()
    save('figM_1_multiplier')


def figM_1b_attenuation():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rows = load('s5_attenuation')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[fam_key(c)][c['nu_rho']].append(r)
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        xs = sorted(d)
        errline(ax, xs, [[r['ratio'] for r in d[x]] for x in xs], color=FAM_COLORS[f['family']],
                label=label(f))
        st = gstat(f)
        if st:
            xg = np.linspace(min(xs), max(xs), 60)
            pred = [theory.attenuation(x / st['rho'], st['rho'], st['kbar'], st['b'], st['n'],
                                       deg_hist=st['hist'], triangles_per_node=st['tri']) for x in xg]
            ax.plot(xg, pred, '--', color=FAM_COLORS[f['family']], lw=0.8)
    ax.plot([0, 1], [1, 0], color='0.6', lw=0.8, ls=':', label='1 − νρ')
    ax.plot([], [], 'k--', lw=0.8, label='closed form (A.3)')
    ax.set_xlabel('peer-effect strength, νρ(A)')
    ax.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(ax)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save('figM_1b_attenuation')


def figM_1c_treated_fraction():
    """E[DM]/GATE against the treated fraction under random assignment, treatment-induced DM
    (the baseline difference between the arms removed), nu rho = 0.5."""
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    rows = load('s5_attenuation_p')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[fam_key(c)][c['p']].append(r)
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        ps = sorted(d)
        errline(ax, ps, [[r['ratio'] for r in d[p]] for p in ps], color=FAM_COLORS[f['family']],
                label=label(f))
    ax.set_xscale('log')
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9])
    ax.set_xticklabels(['1%', '2%', '5%', '10%', '20%', '50%', '90%'])
    ax.set_xlabel('share of users treated, p')
    ax.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(ax)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    fig.tight_layout()
    save('figM_1c_treated_fraction')


def figM_1d_attenuation_nukbar():
    """E[DM]/GATE against nu kbar (same peer effect per friend on every family), 10% treated;
    each heavy-tailed curve ends where nu rho reaches 1."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rows = load('s5_attenuation_kbar')
    by = defaultdict(lambda: defaultdict(list))
    lim = defaultdict(list)
    for c, r in rows:
        if r.get('stable'):
            by[fam_key(c)][c['nu_kbar']].append(r)
        lim[fam_key(c)].append(r['kbar'] / r['rho'])
    for k in fam_sorted(by):
        d = by[k]
        f = fam_of(k)
        xs = sorted(x for x in d if len(d[x]) == max(len(v) for v in d.values()))
        col = FAM_COLORS[f['family']]
        errline(ax, xs, [[r['ratio'] for r in d[x]] for x in xs], color=col, label=label(f))
        if f['family'] in ('ba', 'cm'):
            ax.axvline(np.mean(lim[k]), color=col, lw=0.8, ls='--')
    ax.plot([], [], 'k--', lw=0.8, label='stability limit νρ = 1')
    ax.set_xlabel('peer effect per friend times mean degree, ν k̄')
    ax.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(ax)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    save('figM_1d_attenuation_nukbar')


def figM_2_size_fixed_nurho():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rows = load('s5_size')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[c['family']][c['n']].append(r)
    for fam in [f for f in FAM_ORDER if f in by]:
        d = by[fam]
        ns = sorted(d)
        errline(ax, ns, [[r['ratio'] for r in d[n]] for n in ns], color=FAM_COLORS[fam],
                label=label({'family': fam}))
    ax.set_xscale('log')
    ax.set_xlabel('number of users, N')
    ax.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(ax)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=6.5, loc='center right')
    fig.tight_layout()
    save('figM_2_size_fixed_nurho')


def figM_9_threshold():
    """Threshold contagion, main text: E[DM]/GATE against the treated fraction, one color per
    threshold theta; Erdos-Renyi (solid) and Watts-Strogatz (dashed); ticks at the tree-recursion p*
    on Erdos-Renyi (from fig10_2)."""
    rows = load('s10_threshold2')
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    thetas = sorted({c['theta'] for c, _ in rows})
    for fam, ls in [('er', '-'), ('ws', '--')]:
        ks = [k for k in fam_sorted({fam_key(c) for c, _ in rows}) if fam_of(k)['family'] == fam]
        if not ks:
            continue
        k = ks[0]
        for i, th in enumerate(thetas):
            d = defaultdict(list)
            for c, r in rows:
                if fam_key(c) == k and c['theta'] == th:
                    for row in r['rows']:
                        d[row['p']].append(row['dm'] / r['gate'])
            ps = sorted(d)
            ax.plot(ps, [np.nanmean(d[p]) for p in ps], ls, marker='o', ms=3, color=f'C{i}',
                    label=f'θ = {th}' if fam == 'er' else None)
            if fam == 'er':
                st = gstat(fam_of(k))
                if st:
                    pg = np.logspace(np.log10(min(ps)), np.log10(max(ps)), 400)
                    pred = np.array([theory.threshold_cascade(p, th, st['hist'])['dm_over_gate'] for p in pg])
                    below = np.where(pred < 0.5)[0]
                    if len(below):
                        ax.axvline(pg[below[0]], color=f'C{i}', lw=0.8, ls=':', alpha=0.9)
    ax.plot([], [], 'k-', label='Erdős-Rényi')
    ax.plot([], [], 'k--', label='Watts-Strogatz (p=0.1)')
    ax.plot([], [], color='k', lw=0.8, ls=':', label='predicted critical fraction p*')
    ax.set_xscale('log')
    ax.set_xlabel('share of users treated, p')
    ax.set_ylabel('measured launch effect ratio,\nE[DM] / GATE')
    better(ax)
    ax.set_ylim(-0.02, 1.04)
    ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    save('figM_9_threshold')

def figM_2b_size_fixed_nukbar():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rows = load('s5_size_fixednu')
    by = defaultdict(lambda: defaultdict(list))
    for c, r in rows:
        by[c['family']][c['n']].append(r)
    for fam in [f for f in FAM_ORDER if f in by]:
        d = by[fam]
        ns = sorted(d)
        errline(ax, ns, [[r['nu_rho'] for r in d[n]] for n in ns], color=FAM_COLORS[fam],
                label=label({'family': fam}))
    ax.axhline(1, color='k', ls='--', lw=0.8)
    ax.text(1.5e6, 1.08, 'unstable above this line', fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_yticks([0.3, 0.5, 1, 2, 3])
    ax.set_yticklabels(['0.3', '0.5', '1', '2', '3'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('number of users, N')
    ax.set_ylabel('peer-effect strength, νρ(A)')
    ax.legend(fontsize=6.5, loc='lower right', bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()
    save('figM_2b_size_fixed_nukbar')


def sM4():
    for f in (figM_1_multiplier, figM_1b_attenuation, figM_1c_treated_fraction, figM_1d_attenuation_nukbar,
              figM_2_size_fixed_nurho,
              figM_2b_size_fixed_nukbar):
        f()


# Sections 5 to 8: the stacked main-text figures, drawn as single-column figures, one panel each.

class _Pair:
    """Stands in for the figure of a two-panel _stack: two separate figures."""
    def __init__(self, figs):
        self.figs = figs

    def tight_layout(self, *a, **k):
        for f in self.figs:
            f.tight_layout(*a, **k)


def _split(func, names, h=3.8):
    """Run a two-panel figure function with each panel on its own figure, saved under names[i]."""
    global _stack, save
    orig_stack, orig_save = _stack, save
    made = {}

    def stack(n, hh=3.0):
        assert n == 2
        figs, axes = zip(*[plt.subplots(figsize=(6.4, h)) for _ in range(n)])
        made['figs'], made['axes'] = figs, axes
        return _Pair(list(figs)), list(axes)

    def save_each(_name):
        axes = made['axes']
        donor = next((a.get_legend() for a in axes if a.get_legend() is not None), None)
        for fig, ax, name in zip(made['figs'], axes, names):
            if ax.get_legend() is None and donor is not None:
                # a panel without a legend gets the legend of the other panel
                ax.legend(handles=donor.legend_handles, labels=[t.get_text() for t in donor.get_texts()],
                          fontsize=donor.get_texts()[0].get_fontsize(), loc='best')
                fig.tight_layout()
            plt.figure(fig.number)
            FIGURES.mkdir(parents=True, exist_ok=True)
            fig.savefig(FIGURES / f'{name}.png')
            fig.savefig(FIGURES / f'{name}.pdf')
            print('  wrote', name)
        plt.close('all')

    _stack, save = stack, save_each
    try:
        func()
    finally:
        _stack, save = orig_stack, orig_save


SPLIT58 = [(lambda: figM_3_paths(), ('figM_3_gate_paths', 'figM_3b_dm_paths')),
           (lambda: figM_4_clusters(), ('figM_4_clusters_phi', 'figM_4b_clusters_size')),
           (lambda: figM_6_critical(), ('figM_6_critical_effects', 'figM_6b_critical_power')),
           (lambda: figM_7_graph_change(), ('figM_7_new_edges', 'figM_7b_new_users')),
           (lambda: figM_8_concave(), ('figM_8_concave_multiplier', 'figM_8b_concave_attenuation'))]


def sM58():
    for func, names in SPLIT58:
        _split(func, names)


def sM():
    for f in (figM_1_multiplier_attenuation, figM_2_size, figM_3_paths, figM_4_clusters,
              figM_5_extrapolation, figM_6_critical, figM_7_graph_change, figM_8_concave):
        f()


if __name__ == '__main__':
    parts = sys.argv[1:] or ['s4', 's5', 's6', 's7', 's8', 's9', 's10', 's12', 'sM', 'sA']
    for p in parts:
        print(p)
        globals()[p]()
