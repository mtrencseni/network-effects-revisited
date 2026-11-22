"""Reproduce every number and figure of Trencseni 2023 (arXiv:2312.01607).

Two modes:

  --mode exact  Replays the 2023 notebook's random number sequence: graphs from
                networkx and treatment groups from Python's random module, in
                notebook order on one thread, then all simulations in parallel.
                The noisy (uniform, Poisson) runs share numpy's legacy random
                stream, so they run as one sequential chain in a worker.
                Every printed number should match the notebook to floating point.

  --mode fast   Same experiments with the fast generators in netfx.generators
                and an independent seed per task, fully parallel. Results should
                agree with the notebook within sampling noise.

Outputs go to out/<mode>/: results.json, figures/, report.md.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from netfx import generators, graph, metrics, runner  # noqa: E402

SEED = 342423
NU = 0.01
T = 25
NOTEBOOK = ROOT.parent / 'mc-network-effects-rct' / 'notebook.ipynb'


# ---------------------------------------------------------------------------
# The 2023 notebook as a list of steps, in execution order. Each step makes one
# graph (and possibly a treatment group) and lists the simulations run on it.
# 'cell' is the notebook cell the step comes from.
# ---------------------------------------------------------------------------

def steps():
    s = []
    s.append(dict(id='c02', n=100, k=6, p=0.1, treat=('random', 20), sims=[], layout=True))
    for p, k in product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [10, 20, 30, 40, 50]):
        s.append(dict(id=f'c04_p{p}_k{k}', n=10_000, k=k, p=p, treat=None,
                      sims=[dict(name='mean')]))
    for cell, k, nu_fn in (('c05', 50, 'linear'), ('c08', 200, 'linear'),
                           ('c10', 100, 'linear'), ('c12', 100, 'sigmoid')):
        keep = cell == 'c05'
        s.append(dict(id=cell, n=10_000, k=k, p=0.1, treat=None,
                      sims=[dict(name=smp, sampler=smp, nu_fn=nu_fn, keep_final=keep)
                            for smp in ('mean', 'uniform', 'poisson')]))
    for cell, k, treat in (('c14', 50, ('random', 10_000)), ('c17', 10, ('random', 5_000)),
                           ('c21', 50, ('first', 10_000))):
        s.append(dict(id=cell, n=500_000, k=k, p=0.1, treat=treat, dlam=0.05,
                      sims=[dict(name=smp, sampler=smp, groups=True)
                            for smp in ('mean', 'uniform', 'poisson')]))
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s.append(dict(id=f'c26_p{p}', n=10_000, k=50, p=p, treat=('random', 0), dlam=0.05,
                      sims=[dict(name='mean', groups=True)]))
    for p in (0.1, 0.9):
        s.append(dict(id=f'c30_p{p}', n=10_000, k=50, p=p, treat=None,
                      sims=[dict(name='mean', keep_final=True)]))
    for N, k in product([100, 500, 1000, 5000, 9000], [10, 20, 30, 40, 50]):
        s.append(dict(id=f'c32_N{N}_k{k}', n=10_000, k=k, p=0.1, treat=('random', N),
                      dlam=0.05, sims=[dict(name='mean', groups=True)]))
    for N, p in product([100, 500, 1000, 5000, 9000], [0.1, 0.2, 0.3, 0.4, 0.5]):
        s.append(dict(id=f'c36_N{N}_p{p}', n=10_000, k=50, p=p, treat=('random', N),
                      dlam=0.05, sims=[dict(name='mean', groups=True)]))
    s.append(dict(id='c39', n=10_000, k=50, p=0.1, treat=('random', 1000), dlam=0.05,
                  sims=[dict(name='mean', groups=True)]))
    s.append(dict(id='c40', n=10_000, k=10, p=0.1, treat=('random', 5000), dlam=0.05,
                  sims=[dict(name='mean', groups=True)]))
    s.append(dict(id='c42', n=500_000, k=50, p=0.1, treat=None, sims=[dict(name='mean')]))
    s.append(dict(id='c43', n=10_000, k=50, p=0.1, treat=('random', 1000), sims=[],
                  rest_degree=True))
    return s


def realize_exact(step, cache):
    """Make the step's graph and treatment group with the 2023 random sequence."""
    g = nx.connected_watts_strogatz_graph(n=step['n'], k=step['k'], p=step['p'])
    treat = None
    if step['treat'] is not None:
        kind, N = step['treat']
        if kind == 'random':
            treat = np.array(sorted(random.sample(list(g.nodes), N)), dtype=np.int64)
        else:
            treat = np.array(list(g.nodes)[:N], dtype=np.int64)
    extra = {}
    if step.get('layout'):
        pos = nx.spring_layout(g)  # consumes numpy's global stream, as in 2023
        extra['pos'] = {int(a): b.tolist() for a, b in pos.items()}
        extra['small_graph'] = [list(map(int, e)) for e in g.edges()]
    A = graph.from_networkx(g)
    path = str(cache / f"{step['id']}.npz")
    graph.save(path, A)
    return path, treat, A, extra


def realize_fast(step, cache, seed):
    rng = np.random.default_rng(seed)
    A = generators.connected_watts_strogatz(step['n'], step['k'], step['p'], rng)
    treat = None
    if step['treat'] is not None:
        kind, N = step['treat']
        treat = (np.sort(rng.choice(step['n'], N, replace=False)) if kind == 'random'
                 else np.arange(N, dtype=np.int64))
    extra = {}
    if step.get('layout'):
        g = nx.Graph()
        g.add_nodes_from(range(step['n']))
        coo = A.to_scipy().tocoo()
        g.add_edges_from((int(a), int(b)) for a, b in zip(coo.row, coo.col) if a < b)
        pos = nx.spring_layout(g, seed=int(rng.integers(0, 2**31 - 1)))
        extra['pos'] = {int(a): b.tolist() for a, b in pos.items()}
        extra['small_graph'] = [list(e) for e in g.edges()]
    path = str(cache / f"{step['id']}.npz")
    graph.save(path, A)
    return path, treat, A, extra


def make_task(step, sim, path, treat, seed=None):
    return dict(graph=path, T=T, nu=NU, lam_int=1.0, dlam=step.get('dlam', 0.0),
                treat=treat, sampler=sim.get('sampler', 'mean'),
                nu_fn=sim.get('nu_fn', 'linear'), groups=sim.get('groups', False),
                keep_final=sim.get('keep_final', False), seed=seed)


def run(mode, workers):
    out = ROOT / 'out' / mode
    cache = ROOT / 'cache' / mode
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plan = steps()
    futures, info = {}, {}
    chain = []
    with runner.pool(workers) as ex:
        if mode == 'exact':
            random.seed(SEED)
            np.random.seed(SEED)
        for i, step in enumerate(plan):
            if mode == 'exact':
                path, treat, A, extra = realize_exact(step, cache)
            else:
                path, treat, A, extra = realize_fast(step, cache, seed=[SEED, i])
            info[step['id']] = dict(extra, degrees=np.bincount(graph.degrees(A)).tolist())
            if step.get('layout'):
                info[step['id']]['treat_random'] = treat.tolist()
            if step.get('rest_degree'):
                grp = metrics.groups(A, np.isin(np.arange(A.shape[0]), treat))
                info[step['id']]['rest_degree'] = float(graph.degrees(A)[grp['rest']].mean())
            for j, sim in enumerate(step['sims']):
                key = (step['id'], sim['name'])
                if mode == 'exact' and sim.get('sampler', 'mean') != 'mean':
                    chain.append((key, make_task(step, sim, path, treat)))
                else:
                    futures[key] = ex.submit(runner.run_sim,
                                             make_task(step, sim, path, treat, seed=[SEED, i, j]))
            if mode == 'exact' and step['id'] == 'c21':
                # all noisy runs are known now; numpy's stream is where cell 2 left it
                tasks = [t for _, t in chain]
                tasks[0]['rng_state'] = np.random.get_state()
                chain_future = ex.submit(runner.run_chain, tasks)
        t_graphs = time.time() - t0
        res = {k: f.result() for k, f in futures.items()}
        if mode == 'exact':
            for (key, _), rec in zip(chain, chain_future.result()):
                res[key] = rec
    t_total = time.time() - t0
    return plan, res, info, dict(graphs_s=t_graphs, total_s=t_total)


# ---------------------------------------------------------------------------
# 2023 model_stats() outputs and figures
# ---------------------------------------------------------------------------

def stats_for(res, step_id, c_base_map, p, k):
    rec = res[(step_id, 'mean')]
    final = {g: v[-1] for g, v in rec['groups'].items()}
    return metrics.stats(rec['counts'], final, c_base_map[(p, k, NU)], rec['k_mean'],
                         NU, 1.0, 0.05)


def printed_numbers(d):
    """The numbers model_stats() printed, in print order."""
    return [d['treatment'] + d['control'], d['treatment'], d['control'], d['neighbours'],
            d['rest'], d['c_base'], d['c_base_prime'], d['c_graph'], d['c_treatment'],
            d['c_control'], d['c_neighbours'], d['c_rest'],
            d['c_graph'] / d['c_base_prime'], d['c_treatment'] / d['c_base_prime'],
            d['c_control'] / d['c_base_prime'], d['c_neighbours'] / d['c_base_prime'],
            d['c_rest'] / d['c_base_prime'], d['e_spillover'], d['e_treatment'],
            d['e_dampening'], d['e_intrinsic']]


def collect(plan, res, info):
    by_id = {s['id']: s for s in plan}
    c_base_map = {}
    for s in plan:
        if s['id'].startswith('c04'):
            c_base_map[(s['p'], s['k'], NU)] = res[(s['id'], 'mean')]['mean'][-1]
    cells = {}
    for cid in ('c14', 'c17', 'c21', 'c39', 'c40'):
        s = by_id[cid]
        cells[cid] = [stats_for(res, cid, c_base_map, s['p'], s['k'])]
    cells['c26'] = [stats_for(res, f'c26_p{p}', c_base_map, p, 50)
                    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    cells['c32'] = [stats_for(res, f'c32_N{N}_k{k}', c_base_map, 0.1, k)
                    for N, k in product([100, 500, 1000, 5000, 9000], [10, 20, 30, 40, 50])]
    cells['c36'] = [stats_for(res, f'c36_N{N}_p{p}', c_base_map, p, 50)
                    for N, p in product([100, 500, 1000, 5000, 9000], [0.1, 0.2, 0.3, 0.4, 0.5])]
    k_prime = info['c43']['rest_degree']
    scalars = {
        'c44': [k_prime],
        'c45': [1 / (1 - k_prime * NU)],  # the cell displays only its last expression
        'c46': [res[('c42', 'mean')]['mean'][-1], c_base_map[(0.1, 50, NU)]],
        'c47': [c_base_map[(0.1, 10, NU)]],
    }
    return c_base_map, cells, scalars


def notebook_numbers():
    """Numbers printed by the executed 2023 notebook, per cell, in print order."""
    nb = json.load(open(NOTEBOOK))
    num = re.compile(r'-?\d+\.\d+(?:e-?\d+)?|-?\d+(?:e-?\d+)?|nan')

    def text(i):
        parts = []
        for o in nb['cells'][i].get('outputs', []):
            if o.get('name') == 'stderr':
                continue
            if 'text' in o:
                parts.append(''.join(o['text']))
            elif 'data' in o and 'text/plain' in o['data'] and 'image/png' not in o['data']:
                parts.append(''.join(o['data']['text/plain']))
        return '\n'.join(parts)

    def stat_blocks(i):
        blocks, cur = [], None
        for line in text(i).split('\n'):
            if line.startswith('Total nodes:'):
                cur = []
                blocks.append(cur)
            if cur is None or line.startswith('Simulation at') or ':' not in line and '=' not in line:
                continue
            if line.startswith('(') or line.startswith('np.'):
                continue
            m = num.findall(line.split(':', 1)[-1] if ':' in line else line.split('=', 1)[1])
            cur.extend(float(x) for x in m)
        return blocks

    def scalars(i):
        t = text(i).replace('np.float64(', '')
        return [float(x) for x in num.findall(t)]

    cells = {}
    for name, i in (('c14', 16), ('c17', 19), ('c21', 23), ('c26', 26), ('c32', 32),
                    ('c36', 36), ('c39', 39), ('c40', 40)):
        cells[name] = stat_blocks(i)
    sc = {name: scalars(i) for name, i in (('c44', 44), ('c45', 45), ('c46', 46), ('c47', 47))}
    return cells, sc


def compare(cells, scalars):
    nb_cells, nb_sc = notebook_numbers()
    rows, worst = [], 0.0

    def rel(a, b):
        if np.isnan(a) and np.isnan(b):
            return 0.0
        if a == b:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b))

    for name, ours in cells.items():
        theirs = nb_cells[name]
        assert len(theirs) == len(ours), (name, len(theirs), len(ours))
        r = max(rel(a, b) for d, nb in zip(ours, theirs)
                for a, b in zip(printed_numbers(d), nb))
        assert all(len(printed_numbers(d)) == len(nb) for d, nb in zip(ours, theirs)), name
        rows.append((name, sum(len(b) for b in theirs), r))
        worst = max(worst, r)
    for name, ours in scalars.items():
        theirs = nb_sc[name]
        assert len(theirs) == len(ours), (name, theirs, ours)
        r = max(rel(a, b) for a, b in zip(ours, theirs))
        rows.append((name, len(theirs), r))
        worst = max(worst, r)
    return rows, worst


def figures(plan, res, info, c_base_map, cells, out):
    fig_dir = out / 'figures'

    def save(name):
        plt.savefig(fig_dir / name, dpi=100, bbox_inches='tight')
        plt.close('all')

    # Figures 1 and 12: the small graph with random and contiguous treatment groups
    c02 = info['c02']
    g = nx.Graph()
    g.add_nodes_from(range(100))
    g.add_edges_from(c02['small_graph'])
    pos = {int(k): np.array(v) for k, v in c02['pos'].items()}
    for name, members in (('figure-1.png', set(c02['treat_random'])),
                          ('figure-12.png', set(range(20)))):
        plt.figure()
        nx.draw(g, pos, node_color=['blue' if v in members else 'red' for v in g],
                node_size=50)
        save(name)

    # Figures 2, 4, 5, 6: mean content production over time, three samplers
    for name, cid in (('figure-2.png', 'c05'), ('figure-4.png', 'c08'),
                      ('figure-5.png', 'c10'), ('figure-6.png', 'c12')):
        x = np.arange(T)
        plt.figure(figsize=(10, 5))
        plt.xlabel('t')
        plt.ylabel('mean content production')
        plt.plot(x, res[(cid, 'mean')]['mean'], 'r-')
        plt.plot(x, res[(cid, 'uniform')]['mean'], 'g--')
        plt.plot(x, res[(cid, 'poisson')]['mean'], 'b:')
        plt.legend(['constant', 'uniform', 'poisson'], loc='lower right')
        save(name)

    # Figure 3: distribution of content production at the last step
    plt.figure()
    plt.yscale('log')
    for smp, col in (('uniform', 'g'), ('poisson', 'b'), ('mean', 'r')):
        plt.hist(res[('c05', smp)]['final'], bins=20, density=1, color=col)
    plt.xlabel('content production')
    plt.ylabel('density')
    plt.legend(['uniform', 'poisson', 'constant'], loc='lower right')
    save('figure-3.png')

    # Figure 7: degree distributions
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.figure(figsize=(10, 5))
    plt.xlabel('degree')
    plt.ylabel('node count')
    for p in ps:
        counts = np.array(info[f'c26_p{p}']['degrees'])
        d = np.flatnonzero(counts)[::-1]
        plt.plot(d, counts[d], marker='.')
    plt.legend(['p=%s' % p for p in ps])
    save('figure-7.png')

    # Figure 8: degree distribution effect relative to Equation (2)
    plt.figure(figsize=(10, 5))
    plt.xlabel('p')
    plt.ylabel('e_degree_distribution')
    c_rest = [d['c_rest'] for d in cells['c26']]
    plt.plot([0] + ps, [0] + [c / (1 / (1 - 50 * 0.01)) - 1 for c in c_rest], marker='o')
    save('figure-8.png')

    # Figure 9: content production as a function of degree
    plt.figure(figsize=(10, 5))
    plt.xlabel('degree')
    plt.ylabel('mean content production')
    for p in (0.9, 0.1):
        A = graph.load(str(ROOT / 'cache' / out.name / f'c30_p{p}.npz'))
        deg = graph.degrees(A)
        final = res[(f'c30_p{p}', 'mean')]['final']
        ds = np.unique(deg)
        plt.plot(ds, [final[deg == d].mean() for d in ds], marker='o')
    plt.axhline(y=2.0, color='r', linestyle='--')
    plt.axvline(x=50, color='r', linestyle='--', label='mean degree')
    plt.legend(['p=0.9', 'p=0.1', 'base'], loc='upper left')
    save('figure-9.png')

    # Figures 10, 11, 13: group means over time, steps 5..24
    for name, cid in (('figure-10.png', 'c14'), ('figure-11.png', 'c17'),
                      ('figure-13.png', 'c21')):
        plt.figure(figsize=(10, 5))
        plt.xlabel('t')
        plt.ylabel('mean content production')
        for smp, style in (('mean', '-'), ('uniform', '--'), ('poisson', ':')):
            gr = res[(cid, smp)]['groups']
            plt.plot(gr['treatment'][5:], 'b' + style)
            plt.plot(gr['neighbours'][5:], 'g' + style)
            plt.plot(gr['rest'][5:], 'r' + style)
        plt.legend(['treatment', 'neighbours', 'rest'], loc='upper right')
        plt.xticks([])
        save(name)

    # Figures 14 and 15, and the supplementary p-comparison plots
    Ns = [100, 500, 1000, 5000, 9000]
    x = [N / 10_000 for N in Ns]
    ks = [10, 20, 30, 40, 50]
    by = {(N, k): d for (N, k), d in zip(product(Ns, ks), cells['c32'])}
    for name, key, ylab, ylim in (('figure-15.png', 'e_dampening', 'e_dampening', (0.0, 1.0)),
                                  ('figure-14.png', 'e_intrinsic', 'e_intrinsic', (0.0, 0.055))):
        plt.figure(figsize=(10, 5))
        plt.xlabel('N/n')
        plt.ylabel(ylab)
        for k in ks:
            plt.plot(x, [by[(N, k)][key] for N in Ns], marker='o')
        legend = [f'k={k}' for k in ks]
        if key == 'e_intrinsic':
            plt.plot(x, [0.05 for _ in Ns], 'r--')
            legend.append('Δλ=0.05')
        plt.legend(legend)
        plt.ylim(ylim)
        save(name)
    ps5 = [0.1, 0.2, 0.3, 0.4, 0.5]
    byp = {(N, p): d for (N, p), d in zip(product(Ns, ps5), cells['c36'])}
    for name, key, ylim in (('supp-dampening-by-p.png', 'e_dampening', (0.0, 1.0)),
                            ('supp-intrinsic-by-p.png', 'e_intrinsic', (0.0, 0.05))):
        plt.figure(figsize=(10, 5))
        plt.xlabel('N/n')
        plt.ylabel(key)
        for p in ps5:
            plt.plot(x, [byp[(N, p)][key] for N in Ns], marker='o')
        plt.legend([f'p={p}' for p in ps5])
        plt.ylim(ylim)
        save(name)


# figure file -> (notebook cell, image index) in the executed 2023 notebook
FIGURE_CELLS = {'figure-1.png': (2, 0), 'figure-12.png': (2, 1), 'figure-2.png': (6, 0),
                'figure-3.png': (7, 0), 'figure-4.png': (9, 0), 'figure-5.png': (11, 0),
                'figure-6.png': (13, 0), 'figure-7.png': (27, 0), 'figure-8.png': (29, 0),
                'figure-9.png': (31, 0), 'figure-10.png': (15, 0), 'figure-11.png': (18, 0),
                'figure-13.png': (22, 0), 'figure-14.png': (35, 0), 'figure-15.png': (34, 0),
                'supp-dampening-by-p.png': (37, 0), 'supp-intrinsic-by-p.png': (38, 0)}


def compare_figures(out):
    import base64
    import io
    import matplotlib.image as mpimg
    nb = json.load(open(NOTEBOOK))
    rows = []
    for name, (cell, k) in FIGURE_CELLS.items():
        imgs = [o['data']['image/png'] for o in nb['cells'][cell].get('outputs', [])
                if 'data' in o and 'image/png' in o['data']]
        theirs = mpimg.imread(io.BytesIO(base64.b64decode(imgs[k])), format='png')
        ours = mpimg.imread(out / 'figures' / name)
        if ours.shape != theirs.shape:
            rows.append((name, f'size differs {ours.shape[:2]} vs {theirs.shape[:2]}'))
            continue
        diff = np.abs(ours.astype(float) - theirs.astype(float))
        frac = float((diff.max(axis=2) > 0).mean())
        rows.append((name, 'identical' if frac == 0 else f'{100 * frac:.3f}% of pixels differ'))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['exact', 'fast'], required=True)
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()
    out = ROOT / 'out' / args.mode
    plan, res, info, timing = run(args.mode, args.workers)
    c_base_map, cells, scalars = collect(plan, res, info)
    t_fig = time.time()
    figures(plan, res, info, c_base_map, cells, out)
    timing['figures_s'] = time.time() - t_fig
    lines = [f'# Reproduction of Trencseni 2023, mode = {args.mode}', '',
             f"Graph construction (serial part): {timing['graphs_s']:.1f} s; "
             f"all simulations done: {timing['total_s']:.1f} s; "
             f"figures: {timing['figures_s']:.1f} s.", '']
    rows, worst = compare(cells, scalars)
    lines += ['| notebook cell | numbers compared | max relative difference |',
              '|---|---|---|']
    lines += [f'| {n} | {c} | {r:.2e} |' for n, c, r in rows]
    lines += ['', f'Largest relative difference: {worst:.2e}', '',
              '| figure | vs executed 2023 notebook |', '|---|---|']
    lines += [f'| {n} | {r} |' for n, r in compare_figures(out)]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    json.dump({'timing': timing,
               'c_base_map': {f'{p},{k}': v for (p, k, _), v in c_base_map.items()},
               'cells': {n: [{k: v for k, v in d.items() if k != 'tuple'} for d in ds]
                         for n, ds in cells.items()},
               'scalars': scalars},
              open(out / 'results.json', 'w'), indent=1, default=float)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
