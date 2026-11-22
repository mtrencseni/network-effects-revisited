import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.stats import poisson, uniform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from netfx import generators as G  # noqa: E402
from netfx import graph, metrics, model  # noqa: E402


def old_simulate(g, T, nu, dlam, treat, sampler):
    """The 2023 notebook's per-node simulation (step_posts), for comparison."""
    series = []
    for t in range(T):
        prev = [0] * g.number_of_nodes() if not series else series[-1]
        today = defaultdict(int)
        for i in g.nodes:
            lam = 1 + nu * sum(prev[j] for j in g.neighbors(i))
            lam = lam + dlam * (1 if i in treat else 0)
            if sampler == 'mean':
                today[i] = lam
            elif sampler == 'uniform':
                today[i] = uniform.rvs(0, 2 * lam)
            else:
                today[i] = poisson.rvs(lam)
        series.append(today)
    return np.array([series[-1][i] for i in range(g.number_of_nodes())])


@pytest.mark.parametrize('sampler', ['mean', 'uniform', 'poisson'])
def test_matches_2023_step_by_step(sampler):
    g = nx.connected_watts_strogatz_graph(300, 6, 0.2, seed=3)
    A = graph.from_networkx(g)
    treat = set(range(0, 300, 7))
    np.random.seed(11)
    old = old_simulate(g, 12, 0.1, 0.05, treat, sampler)
    z = np.zeros(300, dtype=bool)
    z[list(treat)] = True
    rng = np.random.RandomState(11)
    new = model.simulate(A, 12, nu=0.1, dlam=0.05, treat=z, sampler=sampler, rng=rng,
                         keep_final=True)['final']
    assert np.allclose(old, new, rtol=1e-12, atol=0)


def test_regular_graph_closed_form():
    A = graph.from_networkx(nx.circulant_graph(1000, [1, 2, 3, 4, 5]))  # 10-regular
    r = model.simulate(A, 200, nu=0.05)
    assert r['mean'][-1] == pytest.approx(1 / (1 - 10 * 0.05), rel=1e-12)


def test_steady_state_equals_iteration():
    A = G.connected_watts_strogatz(2000, 10, 0.3, seed=1)
    b = 1 + 0.05 * (np.arange(2000) % 3 == 0)
    ss = model.steady_state(A, b, 0.04)
    c = np.zeros(2000)
    for _ in range(2000):
        c = b + 0.04 * (A @ c)
    assert np.allclose(ss, c, rtol=1e-10)


def test_stability_uses_spectral_radius():
    A = G.barabasi_albert(3000, 5, seed=2)
    k_mean = A.nnz / 3000
    rho = model.spectral_radius(A)
    assert rho > 1.5 * k_mean


def test_groups():
    g = nx.path_graph(6)
    A = graph.from_networkx(g)
    z = np.zeros(6, dtype=bool)
    z[[0, 1]] = True
    grp = metrics.groups(A, z)
    assert grp['neighbours'].tolist() == [False, False, True, False, False, False]
    assert grp['rest'].tolist() == [False, False, False, True, True, True]


def test_ws_matches_networkx_in_distribution():
    n, k, p = 20_000, 20, 0.3
    ours = [graph.degrees(G.watts_strogatz(n, k, p, seed=s)).var() for s in range(5)]
    theirs = [graph.degrees(graph.from_networkx(nx.watts_strogatz_graph(n, k, p, seed=s))).var()
              for s in range(5)]
    assert np.mean(ours) == pytest.approx(np.mean(theirs), rel=0.05)
    A = G.watts_strogatz(n, k, p, seed=0)
    S = A.to_scipy()
    assert A.nnz == n * k and (S != S.T).nnz == 0 and S.diagonal().sum() == 0


def test_erdos_renyi():
    A = G.erdos_renyi(50_000, 12.0, seed=1)
    assert A.nnz / 50_000 == pytest.approx(12.0, rel=0.02)
    assert A.to_scipy().diagonal().sum() == 0


def test_barabasi_albert():
    n, m = 20_000, 4
    A = G.barabasi_albert(n, m, seed=1)
    assert A.nnz == 2 * (n - m) * m
    assert graph.degrees(A).max() > 20 * m


def test_configuration():
    deg = G.power_law_degrees(50_000, 2.5, 3, seed=1)
    A = G.configuration(deg, seed=2)
    assert A.to_scipy().diagonal().sum() == 0
    assert A.nnz / 50_000 == pytest.approx(deg.mean(), rel=0.05)


def test_sbm():
    A = G.sbm([5000, 5000], [[0.002, 0.0002], [0.0002, 0.002]], seed=1).to_scipy()
    inside = A[:5000, :5000].nnz + A[5000:, 5000:].nnz
    across = A[:5000, 5000:].nnz * 2
    assert inside / across == pytest.approx(10 * 4999 / 5000 / 2 * 2, rel=0.1)
