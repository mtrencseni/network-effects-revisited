"""Run simulation tasks in parallel worker processes.

A task is a plain dict, so it pickles cheaply: graphs travel as file paths and
are loaded by the worker. See run_sim() for the fields.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import graph, metrics, model


def run_sim(task):
    """Run one simulation.

    task fields: graph (path), T, nu, lam_int, dlam, treat (index array or None),
    sampler, nu_fn, rng_state (legacy RandomState state, for random samplers),
    groups (bool: record treatment/neighbours/rest/control), keep_final.
    """
    A = graph.load(task['graph'])
    n = A.shape[0]
    treat = None
    if task.get('treat') is not None:
        treat = np.zeros(n, dtype=bool)
        treat[np.asarray(task['treat'], dtype=np.int64)] = True
    grp = None
    if task.get('groups'):
        grp = metrics.groups(A, treat if treat is not None else np.zeros(n, dtype=bool))
    rng = None
    if task.get('rng_state') is not None:
        rng = np.random.RandomState()
        rng.set_state(task['rng_state'])
    elif task.get('seed') is not None:
        rng = np.random.default_rng(task['seed'])
    rec = model.simulate(A, task['T'], lam_int=task.get('lam_int', 1.0), nu=task['nu'],
                         dlam=task.get('dlam', 0.0), treat=treat,
                         sampler=task.get('sampler', 'mean'), nu_fn=task.get('nu_fn', 'linear'),
                         rng=rng, groups=grp, keep_final=task.get('keep_final', False))
    if grp is not None:
        rec['counts'] = {g: int(m.sum()) for g, m in grp.items()}
    rec['k_mean'] = A.nnz / n
    if rng is not None and isinstance(rng, np.random.RandomState):
        rec['rng_state'] = rng.get_state()
    return rec


def run_chain(tasks):
    """Run tasks one after another, passing the legacy RandomState from each to the next."""
    out = []
    state = tasks[0]['rng_state']
    for t in tasks:
        t = dict(t, rng_state=state)
        rec = run_sim(t)
        state = rec['rng_state']
        out.append(rec)
    return out


def _single_thread():
    import numba
    numba.set_num_threads(1)


def pool(workers=None):
    """Process pool; each worker runs numba kernels on one thread."""
    return ProcessPoolExecutor(max_workers=workers or max(1, os.cpu_count() - 1),
                               initializer=_single_thread)
