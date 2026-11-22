"""Resumable parameter sweeps.

Each configuration is a JSON-serializable dict. Results are appended to a JSONL
file as {'config': ..., 'result': ...}; configurations already in the file are
skipped, so an interrupted sweep continues where it stopped.
"""
import json
import os
import time
import traceback
from concurrent.futures import as_completed

from . import runner


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def done(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[key(r['config'])] = r['result']
    return out


def _call(fn, cfg):
    t = time.time()
    try:
        res = fn(cfg)
    except Exception as e:  # recorded, so one failure does not stop a sweep
        res = {'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}
    res['seconds'] = time.time() - t
    return res


def run(fn, configs, path, workers=None, inline=False, log=print):
    """Run fn on every configuration not yet in path. inline=True runs in this process
    (for large graphs, using all numba threads); otherwise a process pool."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = done(path)
    todo = [c for c in configs if key(c) not in have]
    log(f'{os.path.basename(path)}: {len(configs) - len(todo)} done, {len(todo)} to run')
    t0 = time.time()
    with open(path, 'a') as f:
        def write(cfg, res):
            f.write(json.dumps({'config': cfg, 'result': res}, default=_json) + '\n')
            f.flush()
            have[key(cfg)] = res
        if inline:
            for i, c in enumerate(todo):
                write(c, _call(fn, c))
                log(f'  {i + 1}/{len(todo)} {time.time() - t0:.0f}s')
        else:
            with runner.pool(workers) as ex:
                futs = {ex.submit(_call, fn, c): c for c in todo}
                for i, fut in enumerate(as_completed(futs)):
                    write(futs[fut], fut.result())
                    if (i + 1) % max(1, len(todo) // 20) == 0:
                        log(f'  {i + 1}/{len(todo)} {time.time() - t0:.0f}s')
    return [(c, have[key(c)]) for c in configs]


def _json(x):
    import numpy as np
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))
