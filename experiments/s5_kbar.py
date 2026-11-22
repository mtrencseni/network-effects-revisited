"""Section 5.4: mean-degree correction to the attenuation ratio, exact for small graphs.

E[DM]/GATE = (N tr M - 1'M1) / ((N - 1) 1'M1) with M = (I - nu A)^{-1} computed densely,
on Erdos-Renyi graphs with N = 3000 at nu rho = 0.5.
"""
import numpy as np

from common import RESULTS
from netfx import generators, solve, sweep


def kbar(cfg):
    A = generators.erdos_renyi(cfg['n'], cfg['kbar'], seed=cfg['seed'])
    rho = solve.spectral_radius(A)
    M = np.linalg.inv(np.eye(cfg['n']) - cfg['nu_rho'] / rho * A.to_scipy().toarray())
    N = cfg['n']
    return dict(ratio=float((N * np.trace(M) - M.sum()) / ((N - 1) * M.sum())),
                one_minus_nu_rho=1 - cfg['nu_rho'], tr_over_n=float(np.trace(M) / N))


if __name__ == '__main__':
    cfgs = [dict(n=3000, kbar=k, nu_rho=x, seed=s) for k in [5, 10, 20, 40]
            for x in [0.5, 0.9] for s in range(3)]
    sweep.run(kbar, cfgs, str(RESULTS / 's5_kbar.jsonl'), workers=8)
