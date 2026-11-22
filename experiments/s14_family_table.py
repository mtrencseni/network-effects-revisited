"""Statistics for the family table of Section 3 (Table [families]): N = 10^5, kbar = 20, seeds 0 to 4 (the graphs of Table [spectra]).

clustering: global clustering coefficient (transitivity), 3 x triangles / connected triples.
inside: share of edges inside METIS clusters of 1,000 nodes (100 clusters).
cv: degree standard deviation / mean degree; kmax: largest degree; rho/kbar.
Output: results/s14_family_table.json
"""
import json

import numpy as np

from common import FAMILIES, RESULTS, spec
from netfx import designs, families, solve

out = {}
for fam in FAMILIES:
    rows = []
    for seed in range(5):
        A = families.make(spec(fam, seed=seed))
        S = A.to_scipy().tocsr().astype(np.float64)
        d = np.asarray(S.sum(axis=1)).ravel()
        tri6 = (S @ S).multiply(S).sum()          # 6 x triangles
        triples = (d * (d - 1)).sum()               # 2 x connected triples
        lab = designs.metis(A, A.shape[0] // 1000, seed=seed)
        r, c = S.nonzero()
        rows.append(dict(clustering=float(tri6 / triples), inside=float((lab[r] == lab[c]).mean()),
                         cv=float(d.std() / d.mean()), kmax=int(d.max()), kbar=float(d.mean()),
                         rho_kbar=float(solve.spectral_radius(A) / d.mean())))
    out[fam['family']] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    print(fam['family'], {k: round(v, 3) for k, v in out[fam['family']].items()}, flush=True)
json.dump(out, open(RESULTS / 's14_family_table.json', 'w'), indent=1)
