"""Content production on a graph: c^t = R(lambda_int + Delta-lambda z + nu A c^{t-1}).

The 2023 notebook stepped through nodes one at a time in Python. Here one time
step is a single sparse matrix-vector product, and the random samplers draw one
value per node with a single numpy call. The draws are made in node order, so a
legacy numpy RandomState in the same state produces the same values as the
2023 per-node scipy.stats calls.
"""
import numpy as np
from scipy.special import expit


def simulate(A, T, lam_int=1.0, nu=0.01, dlam=0.0, treat=None, sampler='mean',
             nu_fn='linear', nu_max=1.0, rng=None, groups=None, keep_final=False):
    """Run T steps from c = 0 and record per-step statistics.

    treat:   boolean mask of treatment nodes (or None)
    sampler: 'mean' (c = lambda), 'uniform' (U(0, 2 lambda)), 'poisson'
    nu_fn:   'linear' (lambda_int + nu * sum) or 'sigmoid'
             (lambda_int + nu_max * expit(nu * sum)), the 2023 regularized model
    rng:     numpy RandomState or Generator, required for the random samplers
    groups:  dict name -> boolean mask; the mean over each group is recorded per step

    Returns a dict with per-step 'mean' and 'std' over all nodes, per-step
    'groups' means, and the final vector if keep_final.
    """
    n = A.shape[0]
    boost = None
    if treat is not None and dlam != 0.0:
        boost = dlam * treat.astype(np.float64)
    groups = groups or {}
    rec = {'mean': np.empty(T), 'std': np.empty(T),
           'groups': {g: np.empty(T) for g in groups}}
    c = np.zeros(n)
    for t in range(T):
        s = A @ c
        if nu_fn == 'linear':
            lam = lam_int + nu * s
        elif nu_fn == 'sigmoid':
            lam = lam_int + nu_max * expit(nu * s)
        else:
            raise ValueError(nu_fn)
        if boost is not None:
            lam = lam + boost
        if sampler == 'mean':
            c = lam
        elif sampler == 'uniform':
            c = rng.uniform(0.0, 1.0, n) * (2 * lam)
        elif sampler == 'poisson':
            c = rng.poisson(lam).astype(np.float64)
        else:
            raise ValueError(sampler)
        rec['mean'][t] = c.mean()
        rec['std'][t] = c.std()
        for g, mask in groups.items():
            rec['groups'][g][t] = c[mask].mean() if mask.any() else np.nan
    if keep_final:
        rec['final'] = c
    return rec


def steady_state(A, b, nu, tol=1e-10):
    """Exact steady state c = (I - nu A)^{-1} b of the linear mean model (nu rho(A) < 1)."""
    from . import solve
    if getattr(A, 'directed', False):
        return solve.neumann(A, b, nu, tol)[0]
    return solve.cg(A, b, nu, tol)


def spectral_radius(A):
    from . import solve
    return solve.spectral_radius(A)
