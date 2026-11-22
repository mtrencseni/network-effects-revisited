"""Concave peer response: c = lam + dlam z + nu g(A c), g increasing, concave, g(0) = 0.

Responses:
  'linear'  g(s) = s                      (the model of the rest of the paper)
  'log'     g(s) = K log(1 + s/K)          (linear for s << K, logarithmic beyond; K = 1 gives log(1 + s))
  'sat'     g(s) = s / (1 + s/K)           (saturates at K; K -> infinity is linear)
K is the saturation scale, in units of content seen; both concave forms tend to the linear model as
K -> infinity, and the experiments set K relative to the typical exposure kbar * lam.

With a positive intercept and g(s)/s -> 0, the fixed point exists and is unique for every nu and
every graph (Kennan 2001; Belhaj, Bramoullé and Deroïan 2014), and plain iteration from c = lam
increases monotonically to it. Around the fixed point the response to a small change is linear with
the matrix J = nu diag(g'(s)) A, s = A c.
"""
import numpy as np
import scipy.sparse.linalg as spla

from .solve import Unstable


def g(s, kind, K=None):
    if kind == 'linear':
        return s
    if kind == 'log':
        K = 1.0 if K is None else K
        return K * np.log1p(s / K)
    if kind == 'sat':
        return s / (1 + s / K)
    raise ValueError(kind)


def gprime(s, kind, K=None):
    if kind == 'linear':
        return np.ones_like(s)
    if kind == 'log':
        K = 1.0 if K is None else K
        return 1 / (1 + s / K)
    if kind == 'sat':
        return 1 / (1 + s / K) ** 2
    raise ValueError(kind)


def steady(A, b, nu, kind, K=None, tol=1e-10, maxit=200_000):
    """Fixed point of c = b + nu g(A c) by iteration from c = b (monotone for b >= 0).
    b may be a vector or an (N, R) matrix of right-hand sides. Returns (c, iterations)."""
    c = np.array(b, dtype=np.float64)
    for it in range(maxit):
        cn = b + nu * g(A @ c, kind, K)
        if not np.all(np.isfinite(cn)) or cn.max() > 1e12:
            raise Unstable('iteration diverges')
        if np.max(np.abs(cn - c)) <= tol * max(1.0, float(np.max(np.abs(cn)))):
            return cn, it + 1
        c = cn
    raise RuntimeError('concave steady state did not converge')


def jacobian_radius(A, c, nu, kind, K=None):
    """Spectral radius of J = nu diag(g'(A c)) A, the local amplification at the fixed point.
    J is similar to the symmetric D^{1/2} A D^{1/2}, D = nu diag(g'), for symmetric A."""
    d = nu * gprime(A @ c, kind, K)
    sd = np.sqrt(d)
    n = A.shape[0]
    op = spla.LinearOperator((n, n), matvec=lambda x: sd * (A @ (sd * x)), dtype=np.float64)
    val = spla.eigsh(op, k=1, which='LA', tol=1e-6, maxiter=5000)[0][0]
    return float(val)


def linear_response(A, c, nu, kind, rhs, K=None, tol=1e-10, maxit=100_000):
    """x = rhs + J x, J = nu diag(g'(A c)) A: the first-order response of the fixed point to an
    intercept change rhs (rho(J) < 1 always holds at the fixed point of a concave model)."""
    d = nu * gprime(A @ c, kind, K)
    if rhs.ndim == 2:
        d = d[:, None]
    x = np.array(rhs, dtype=np.float64)
    for it in range(maxit):
        xn = rhs + d * (A @ x)
        if np.max(np.abs(xn - x)) <= tol * max(1.0, float(np.max(np.abs(xn)))):
            return xn
        x = xn
    raise RuntimeError('linear response did not converge')


# ----------------------------------------------------------------------------------------
# Annealed closed form (degree statistics only)
# ----------------------------------------------------------------------------------------

def annealed_baseline(nu, kind, deg_hist, lam=1.0, K=None, iters=10_000, tol=1e-13):
    """Annealed fixed point: a node of degree k sees s_k = k theta, theta the edge-end weighted
    mean production, so c_k = lam + nu g(k theta) and theta = sum_k k P(k) c_k / kbar.
    Returns (theta, ks, P, c_k)."""
    h = np.asarray(deg_hist, dtype=float)
    ks = np.arange(h.size, dtype=float)
    P = h / h.sum()
    kbar = (ks * P).sum()
    theta = lam
    for _ in range(iters):
        ck = lam + nu * g(ks * theta, kind, K)
        tn = (ks * P * ck).sum() / kbar
        if abs(tn - theta) < tol * max(1.0, tn):
            theta = tn
            break
        theta = tn
    ck = lam + nu * g(ks * theta, kind, K)
    return theta, ks, P, ck


def annealed_multiplier(nu, kind, deg_hist, lam=1.0, K=None):
    """Small-effect multiplier m = 1'(I - J)^{-1} 1 / N under the annealed adjacency A ~ k k'/(N kbar),
    with w_k = nu g'(k theta) the local peer effect of a degree-k node:
        m = 1 + <k w> / (1 - <k^2 w>/<k>).
    For g = K log(1 + s/K), k w_k = nu k/(1 + k theta/K) < nu K/theta, so a node's weight stays
    bounded as its degree grows (at fixed K, nu, theta) and the denominator stays away from 0 on
    every degree distribution; the bound alone does not bound the multiplier. Also returns the annealed
    local amplification rho_J ~ <k^2 w>/<k>, and the mean baseline production."""
    theta, ks, P, ck = annealed_baseline(nu, kind, deg_hist, lam, K)
    w = nu * gprime(ks * theta, kind, K)
    kbar = (ks * P).sum()
    rhoJ = (ks * ks * w * P).sum() / kbar
    if rhoJ >= 1:
        return np.inf, rhoJ, float((P * ck).sum())
    m = 1 + (ks * w * P).sum() / (1 - rhoJ)
    return float(m), float(rhoJ), float((P * ck).sum())
