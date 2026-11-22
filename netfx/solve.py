"""Exact steady states and spectra for c = b + nu A c.

cg():      (I - nu A) X = B for symmetric A, all columns at once (block of
           independent conjugate-gradient iterations sharing one matrix product).
neumann(): the same by fixed-point iteration, for directed A (nu rho(A) < 1).
"""
import numpy as np
import scipy.sparse.linalg as spla


class Unstable(RuntimeError):
    """The model has no steady state: nu * rho(A) >= 1."""


def cg(A, B, nu, tol=1e-10, maxit=100_000):
    vec = B.ndim == 1
    B = np.ascontiguousarray(B, dtype=np.float64).reshape(B.shape[0], -1)
    X = B.copy()                       # start from the nu = 0 solution
    R = B - (X - nu * (A @ X))
    P = R.copy()
    rs = (R * R).sum(axis=0)
    bnorm = np.sqrt((B * B).sum(axis=0))
    for it in range(maxit):
        if np.all(np.sqrt(rs) <= tol * bnorm):
            break
        AP = P - nu * (A @ P)
        pAp = (P * AP).sum(axis=0)
        if np.any(pAp <= 0):
            raise Unstable('I - nu A is not positive definite: nu * rho(A) >= 1')
        alpha = rs / pAp
        X += alpha * P
        R -= alpha * AP
        rs_new = (R * R).sum(axis=0)
        P = R + (rs_new / np.where(rs == 0, 1, rs)) * P
        rs = rs_new
    else:
        raise RuntimeError('cg did not converge')
    return X[:, 0] if vec else X


def neumann(A, B, nu, tol=1e-10, maxit=1_000_000):
    """X = B + nu A X by iteration; returns X and the number of steps."""
    X = np.array(B, dtype=np.float64)
    for it in range(maxit):
        Xn = B + nu * (A @ X)
        if np.max(np.abs(Xn - X)) <= tol * max(1.0, np.max(np.abs(Xn))):
            return Xn, it + 1
        if not np.all(np.isfinite(Xn)) or np.max(np.abs(Xn)) > 1e100:
            raise Unstable('iteration diverges: nu * rho(A) >= 1')
        X = Xn
    raise RuntimeError('neumann did not converge')


def _op(A):
    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=lambda x: A @ x, dtype=np.float64)


def spectral_radius(A, return_vector=False, ncv=None, v0=None, tol=0):
    """Largest eigenvalue (Perron root) of A, and optionally its unit eigenvector.
    ncv: number of Lanczos vectors (scipy default about 20; a small value such as 6 keeps the memory
    of a 10^9-node solve in bounds at the cost of more iterations). v0: starting vector (for example
    the degree vector, which is close to the Perron vector)."""
    if getattr(A, 'directed', False):
        w, v = spla.eigs(_op(A), k=1, which='LR')
        w, v = float(w[0].real), np.abs(v[:, 0].real)
    else:
        w, v = spla.eigsh(_op(A), k=1, which='LA', ncv=ncv, v0=v0, tol=tol, maxiter=100_000)
        w, v = float(w[0]), np.abs(v[:, 0])
    v = v / np.linalg.norm(v)
    return (w, v) if return_vector else w


def lanczos_radius(A, v0, tol=1e-10, maxit=5000, check=10):
    """Largest eigenvalue of a symmetric A by the Lanczos recursion without reorthogonalization.

    Keeps three vectors of length n (eigsh keeps about ncv + 6), so it fits a 10^9-node graph in memory.
    Loss of orthogonality adds spurious copies of converged Ritz values but does not change the largest
    one. Stops when the largest eigenvalue of the tridiagonal matrix changes by less than tol (relative)
    over `check` steps. Returns (rho, steps). No eigenvector."""
    from scipy.linalg import eigvalsh_tridiagonal
    q = np.array(v0, dtype=np.float64)
    q /= np.linalg.norm(q)
    q_prev = np.zeros_like(q)
    alpha, beta = [], []
    b, last = 0.0, None
    for j in range(maxit):
        w = A @ q
        a = float(q @ w)
        w -= a * q
        if b:
            w -= b * q_prev
        alpha.append(a)
        b = float(np.linalg.norm(w))
        if (j + 1) % check == 0 or b == 0:
            theta = eigvalsh_tridiagonal(np.array(alpha), np.array(beta), select='i',
                                         select_range=(j, j))[0] if j else alpha[0]
            if b == 0 or (last is not None and abs(theta - last) <= tol * abs(theta)):
                return float(theta), j + 1
            last = theta
        beta.append(b)
        q_prev, q = q, w
        q /= b
        del w
    raise RuntimeError('lanczos did not converge')
