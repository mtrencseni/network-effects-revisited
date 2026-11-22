"""Ground truth and estimators of the global average treatment effect (GATE).

Model: steady state c = (I - nu A)^{-1} (lam + dlam z). The GATE is the mean
production with everyone treated minus with no one treated,
dlam * mean((I - nu A)^{-1} 1) for constant dlam.
"""
import numpy as np

from . import solve


def steady(A, b, nu):
    if getattr(A, 'directed', False):
        return solve.neumann(A, b, nu)[0]
    return solve.cg(A, b, nu)


def gate(A, nu, dlam, lam=None):
    n = A.shape[0]
    return dlam * steady(A, np.ones(n), nu).mean()


def dm(y, z):
    """Difference in means, treatment minus control."""
    return y[z].mean() - y[~z].mean()


def ratio_lift(y, z):
    """Relative lift c_T / c_C - 1, as experimentation platforms report it."""
    return y[z].mean() / y[~z].mean() - 1


def cuped(y, x, z):
    """Difference in means of y - theta (x - mean x), theta pooled over both arms."""
    xc = x - x.mean()
    theta = (xc * (y - y.mean())).sum() / (xc * xc).sum()
    return dm(y - theta * xc, z)


def neighbourhood_regression(y, z, A):
    """OLS of y on [1, z, f] with f the treated fraction of neighbours; the effect of
    moving from (z=0, f=0) to (z=1, f=1) estimates the GATE (linear exposure model)."""
    deg = np.diff(A.indptr).astype(np.float64)
    f = np.divide(A @ z.astype(np.float64), deg, out=np.zeros(deg.size), where=deg > 0)
    X = np.column_stack([np.ones(y.size), z.astype(np.float64), f])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1] + beta[2]


def neighbourhood_regression2(y, z, A):
    """OLS of y on [1, z, f1, f2], f1 the treated fraction of neighbours, f2 the treated
    fraction among walks of length two; GATE estimate is the (z, f1, f2) = (1, 1, 1) vs
    (0, 0, 0) contrast (Chin 2019 with two exposure features)."""
    zf = z.astype(np.float64)
    deg = np.diff(A.indptr).astype(np.float64)
    a1 = A @ zf
    f1 = np.divide(a1, deg, out=np.zeros(deg.size), where=deg > 0)
    w2 = A @ deg
    f2 = np.divide(A @ a1, w2, out=np.zeros(deg.size), where=w2 > 0)
    X = np.column_stack([np.ones(y.size), zf, f1, f2])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[1] + beta[2] + beta[3]


def two_fraction(y1, z1, y2, z2):
    """Slope of mean outcome in the treated fraction between two experiments; the mean
    outcome is linear in the treated fraction, so the slope is the GATE."""
    p1, p2 = z1.mean(), z2.mean()
    return (y2.mean() - y1.mean()) / (p2 - p1)


def plug_in(y, z, A, nu_true=None):
    """Estimate nu and dlam by two-stage least squares of y on [1, z, A y] with
    instruments [1, z, A z, A^2 z], then GATE = dlam_hat * mean((I - nu_hat A)^{-1} 1).
    Returns (gate_hat, nu_hat)."""
    zf = z.astype(np.float64)
    Ay = A @ y
    Az = A @ zf
    AAz = A @ Az
    X = np.column_stack([np.ones(y.size), zf, Ay])
    Z = np.column_stack([np.ones(y.size), zf, Az, AAz])
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    beta = np.linalg.lstsq(Xhat, y, rcond=None)[0]
    lam_hat, dlam_hat, nu_hat = beta
    try:
        g = dlam_hat * steady(A, np.ones(y.size), nu_hat).mean()
    except solve.Unstable:
        g = np.inf
    return g, nu_hat
