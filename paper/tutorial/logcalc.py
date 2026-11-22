"""Numerics for the logarithmic-response section of tutorial.tex (three-user path graph)."""
import itertools
import numpy as np

A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
N = 3
lam, dl = 1.0, 0.1


def g(s, K):
    return K * np.log1p(s / K)


def gp(s, K):
    return 1.0 / (1.0 + s / K)


def solve(z, nu, K, tol=1e-15, maxit=100000, trace=False):
    c = np.full(N, lam)
    hist = [c.copy()]
    for t in range(maxit):
        cn = lam + dl * np.asarray(z, float) + nu * (A @ c if K is None else g(A @ c, K))
        hist.append(cn.copy())
        if np.max(np.abs(cn - c)) < tol:
            return (cn, hist) if trace else cn
        c = cn
    raise RuntimeError('no convergence')


def effects(nu, K):
    c0 = solve([0, 0, 0], nu, K)
    c1 = solve([1, 1, 1], nu, K)
    gate = (c1 - c0).mean()
    ade = np.mean([solve(np.eye(N)[i], nu, K)[i] - c0[i] for i in range(N)])
    dms = []
    for z in itertools.product([0, 1], repeat=N):
        if 0 < sum(z) < N:
            c = solve(z, nu, K)
            z = np.array(z, bool)
            dms.append(c[z].mean() - c[~z].mean())
    return c0, c1, gate, ade, np.mean(dms)


if __name__ == '__main__':
    nu, K = 0.25, 1.0
    c0, hist = solve([0, 0, 0], nu, K, trace=True)
    print('iteration z=0:', [np.round(h, 4).tolist() for h in hist[:6]], 'steps', len(hist))
    c0, c1, gate, ade, edm = effects(nu, K)
    print('c0', c0, 'c1', c1)
    for i in range(3):
        print('c(e%d)' % (i + 1), solve(np.eye(3)[i], nu, K))
    print('GATE %.5f ADE %.5f EDM %.5f formula %.5f' % (gate, ade, edm, ade - (gate - ade) / (N - 1)))
    s = A @ c0
    J = nu * np.diag(gp(s, K)) @ A
    MJ = np.linalg.inv(np.eye(3) - J)
    print('s', s, "g'(s)", gp(s, K), 'rhoJ', max(abs(np.linalg.eigvals(J))))
    print('J', np.round(J, 4)); print('MJ', np.round(MJ, 4))
    print('lin GATE %.5f ADE %.5f EDM %.5f' % (dl * MJ.sum() / N, dl * np.trace(MJ) / N,
                                               dl * (np.trace(MJ) / N - (MJ.sum() - np.trace(MJ)) / N / (N - 1))))
    # beyond the linear limit
    for nu2 in (0.25, 0.5, 0.7, 1.0, 2.0, 5.0):
        c0, c1, gate, ade, edm = effects(nu2, K)
        s = A @ c0; J = nu2 * np.diag(gp(s, K)) @ A
        print('nu %.2f nurho %.3f c0 %s GATE/dl %.4f ADE/dl %.4f EDM/GATE %.4f rhoJ %.4f 1-rhoJ %.4f' % (
            nu2, nu2 * np.sqrt(2), np.round(c0, 3), gate / dl, ade / dl, edm / gate,
            max(abs(np.linalg.eigvals(J))), 1 - max(abs(np.linalg.eigvals(J)))))
    # linear model comparison
    for nu2 in (0.25, 0.5, 0.7):
        c0, c1, gate, ade, edm = effects(nu2, None)
        print('linear nu %.2f GATE/dl %.4f EDM/GATE %.4f' % (nu2, gate / dl, edm / gate))
