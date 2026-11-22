"""Graphs as compressed sparse rows without a value array.

All graphs are unweighted, so a graph is just (indptr, indices): row i lists the
nodes whose content node i sees. Undirected graphs store both directions.
Matrix products run in numba-compiled, multi-threaded kernels, which keeps a
10^8-node graph with mean degree 10 at about 4 GB (int32 indices) plus 0.8 GB
(int64 indptr).

Graph supports A @ x for vectors and (n, m) matrices, .shape, .nnz, .indptr,
.indices, so it can stand in for a scipy CSR matrix in this package.
"""
import os

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@njit(parallel=True, cache=True)
def _matvec(indptr, indices, x, out):
    for i in prange(indptr.size - 1):
        s = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            s += x[indices[j]]
        out[i] = s


@njit(parallel=True, cache=True)
def _matmat(indptr, indices, X, out):
    m = X.shape[1]
    for i in prange(indptr.size - 1):
        for c in range(m):
            out[i, c] = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            r = indices[j]
            for c in range(m):
                out[i, c] += X[r, c]


@njit(cache=True)
def _scatter(n, u, v, symmetric):
    deg = np.zeros(n + 1, dtype=np.int64)
    for e in range(u.size):
        deg[u[e] + 1] += 1
        if symmetric:
            deg[v[e] + 1] += 1
    for i in range(n):
        deg[i + 1] += deg[i]
    fill = deg[:-1].copy()
    indices = np.empty(deg[n], dtype=np.int32)
    for e in range(u.size):
        indices[fill[u[e]]] = v[e]
        fill[u[e]] += 1
        if symmetric:
            indices[fill[v[e]]] = u[e]
            fill[v[e]] += 1
    return deg, indices


@njit(parallel=True, cache=True)
def _dedup(indptr, indices):
    """Sort each row in place; count entries that are neither duplicates nor self-loops."""
    n = indptr.size - 1
    cnt = np.zeros(n + 1, dtype=np.int64)
    for i in prange(n):
        row = indices[indptr[i]:indptr[i + 1]]
        row.sort()
        last = -1
        c = 0
        for x in row:
            if x != last and x != i:
                c += 1
            last = x
        cnt[i + 1] = c
    return cnt


@njit(parallel=True, cache=True)
def _compact(indptr, indices, out_ptr, out):
    n = indptr.size - 1
    for i in prange(n):
        w = out_ptr[i]
        last = -1
        for j in range(indptr[i], indptr[i + 1]):
            x = indices[j]
            if x != last and x != i:
                out[w] = x
                w += 1
            last = x


def _build(n, u, v, symmetric):
    """CSR rows from edge list (u -> v; both directions if symmetric), sorted, deduplicated."""
    indptr, indices = _scatter(n, u, v, symmetric)
    cnt = _dedup(indptr, indices)
    out_ptr = np.cumsum(cnt)
    out = np.empty(out_ptr[-1], dtype=np.int32)
    _compact(indptr, indices, out_ptr, out)
    return out_ptr, out


@njit(cache=True)
def _components(indptr, indices):
    n = indptr.size - 1
    label = np.full(n, -1, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    c = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        top = 0
        stack[top] = s
        label[s] = c
        while top >= 0:
            x = stack[top]
            top -= 1
            for j in range(indptr[x], indptr[x + 1]):
                y = indices[j]
                if label[y] < 0:
                    label[y] = c
                    top += 1
                    stack[top] = y
        c += 1
    return c, label


class Graph:
    def __init__(self, indptr, indices, directed=False):
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.n = self.indptr.size - 1
        self.directed = directed

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def nnz(self):
        return int(self.indptr[-1])

    def __matmul__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim == 1:
            out = np.empty(self.n)
            _matvec(self.indptr, self.indices, x, out)
        else:
            out = np.empty((self.n, x.shape[1]))
            _matmat(self.indptr, self.indices, x, out)
        return out

    def degrees(self):
        return np.diff(self.indptr)

    def components(self):
        """Number of connected components and a label per node (undirected graphs)."""
        return _components(self.indptr, self.indices)

    def to_scipy(self):
        data = np.ones(self.nnz, dtype=np.float64)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def transpose(self):
        T = self.to_scipy().T.tocsr()
        T.sort_indices()
        return Graph(T.indptr, T.indices, self.directed)


def from_edges(n, u, v, directed=False):
    """Graph from edge endpoints; undirected unless directed (then u sees v)."""
    indptr, indices = _build(n, np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64),
                             not directed)
    return Graph(indptr, indices, directed)


def from_scipy(A, directed=False):
    A = A.tocsr()
    A.sort_indices()
    return Graph(A.indptr, A.indices, directed)


def from_networkx(g):
    """Graph of an undirected networkx graph with nodes 0..n-1."""
    n = g.number_of_nodes()
    nodes = np.fromiter(g.nodes, dtype=np.int64, count=n)
    assert np.array_equal(nodes, np.arange(n)), 'nodes must be 0..n-1 in order'
    m = g.number_of_edges()
    e = np.fromiter((x for edge in g.edges() for x in edge), dtype=np.int64, count=2 * m)
    return from_edges(n, e[0::2], e[1::2])


def degrees(A):
    return np.diff(A.indptr)


def save(path, A):
    tmp = path + '.tmp.npz'
    np.savez(tmp, indptr=A.indptr, indices=A.indices, directed=getattr(A, 'directed', False))
    os.replace(tmp, path)


def load(path):
    f = np.load(path)
    return Graph(f['indptr'], f['indices'], bool(f['directed']))
