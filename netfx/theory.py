"""Closed forms and controlled approximations for the model (paper Appendix A).

Model: c = M (lam + dlam z), M = (I - nu A)^{-1}, stable iff nu rho(A) < 1.
GATE = dlam 1'M1/N = dlam m, with m the multiplier; ADE = dlam tr(M)/N = dlam tau.

Notation used throughout
  kbar   mean degree <k>
  k2     second moment <k^2>
  kappa  <k^2>/<k>, the mean degree of a neighbour (annealed spectral radius)
  b      branching factor of the local tree, the mean excess degree kappa - 1
  rho    spectral radius of A (measured, or approximated by rho_annealed)
  nu     peer effect; the experiments set nu = (nu rho) / rho with rho measured
  deg_hist  degree histogram (deg_hist[k] = number of nodes of degree k), optional;
            when given, node-level quantities are averaged over it

Every function is pure numpy; nothing here touches a graph. experiments/graph_stats.py
computes the graph statistics they need (results/graph_stats.json).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# ----------------------------------------------------------------------------------
# T1  multiplier and stability
# ----------------------------------------------------------------------------------


def kappa(kbar, k2):
    """Mean degree of a random neighbour, <k^2>/<k>."""
    return k2 / kbar


def rho_annealed(kbar, k2, kmax):
    """Spectral radius from the degree sequence: max(<k^2>/<k>, sqrt(k_max)).

    <k^2>/<k> is the Perron root of the annealed (Chung-Lu) adjacency k k'/(N kbar);
    sqrt(k_max) is the root of the star around the maximum-degree node. Chung, Lu and
    Vu (2003): for expected-degree graphs the larger of the two is the leading order of
    rho when the two terms are a power of log n apart; for power laws, sqrt(k_max) for
    exponent gamma > 2.5 and <k^2>/<k> for 2 < gamma < 2.5. Krivelevich and Sudakov
    (2003) give the max form for G(n, p); Castellano and Pastor-Satorras (2010) use it
    as a heuristic for configuration models. On the paper's power-law graph (gamma = 2.5,
    k_max = sqrt N) the two terms are within a factor of 2 and no theorem applies."""
    return max(k2 / kbar, np.sqrt(kmax))


def rho_hub(kmax, kappa_):
    """Hub eigenvalue, next order: the two-level tree around the hub whose neighbours
    have the mean excess degree kappa - 1 gives lam^2 = k_max + kappa - 1."""
    return np.sqrt(kmax + kappa_ - 1)


def multiplier_annealed(nu, kbar, kappa_or_rho):
    """1'M1/N under the rank-one (annealed) adjacency A ~ k k'/(N kbar):
    A^t 1 = k kappa^{t-1}, so 1'M1/N = 1 + nu kbar / (1 - nu kappa).

    Passing the measured rho for kappa gives the version used in the tables: it keeps
    the exact first-order term nu kbar and the exact pole at nu rho = 1."""
    x = nu * kappa_or_rho
    return np.where(x < 1, 1 + nu * kbar / (1 - x), np.inf)


def multiplier_mean_field(nu, kbar):
    """Regular-graph (mean-field) multiplier 1/(1 - nu kbar); a lower bound for
    undirected graphs since 1'A^t1/N >= kbar^t."""
    return 1 / (1 - nu * kbar)


def gamma_kappa(kbar, cv):
    """<k^2>/<k> for gamma-distributed degrees with mean kbar and coefficient of
    variation cv: kbar (1 + cv^2)."""
    return kbar * (1 + cv * cv)


def cv_critical(nu_kbar):
    """Stability boundary of the gamma configuration model at fixed nu kbar:
    nu kappa = nu kbar (1 + cv^2) = 1  <=>  cv* = sqrt(1/(nu kbar) - 1)."""
    return np.sqrt(1 / nu_kbar - 1)


# ----------------------------------------------------------------------------------
# T2  Katz resolvent on a locally tree-like graph (Green's function)
# ----------------------------------------------------------------------------------


def tree_green(nu, b):
    """Walk generating functions on an infinite tree with branching factor b.

    g: walks that leave a node into one given child subtree and first return to the node
       (closed excursion into one branch), g = nu^2 T, with
    T: walks from a node that stay in its own subtree and return to it, T = 1/(1 - b g).
    Solving b g^2 - g + nu^2 = 0 gives g = (1 - sqrt(1 - 4 b nu^2)) / (2 b).
    Requires 4 b nu^2 < 1 (nu below the edge of the Kesten-McKay band 2 sqrt(b))."""
    nu = np.asarray(nu, dtype=float)
    disc = 1 - 4 * b * nu * nu
    if np.any(disc < 0):
        raise ValueError('nu beyond the spectral band edge 1/(2 sqrt(b))')
    g = np.where(b > 0, (1 - np.sqrt(disc)) / (2 * b if b > 0 else 1), nu * nu)
    T = 1 / (1 - b * g)
    return g, T


def green_diagonal(nu, k, b):
    """M_ii for a node of degree k on a tree with branching b: G(0) = 1/(1 - k g)."""
    g, _ = tree_green(nu, b)
    return 1 / (1 - k * g)


def green_distance(nu, k_source, b, h):
    """M_ij for nodes at tree distance h, j the source node of degree k_source:
    G(h) = G(0) (nu T)^h. Decomposition at the last visit to the source and then at the
    last visit to each node on the path; every step contributes nu T."""
    g, T = tree_green(nu, b)
    return green_diagonal(nu, k_source, b) * (nu * T) ** h


def _degree_weights(deg_hist, kbar):
    if deg_hist is None:
        return np.array([kbar], dtype=float), np.array([1.0])
    h = np.asarray(deg_hist, dtype=float)
    ks = np.arange(h.size, dtype=float)
    keep = h > 0
    return ks[keep], h[keep] / h.sum()


def hop_lift(nu, kbar, b, p, hops=(0, 1, 2, 3), deg_hist=None):
    """Expected lift / dlam of a node at hop distance h from the treated set, and the
    expected share of nodes at each hop, when a fraction p is treated at random.

    Locally tree-like graph: a node of degree k has n_h = k b^{h-1} nodes at distance h.
    Conditional on being at distance h (nobody closer treated, at least one node at
    distance h treated, nodes farther away treated independently with probability p):
      lift_h / dlam = G(0) [ E[#treated at h | >= 1] (nu T)^h
                             + sum_{d > h} p n_d (nu T)^d ]
                    = G(0) [ n_h p / (1 - (1-p)^{n_h}) (nu T)^h
                             + p k (nu T) (b nu T)^h / (1 - b nu T) ]
    and for h = 0 (a treated node): G(0) [1 + p k nu T / (1 - b nu T)].
    Averaged over degrees with the probability of being at distance h given k. The
    identity G(0) [1 + k nu T/(1 - b nu T)] = 1/(1 - nu k) for b = k - 1 recovers the
    regular multiplier.

    Returns (lift_by_hop, share_by_hop) as arrays over `hops`."""
    g, T = tree_green(nu, b)
    x = nu * T
    ks, w = _degree_weights(deg_hist, kbar)
    lifts = np.zeros(len(hops))
    shares = np.zeros(len(hops))
    for idx, h in enumerate(hops):
        G0 = 1 / (1 - ks * g)
        if h == 0:
            prob = np.full(ks.size, p)
            lift = G0 * (1 + p * ks * x / (1 - b * x))
        else:
            n_h = ks * b ** (h - 1)
            n_closer = ks * (b ** (h - 1) - 1) / (b - 1) if b != 1 else ks * (h - 1)
            prob = (1 - p) * (1 - p) ** n_closer * (1 - (1 - p) ** n_h)
            miss = (1 - p) ** n_h
            e_treated = np.divide(n_h * p, 1 - miss, out=np.zeros_like(n_h), where=miss < 1)
            tail = p * ks * x * (b * x) ** h / (1 - b * x)
            lift = G0 * (e_treated * x ** h + tail)
        shares[idx] = (w * prob).sum()
        lifts[idx] = (w * prob * lift).sum() / max((w * prob).sum(), 1e-300)
    return lifts, shares


# ----------------------------------------------------------------------------------
# T3  attenuation E[DM]/GATE
# ----------------------------------------------------------------------------------


def tr_over_n_tree(nu, kbar, b, deg_hist=None, triangles_per_node=0.0):
    """tr(M)/N on a locally tree-like graph: the degree average of G(0) = 1/(1 - k g).

    For a k-regular graph (b = k - 1) this is the Stieltjes transform of the Kesten-McKay
    density at 1/nu (Kesten 1959, McKay 1981): tr(M)/N = (1/nu) G_KM(1/nu), the same
    number; see tr_over_n_kesten_mckay. The optional triangle term adds the leading
    non-tree closed walks, nu^3 tr(A^3)/N = 2 nu^3 (triangles per node), dressed by the
    same factor (nu T)^3 G(0) as a tree path (Watts-Strogatz has about 100 triangles per
    node at kbar = 20, p = 0.1)."""
    g, T = tree_green(nu, b)
    ks, w = _degree_weights(deg_hist, kbar)
    G0 = (w / (1 - ks * g)).sum()
    tri = 2 * triangles_per_node * (nu * T) ** 3 * G0
    return G0 + tri


def tr_over_n_kesten_mckay(nu, k):
    """tr(M)/N for the random k-regular graph from the Kesten-McKay spectral density
    rho(x) = k sqrt(4(k-1) - x^2) / (2 pi (k^2 - x^2)) on |x| <= 2 sqrt(k-1), integrated
    against 1/(1 - nu x). Closed form via the Stieltjes transform
    G(z) = 2 (k-1) / ((k-2) z + k sqrt(z^2 - 4(k-1))), evaluated at z = 1/nu."""
    z = 1 / nu
    G = 2 * (k - 1) / ((k - 2) * z + k * np.sqrt(z * z - 4 * (k - 1)))
    return G / nu


def attenuation_exact_from_parts(tau, m, N):
    """E[DM]/GATE under Bernoulli assignment: (N tau - m) / ((N - 1) m), with
    tau = tr(M)/N and m = 1'M1/N (exact for a given graph; see Section 3.2)."""
    return (N * tau - m) / ((N - 1) * m)


def attenuation(nu, rho, kbar, b, N, deg_hist=None, triangles_per_node=0.0):
    """E[DM]/GATE from the tree tr(M)/N and the annealed multiplier with the measured rho.
    First order: (1 - nu rho)(1 + nu^2 kbar + ...)."""
    tau = tr_over_n_tree(nu, kbar, b, deg_hist, triangles_per_node)
    m = multiplier_annealed(nu, kbar, rho)
    return attenuation_exact_from_parts(tau, m, N)


def attenuation_regular(nu_rho, k):
    """Random k-regular graph, large N: (1 - nu k) / (1 - k g(nu)), nu = nu_rho / k."""
    nu = nu_rho / k
    g, _ = tree_green(nu, k - 1)
    return (1 - nu * k) / (1 - k * g)


# ----------------------------------------------------------------------------------
# T4  cluster designs
# ----------------------------------------------------------------------------------


def cluster_ratio_mean_field(nu_rho, phi, C=np.inf):
    """Two-type mean field for whole-cluster assignment with probability 1/2.

    E[DM] = dlam (C sigma - m)/(C - 1) exactly, where sigma = (1/N) sum_c 1_c' M 1_c is the
    within-cluster Katz mass per node and C the number of clusters (the cluster-level
    analogue of (N ADE - GATE)/(N - 1)). Mean field: a walk stays in its cluster with
    probability phi per step (phi = share of edge ends inside clusters = 1 - cut), so
    sigma = sum_t (nu kbar)^t q_t with q_t = 1/C + (1 - 1/C) phi_C^t, where the mixing
    ratio phi_C = (C phi - 1)/(C - 1) (the paper's phi_C; the local variable is r), and
    E[DM]/dlam = 1/(1 - nu kbar phi_C). With kbar -> rho and GATE = dlam/(1 - nu rho):
        E[DM]/GATE = (1 - nu rho) / (1 - nu rho phi_C),   phi_C -> phi for many clusters.
    The exact relation E[DM] = dlam (C sigma - m)/(C - 1) needs equal-size clusters."""
    r = phi if not np.isfinite(C) else (C * phi - 1) / (C - 1)
    return (1 - nu_rho) / (1 - nu_rho * r)


def cluster_contrast_mass_tree(nu, kbar, b, phi, deg_hist=None, C=np.inf):
    """F = E_k[ G(0) (1 + k r nu T / (1 - b r nu T)) ], r = phi_C = (C phi - 1)/(C - 1): the tree
    sum over nodes j of M_ij r^{d(i,j)}. Under the cluster-mixing approximation the probability
    that a walk of length d ends in its start cluster is q_d = 1/C + (1 - 1/C) r^d, so F is the
    cluster contrast (C sigma - m)/(C - 1), not the within-cluster mass sigma itself.
    phi = 1 gives the multiplier, phi = 0 (C infinite) gives tr(M)/N."""
    r = phi if not np.isfinite(C) else (C * phi - 1) / (C - 1)
    g, T = tree_green(nu, b)
    x = nu * T
    ks, w = _degree_weights(deg_hist, kbar)
    G0 = 1 / (1 - ks * g)
    return (w * G0 * (1 + ks * r * x / (1 - b * r * x))).sum()


def cluster_within_mass_tree(nu, kbar, b, phi, deg_hist=None, C=np.inf, rho=None):
    """sigma = (1/N) sum_c 1_c' M 1_c on a locally tree-like graph:
    sigma = m/C + (1 - 1/C) F, with F from cluster_contrast_mass_tree and m the annealed
    multiplier (with the measured rho if given, else kappa = b + 1)."""
    F = cluster_contrast_mass_tree(nu, kbar, b, phi, deg_hist, C)
    if not np.isfinite(C):
        return F
    m = multiplier_annealed(nu, kbar, rho if rho is not None else b + 1)
    return m / C + (1 - 1 / C) * F


def cluster_ratio_tree(nu, rho, kbar, b, phi, N, size, deg_hist=None):
    """E[DM]/GATE under cluster assignment, tree Green's function version:
    (C sigma - m)/((C - 1) m) = F/m, with F from cluster_contrast_mass_tree and the annealed
    multiplier m with the measured rho."""
    C = max(2.0, N / size)
    F = cluster_contrast_mass_tree(nu, kbar, b, phi, deg_hist, C)
    m = multiplier_annealed(nu, kbar, rho)
    return F / m


def dm_sd(N, nu, kbar, b, phi, m, W, var_lam=0.5, base_sd=0.0, lam=1.0, dlam=0.05,
          size=1, deg_hist=None, var_k=None, rho=None):
    """Standard deviation of DM over cluster assignments (probability 1/2 per cluster) with
    heterogeneous intrinsic production lam_i (variance var_lam) and Poisson observation
    over W periods (y_i = Poisson(W c_i)/W):

      Var(DM) = (4/N) [ var_lam * omega(phi) + base_sd^2 + cbar / W
                        + 2 nu^2 Var(k) phi cbar^2 / (1 - nu kbar r)^2 ]

    omega(phi) = (1/N) sum_c 1_c' M^2 1_c, the within-cluster mass of M^2 (the baseline
    (M lam)_i of nodes in one cluster moves together), computed as d/dnu [nu sigma(nu)]
    since M^2 = sum_t (t + 1) nu^t A^t (mean field: 1/(1 - nu kbar phi)^2);
    base_sd^2 = variance over nodes of the mean baseline (M 1)_i (degree-driven, from graph
    statistics or nu^2 Var(k)/(1 - nu rho)^2), treating degrees as independent across nodes;
    cbar = lam m (1 + dlam/(2 lam)), the mean production;
    the last term (only with var_k given, and only for clusters) is the shared part of the
    degree fluctuation: a cluster's realized internal mean degree has variance
    2 phi Var(k)/size, and the cluster's production responds to it with
    d c/d k_in = nu c/(1 - nu k_in), so each cluster sum moves by size * nu cbar/(1 - nu kbar r)
    times that, summed over N/size clusters. It is a heuristic (first-order response of a
    cluster-level mean field), not a derivation from the spectrum.
    The term quadratic in dlam (variation of the treatment lift itself) is omitted; it is
    small (see T4a, sd of the ratio)."""
    C = max(2.0, N / size)
    r = (C * phi - 1) / (C - 1)
    h = 1e-6 * nu

    def nu_sigma(x):
        return x * cluster_within_mass_tree(x, kbar, b, phi, deg_hist, C, rho)

    omega = (nu_sigma(nu + h) - nu_sigma(nu - h)) / (2 * h)
    cbar = lam * m * (1 + dlam / (2 * lam))
    var = var_lam * omega + base_sd ** 2 + cbar / W
    if var_k is not None and size > 1:
        var += 2 * nu ** 2 * var_k * phi * cbar ** 2 / (1 - nu * kbar * r) ** 2
    return np.sqrt(4 / N * var)


# ----------------------------------------------------------------------------------
# T5  power near the stability limit
# ----------------------------------------------------------------------------------


def power_z(N, m, tau, base_cv, W, lam=1.0, dlam=0.05, alpha=0.05, cuped=False, drift=0.0):
    """Expected z-statistic and power of the one-sided test on DM, 50/50 Bernoulli.

    Signal: E[DM] = dlam tau (tau = tr(M)/N, about 1).
    Noise:  se^2 = (4/N) Var(y_i),  y_i = Poisson(W c_i)/W,  c_i = lam (M1)_i,
            Var(y_i) = (lam m base_cv)^2 + lam m / W
            (baseline spread across nodes, then Poisson counting; cbar = lam m).
    So  z = (dlam/lam) (tau/m) sqrt(N) / (2 sqrt(base_cv^2 + 1/(W lam m))),
    and with m = 1/(1 - nu rho):  z = (dlam/lam)(1 - nu rho) sqrt(N) / (2 sqrt(...)).
    Power = Phi(z - z_alpha).

    CUPED with a pre-period of the same length: the residual variance
    Var(y) - Cov(y, x)^2/Var(x) replaces Var(y_i), with Cov(y, x) = Var(c) and
    Var(x) = Var(c) + cbar/W + drift^2 E[c^2] (node-level baseline drift between the periods,
    c_i (1 + drift * eps_i)); with no drift and Var(c) >> cbar/W this leaves 2 cbar/W (the
    two Poisson terms)."""
    cbar = lam * m
    var_c = (cbar * base_cv) ** 2
    var_y = var_c + cbar / W
    if cuped:
        var_x = var_c + cbar / W + drift ** 2 * (cbar ** 2 + var_c)
        var_y = var_y - var_c ** 2 / var_x
    se = np.sqrt(4 / N * var_y)
    z = dlam * tau / se
    return z, norm.cdf(z - norm.isf(alpha)), se


# ----------------------------------------------------------------------------------
# T6  treatment adds edges
# ----------------------------------------------------------------------------------


def kappa_after_edges(kbar, kappa_, r):
    """Mean neighbour degree <k'^2>/<k'> after every node gains Poisson(2r) uniformly random
    edges (degree k' = k + X, X ~ Poisson(2r) independent of k):
        kappa' = (kappa kbar + 4 r kbar + 2r + 4r^2) / (kbar + 2r).
    About kappa + 2r for homogeneous degrees; below kappa on heavy-tailed graphs, where the
    random endpoints dilute the hubs (paper A.6, rho')."""
    return (kappa_ * kbar + 4 * r * kbar + 2 * r + 4 * r * r) / (kbar + 2 * r)


def edges_gate(nu, rho, kbar, r, lam=1.0, dlam=0.05, boost=True, rho_new=None):
    """GATE when everyone forms Poisson(r) new undirected edges: the mean degree rises by 2r
    and the spectral radius to rho_new (paper A.6: rho' ~ kappa_after_edges; the default
    rho + 2r assumes a flat Perron vector and is 43 to 64% high for the GATE on the
    heavy-tailed graphs). With the annealed multiplier m(k, rho) = 1 + nu k/(1 - nu rho):
        GATE = (lam + dlam) m(kbar + 2r, rho') - lam m(kbar, rho).
    The paper's comparisons (Section 8.1) use edges_annealed_effects instead."""
    x = nu * ((rho + 2 * r) if rho_new is None else rho_new)
    if x >= 1:
        return np.inf
    full = (lam + (dlam if boost else 0.0)) * (1 + nu * (kbar + 2 * r) / (1 - x))
    return full - lam * (1 + nu * kbar / (1 - nu * rho))


def edges_dm(nu, kappa_eff, r, phi=0.0, psi=0.5, lam=1.0, dlam=0.05, boost=True):
    """Two-type mean field for DM when treated nodes form Poisson(r) new edges.

    phi  share of original edge ends inside clusters (0 for the user-level split), so the
         same-arm share of original neighbours is (1 + phi)/2
    psi  share of new edges that connect the two arms (1/2 for uniformly random targets)
    kappa_eff  effective degree of the original graph (kbar, or rho for the spectral
         version); per treated node: r new edges of its own, r (1 - psi) received from
         other treated nodes; per control node: r psi received.

      c_T = lam + dlam + nu [ k ((1+phi)/2 c_T + (1-phi)/2 c_C) + r((1-psi) c_T + psi c_C)
                                + r (1-psi) c_T ]
      c_C = lam        + nu [ k ((1+phi)/2 c_C + (1-phi)/2 c_T) + r psi c_T ]
    Returns (DM, c_T, c_C)."""
    k = kappa_eff
    d = dlam if boost else 0.0
    a_tt = nu * (k * (1 + phi) / 2 + r * (1 - psi) + r * (1 - psi))
    a_tc = nu * (k * (1 - phi) / 2 + r * psi)
    a_ct = nu * (k * (1 - phi) / 2 + r * psi)
    a_cc = nu * (k * (1 + phi) / 2)
    A = np.array([[1 - a_tt, -a_tc], [-a_ct, 1 - a_cc]])
    if np.linalg.eigvals(np.eye(2) - A).real.max() >= 1:
        return np.inf, np.inf, np.inf
    cT, cC = np.linalg.solve(A, [lam + d, lam])
    return cT - cC, cT, cC


# ----------------------------------------------------------------------------------
# T7  treatment adds users
# ----------------------------------------------------------------------------------


def users_three_type(nu, kappa_eff, r, q, a, p, lam=1.0, dlam=0.05):
    """Three-type mean field (treated T, control C, new users U) when a fraction p of the
    original users is treated, each treated user invites Poisson(r) new users, and a new
    user links to its inviter and to q of the inviter's neighbours, producing a lam.

    Degrees: T: k p to T, k (1-p) to C, r (1 + p q) to U (own invitees plus the invitees of
    treated neighbours, p k of them, each sending r q links spread over k neighbours);
    C: k p, k (1-p), p r q to U; U: 1 + q p to T, q (1-p) to C.
      c_T = lam + dlam + nu [k p c_T + k (1-p) c_C + r (1 + p q) c_U]
      c_C = lam        + nu [k p c_T + k (1-p) c_C + p r q c_U]
      c_U = a lam      + nu [(1 + q p) c_T + q (1-p) c_C]
    Returns (c_T, c_C, c_U); for p = 1 the control row is unused."""
    k = kappa_eff
    A = np.array([[1 - nu * k * p, -nu * k * (1 - p), -nu * r * (1 + p * q)],
                  [-nu * k * p, 1 - nu * k * (1 - p), -nu * p * r * q],
                  [-nu * (1 + q * p), -nu * q * (1 - p), 1.0]])
    rhs = np.array([lam + dlam, lam, a * lam])
    if np.linalg.eigvals(np.eye(3) - A).real.max() >= 1:
        return np.inf, np.inf, np.inf
    return tuple(np.linalg.solve(A, rhs))


def users_annealed(nu, kbar, kappa_, r, q, a, p, lam=1.0, dlam=0.05):
    """Degree-corrected (annealed) three-type model. Original users have degree k from the
    graph's degree distribution; over the original edges the annealed approximation
    replaces a neighbour's production by the edge-end weighted mean Theta = p theta_T +
    (1-p) theta_C, theta_tau = sum_{i in tau} k_i c_i / sum_{i in tau} k_i. A new user links
    to its inviter (a uniformly random treated user, production cbar_T) and to q random
    neighbours of the inviter (edge-end weighted, production Theta). An original user of
    degree k receives p r q k/kbar links from the invitees of its treated neighbours.
      c_tau(k) = lam_tau + nu k Theta + nu (r 1[tau = T] + p r q k/kbar) cbar_U
      theta_tau = lam_tau + nu kappa Theta + nu (r 1[T] + p r q kappa/kbar) cbar_U
      cbar_tau  = lam_tau + nu kbar  Theta + nu (r 1[T] + p r q) cbar_U
      cbar_U    = a lam + nu (cbar_T + q Theta)
    with kappa = <k^2>/<k> (passed as kappa_; the measured rho keeps the pole right).
    Unknowns (theta_T, theta_C, cbar_T, cbar_U); returns (cbar_T, cbar_C, cbar_U)."""
    lT, lC = lam + dlam, lam
    # order: theta_T, theta_C, cT, cU
    A = np.array([
        [1 - nu * kappa_ * p, -nu * kappa_ * (1 - p), 0, -nu * (r + p * r * q * kappa_ / kbar)],
        [-nu * kappa_ * p, 1 - nu * kappa_ * (1 - p), 0, -nu * p * r * q * kappa_ / kbar],
        [-nu * kbar * p, -nu * kbar * (1 - p), 1, -nu * (r + p * r * q)],
        [-nu * q * p, -nu * q * (1 - p), -nu, 1]])
    rhs = np.array([lT, lC, lT, a * lam])
    if np.linalg.eigvals(np.eye(4) - A).real.max() >= 1:
        return np.inf, np.inf, np.inf
    thT, thC, cT, cU = np.linalg.solve(A, rhs)
    cC = lC + nu * kbar * (p * thT + (1 - p) * thC) + nu * p * r * q * cU
    return cT, cC, cU


def users_effects(nu, kappa_eff, r, q, a, lam=1.0, dlam=0.05, kbar=None):
    """GATE (total per original user, and per user) and the experiment estimates at p = 1/2
    (inherit_total, inherit_per_user, exclude). With kbar=None the homogeneous three-type
    mean field with degree kappa_eff; with kbar given, the annealed model users_annealed
    with mean degree kbar and kappa_eff for <k^2>/<k> (or the measured rho). The untreated
    baseline uses the same model at r = 0, p = 0, so that the GATE is internally consistent."""
    if kbar is None:
        solve = lambda rr, pp: users_three_type(nu, kappa_eff, rr, q, a, pp, lam, dlam)
    else:
        solve = lambda rr, pp: users_annealed(nu, kbar, kappa_eff, rr, q, a, pp, lam, dlam)
    c0 = solve(0.0, 0.0)[1]                      # control production with nobody treated
    cT1, _, cU1 = solve(r, 1.0)
    out = dict(gate_total=cT1 + r * cU1 - c0, gate_per_user=(cT1 + r * cU1) / (1 + r) - c0)
    cT, cC, cU = solve(r, 0.5)
    out.update(inherit_total=cT + r * cU - cC, inherit_per_user=(cT + r * cU) / (1 + r) - cC,
               exclude=cT - cC)
    return out


# ----------------------------------------------------------------------------------
# T8  treatment raises the peer effect
# ----------------------------------------------------------------------------------


def nutreat(nu_rho, delta, p, lam=1.0):
    """nu_i = nu (1 + delta z_i), fraction p treated at random, flat Perron vector.

    Stability: rho(diag(nu_i) A) = nu rho v' diag(1 + delta z) v + O(delta^2 p (1-p))
               = nu rho (1 + delta p); the partial experiment is stable iff
               nu rho (1 + delta p) < 1, the full rollout iff nu rho (1 + delta) < 1.
    Mean field: c_T = lam + nu (1+delta) rho cbar, c_C = lam + nu rho cbar,
                cbar = p c_T + (1-p) c_C = lam / (1 - nu rho (1 + delta p)), so
      DM = delta nu rho lam / (1 - nu rho (1 + delta p)),
      mean lift = cbar - lam/(1 - nu rho),
      GATE (full rollout) = lam [1/(1 - nu rho (1 + delta)) - 1/(1 - nu rho)].
    Returns dict(stable, dm, mean_lift, gate, gate_stable)."""
    x = nu_rho * (1 + delta * p)
    xf = nu_rho * (1 + delta)
    c0 = lam / (1 - nu_rho)
    out = dict(stable=x < 1, gate_stable=xf < 1, partial_nu_rho=x, full_nu_rho=xf)
    out['dm'] = delta * nu_rho * lam / (1 - x) if x < 1 else np.inf
    out['mean_lift'] = lam / (1 - x) - c0 if x < 1 else np.inf
    out['gate'] = lam / (1 - xf) - c0 if xf < 1 else np.inf
    return out


def nutreat_annealed(nu_rho, delta, p, kbar, rho, lam=1.0):
    """Annealed version of nutreat for heterogeneous degrees: c_i = lam + nu_i k_i theta with
    theta the mean production per edge end, theta = lam / (1 - nu kappa (1 + delta p)),
    kappa -> rho. A node's own gain is proportional to its degree, so
      DM = delta nu kbar theta = delta (nu rho) (kbar/rho) lam / (1 - nu rho (1 + delta p)),
      mean lift = (kbar/rho) (nu rho) lam [(1 + delta p)/(1 - nu rho (1 + delta p)) - 1/(1 - nu rho)]
                  (cbar = p c_T + (1-p) c_C = lam + nu kbar theta (1 + delta p)),
      GATE = (kbar/rho) (nu rho) lam [(1 + delta)/(1 - nu rho (1 + delta)) - 1/(1 - nu rho)]
                  (= lam (m_ann(nu (1 + delta)) - m_ann(nu)) with the annealed multiplier).
    The stability condition is unchanged, nu rho (1 + delta w_T) < 1 with w_T the Perron
    weight sum_{i treated} v_i^2 (mean p, sd sqrt(p (1-p) sum_i v_i^4))."""
    f = kbar / rho
    x = nu_rho * (1 + delta * p)
    xf = nu_rho * (1 + delta)
    out = dict(stable=x < 1, partial_nu_rho=x, full_nu_rho=xf)
    out['dm'] = delta * nu_rho * f * lam / (1 - x) if x < 1 else np.inf
    out['mean_lift'] = f * nu_rho * lam * ((1 + delta * p) / (1 - x) - 1 / (1 - nu_rho)) if x < 1 else np.inf
    out['gate'] = f * nu_rho * lam * ((1 + delta) / (1 - xf) - 1 / (1 - nu_rho)) if xf < 1 else np.inf
    return out


# ----------------------------------------------------------------------------------
# T9  threshold contagion
# ----------------------------------------------------------------------------------


def threshold_cascade(p, theta, deg_hist, kbar=None, iters=5000, tol=1e-13):
    """Final active fraction of the fractional threshold model (Watts 2002) with a random
    seed fraction p, on a locally tree-like graph with the given degree distribution,
    by the Gleeson-Cahalane (2007) recursion.

    q_{n+1} = p + (1 - p) sum_k (k P(k)/kbar) P[Bin(k-1, q_n) >= ceil(theta k)]
    from q_0 = p (q = probability that a child in the tree is active given its parent is
    not; a node of degree k has k - 1 children and needs m/k >= theta of them active), and
    the final fraction
    rho = p + (1 - p) sum_k P(k) P[Bin(k, q) >= ceil(theta k)].
    Returns dict(active, active_control = (active - p)/(1 - p), dm_over_gate = 1 - active_control, q)."""
    from scipy.stats import binom
    h = np.asarray(deg_hist, dtype=float)
    ks = np.arange(h.size)
    P = h / h.sum()
    keep = (P > 0) & (ks > 0)                      # isolated nodes activate only as seeds
    ks, P = ks[keep], P[keep]
    kbar = kbar or (ks * P).sum()
    excess = ks * P / kbar
    need = np.ceil(theta * ks - 1e-9)               # active neighbours needed
    q = p
    for _ in range(iters):
        qn = min(1.0, p + (1 - p) * (excess * binom.sf(need - 1, ks - 1, q)).sum())
        if abs(qn - q) < tol:
            q = qn
            break
        q = qn
    rho = (P * binom.sf(need - 1, ks, q)).sum()
    active = p + (1 - p) * rho
    ac = (active - p) / (1 - p)
    return dict(active=active, active_control=ac, dm_over_gate=1 - ac, q=q)


def threshold_cascade_poisson(p, theta, kbar, kmax=None):
    """threshold_cascade with Poisson(kbar) degrees (Erdos-Renyi)."""
    from scipy.stats import poisson
    kmax = kmax or int(kbar + 12 * np.sqrt(kbar))
    hist = poisson.pmf(np.arange(kmax + 1), kbar)
    return threshold_cascade(p, theta, hist, kbar)


def threshold_critical_seed(theta, deg_hist, kbar=None, lo=1e-4, hi=0.6):
    """Smallest seed fraction at which the recursion ends in a global cascade (final active
    fraction above 1/2), by bisection."""
    f = lambda p: threshold_cascade(p, theta, deg_hist, kbar)['active'] - 0.5
    if f(hi) < 0:
        return np.nan            # no cascade even at p = hi
    if f(lo) > 0:
        return lo                # cascades from any seed fraction down to lo
    return brentq(f, lo, hi, xtol=1e-5)


# ----------------------------------------------------------------------------------
# T6, heterogeneous degrees: annealed two-type model for new edges
# ----------------------------------------------------------------------------------


def edges_annealed(nu, kbar, kappa_, r, rule='random', phi=0.0, psi=0.5, lam=1.0, dlam=0.05,
                   boost=True, full=False):
    """Degree-corrected (annealed) version of edges_dm / edges_gate for graphs with a wide
    degree distribution (Barabasi-Albert, power-law configuration model).

    Original users have degree k from the graph; over the original edges a neighbour's
    production is replaced by the edge-end weighted mean of its arm, Theta_tau =
    s theta_tau + (1 - s) theta_other with s = (1 + phi)/2 the same-arm share
    (theta_tau = sum_{i in tau} k_i c_i / sum_{i in tau} k_i, cbar_tau the plain mean).
    A treated node forms r new edges; with rule 'random' the target is a uniformly random node
    of the arm (production cbar), with rule 'fof' (friend of a friend) a degree-biased one
    (production theta). Edges received from other treated nodes: r (1 - psi) per treated and
    r psi per control node on average, spread uniformly over nodes ('random') or in
    proportion to degree k/kbar ('fof'); the sender is a uniformly random treated node
    (production cbar_T). For a node of degree k and type tau:
      c_tau(k) = lam_tau + nu [ k Theta_tau + own_tau + recv_tau w(k) cbar_T ],
    w(k) = 1 or k/kbar, so that the plain mean uses kbar and E[w] = 1 and the weighted mean
    uses kappa and E[k w]/kbar = 1 or kappa/kbar. kappa_ -> measured rho keeps the pole.
    full=True: everyone treated (one type, r edges formed and r received per node), used for
    the GATE together with the untreated baseline (r = 0).
    Returns (cbar_T, cbar_C); for full=True cbar_C is nan."""
    d = dlam if boost else 0.0
    fof = rule == 'fof'
    ew = kappa_ / kbar if fof else 1.0        # E[k w(k)] / kbar
    if full:
        # unknowns theta, cbar:  theta = lam+d + nu[kappa theta + r X + r ew cbar]
        #                        cbar  = lam+d + nu[kbar  theta + r X + r cbar],  X = theta (fof) or cbar
        xt, xc = (1.0, 0.0) if fof else (0.0, 1.0)
        A = np.array([[1 - nu * (kappa_ + r * xt), -nu * (r * xc + r * ew)],
                      [-nu * (kbar + r * xt), 1 - nu * (r * xc + r)]])
        if np.linalg.eigvals(np.eye(2) - A).real.max() >= 1:
            return np.inf, np.nan
        th, cb = np.linalg.solve(A, [lam + d, lam + d])
        return cb, np.nan
    s = (1 + phi) / 2
    # unknowns: theta_T, theta_C, cbar_T, cbar_C
    # own_T = r[(1-psi) X_T + psi X_C], X = theta (fof) or cbar (random); own_C = 0
    # recv_T = r (1-psi) cbar_T w(k); recv_C = r psi cbar_T w(k)
    oT = np.zeros(4)
    if fof:
        oT[0], oT[1] = r * (1 - psi), r * psi
    else:
        oT[2], oT[3] = r * (1 - psi), r * psi
    rows = []
    for tau in (0, 1):                       # 0 = T, 1 = C
        for weighted in (True, False):
            kk = kappa_ if weighted else kbar
            e = ew if weighted else 1.0
            row = np.zeros(4)
            # k Theta_tau
            same, other = (0, 1) if tau == 0 else (1, 0)
            row[same] += kk * s
            row[other] += kk * (1 - s)
            if tau == 0:
                row += oT
                row[2] += r * (1 - psi) * e
            else:
                row[2] += r * psi * e
            rows.append(row)
    # rows order: theta_T, cbar_T, theta_C, cbar_C  -> reorder to unknown order
    Bm = np.array([rows[0], rows[2], rows[1], rows[3]])
    A = np.eye(4) - nu * Bm
    if np.linalg.eigvals(nu * Bm).real.max() >= 1:
        return np.inf, np.inf
    rhs = np.array([lam + d, lam, lam + d, lam])
    thT, thC, cT, cC = np.linalg.solve(A, rhs)
    return cT, cC


def edges_annealed_effects(nu, kbar, kappa_, r, rule='random', phi=0.0, psi=0.5, lam=1.0,
                           dlam=0.05, boost=True):
    """GATE and E[DM] from edges_annealed: GATE = cbar(full, r) - cbar(full, r = 0, no boost),
    DM = cbar_T - cbar_C at p = 1/2. Returns dict(gate, dm, cT, cC)."""
    c0 = edges_annealed(nu, kbar, kappa_, 0.0, rule, 0.0, 0.5, lam, dlam, False, full=True)[0]
    c1 = edges_annealed(nu, kbar, kappa_, r, rule, 0.0, 0.5, lam, dlam, boost, full=True)[0]
    cT, cC = edges_annealed(nu, kbar, kappa_, r, rule, phi, psi, lam, dlam, boost)
    return dict(gate=c1 - c0, dm=cT - cC, cT=cT, cC=cC)
