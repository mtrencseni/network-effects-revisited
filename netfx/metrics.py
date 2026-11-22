"""Node groups and effect metrics, as defined in the 2023 paper (model_stats)."""
from math import isnan

import numpy as np


def groups(A, treat):
    """Treatment, its non-treatment neighbours, the rest, and control (= neighbours + rest)."""
    treat = np.asarray(treat, dtype=bool)
    touches = (A @ treat.astype(np.float64)) > 0
    neighbours = touches & ~treat
    rest = ~treat & ~neighbours
    return {'treatment': treat, 'neighbours': neighbours, 'rest': rest, 'control': ~treat}


def stats(counts, final, c_base_prime, k_mean, nu, lam_int, dlam):
    """Port of the 2023 model_stats().

    counts: dict with sizes of 'treatment', 'neighbours', 'rest'
    final:  dict with final-step means of 'treatment', 'neighbours', 'rest'
    Returns a dict, plus the 10-tuple the 2023 function returned.
    """
    c_base = lam_int / (1 - k_mean * nu)
    t, n, r = counts['treatment'], counts['neighbours'], counts['rest']
    c_treatment = final['treatment']
    c_neighbours = final['neighbours']
    c_rest = final['rest']
    if isnan(c_rest):
        c_rest = 0
    c_graph = (t * c_treatment + n * c_neighbours + r * c_rest) / (t + n + r)
    c_control = (n * c_neighbours + r * c_rest) / (n + r)
    e_spillover = c_control / c_base_prime - 1
    e_treatment = c_treatment / c_control - 1
    e_dampening = e_treatment / dlam
    e_intrinsic = c_treatment / c_base_prime - 1
    d = dict(c_base=c_base, c_base_prime=c_base_prime, c_graph=c_graph,
             c_treatment=c_treatment, c_control=c_control, c_neighbours=c_neighbours,
             c_rest=c_rest, e_spillover=e_spillover, e_treatment=e_treatment,
             e_dampening=e_dampening, e_intrinsic=e_intrinsic,
             treatment=t, control=n + r, neighbours=n, rest=r)
    d['tuple'] = (c_base_prime, c_graph, c_treatment, c_control, c_neighbours, c_rest,
                  e_spillover, e_treatment, e_dampening, e_intrinsic)
    return d
