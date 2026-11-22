"""Figure for the logarithmic-response section of tutorial.tex."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from logcalc import effects, g

plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': 0.3, 'legend.fontsize': 6.5})
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.25))
s = np.linspace(0, 6, 200)
ax[0].plot(s, s, 'k-', lw=1.2, label='linear, g(s) = s')
for K, ls in ((10, '--'), (3, '-.'), (1, ':')):
    ax[0].plot(s, g(s, K), 'k', ls=ls, lw=1.2, label=f'logarithmic, K = {K}')
ax[0].set_xlabel('content seen, s')
ax[0].set_ylabel('response g(s)')
ax[0].legend(loc='upper left')

nus = np.linspace(0.02, 2.0, 60)
lin = [n for n in nus if n * np.sqrt(2) < 0.985]
res = {None: [effects(n, None) for n in lin]}
for K in (10, 3, 1):
    res[K] = [effects(n, K) for n in nus]
styles = {None: '-', 10: '--', 3: '-.', 1: ':'}
for K, rs in res.items():
    xs = lin if K is None else nus
    lab = 'linear' if K is None else f'logarithmic, K = {K}'
    ax[1].plot(xs, [r[2] / 0.1 for r in rs], 'k', ls=styles[K], lw=1.2, label=lab)
    ax[2].plot(xs, [r[4] / r[2] for r in rs], 'k', ls=styles[K], lw=1.2, label=lab)
for a in ax[1:]:
    a.axvline(1 / np.sqrt(2), color='gray', lw=0.8)
    a.set_xlabel('peer effect ν')
ax[1].text(1 / np.sqrt(2) + 0.04, 7.2, 'linear model unstable\nto the right of this line', fontsize=7, color='gray', va='top')
ax[1].set_ylim(1, 8)
ax[1].set_ylabel('social multiplier, GATE / Δλ')
ax[2].set_ylim(0, 1)
ax[2].set_ylabel('share the A/B test reports, E[DM] / GATE')
fig.tight_layout()
fig.savefig('log_response.pdf')
print('wrote log_response.pdf')
