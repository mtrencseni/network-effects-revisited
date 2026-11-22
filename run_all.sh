#!/bin/bash
# Runs every simulation of the paper, then the figures. Resumable: finished configurations are
# skipped. Most parts take minutes to hours on a laptop; s5_effects.py size runs single graphs
# with 10^7 and 10^8 nodes, and s13_giant.py graphs with up to 10^9 nodes (about 80 GB of memory).
set -e
cd "$(dirname "$0")/experiments"
PY=../.venv/bin/python

$PY graph_stats.py                                   # graph statistics used by the closed forms
$PY s14_family_table.py                              # Table 1
$PY s5_effects.py multiplier hetero hetero_base hops attenuation attenuation_p attenuation_kbar \
                  clustering clustering_mu saturation   # Section 4
$PY s5_kbar.py                                       # Appendix F.3
$PY s6_duration.py                                   # Section 5
$PY s8_designs.py                                    # Sections 6.1 and 6.2
$PY s7_critical.py                                   # Section 6.3
$PY s9_graph_change.py                               # Section 7
$PY s10_nonlinear.py threshold                       # Section 8.1
$PY s12_concave.py                                   # Section 8.2
$PY theory_check.py                                  # Appendix B
$PY s11_sizecheck.py small                           # Appendix E (process pool)
$PY s11_sizecheck.py big                             # Appendix E (10^7 nodes, inline)
$PY s5_effects.py size                               # Section 4.3, up to 10^8 nodes
$PY s5_effects.py size_fixednu                       # Section 4.3 (separate process: pool after inline runs)
$PY s13_giant.py                                     # Appendix F.4, 10^4 to 10^9 nodes, kbar = 2
$PY figures.py
