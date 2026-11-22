"""Generate paper/paper.tex (elsarticle) from paper/paper.md, with the figures placed in their
sections: `python make_tex.py paper.md paper.tex --paper`. `make paper` runs it and latexmk.

Converts the subset of Markdown used by the paper: headings, lists, tables, display equations,
citations, footnotes, bold, italics, inline code.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'figures'

# figure keys of draft.md (section numbers of the draft)
DRAFT_FIGURES = {
    '4': [('fig4_1_families', 'Top: examples of the five graph families, each with $N = 300$ nodes and average degree 4 (largest connected component; node size grows with degree). Watts-Strogatz is drawn on its ring: most edges join near neighbors on the ring, and about 10\\% are shortcuts across it. The SBM has dense communities with few edges between them; Barab\\\'asi-Albert has a few hubs with many links. At this small $N$ the power-law generator caps degrees at $\\sqrt{N}$, which hides most of its tail. Bottom: the gamma configuration model used in Section 5.2, at the same average degree and increasing spread of friend counts, and its degree distributions at the experiment scale.')],
    '5.1': [('fig5_1_multiplier', 'The average number of friends does not predict when content explodes. Every simulated curve (solid) diverges at $\\nu\\rho(A) = 1$, the true stability limit; the prediction from the average number of friends, $1/(1-\\nu\\bar k)$ (dashed), is close on graphs without hubs but stays near 1.5 on graphs with hubs, because there $\\rho(A)$ is 2.6 to 3.2 times $\\bar k$. Social multiplier = GATE$/\\Delta\\lambda$. Erd\\H{o}s-R\\\'enyi and SBM differ by less than 1\\% at every point, and the Erd\\H{o}s-R\\\'enyi curve stands for both.')],
    '5.2': [('fig5_2_heterogeneity', 'Social multiplier against the spread of friend counts, gamma configuration model, $N = 10^5$, at a fixed average of $\\bar k = 20$ friends. Dashed: the prediction from the average number of friends, $1/(1-\\nu\\bar k)$, which ignores the spread and is therefore flat; the gap to the simulated curve is the error from ignoring the spread. Open markers: the five base families at their own spread. Shaded: spreads at which $\\nu\\rho(A) \\geq 1$ for that peer effect, so the model is unstable (posting grows without limit and there is no multiplier); since $\\nu\\bar k$ is fixed, spreading the friend counts raises $\\rho(A)/\\bar k$ until $\\nu\\rho(A)$ crosses 1. Dash-dot: the closed form $1 + \\nu\\bar k/(1 - \\nu\\bar k(1 + cv^2))$ (Appendix A.1), which accounts for the spread and diverges at $cv = \\sqrt{1/(\\nu\\bar k) - 1}$.')],
    '5.3': [('fig5_3_spillover_hops', 'Lift by hop distance from the nearest treated node, 2\\% treated. Dashed: the closed form from the Green\\textquoteright s function of a locally tree-like graph (Appendix A.2), drawn up to hop 2; beyond it the tree predicts that fewer than 0.1\\% of nodes remain. It fails on Watts-Strogatz, where neighbors share most of their neighbors.')],
    '5.4': [('fig5_4_attenuation', 'Attenuation ratio E[DM]/GATE and its decomposition, 10\\% treated. Dashed (left): the closed form of Appendix A.3.')],
    '5.5': [('fig5_5_size', 'Attenuation, $\\rho/\\bar k$ and eigenvector localization against graph size at fixed $\\nu\\rho = 0.5$, $\\bar k = 10$ (the $\\bar k = 2$ runs of the text are not shown); SBM up to $10^7$ nodes.'),
            ('fig5_5b_size_fixed_nu', 'Graph size at fixed $\\nu\\bar k = 0.3$, $\\bar k = 10$: $\\nu\\rho$ and attenuation on stable graphs.')],
    '5.6': [('fig5_6_clustering', 'Attenuation against the share of same-arm neighbors for random and clustered assignment, $\\nu\\rho = 0.5$. Crosses: SBM graphs with a share $\\mu = 0.2$, 0.3, 0.5 and 0.7 of edges between blocks, which fill the range between graphs without communities and graphs with strong ones.')],
    '5.7': [('fig5_7_saturation', 'Saturation and targeted treatment ($\\nu\\rho = 0.5$).')],
    '6': [('fig6_1_paths', 'GATE(t) and treatment-induced DM(t) after launch, relative to their limits.'),
            ('fig6_2_time_to_95', 'Periods until GATE(t) reaches 95\\%, against $\\log 0.05/\\log \\nu\\rho$.'),
            ('fig6_3_directed', 'Directed follower graph: GATE(t) for several $\\nu\\bar k$.')],
    '7.1': [('fig8_1_clusters', 'Absolute bias (dotted) and RMSE (solid) of the DM, and absolute bias (dash-dot) and RMSE (dashed) of the H\\\'ajek exposure estimator, relative to the GATE, against cluster size.')],
    '7.2': [('fig8_2_wrong_graph', 'RMSE when clusters are computed from a perturbed graph.')],
    '7.3': [('fig8_3_extrapolation', 'RMSE of rollout estimators per family and scenario.')],
    '7.4': [('fig8_4_cuped', 'CUPED and exposure adjustment.')],
    '7.5': [('fig7_1_critical', 'Near the stability limit: GATE and E[DM], power (with and without CUPED), and SD calibration.'),
            ('fig7_1b_null_rejection', 'Rejection rate with no effect.')],
    '7.6': [('fig8_5_bayes', 'Posterior probability against p-value, and false positive rates by design, pooled over the five families; per-family values are in the text.')],
    '8.1': [('fig9_1_new_edges', 'Share of the launch effect that the A/B test sees (E[DM]/GATE) when treatment makes users add new friends, against the number of new friends per treated user. Top: treatment also raises posting by $\\Delta\\lambda$; bottom: treatment only adds friends. Dotted: user-level split; solid: cluster split, with clusters computed before the test. Green: new friends are friends of friends, which mostly stay inside a cluster; red: new friends are random users, about half of them in the other arm.')],
    '8.2': [('fig9_2_new_users', 'Share of the launch effect the A/B test sees when treatment raises posting by $\\Delta\\lambda$ and makes treated users invite new users, one panel per family. New users post half as much as existing users ($a = 0.5$); $\\nu\\rho(A) = 0.5$. The launch effect (GATE) is the change in total posts divided by the number of original users; the new users exist only in the launched world. Blue: the A/B estimate with the new users\\textquoteright{} posts added to the treated arm\\textquoteright s total per original treated user (only treated users invite). Gray: the A/B estimate when new users are left out, which sees only the $\\Delta\\lambda$ boost. Crosses: the three-type closed form of Appendix A.7. User-level 50/50 split.')],
    '9.1': [('fig10_1_nu_treatment', 'E[DM] when treatment raises a treated user\\textquoteright s peer effect from $\\nu$ to $\\nu(1+\\delta)$, against the treated fraction, $\\nu\\rho = 0.8$ before treatment, one panel per family. For $\\delta \\geq 0.25$ the launch to everyone has no steady state while the partial experiment reports a finite DM. Dashed: the closed form of Appendix A.8. Crosses: settings at which some of the 5 graphs diverge ($\\delta = 0.5$ with half the nodes treated on the heavy-tailed graphs, 3 of 5), plotted as the mean over the graphs that stay stable; settings at which every graph diverges are not plotted.')],
    '9.2': [('fig10_2_threshold', 'Threshold contagion: frequency of a full cascade (solid) and E[DM]/GATE (dotted) against the treated fraction, one panel per family. Dashed: E[DM]/GATE from the Gleeson-Cahalane recursion with each family\\textquoteright s own degree distribution (Appendix A.9). It places the transition at the simulated point on Erd\\H{o}s-R\\\'enyi, Barab\\\'asi-Albert and the power-law graph (at $\\theta = 0.5$ the simulated transition is one grid step later, since the graphs are bistable at $p^*$ itself), 15 to 17\\% early on the SBM (blocks of 1,000 seed unevenly), and 2 to 5 times late on Watts-Strogatz.')],
    '9.3': [('fig11_1_concave', 'Concave peer response $g(s) = K\\log(1 + s/K)$, $K = \\sigma\\bar k\\lambda$. Left: social multiplier against the nominal peer effect $\\nu\\rho(A)$ on Erd\\H{o}s-R\\\'enyi and Barab\\\'asi-Albert graphs for $\\sigma = 1$, 3, 10 (dotted, dashed, solid), against the linear model, which diverges at $\\nu\\rho = 1$. Middle: $E[\\mathrm{DM}]/\\mathrm{GATE}$, 10\\% treated; crosses: the annealed closed form of Appendix A.10. Right: $E[\\mathrm{DM}]/\\mathrm{GATE}$ against the local amplification $\\rho(J)$ for every configuration of the grid (both responses, all families), with the linear-model relation $1 - \\nu\\rho$.')],
    'Appendices': [('figA_1_closed_forms', 'Appendix A: closed form (horizontal axis) against simulation (vertical axis) for the approximations A.1 to A.9 (A.10 is compared in Figure~\\ref{fig:fig11_1_concave}), all families; each point is one parameter setting averaged over graphs. The line is equality. The panels show one quantity per entry: A.1 the multiplier (the spectral radius is checked in Table~\\ref{tab:families}); A.2 the lift by hop distance; A.3 $E[\\mathrm{DM}]/\\mathrm{GATE}$ under Bernoulli assignment; A.4 $E[\\mathrm{DM}]/\\mathrm{GATE}$ under cluster designs, with and without observation noise and with clusters from the wrong graph (the standard deviation of A.4 is compared in Section 7.1, not here); A.5 power, with and without CUPED; A.6 $E[\\mathrm{DM}]/\\mathrm{GATE}$ with new edges; A.7 the total GATE and the two estimates with new users, in production units; A.8 $E[\\mathrm{DM}]$ with a raised peer effect; A.9 the active fraction under threshold contagion. Outliers: Watts-Strogatz cascades (A.9) and hop distances (A.2), Barab\\\'asi-Albert cluster designs near the stability limit (A.4).'),
            ('fig6_4_peeking', 'Appendix D: rejection rates with daily peeking and with one test on day 14.'),
            ('fig7_2_noise', 'Appendix E: stationary variance of production on the highest-degree nodes under propagated noise.')],
}

# figure keys of paper.md (its own section numbers; supplementary figures under the appendix heading)
PAPER_FIGURES = {
    # main text (paper.md); supplementary figures are keyed by the full appendix heading
    '4.1': [('figM_1_multiplier', "The network multiplies the treatment effect, and on every graph family the multiplier grows without bound as $\\nu\\rho(A)$ approaches 1. Solid: the social multiplier GATE$/\\Delta\\lambda$ from simulation. Dashed: the prediction from the average number of friends, $1/(1 - \\nu\\bar k)$, which is close on graphs without hubs and badly wrong on Barab\\'asi-Albert and power-law graphs. The Erd\\H{o}s-R\\'enyi curve also stands for SBM.")],
    '4.2': [('figM_1b_attenuation', 'An A/B test with 10\\% of users treated reports about $1 - \\nu\\rho$ of the launch effect (dotted) on graphs without hubs; at the same $\\nu\\rho$, graphs with hubs lie above the line. Dashed: the closed form of Appendix B.3.'),
             ('figM_1d_attenuation_nukbar', "At the same peer effect per friend, $\\nu\\bar k$, the test reports less of the launch effect on graphs with hubs, and almost nothing as they approach their stability limit (dashed vertical lines, where $\\nu\\rho = 1$). Erd\\H{o}s-R\\'enyi lies under SBM."),
             ('figM_1c_treated_fraction', 'The share of the launch effect the test reports does not depend on the share of users treated ($\\nu\\rho = 0.5$).')],
    '4.3': [('figM_2_size_fixed_nurho', 'On graphs without hubs the test reports the same share of the launch effect at 10 thousand and at 100 million users. On graphs with hubs the share rises, because at a fixed $\\nu\\rho = 0.5$ the peer effect per friend falls as the graph grows. $\\bar k = 10$; SBM up to $10^7$ users.'),
             ('figM_2b_size_fixed_nukbar', 'On graphs with hubs, a peer effect per friend that is stable on a small network is unstable on a larger one: at a fixed $\\nu\\bar k = 0.3$, $\\nu\\rho(A)$ grows with the number of users and crosses the stability limit (dashed). $\\bar k = 10$.')],
    '5': [('figM_3_gate_paths', "After launch, the launch effect keeps building up for many periods, the longer the closer $\\nu\\rho$ is to 1. GATE(t) relative to its limit, on Watts-Strogatz (solid) and Barab\\'asi-Albert (dashed) graphs."),
           ('figM_3b_dm_paths', 'The A/B difference settles within the first periods, except very close to the stability limit, so running the test longer does not close the gap. DM(t) relative to its limit, on the same graphs.')],
    '6.1': [('figM_4_clusters_phi', 'Treating whole clusters of friends together brings the test closer to the launch effect, the more so the larger the share $\\phi$ of friendships kept inside clusters ($\\nu\\rho = 0.5$). Dashed: the mean field of Eq.~\\ref{eq:cluster}, which holds on graphs without hubs. Crosses and plus signs: SBM graphs with 20\\% to 70\\% of edges between blocks, with the true blocks and with METIS clusters of 1,000.'),
             ('figM_4b_clusters_size', "Clusters cut the error on graphs with communities or local clustering, barely help on Erd\\H{o}s-R\\'enyi graphs, and with very large clusters make it worse on graphs with hubs. Error (RMSE, solid) and bias (dotted) of the DM relative to the GATE against cluster size, $\\nu\\rho = 0.6$, weekly Poisson counts.")],
    '6.2': [('figM_5_extrapolation', 'Two simple estimators recover the launch effect when the observation windows are comparable; with a small drift in baseline posting the slope is far off, and with wrong friendships the plug-in is far off. Typical error (RMSE) of the DM, the slope estimator (a window before and one during the test, or two A/B tests at 10\\% and 90\\% treated) and the plug-in, as a percentage of the launch effect, in three settings (a) to (c) described in the text; mean over the five families.')],
    '6.3': [('figM_6_critical_effects', 'As $\\nu\\rho$ approaches 1 the launch effect (solid) grows without bound, while the A/B difference (dotted) stays near the direct effect $\\Delta\\lambda$. The axis is logarithmic in the distance from the stability limit: each tick is ten times closer to 1.'),
             ('figM_6b_critical_power', "The test loses power near the stability limit; pre-period adjustment (CUPED, dotted) recovers part of it when each user's baseline is the same before and during the test. One-sided test, $\\Delta\\lambda/\\lambda = 0.05$. The axis is logarithmic in the distance from the stability limit.")],
    '7.1': [('figM_7_new_edges', "When the feature makes users add friends, the test reports a shrinking share of the launch effect as more friends are added. A cluster split (solid) keeps most of the effect only when the new friends are inside the cluster (friends of friends, circles); with random users (squares) its share falls as fast as under the user-level split. Dotted: user-level split. SBM (green) and Barab\\'asi-Albert (red) graphs. SBM and Barab\\'asi-Albert stand for graphs with and without communities; the other families are described in Appendix F.14.")],
    '7.2': [('figM_7b_new_users', "When the feature makes users invite new users, adding the new users' posts to the treated arm's total per original user (solid) captures the new users' own posts, which a launch adds too, and misses only the network effect, as in Section 4.2; the more users are invited, the larger the share it recovers. Leaving them out (dashed) captures only the treatment's effect on existing users. New users post half as much as existing users and befriend the inviter and two draws from the inviter's friends.")],
    '8.1': [('figM_9_threshold', "A behavior that spreads once enough friends adopt it: the test reports almost all of the launch effect below a critical treated fraction, and almost nothing above it. One color per threshold $\\theta$; solid: Erd\\H{o}s-R\\'enyi, where the tree formula of Appendix B.7 predicts the critical fraction (dotted vertical lines); dashed: Watts-Strogatz, where friends are often friends of each other and the switch comes earlier. The other families are described in Appendix F.16.")],
    '8.2': [('figM_8_concave_multiplier', "If each additional post a user sees adds less than the previous one, posting never grows without bound. Social multiplier with the response $g(s) = K\\log(1 + s/K)$, $K = \\sigma\\bar k\\lambda$, on Erd\\H{o}s-R\\'enyi (orange) and Barab\\'asi-Albert (red) graphs for $\\sigma = 1$, 3, 10 (dotted, dashed, solid), against the linear model, which diverges at $\\nu\\rho = 1$."),
             ('figM_8b_concave_attenuation', 'With a concave response the test reports about $1 - \\rho(J)$ of the launch effect, as in the linear model with $\\rho(J)$ in place of $\\nu\\rho$. Every configuration of the grid: both responses, all families, 10\\% treated.')],
    'B. Closed forms and approximations': [('figA_1_closed_forms', "The closed forms of Appendix B (horizontal) against the simulation (vertical): points on the diagonal mean that the formula predicts the simulated value. Six panels, for B.1 to B.4, B.6 and B.7. The main misses are Watts-Strogatz cascades (B.7) and hop distances (B.2), and Barab\\'asi-Albert cluster designs near the stability limit (B.4).")],
    'F.1 Degree heterogeneity': [('fig5_2_heterogeneity', 'At the same average number of friends, the more unequal the friend counts, the larger the multiplier, until production grows without bound (shaded). Gamma configuration model, $\\bar k = 20$. Dashed: $1/(1 - \\nu\\bar k)$, which ignores the spread; dash-dot: the closed form of Appendix B.1, which includes it. Open markers: the five base families at their own spread.')],
    'F.2 Spillover by distance': [('fig5_3_spillover_hops', 'Friends of treated users gain a little from the treatment, friends of friends much less. Lift of control users by hop distance from the nearest treated user, 2\\% treated. Dashed: the tree closed form of Appendix B.2, which fails on Watts-Strogatz graphs, where neighbors share most of their neighbors.')],
    'F.6 Saturation and targeted treatment': [('fig5_7_saturation', 'With random assignment the treated fraction does not change the share of the launch effect the test reports (left). Treating the top users by degree (dashed) or by untreated production (dotted) mostly measures that they post more anyway (middle); their own lift over their untreated production is close to the GATE on the heavy-tailed families and between half of the GATE and the GATE on the homogeneous ones (right). $\\nu\\rho = 0.5$.')],
    'F.9 Clusters from the wrong graph': [('fig8_2_wrong_graph', 'Clusters built from an inaccurate friendship graph keep most of their benefit as long as the communities can still be found. Error relative to the GATE against the share of edges replaced at random.')],
}



STRONG = set('∞≪λΔνρσγδθαμκτφψωΣᵀᵗ⁴≈≥≤≠−×·√²₁₂⟨⟩⌈⌉←\u0304^')
MATHWORDS = {'tr', 'log', 'max', 'min', 'exp', 'GATE', 'DM', 'ADE', 'E', 'O'}
TRAIL = ',.;:'


def _weak(tok):
    t = tok.strip('()[]' + TRAIL)
    if t == '':
        return tok.strip(TRAIL) in ('(', ')', '[', ']') or tok in '=+/<>'
    if t in '=+/<>-':
        return True
    if re.fullmatch(r'[A-Za-z]', t):
        return True
    if re.fullmatch(r'[A-Za-z](_[A-Za-z0-9]+)?', t):
        return True
    if re.fullmatch(r'(19|20)[0-9]{2}', t):        # years are text
        return False
    if tok.startswith('(') and re.fullmatch(r'\(?[0-9.]+', tok.rstrip(TRAIL)):   # '(0 for ...' starts text
        return False
    if re.fullmatch(r'[0-9.]+(/[0-9.]+)?', t):
        return True
    if re.fullmatch(r'[A-Za-z0-9()\[\]/_^{}.+=−-]+', t) and any(w in t for w in ('tr(', 'E[', '1ᵀ', 'M1', ')/', '/(')):
        return True
    return False


def _strong(tok):
    return any(c in STRONG for c in tok)


def _to_math(m):
    m = m.replace('1ᵀ', '\\mathbf{1}^{\\top}').replace('M1', 'M\\mathbf{1}').replace('^t1', '^t\\mathbf{1}')
    m = m.replace('k\u0304', '\\bar k').replace('c\u0304', '\\bar c')
    m = re.sub(r'k_max', r'k_{\\max}', m)
    m = re.sub(r'\btr\(', r'\\operatorname{tr}(', m)
    m = re.sub(r'\blog\b', r'\\log', m)
    m = re.sub(r'\b(GATE|DM|ADE)\b', r'\\mathrm{\1}', m)
    m = re.sub(r'([A-Za-z])_([A-Za-z0-9]{2,})', r'\1_{\2}', m)
    m = re.sub(r'\^([0-9]+)', r'^{\1}', m)
    # √(...) with one level of nested parentheses, then √ followed by one symbol with an optional
    # subscript; both get braces, so the radical covers the whole argument.
    m = re.sub(r'√\(((?:[^()]|\([^()]*\))*)\)', r'\\sqrt{\1}', m)
    m = re.sub(r'√(\\?[A-Za-z0-9]+(?:_(?:\{[^{}]*\}|[A-Za-z0-9]))?)', r'\\sqrt{\1}', m)
    for a, b in (('λ', '\\lambda '), ('Δ', '\\Delta '), ('ν', '\\nu '), ('ρ', '\\rho '),
                 ('σ', '\\sigma '), ('γ', '\\gamma '), ('δ', '\\delta '), ('θ', '\\theta '),
                 ('α', '\\alpha '), ('μ', '\\mu '), ('ᵀ', '^{\\top}'), ('≈', '\\approx '),
                 ('≥', '\\geq '), ('≤', '\\leq '), ('≠', '\\neq '), ('−', '-'), ('×', '\\times '),
                 ('·', '\\cdot '), ('√', '\\sqrt '), ('²', '^{2}'), ('₁', '_{1}'), ('₂', '_{2}'),
                 ('%', '\\%')):
        m = m.replace(a, b)
    return m


def mathify(text):
    """Split text into ('text', s) and ('math', s) pieces; formula spans become math."""
    toks = re.split(r'( )', text)
    words = [t for t in toks if t != ' ']
    n = len(words)
    is_m = [False] * n
    for i, w in enumerate(words):
        if _strong(w):
            is_m[i] = True
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if is_m[i]:
                continue
            if _weak(words[i]) and ((i > 0 and is_m[i - 1]) or (i + 1 < n and is_m[i + 1])):
                if not (i > 0 and is_m[i - 1] and words[i - 1][-1:] in TRAIL):
                    is_m[i] = True
                    changed = True
    pieces, i = [], 0
    while i < n:
        if not is_m[i]:
            pieces.append(('text', words[i]))
            i += 1
            continue
        j = i
        while j + 1 < n and is_m[j + 1] and words[j][-1:] not in TRAIL:
            j += 1
        span = ' '.join(words[i:j + 1])
        lead = ''
        trail = ''
        while span and span[-1] in TRAIL:
            trail = span[-1] + trail
            span = span[:-1]
        if span.startswith('(') and span.count('(') > span.count(')'):
            lead, span = '(', span[1:]
        if span.endswith(')') and span.count(')') > span.count('('):
            trail, span = ')' + trail, span[:-1]
        if lead:
            pieces.append(('text', lead))
        pieces.append(('math', span))
        if trail:
            pieces.append(('text', trail))
        i = j + 1
    out, prev = [], None
    for kind, sx in pieces:
        out.append((kind, sx))
    return out


FIGID = {}


def _set_figures(figs):
    """Select the figure table (draft or paper) and rebuild the label map 'x.y' -> figure file."""
    global FIGURES
    FIGURES = figs
    FIGID.clear()
    for _figs in FIGURES.values():
        for _name, _ in _figs:
            _m = re.match(r'fig([0-9A-Z]+)_(\d+)([bcd]?)_', _name)
            FIGID[f'{_m.group(1)}.{_m.group(2)}{_m.group(3)}'] = _name
    if figs is PAPER_FIGURES:
        # supplementary figures are referred to by their appendix label in paper.md
        FIGID.update({'B.1': 'figA_1_closed_forms', 'F.1': 'fig5_2_heterogeneity', 'F.2': 'fig5_3_spillover_hops', 'F.6': 'fig5_7_saturation',
                      'F.9': 'fig8_2_wrong_graph'})


_set_figures(DRAFT_FIGURES)


def _protect_refs(p, stash):
    """Replace figure and section references by placeholders before math detection, so that
    'Figure 6.3:' or 'Section 7.5:' is never absorbed into a formula; the placeholder is restored
    after the text has been escaped."""
    def fig(m):
        nums = m.group(2)
        word = 'Figures ' if (',' in nums or ' and ' in nums) else 'Figure '
        refs = re.sub(r'[0-9A-Z]+\.\d+[bcd]?', lambda n: '\\ref{fig:' + FIGID.get(n.group(0), '?') + '}', nums)
        stash.append(refs)
        return word + '\x02%d\x02' % (len(stash) - 1)

    def sec(m):
        stash.append(m.group(2))
        return m.group(1) + ' \x02%d\x02' % (len(stash) - 1)

    p = re.sub(r'(Figures?|figure) ((?:[0-9A-Z]+\.\d+[bcd]?)(?:(?:, | and )[0-9A-Z]+\.\d+[bcd]?)*)', fig, p)
    # a number followed by '/', '.' or a digit is a formula ('Section 7.5, 1/√(7c̄)'), not a section
    # equation references: 'Eq. [eq:x]' -> Eq.~\ref (tie, no sentence space); a bare '[eq:x]' -> \ref
    p = re.sub(r'(Eqs?\.) \[(eq:[\w-]+)\]', lambda m: (stash.append(m.group(1) + '~\\ref{' + m.group(2) + '}'), '\x02%d\x02' % (len(stash) - 1))[1], p)
    p = re.sub(r'\[(eq:[\w-]+)\]', lambda m: (stash.append('\\ref{' + m.group(1) + '}'), '\x02%d\x02' % (len(stash) - 1))[1], p)
    p = re.sub(r'\b(Sections?) (\d+(?:\.\d+)?(?:(?:, | and | to )\d+(?:\.\d+)?(?![\d./]))*)', sec, p)
    return p


def inline(t):
    # footnotes: ^[text]
    if '^[' in t:
        notes = []

        def _fn(m):
            notes.append(m.group(1))
            return 'QFOOTNOTE' + chr(65 + len(notes) - 1) + 'Q'
        t = re.sub(r'\^\[((?:[^\[\]]|\[[^\[\]]*\])+)\]', _fn, t)   # one level of [] inside, e.g. E[DM]
        res = _inline(t)
        for j, n in enumerate(notes):
            res = res.replace('QFOOTNOTE' + chr(65 + j) + 'Q', '\\footnote{' + _inline(n) + '}')
        return res
    return _inline(t)


def _inline(t):
    out, parts = [], re.split(r'(`[^`]*`|\$[^$]+\$)', t)
    expanded = []
    stash = []
    for p in parts:
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            expanded.append(p)
            continue
        if p.startswith('$') and p.endswith('$') and len(p) > 2:   # explicit LaTeX math, verbatim
            expanded.append('\x00' + p[1:-1] + '\x00')
            continue
        txt = []
        p = _protect_refs(p, stash)
        for kind, sx in mathify(p.replace('\\textbackslash{}', '\\')):
            if kind == 'math':
                if txt:
                    expanded.append(' '.join(txt) + ' ')
                    txt = []
                expanded.append('\x00' + _to_math(sx) + '\x00')
            else:
                txt.append(sx)
        if txt:
            expanded.append(' '.join(txt))
    parts = expanded
    for p in parts:
        if p.startswith('\x00'):
            out.append('$' + p.strip('\x00') + '$ ')
            continue
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            q = p[1:-1]
            for a, b in (('_', '\\_'), ('%', '\\%'), ('#', '\\#'), ('&', '\\&'), ('$', '\\$')):
                q = q.replace(a, b)
            out.append('\\texttt{' + q + '}')
            continue
        for a, b in (('&', '\\&'), ('%', '\\%'), ('#', '\\#'), ('_', '\\_'), ('$', '\\$'),
                     ('^', '\\textasciicircum{}'), ('~', '\\textasciitilde{}'),
                     ('{', '\\{'), ('}', '\\}')):
            p = p.replace(a, b)
        p = p.replace('\\textbackslash\\{\\}', '\\textbackslash{}')
        p = p.replace('\\textasciicircum\\{\\}', '\\textasciicircum{}')
        p = p.replace('\\textasciitilde\\{\\}', '\\textasciitilde{}')
        p = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'\\emph{\1}', p)
        p = p.replace('k\u0304', '\\ensuremath{\\bar k}').replace('c\u0304', '\\ensuremath{\\bar c}')
        p = re.sub(r'\\textasciicircum\{\}\\\{(.*?)\\\}', r'\\ensuremath{^{\1}}', p)
        p = re.sub(r'\\textasciicircum\{\}(-?[\w.]+)', r'\\ensuremath{^{\1}}', p)
        p = re.sub(r'\\\_(0)\b', r'\\ensuremath{_{\1}}', p)
        p = re.sub(r'(?<![\w\\])([A-Za-z])\\_([A-Za-z0-9]{1,3})\b', r'\\ensuremath{\1_{\2}}', p)
        p = re.sub(r'\x02(\d+)\x02', lambda m: stash[int(m.group(1))], p)
        p = re.sub(r'Table \[(\w+)\]', lambda m: 'Table~\\ref{tab:' + m.group(1) + '}', p)
        out.append(p)
    s = ''.join(out)
    # bold that spans a math segment (**text $x$ text**): the pairs were split across parts
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'\x02(\d+)\x02', lambda m: stash[int(m.group(1))], s)   # a placeholder that landed in math
    return re.sub(r'\$ +([,.;:)\]-])', r'$\1', s)


def table(rows, small=False):
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    head, body = cells[0], cells[2:]
    n = len(head)
    spec = '>{\\raggedright\\arraybackslash}X' * n
    s = '\\centering\n' + ('\\footnotesize\n\\renewcommand{\\arraystretch}{1.2}\n' if small else '\\small\n\\renewcommand{\\arraystretch}{1.35}\n')
    compact = any(re.match(r'^IMG\[', c) for r in body for c in r)
    if compact:
        # one-line cells (explicit <br> breaks only), spread over the full width
        s = '\\centering\n\\scriptsize\n\\renewcommand{\\arraystretch}{1.15}\n'
        s += '\\begin{tabular*}{\\linewidth}{@{}l@{\\extracolsep{\\fill}}' + 'c' * (n - 1) + '@{}}\n\\toprule\n'
    else:
        s += '\\begin{tabularx}{\\linewidth}{' + spec + '}\n\\toprule\n'
    s += ' & '.join('\\textbf{' + inline(c) + '}' for c in head) + ' \\\\\n\\midrule[\\heavyrulewidth]\n'
    def cell(c):
        mi = re.match(r'^IMG\[(\w+)\]$', c)
        if mi:
            return f'\\raisebox{{-0.5\\height}}{{\\includegraphics[height=1.7cm]{{../figures/{mi.group(1)}.png}}}}'
        if '<br>' in c:
            return '\\begin{tabular}[c]{@{}c@{}}' + ' \\\\ '.join(inline(x.strip()) for x in c.split('<br>')) + '\\end{tabular}'
        return inline(c)
    s += '\\\\\n\\hline\n'.join(' & '.join(cell(c) for c in r) for r in body)
    return s + ' \\\\\n\\bottomrule\n' + ('\\end{tabular*}\n' if compact else '\\end{tabularx}\n')


# paper only: wide figures placed at the bottom of their page, below the text
BOTTOM = {'figA_1_closed_forms'}
WIDE = {'fig4_1_families', 'fig8_3_extrapolation', 'figA_1_closed_forms', 'figM_5_extrapolation'}   # multi-row grids


def _wide(name):
    from PIL import Image
    w, h = Image.open(FIG / f'{name}.png').size
    return name in WIDE or w / h > 1.9


# paper only: wide appendix figures drawn narrower, so that Figures 30 to 33 share one page
NARROW = {'fig5_3_spillover_hops': '0.88', 'fig8_5_bayes': '0.88'}   # two-panel: panels as wide as a main-text figure


# review copy: sections whose figures go right after the paragraph that first cites them, so the
# section can be read linearly (elsewhere the review copy collects figures at the section end)
# paper: sections whose figures go right after the paragraph that first cites them
PAPER_INLINE = {'6.3'}
REVIEW_INLINE = {'4.1', '4.2', '4.3', '5', '6.1', '6.2', '6.3', '7.1', '7.2', '8.1', '8.2'}


def figures(key, only=None):
    s = ''
    for name, cap in FIGURES.get(key, []):
        if only is not None and name != only:
            continue
        env = 'figure*' if _wide(name) else 'figure'
        place = '!tp' if (env == 'figure*' and PAPER_MODE) else '!htbp'
        if PAPER_MODE and not REVIEW_MODE and name in BOTTOM:
            place = '!b'   # stfloats: a two-column figure at the bottom of the page
        width = '\\textwidth' if env == 'figure*' else '\\columnwidth'
        if PAPER_MODE and not REVIEW_MODE and name in NARROW:
            width = NARROW[name] + width
        elif PAPER_MODE and env == 'figure':
            # main-text column figures at 90% width, so the main text fits in 10 pages
            width = '0.9' + width
        s += (f'\\begin{{{env}}}[{place}]\\centering\n'
              f'\\includegraphics[width={width}]{{../figures/{name}.png}}\n'
              f'\\caption{{{cap}}}\\label{{fig:{name}}}\n\\end{{{env}}}\n')
    return s


def convert(md):
    lines = md.split('\n')
    out, depth, i, pending = [], 0, 0, None
    title, abstract, in_abstract, appendix = '', [], False, False
    pending_table = None
    in_refs = False
    held, seen_section = [], False   # review mode: floats of the current section
    subs_seen = 0                    # review mode: subsections seen in the current section

    in_enum = False

    def close_lists(to=0):
        nonlocal depth, in_enum
        while depth > to:
            out.append('\\end{itemize}')
            depth -= 1
        if to == 0 and in_enum:
            out.append('\\end{enumerate}')
            in_enum = False

    inline_wait = []   # review copy: (key, figure name, reference id) waiting for their first citation

    def flush_figs():
        nonlocal pending
        if pending:
            close_lists()
            if (REVIEW_MODE and pending in REVIEW_INLINE) or (PAPER_MODE and not REVIEW_MODE and pending in PAPER_INLINE):
                ids = {v: k for k, v in FIGID.items()}
                inline_wait.extend((pending, n, ids.get(n)) for n, _ in FIGURES.get(pending, []))
            else:
                (held if REVIEW_MODE else out).append(figures(pending))
            pending = None

    def flush_inline(line=None):
        # with a line: the waiting figures it cites; without: all of them (at the next heading)
        for item in list(inline_wait):
            k, n, fid = item
            if line is None or (fid and re.search(re.escape(fid) + r'(?![0-9a-z])', line)):
                out.append(figures(k, only=n))
                inline_wait.remove(item)

    while i < len(lines):
        ln = lines[i]
        if ln.startswith('#'):
            close_lists()
            flush_inline()
            # the figure under the bare 'Appendices' heading is emitted after the next heading:
            # a figure* right after 'Appendices' would leave that page with only the heading
            carry = PAPER_MODE and not REVIEW_MODE and pending == 'Appendices'
            if not carry:
                flush_figs()
            level = len(ln) - len(ln.lstrip('#'))
            text = ln.lstrip('#').strip()
            if REVIEW_MODE and text != 'Abstract' and (level == 2 or (appendix and level == 3)):
                if seen_section:
                    # the previous section's figures and tables, then a new page
                    out.extend(held)
                    held.clear()
                    out.append('\\FloatBarrier\\clearpage')
                seen_section = True
                subs_seen = 0
            elif REVIEW_MODE and not appendix and level == 3:
                # review copy: every main-text subsection after the first starts a new page;
                # X.1 stays on the page of its section heading
                if subs_seen:
                    out.extend(held)
                    held.clear()
                    out.append('\\FloatBarrier\\clearpage')
                subs_seen += 1
            if in_abstract:
                in_abstract = False
                out.append('\\end{abstract}\n\\end{frontmatter}\n')
            if level == 1:
                title = text
                out.append('\\begin{frontmatter}\n\\title{' + inline(title) + '}\n'
                           '\\author{M\\\'arton Trencs\\\'eni (\\texttt{mtrencseni@gmail.com})}\n')
            elif text == 'Abstract':
                in_abstract = True
                out.append('\\begin{abstract}')
                i += 1
                continue
            else:
                m = re.match(r'(\d+(?:\.\d+)?)\.?\s+(.*)$', text)
                key, name = (m.group(1), m.group(2)) if m else (text, text)
                if not text.startswith('References'):
                    in_refs = False
                if text.startswith('Appendices'):
                    if PAPER_MODE:
                        # main text ends with balanced columns; appendices start on a new page, smaller
                        out.append('\\flushcolsend\n\\clearpage')
                    out.append('\\appendix\n\\renewcommand{\\thefigure}{\\arabic{figure}}\\renewcommand{\\thetable}{\\arabic{table}}\n\\section*{Appendices}')
                    if PAPER_MODE:
                        out.append('\\footnotesize')
                    if PAPER_MODE and not REVIEW_MODE:
                        # appendix only: up to three wide figures may share a page, so that the
                        # figures keep up with their subsections (the default 0.7 lets one through)
                        out.append('\\renewcommand{\\dbltopfraction}{0.95}'
                                   '\\renewcommand{\\dblfloatpagefraction}{0.8}'
                                   '\\setcounter{dbltopnumber}{3}'
                                   '\\setlength{\\dblfloatsep}{10pt plus 2pt minus 2pt}')
                    appendix = True
                elif text.startswith('References'):
                    out.append(('\\FloatBarrier\n' if PAPER_MODE else '') + '\\section*{References}\\small')
                    in_refs = True
                elif text.startswith('Open items'):
                    out.append('\\section*{' + inline(text) + '}')
                else:
                    cmd = {2: 'section', 3: 'subsection'}.get(level, 'subsubsection')
                    if PAPER_MODE and appendix and REVIEW_MODE:
                        # review copy only: figures stay within their own subsection; in the paper a
                        # barrier per appendix subsection left most appendix pages half empty
                        out.append('\\FloatBarrier')
                    out.append(f'\\{cmd}{{{inline(name)}}}')
                    if carry:
                        flush_figs()
                if key in FIGURES:
                    pending = key
                    if PAPER_MODE and (not appendix or not REVIEW_MODE) and key != 'Appendices':
                        # main-text figures go right after their heading, so that LaTeX meets them
                        # (and can place them) a page earlier than at the end of the section
                        flush_figs()
            i += 1
            continue
        if ln.startswith('*Draft'):
            i += 1
            continue
        if ln.startswith('<!--'):
            i += 1
            continue
        mcap = re.match(r'^TABLE\[(\w+)\]: (.*)$', ln)
        if mcap:
            pending_table = (mcap.group(1), mcap.group(2))
            i += 1
            continue
        if ln.strip().startswith('|'):
            close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i])
                i += 1
            ncol = len(rows[0].strip().strip('|').split('|'))
            env = 'table*'
            body_t = table(rows, small=appendix)
            if PLAIN_MODE:
                body_t = '\\color{gray}' + body_t
            cap = ''
            if pending_table:
                cap = f'\\caption{{{inline(pending_table[1])}}}\\label{{tab:{pending_table[0]}}}\n'
                pending_table = None
            (held if REVIEW_MODE else out).append(f'\\begin{{{env}}}[{"!tp" if PAPER_MODE else "!htbp"}]\n'
                + ((body_t + '\\vspace{4pt}\n' + cap) if 'IMG[' in ''.join(rows) else (cap + body_t)) + f'\\end{{{env}}}\n')
            continue
        me = re.match(r'^\d+\. (.*)$', ln)
        if me and not in_refs:
            if not in_enum:
                close_lists()
                out.append('\\begin{enumerate}[itemsep=1pt,topsep=2pt,leftmargin=*]')
                in_enum = True
            out.append('\\item ' + inline(me.group(1)))
            i += 1
            continue
        m = re.match(r'^(\s*)- (.*)$', ln)
        if m:
            d = len(m.group(1)) // 2 + 1
            while depth < d:
                # references: a plain list with hanging indent, no bullets
                out.append('\\begin{itemize}[label={},leftmargin=1.5em,itemindent=-1.5em,itemsep=2pt]'
                           if in_refs else '\\begin{itemize}')
                depth += 1
            close_lists(d)
            txt = m.group(2)
            if txt.startswith('PLAIN: '):
                out.append('\\item ' + inline(txt[7:]))
            elif PLAIN_MODE:
                out.append('\\item[\\textcolor{gray}{\\small\\textbullet}] \\tech{' + inline(txt) + '}')
            else:
                out.append('\\item ' + inline(txt))
            i += 1
            continue
        mdisp = re.match(r'^\$([^$]+)\$([,.;:]?)\s*(?:\{#(eq:[\w-]+)\})?\s*$', ln) if PAPER_MODE else None
        if mdisp:
            # a line that is only math is a displayed equation inside the surrounding paragraph
            close_lists()
            while out and out[-1] == '':
                out.pop()
            if out:
                out[-1] = out[-1].rstrip('\n')
            out.append('\\begin{equation}' + mdisp.group(1) + ('\\label{' + mdisp.group(3) + '}' if mdisp.group(3) else '') + '\\end{equation}')   # numbered; no trailing punctuation on displays
            i += 1
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            continue
        if ln.strip() == '':
            if PAPER_MODE:
                close_lists()
                out.append('')
            i += 1
            continue
        if PAPER_MODE and ln.startswith('>'):
            close_lists()
            if in_abstract:
                in_abstract = False
                out.append('\\end{abstract}\n\\end{frontmatter}\n')
            box = []
            while i < len(lines) and lines[i].startswith('>'):
                box.append(lines[i][1:].strip())
                i += 1
            paras = '\n'.join(box).split('\n\n') if '' in box else [' '.join(box)]
            title = 'For practitioners'
            first = paras[0]
            mt = re.match(r'^\*\*(.+?)\*\*\s*(.*)$', first, re.S)
            if mt:
                title, paras[0] = mt.group(1).rstrip('.'), mt.group(2)
            out.append('\\begin{practbox}{' + inline(title) + '}\n'
                       + '\n\n'.join(inline(' '.join(p.split('\n'))) for p in paras if p.strip())
                       + '\n\\end{practbox}\n')
            continue
        close_lists()
        if ln.startswith('PLAINPARA: '):
            out.append(inline(ln[11:]) + '\n\n')
        else:
            out.append(('\\tech{' + inline(ln) + '}' if PLAIN_MODE else inline(ln)) + '\n')
        if inline_wait:
            flush_inline(ln)
        i += 1
    close_lists()
    flush_figs()
    out.extend(held)
    return '\n'.join(out)


PRE = r"""\pdfminorversion=7
\documentclass[final,5p,times,twocolumn,number]{elsarticle}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

% Side margins widened by 50% over the elsarticle 5p defaults (as in PaxosLease Revisited).
\setlength{\textwidth}{484.246pt}
\setlength{\oddsidemargin}{-15.639pt}
\setlength{\evensidemargin}{-15.639pt}

\makeatletter
\def\ps@pprintTitle{%
  \let\@oddhead\@empty
  \let\@evenhead\@empty
  \let\@oddfoot\@empty
  \let\@evenfoot\@oddfoot
}
\makeatother

\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{flushend}
\usepackage{placeins}
\usepackage{stfloats}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{newunicodechar}
\usepackage{enumitem}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{4}
\setlength{\dblfloatsep}{20pt plus 2pt minus 2pt}
% gap above footnotes; the elsarticle default of 24pt costs two lines per footnote column
\setlength{\skip\footins}{10pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{20pt plus 2pt minus 2pt}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}
\urlstyle{rm}
\setlist[itemize]{itemsep=1pt,topsep=2pt,leftmargin=*}
\newunicodechar{²}{\ensuremath{^2}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{ν}{\ensuremath{\nu}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{ᵀ}{\ensuremath{^{\top}}}
\newunicodechar{₁}{\ensuremath{_1}}
\newunicodechar{₂}{\ensuremath{_2}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{≠}{\ensuremath{\neq}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{ψ}{\ensuremath{\psi}}
\newunicodechar{ω}{\ensuremath{\omega}}
\newunicodechar{ᵗ}{\ensuremath{^{t}}}
\newunicodechar{⁴}{\ensuremath{^{4}}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{⌈}{\ensuremath{\lceil}}
\newunicodechar{⌉}{\ensuremath{\rceil}}
\newunicodechar{⟨}{\ensuremath{\langle}}
\newunicodechar{⟩}{\ensuremath{\rangle}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{≪}{\ensuremath{\ll}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\usepackage[most]{tcolorbox}
% Practitioner boxes: full page width (a two-column float, so never split across pages),
% black and white, square corners.
\newtcolorbox{practinner}[1]{enhanced,sharp corners,colback=white,colframe=black,boxrule=0.5pt,
  colbacktitle=white,coltitle=black,titlerule=0pt,fonttitle=\bfseries\small,title={#1},
  attach title to upper={\ \ },left=5pt,right=5pt,top=4pt,bottom=4pt,fontupper=\small\setlength{\parskip}{4pt}}
% Boxes go only to the bottom of a page (stfloats), at most one per page; wide figures and tables
% go to the top or to float pages, so two boxes never share a page. The boxes are table* floats
% (without captions), so that they do not hold back the column figures in the figure queue.
\setcounter{dblbotnumber}{1}
\newenvironment{practbox}[1]{\begin{table*}[!b]\begin{practinner}{#1}}{\end{practinner}\end{table*}}
\begin{document}
"""


# ---- numbered citations (paper mode): "Eckles, Karrer and Ugander (2017)" -> "Eckles, Karrer and
# Ugander [12]"; "(Aronow and Samii 2017)" -> "(Aronow and Samii [3])"; the reference list moves to
# the end of the document (after the appendices), numbered in its alphabetical order.
_INITIAL = re.compile(r"^(?:[A-Z][a-z]?\.\s?-?)+$")


def _surnames(authors):
    return {t.strip() for t in authors.split(',') if t.strip() and not _INITIAL.match(t.strip())}


def _cite_token(k, opt=''):
    tok = 'QCITE' + ''.join(chr(65 + int(d)) for d in str(k)) + 'Q'
    return tok + ('QOPT' + opt + 'QENDQ' if opt else '')


def number_citations(md):
    lines = md.split('\n')
    r0 = next(i for i, l in enumerate(lines) if l.startswith('## References'))
    a0 = next(i for i, l in enumerate(lines) if l.startswith('## Appendices'))
    refs = [l[2:] for l in lines[r0 + 1:a0] if l.startswith('- ')]
    text = '\n'.join(lines[:r0] + lines[a0:])
    used = set()

    # bare citations without the author name: [cite: Trencseni 2023] -> [66]
    def _bare(m):
        name, year = m.group(1).strip(), m.group(2)
        for k, e in enumerate(refs, 1):
            if e.startswith(name + ',') and f'({year})' in e:
                used.add(k)
                return _cite_token(k)
        raise ValueError(f'no reference for {m.group(0)}')
    text = re.sub(r'\[cite:\s*([^\]]+?)\s+(\d{4}[a-z]?)\]', _bare, text)
    cands = []
    for k, e in enumerate(refs, 1):
        m = re.match(r'^(.*?)\((\d{4}[a-z]?)\)', e)
        authors, year = m.group(1), m.group(2)
        first = authors.split(',')[0].strip()
        names = _surnames(authors) | {'et', 'al', 'and'}
        pat = (r'(?<![\w-])(' + re.escape(first) + r"(?:[^\W\d]|[\s,.'\-])*?)"
               r'(?:\s\((' + year + r')(?:[,;]\s*([^)]*))?\)|\s(' + year + r')(?![\d\w]))')
        for mm in re.finditer(pat, text):
            words = re.findall(r'[^\W\d_][\w\-]*', mm.group(1))
            if all(w in names or any(w in n.split('-') or n.endswith(w) for n in names) for w in words):
                cands.append((mm.start(), mm.end(), k, mm))
    cands.sort(key=lambda c: (c[0], -c[1]))
    out, pos = [], 0
    for st, en, k, mm in cands:
        if st < pos:
            continue
        auth = mm.group(1).rstrip()
        if mm.group(2):
            rep = auth + ' ' + _cite_token(k, mm.group(3) or '')
        else:
            rep = auth + ' ' + _cite_token(k)
        out.append(text[pos:st] + rep)
        pos = en
        used.add(k)
    out.append(text[pos:])
    text = ''.join(out)
    for k, e in enumerate(refs, 1):
        if k not in used:
            print('warning: reference never cited:', e[:70], file=sys.stderr)
    for mm in re.finditer(r'[A-Z][^\s()]+(?: [^\s()]+){0,4}\s\(?(?:19|20)\d\d[a-z]?\)?', text):
        if 'QCITE' not in mm.group(0):
            print('warning: possible uncited reference:', mm.group(0), file=sys.stderr)
    return text, refs


def finish_citations(tex):
    dec = lambda s: ''.join(str(ord(c) - 65) for c in s)
    # a bare citation at the start of a phrase gets no tie before it
    tex = re.sub(r' ?QCITE([A-J]+)QQOPT(.*?)QENDQ', lambda m: '~\\cite[' + m.group(2) + ']{r' + dec(m.group(1)) + '}', tex)
    return re.sub(r' ?QCITE([A-J]+)Q', lambda m: '~\\cite{r' + dec(m.group(1)) + '}', tex)


def bibliography(refs):
    items = '\n'.join(f'\\bibitem{{r{k}}} ' + inline(e) for k, e in enumerate(refs, 1))
    # every remaining figure is placed before the references start
    # after \clearpage the first column keeps the room left by the wide figures of the previous
    # page (stfloats), so the references would fill only part of it; reset the column height
    return ('\n\\FloatBarrier\n'
            '\\makeatletter\\global\\@colroom\\@colht\\global\\vsize\\@colht\\makeatother\n'
            '\\begingroup\\footnotesize\n\\begin{thebibliography}{99}\n') + items + '\n\\end{thebibliography}\n\\endgroup\n'

import sys
args = [a for a in sys.argv[1:] if not a.startswith('--')]
PLAIN_MODE = '--plain' in sys.argv
REVIEW_MODE = '--review' in sys.argv   # paper layout, each section on a new page, floats at section end
PAPER_MODE = '--paper' in sys.argv or REVIEW_MODE
_set_figures(PAPER_FIGURES if PAPER_MODE else DRAFT_FIGURES)
src = HERE / (args[0] if args else 'draft.md')
dst = HERE / (args[1] if len(args) > 1 else 'draft.tex')
md = src.read_text()
refs = None
if PAPER_MODE:
    md, refs = number_citations(md)
body = convert(md)
if refs is not None:
    body = finish_citations(body + bibliography(refs))
# appendix headings carry their own letters (A., E.1); unnumbered, so elsarticle does not prefix 'Appendix .1.'
if '\\section*{Appendices}' in body:
    _a, _b = body.split('\\section*{Appendices}', 1)
    _b = _b.replace('\\subsection{', '\\subsection*{').replace('\\subsubsection{', '\\subsubsection*{')
    body = _a + '\\section*{Appendices}' + _b
pre = PRE.replace('\\begin{document}', '\\newcommand{\\tech}[1]{{\\small\\color{gray}#1}}\n\\begin{document}')
dst.write_text(pre + body + '\n\\end{document}\n')
print('wrote', dst)
