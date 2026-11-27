"""Build the arXiv submission: submission/network-effects-revisited-arxiv.zip.

The zip holds paper.tex (figure paths rewritten to figures/, full-line comments
removed) and every image it includes.  Before zipping, the flat source is
compiled with pdflatex in build/submission/ to check that it is self-contained.
Run from the repository root, after `make paper`.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / 'paper' / 'paper.tex'
WORK = ROOT / 'build' / 'submission'
OUT = ROOT / 'submission' / 'network-effects-revisited-arxiv.zip'


def main():
    src = TEX.read_text()
    figs = sorted(set(re.findall(r'\.\./figures/([^}]+)', src)))
    src = src.replace('../figures/', 'figures/')
    if '../' in src:
        sys.exit('paper.tex still references a file outside paper/')
    src = '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('%')) + '\n'

    if WORK.exists():
        shutil.rmtree(WORK)
    (WORK / 'figures').mkdir(parents=True)
    (WORK / 'paper.tex').write_text(src)
    for f in figs:
        shutil.copy2(ROOT / 'figures' / f, WORK / 'figures' / f)

    # compile twice (cross-references), as arXiv does
    for _ in range(2):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex'],
                           cwd=WORK, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit('pdflatex failed; see build/submission/paper.log')
    log = (WORK / 'paper.log').read_text(errors='replace')
    if re.search(r'undefined|Rerun to get', log):
        sys.exit('undefined references in build/submission/paper.log')

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(WORK / 'paper.tex', 'paper.tex')
        for f in figs:
            z.write(WORK / 'figures' / f, 'figures/' + f)
    print(f'{OUT.relative_to(ROOT)}: paper.tex + {len(figs)} figures; '
          f'test build in {WORK.relative_to(ROOT)}/paper.pdf')


if __name__ == '__main__':
    main()
