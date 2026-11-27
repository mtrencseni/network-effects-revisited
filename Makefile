PYTHON := .venv/bin/python
LATEXMK ?= latexmk

# Generated files land under build/, which is disposable and git-ignored.
BUILD := $(CURDIR)/build
PAPER_DIR := paper
TUTORIAL_DIR := paper/tutorial

.PHONY: all test reproduce-2023 experiments figures paper review pdfs submission tutorial clean

all: test paper tutorial

test:
	$(PYTHON) -m pytest -q tests

# reproduces every number and figure of the 2023 paper (Appendix D)
reproduce-2023:
	$(PYTHON) scripts/reproduce_2023.py --mode exact

# every simulation of the paper (long; resumable)
experiments:
	./run_all.sh

# all figures, from the stored results in results/
figures:
	cd experiments && ../$(PYTHON) figures.py

# paper/paper.md -> paper/paper.tex -> paper/paper.pdf
paper:
	$(PYTHON) $(PAPER_DIR)/make_tex.py paper.md paper.tex --paper
	mkdir -p $(BUILD)/latex
	cd $(PAPER_DIR) && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD)/latex paper.tex
	cp $(BUILD)/latex/paper.pdf $(PAPER_DIR)/paper.pdf

# review copy of the paper: each section on a new page, floats at section end
review:
	$(PYTHON) $(PAPER_DIR)/make_tex.py paper.md review.tex --review
	mkdir -p $(BUILD)/latex
	cd $(PAPER_DIR) && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD)/latex review.tex
	cp $(BUILD)/latex/review.pdf $(PAPER_DIR)/review.pdf

# both paper.pdf and review.pdf
pdfs: paper review

# arXiv upload: submission/network-effects-revisited-arxiv.zip (paper.tex + figures)
submission: paper
	$(PYTHON) scripts/make_submission.py

tutorial:
	cd $(TUTORIAL_DIR) && ../../$(PYTHON) logfig.py
	mkdir -p $(BUILD)/tutorial
	cd $(TUTORIAL_DIR) && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error \
		-outdir=$(BUILD)/tutorial tutorial.tex
	cp $(BUILD)/tutorial/tutorial.pdf $(TUTORIAL_DIR)/tutorial.pdf

clean:
	rm -rf $(BUILD)
