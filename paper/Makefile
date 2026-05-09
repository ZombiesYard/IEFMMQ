# ===========================================================================
#  SimTutor Paper — Makefile
#  Usage:
#    make           — build the PDF (pdflatex → bibtex → pdflatex ×2)
#    make clean     — remove build artifacts, keep PDF
#    make distclean — remove everything including PDF
#    make view      — open PDF with system viewer
# ===========================================================================

MAIN   = main
LATEX  = pdflatex
BIBTEX = bibtex
VIEWER = xdg-open

.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/00_abstract.tex \
              sections/01_introduction.tex \
              sections/02_background_related_work.tex \
              sections/03_system_design.tex \
              sections/04_methodology.tex \
              sections/05_implementation.tex \
              sections/06_evaluation.tex \
              sections/07_discussion.tex \
              sections/08_conclusion.tex \
              references.bib
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	      $(MAIN).nav $(MAIN).snm $(MAIN).vrb $(MAIN).brf

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	$(VIEWER) $(MAIN).pdf 2>/dev/null &
