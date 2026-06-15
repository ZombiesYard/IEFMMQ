# ===========================================================================
#  基于证据的 VR 教学论文 — 中文版 Makefile
#  用法：
#    make           — 使用 tectonic 构建 PDF
#    make clean     — 移除构建产物，保留 PDF
#    make distclean — 移除包括 PDF 在内的所有构建产物
#    make view      — 使用系统查看器打开 PDF
# ===========================================================================

MAIN   = main
TECTONIC = tectonic
VIEWER = xdg-open

.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex Preamble.tex \
              sections/00_abstract.tex \
              sections/00_acknowledgements.tex \
              sections/01_introduction.tex \
              sections/02_background_related_work.tex \
              sections/03_system_design.tex \
              sections/04_methodology.tex \
              sections/05_implementation.tex \
              sections/06_evaluation.tex \
              sections/07_discussion.tex \
              sections/08_conclusion.tex \
              sections/09_appendix.tex \
              references.bib
	$(TECTONIC) --keep-logs --keep-intermediates $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	      $(MAIN).nav $(MAIN).snm $(MAIN).vrb $(MAIN).brf

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	$(VIEWER) $(MAIN).pdf 2>/dev/null &
