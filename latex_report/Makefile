MAIN=rapport_master_serenesense

pdf:
	latexmk -pdf -interaction=nonstopmode -file-line-error $(MAIN).tex
	latexmk -pdf -interaction=nonstopmode -file-line-error $(MAIN).tex
	biber $(MAIN)
	latexmk -pdf -interaction=nonstopmode -file-line-error $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).run.xml $(MAIN).bbl $(MAIN).bcf $(MAIN).blg

.PHONY: pdf clean
