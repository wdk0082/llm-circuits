.PHONY: all count clean watch

all: main.pdf

main.pdf: main.tex refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Word count against the 4,000-word budget (see the header of main.tex).
# "Words in text" is the number to watch; placeholders carry their budgets
# as comments and must be counted mentally until written.
count:
	texcount -sum -brief main.tex

clean:
	latexmk -C
	rm -f main.bbl main.run.xml
