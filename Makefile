MD = $(wildcard *.md)
HTML = $(MD:.md=.html)

all: index.html $(HTML)

%.html: %.md
	pandoc -t html $^ > $@

index.html: README.md
	pandoc -t html README.md > index.html
