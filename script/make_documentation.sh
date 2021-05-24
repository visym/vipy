#!/bin/bash

# Style guide:  https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
# Supported docstring format: https://pdoc3.github.io/pdoc/doc/pdoc/#supported-docstring-formats&gsc.tab=0

if [ "$#" -ge 3 ]; then
    echo "Usage: ./make_documentation.sh --commit --open"
    exit 2
fi


cd ..
pdoc vipy -o ./docs --html --force --template-dir ./docs/templates

for arg in "$@"
do
    case $arg in
        -c|--commit)

	git add ./docs/*
	git commit -m "documentation" ./docs/*
	    
        shift # Remove from processing
        ;;

        -o|--open)
	open ./docs/vipy/index.html
        shift # Remove from processing
        ;;

    esac
done
	
cd script
