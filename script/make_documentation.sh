#!/bin/bash

# Style guide:  https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
# Supported docstring format: https://pdoc3.github.io/pdoc/doc/pdoc/#supported-docstring-formats&gsc.tab=0

if [ "$#" -ge 3 ]; then
    echo "Usage: ./make_documentation.sh --commit --open"
    exit 2
fi

COMMIT=0
OPEN=0

for arg in "$@"
do
    case $arg in
        -c|--commit)
	COMMIT=1
        shift # Remove from processing
        ;;

        -o|--open)
	OPEN=1
        shift # Remove from processing
        ;;

    esac
done



if [ ${COMMIT} == 1 ]; then
    git commit -am "documentation"  # necessary for git blobs to be correct
fi
   
cd ..
pdoc vipy -o ./docs --html --force --template-dir ./docs/templates
cp -r ./docs/vipy/* ./docs
rm -rf ./docs/vipy
sed -i -- "s|\"vipy/|\"|g" ./docs/index.js

if [ ${COMMIT} == 1 ]; then
    git add ./docs/*
    git commit -m "documentation" ./docs/*
fi

if [ ${OPEN} == 1 ]; then
    open ./docs/index.html
fi
	
cd script
