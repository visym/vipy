#!/bin/bash

cd ..
#rm -rf ./docs/dataset ./docs/gui
pdoc vipy -o ./docs --html --force --template-dir ./docs/templates
cp -r ./docs/vipy/* ./docs
rm -rf  ./docs/vipy
git add ./docs/*
cd script

