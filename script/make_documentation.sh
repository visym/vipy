#!/bin/bash

cd ..
pdoc vipy -o ./docs --html --force --template-dir ./docs/templates
git add ./docs/*
cd script

