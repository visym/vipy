#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./make_pypy_release.sh X.Y.Z"
    exit 2
fi

cd ..

# Update docs for tag
pdoc vipy -o ./docs --html --force --template-dir ./docs/templates
mv ./docs/vipy/* ./docs
rmdir ./docs/vipy
git add ./docs/*

# Make release
git tag $1 -m "vipy-$1"
git push --tags origin master
python3 setup.py sdist upload -r pypi
