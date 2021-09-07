#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./make_pypy_release.sh X.Y.Z"
    exit 2
fi

source ./make_documentation.sh --commit
git push

# Make release
cd ..
git tag $1 -m "vipy-$1"
git push --tags origin master

# Upload me
python3 setup.py sdist bdist_wheel
twine upload dist/*

# Cleanup
rm -rf dist/
rm -rf build/
rm -rf vipy.egg-info/

