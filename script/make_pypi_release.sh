#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./make_pypy_release.sh X.Y.Z"
    exit 2
fi

cd ..
git tag $1 -m "vipy-$1"
git push --tags origin master
python3 setup.py sdist upload -r pypi
