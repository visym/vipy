# Distributing on pypi

# Tag

To create a tag in the repo

```bash
    git commit -am "message"
    git push
    git tag X.Y.Z -m "vipy-X.Y.Z"
    git push --tags origin master
```

To delete a tag in the repo

```bash
    git tag -d X.Y.Z
    git push origin :refs/tags/X.Y.Z
```

# PyPI distribution

* Edit vipy/version.py to update the version number to match the tag
* create ~/.pypirc following https://packaging.python.org/guides/migrating-to-pypi-org/  # uploading
* minimum required setuptools >= 38.6.0

```bash
python3 -m pip install --upgrade setuptools wheel twine
python3 setup.py sdist upload -r pypi
```


# Local installation (virtualenv)

```bash
cd /path/to/vipy
pip install -e .
```
