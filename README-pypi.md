# Distributing on pypi

## Tag

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

## PyPI

* edit setup.py to update "version" and "download_url" to reference version X.Y.Z
* create ~/.pypirc following https://packaging.python.org/guides/migrating-to-pypi-org/#uploading

```bash
python3 setup.py sdist upload -r pypi
```


## Local installation

* edit setup.py to update "version" and "download_url" to reference version X.Y.Z
* create ~/.pypirc following https://packaging.python.org/guides/migrating-to-pypi-org/#uploading

```bash
python3 setup.py sdist upload -r pypi
```



