Creating a pypi package
-------------------
```bash
 git commit -am "message"
 git push
 git tag X.Y.Z -m "vipy-X.Y.Z"
 git push --tags origin master
```
edit setup.py to create new version

The git sequence to delete a tag
```bash
   git tag -d x.y
   git push origin :refs/tags/x.y
```

 create ~/.pypirc following https://packaging.python.org/guides/migrating-to-pypi-org/#uploading
 edit setup.py to point to new version
 python3 setup.py register -r pypi
 python3 setup.py sdist upload -r pypi


