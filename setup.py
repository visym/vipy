from setuptools import setup, find_packages

## Distributing on pypi
#
## Tag
#
# To create a tag in the repo
#
# ```bash
#     git commit -am "message"
#     git push
#     git tag X.Y.Z -m "vipy-X.Y.Z"
#     git push --tags origin master
# ```
#
# To delete a tag in the repo
# 
# ```bash
#     git tag -d X.Y.Z
#     git push origin :refs/tags/X.Y.Z
# ```
#
## PyPI distribution
#
# * Edit vipy/version.py to update the version number to match the tag
# * create ~/.pypirc following https://packaging.python.org/guides/migrating-to-pypi-org/  # uploading
# * minimum required setuptools >= 38.6.0
#
# ```bash
# python3 -m pip install --upgrade setuptools wheel twine
# python3 setup.py sdist upload -r pypi
# ```
#
# Local installation (virtualenv)
#
# ```bash
# cd /path/to/vipy
# pip install -e .
# ```

d_version = {}
with open("./vipy/version.py") as fp:
    exec(fp.read(), d_version)
version = d_version['VERSION']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vipy',
    author='Visym Labs',
    author_email='info@visym.com',
    version=version,
    packages=find_packages(),
    description='Visym Python Tools for Visual Dataset Transformation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/visym/vipy',
    download_url='https://github.com/visym/vipy/archive/%s.tar.gz' % version,
    install_requires=[
        "numpy",
        "matplotlib",
        "dill",
        "pillow",
        "ffmpeg-python"
    ],
    extras_require={
        'all': ['scikit-build', 'scipy', 'opencv-python', 'torch', 'ipython', 'scikit-learn', 'boto3', 'youtube-dl', 'dask', 'distributed', 'h5py', 'nltk', 'bs4', 'dropbox', 'pyyaml', 'pytest', 'paramiko', 'scp', 'ujson'],
        'complete': ['scikit-build', 'scipy', 'opencv-python', 'torch', 'ipython', 'scikit-learn', 'boto3', 'youtube-dl', 'dask', 'distributed', 'h5py', 'nltk', 'bs4', 'dropbox', 'pyyaml', 'pytest', 'paramiko', 'scp', 'ujson', 'numba']
        },
    keywords=['computer vision machine learning ML CV privacy video image'],    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)
