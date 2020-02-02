from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'vipy',
    author = 'Jeffrey Byrne',
    author_email = 'jeff@visym.com',    
    version = '0.5',
    packages = find_packages(),
    description = 'Visym Python Tools for Privacy Preserving Computer Vision',
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/visym/vipy',
    download_url = 'https://github.com/visym/vipy/archive/0.5.tar.gz', 
    install_requires=[
        "numpy",  
        "scipy",
        "matplotlib",    
        "dill",
        "pillow",
        "ffmpeg-python"
    ],
    keywords = ['vision', 'learning', 'ML', 'CV'], 
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ]
)

