from setuptools import setup, find_packages
setup(
    name = 'vipy',
    version = '0.3',
    packages = find_packages(),
    description = 'Visym python tools for privacy preserving computer vision',
    author = 'Jeffrey Byrne',
    author_email = 'jeff@visym.com',
    url = 'https://github.com/visym/vipy',
    download_url = 'https://github.com/visym/vipy/archive/0.3.tar.gz', 
    install_requires=[
        "numpy",  
        "scipy",
        "scikit-learn",
        "matplotlib",    
        "dill",
        "pillow==5.4.1",
    ],
    keywords = ['vision', 'learning', 'ML', 'CV'], 
    classifiers = [],
)

