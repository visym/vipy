from setuptools import setup, find_packages
setup(
    name = 'vipy',
    version = '0.5',
    packages = find_packages(),
    description = 'Visym python tools for privacy preserving computer vision',
    long_description = 'Visym Labs provides privacy preserving computer vision that does not leak private information without your consent',
    author = 'Jeffrey Byrne',
    author_email = 'jeff@visym.com',
    url = 'https://github.com/visym/vipy',
    download_url = 'https://github.com/visym/vipy/archive/0.5.tar.gz', 
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

