from setuptools import setup, find_packages
setup(
    name = 'vipy',
    version = '0.2',
    packages = find_packages(),
    description = 'Visym python tools for computer vision and machine learning',
    author = 'Jeffrey Byrne',
    author_email = 'jeff@visym.com',
    url = 'https://github.com/visym/vipy',
    download_url = 'https://github.com/visym/vipy/archive/0.2.tar.gz', 
    install_requires=[
        "opencv-python",
        "numpy",  
        "scipy",
        "scikit-learn",
        "matplotlib",    
        "dill",
        "ipython",
        "h5py", 
    ],
    keywords = ['vision', 'learning', 'ML', 'CV'], 
    classifiers = [],
)

