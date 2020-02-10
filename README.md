Project
-------------------
VIPY: Visym Python Tools for Computer Vision and Machine Learning  
URL: https://github.com/visym/vipy/  

VIPY is a python package for representation, transformation and visualization of annotated videos and images.  Annotations are the ground truth provided by labelers (e.g. object bounding boxes, face identities, temporal activity clips), suitable for training computer vision systems.  VIPY provides tools to easily edit videos and images so that the annotations are transformed along with the pixels.  This enables a clean interface for transforming complex datasets for input to your training and testing pipeline.

VIPY provides:  

* Representation of videos with labeled activities and objects that can be resized, clipped, rotated, scaled and cropped
* Representation of images with object bounding boxes that can be manipulated as easily as editing an image
* Clean visualization of annotated images and videos 
* Lazy loading of images and videos suitable for distributed procesing (e.g. spark, dask)
* Straightforward integration into machine learning toolchains (e.g. torch, numpy)
* Fluent interface for chaining operations on videos and images
* Dataset download, unpack and import (e.g. ActivityNet, Kinetics)
* Video and image web search tools with URL downloading and caching


Requirements
-------------------
python 3.*  
ffmpeg (optional)  


Installation
-------------------

```python
pip install vipy
```

This package has the following required dependencies
```python
pip install numpy scipy matplotlib dill pillow ffmpeg-python
```

Optional dependencies
```python
pip install opencv-python ipython h5py nltk bs4 youtube-dl scikit-learn dropbox torch
```

Contact
-------------------
Jeffrey Byrne <<jeff@visym.com>>
