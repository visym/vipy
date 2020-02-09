Project
-------------------
VIPY: Visym Python Tools for Computer Vision and Machine Learning  
URL: https://github.com/visym/vipy/  

VIPY provides python tools for representation, transformation and visualization of annotated videos and images.  Annotations are the ground truth provided by labelers (e.g. object bounding boxes, face identities, temporal activity clips), suitable for training machine learning systems.  VIPY provides tools to easily edit videos and images so that the annotations are always updated along with them.  This enables a clean interface for transforming complex datasets for input to your training and testing pipeline.

VIPY provides:  

* Representation of videos with activities and objects that can be resized, clipped, rotated, scaled and cropped.
* Representation of images with object bounding boxes that can be manipulated as easily as editing an image
* Clean visualization of labeled images and videos 
* Fluent interface for chaining operations on videos and images
* Lazy loading of images and videos suitable for distributed procesing (e.g. spark, dask)
* Straightforward integration into machine learning toolchains (e.g. torch, numpy)
* Dataset download, unpack and import (e.g. ActivityNet, Kinetics700)


Requirements
-------------------
python 3.*  
ffmpeg(optional)  


Installation
-------------------

Required
```python
pip install numpy scipy matplotlib dill pillow ffmpeg-python
```

Optional
```python
pip install opencv-python ipython h5py nltk bs4 youtube-dl scikit-learn dropbox torch
```

Contact
-------------------
Jeffrey Byrne <<jeff@visym.com>>
