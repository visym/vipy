[![PyPI version](https://badge.fury.io/py/vipy.svg)](https://badge.fury.io/py/vipy)  [![CI](https://github.com/visym/vipy/workflows/vipy%20unit%20tests/badge.svg)](https://github.com/visym/vipy/actions?query=workflow%3A%22vipy+unit+tests%22)

VIPY
-------------------
VIPY: Visym Python Tools for Visual Dataset Transformation    
Documentation: https://visym.github.io/vipy

VIPY is a python package for representation, transformation and visualization of annotated videos and images.  Annotations are the ground truth provided by labelers (e.g. object bounding boxes, face identities, temporal activity clips), suitable for training computer vision systems.  VIPY provides tools to easily edit videos and images so that the annotations are transformed along with the pixels.  This enables a clean interface for transforming complex datasets for input to your computer vision training and testing pipeline.

VIPY provides:  

* Representation of videos with labeled activities that can be resized, clipped, rotated, scaled and cropped
* Representation of images with object bounding boxes that can be manipulated as easily as editing an image
* Clean visualization of annotated images and videos 
* Lazy loading of images and videos suitable for distributed procesing (e.g. dask, spark)
* Straightforward integration into machine learning toolchains (e.g. torch, numpy)
* Fluent interface for chaining operations on videos and images
* Dataset download, unpack and import (e.g. Charades, AVA, ActivityNet, Kinetics, Moments in Time)
* Video and image web search tools with URL downloading and caching
* Minimum dependencies for easy installation (e.g. AWS Lambda, Flask)

[![VIPY MEVA dataset visualization](http://i3.ytimg.com/vi/_jixHQr5dK4/maxresdefault.jpg)](https://youtu.be/_jixHQr5dK4)


Requirements
-------------------
python 3.*  
[ffmpeg](https://ffmpeg.org/download.html) (required for videos)  
numpy, matplotlib, dill, pillow, ffmpeg-python   

Installation
-------------------

```python
pip install vipy
```

Optional dependencies are installable as a complete package:

```python
pip install pip --upgrade
pip install 'vipy[all]'
```

You will receive a friendly warning if attempting to use an optional dependency before installation.


Quickstart
-------------------
```python
import vipy
vipy.image.owl().mindim(512).zeropad(padwidth=150, padheight=0).show()
```
<img src="https://raw.githubusercontent.com/visym/vipy/master/docs/vipy_image_owl.jpg" width="700">

The [demos](https://github.com/visym/vipy/tree/master/demo) provide useful notebook tutorials to help you get started.
