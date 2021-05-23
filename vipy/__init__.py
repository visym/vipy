"""
VIPY is a python package for representation, transformation and visualization of annotated videos and images.  Annotations are the ground truth provided by labelers (e.g. object bounding boxes, face identities, temporal activity clips), suit\
able for training computer vision systems.  VIPY provides tools to easily edit videos and images so that the annotations are transformed along with the pixels.  This enables a clean interface for transforming complex datasets for input to yo\
ur computer vision training and testing pipeline.

VIPY provides:

* Representation of videos with labeled activities that can be resized, clipped, rotated, scaled and cropped
* Representation of images with object bounding boxes that can be manipulated as easily as editing an image
* Clean visualization of annotated images and videos
* Lazy loading of images and videos suitable for distributed procesing (e.g. dask, spark)
* Straightforward integration into machine learning toolchains (e.g. torch, numpy)
* Fluent interface for chaining operations on videos and images
* Dataset download, unpack and import (e.g. Charades, AVA, ActivityNet, Kinetics, Moments in Time)
* Video and image web search tools with URL downloading and caching
* Minimum dependencies for easy installation (e.g. AWS Lambda)

"""

# Import all subpackages
import vipy.show  # matplotlib first
import vipy.activity
import vipy.annotation
import vipy.calibration
import vipy.downloader
import vipy.geometry
import vipy.image
import vipy.linalg
import vipy.math
import vipy.object
import vipy.util
import vipy.version
import vipy.video
import vipy.videosearch
import vipy.visualize

__version__ = vipy.version.VERSION

