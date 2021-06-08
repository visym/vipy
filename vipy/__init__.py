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

# Design Goals

Vipy was created with three design goals.  

* **Simplicity**.  Annotated Videos and images should be as easy to manipulate as the pixels.  We provide a simple fluent API that enables the transformation of media so that pixels are transformed along with the annotations.  We provide a comprehensive unit test suite to validate this pipeline with continuous integration.
* **Portability**.  Vipy was designed with the goal of allowing it to be easily retargeted to new platforms.  For example, deployment on a serverless architecture such as AWS lambda has restrictions on the allowable code that can be executed in layers.  We designed Vipy with minimal dependencies on standard and mature machine learning tool chains (numpy, matplotlib, ffmpeg, pillow) to ensure that it can be ported to new computational environments. 
* **Efficiency**.  Vipy is written in pure python with the goal of performing in place operations and avoiding copies of media whenever possible.  This enables fast video processing by operating on videos as chains of transformations.  The documentation describes when an object is changed in place vs. copied.  Furthermore, loading of media is delayed until explicitly requested by the user (or the pixels are needed) to enable lazy loading for distributed processing.  


# Getting started

See the [demos](https://github.com/visym/vipy/tree/master/demo) as a starting point.


## Import

Vipy was designed to define annotated videos and imagery as collections of python objects.  The core objects for images are:

* [vipy.image.Scene](image.html#vipy.image.Scene)
* [vipy.object.Detection](object.html#vipy.object.Detection)
* [vipy.geometry.BoundingBox](geometry.html#vipy.geometry.BoundingBox)

The core objects for videos:

* [vipy.video.Scene](video.html#vipy.video.Scene)
* [vipy.object.Track](object.html#vipy.object.Track)
* [vipy.activity.Activity](activity.html#vipy.activity.Activity)

See the documentation for each object for how to construct them.  

## Customization

You can set the following environment variables to customize the output of vipy

* **VIPY_CACHE**='/path/to/directory.  This directory will contain all of the cached downloaded filenames when downloading URLs.  For example, the following will download all media to '~/.vipy'.

```python
os.environ['VIPY_CACHE'] = vipy.util.remkdir('~/.vipy')
vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').download()
```

This will output an image object:
```python
<vipy.image: filename="/Users/jebyrne/.vipy/1920px-Bubo_virginianus_06.jpg", filename="/Users/jebyrne/.vipy/1920px-Bubo_virginianus_06.jpg", url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg">
```

This provides control over where large datasets are cached on your local file system.  By default, this will be cached to the system temp directory.

* **VIPY_AWS_ACCESS_KEY_ID**='MYKEY'.  This is the [AWS key](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to download urls of the form "s3://".  
* **VIPY_AWS_SECRET_ACCESS_KEY**='MYKEY'.   This is the [AWS secret key](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to download urls of the form "s3://".


## Parallelization

Vipy includes integration with [Dask Distributed](https://distributed.dask.org) for parallel processing of video and images.   This is useful for video preprocessing of datasets to export cached tensors for training.

For example, to export torch tensors for a list of video objects using four parallel processes:

```python
with vipy.globals.parallel(4):
    vipy.batch.Batch(my_list_of_vipy_videos).map(lambda v: v.torch()).result()
```

This supports integration with distributed schedulers for massively parallel operation.

## Export

All vipy objects can be imported and exported to JSON for interoperatability with other tool chains.  This allows for introspection of the vipy object state providing transparency

```python
vipy.video.RandomScene().json()
```

## Versioning

To determine what vipy version you are running you can use:

>>> vipy.__version__
>>> vipy.version.is_at_least('1.11.1') 

# Contact

<info@visym.com>

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

