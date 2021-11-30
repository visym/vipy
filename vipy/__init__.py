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

The VIPY tools are designed for simple and intuitive interaction with videos and images.  Try to create a `vipy.video.Scene` object:

```python
v = vipy.video.RandomScene()
```

Videos are constructed from URLs (e.g. RTSP/RTMP live camera streams, YouTube videos, public or keyed AWS S3 links), SSH accessible paths, local filenames, `vipy.image.Image` frame lists, numpy arrays or pytorch tensors.  In this example, we create a random video with tracks and activities.  Videos can be natively iterated:


```python
for im in v:
    print(im.numpy())
```

This will iterate and yield `vipy.image.Image` objects corresponding to each frame of the video.  You can use the `vipy.image.Image.numpy` method to extract the numpy array for this frame.  Long videos are streamed to avoid out of memory errors.  Under the hood, we represent each video as a filter chain to an FFMPEG pipe, which yields frames corresponding to the appropriate filter transform and framerate.  The yielded frames include all of the objects that are present in the video at that frame accessible with the `vipy.image.Scene.objects` method.

VIPY supports more complex iterators.  For example, a common use case for activity detection is iterating over short clips in a video.  You can do this using the stream iterator:


```python
for c in v.stream().clip(16):
    print(c.torch())
```
       
This will yield `vipy.video.Scene` objects each containing a `vipy.video.Stream.clip` of length 16 frames.  Each clip overlaps by 15 frames with the next clip, and each clip includes a threaded copy of the pixels.  This is useful to provide clips of a fixed length that are output for every frame of the video.  Each clip contais the tracks and activities within this clip time period.  The method `vipy.video.Video.torch` will output a torch tensor suitable for integration into a pytorch based system.

These python iterators can be combined together in complex ways

```python
for (im, c, imdelay) in (v, v.stream().clip(16), v.stream().frame(delay=10), a_gpu_function(v.stream().batch(16)))
    print(im, c.torch(), imdelay)
```

This will yield the current frame, a video `vipy.video.Stream.clip` of length 16, a `vipy.video.Stream.frame` 10 frames ago and a `vipy.video.Stream.batch` of 16 frames that is designed for computation and transformation on a GPU.  All of the pixels are copied in threaded processing which is efficiently hidden by GPU I/O bound operations.  For more examples of complex iterators in real world use cases, see the [HeyVi package](https://github.com/visym/heyvi) for open source visual analytics.

Videos can be transformed in complex ways, and the pixels will always be transformed along with the annotations.

```python
v.fliplr()          # flip horizontally
v.zeropad(10, 20)   # zero pad the video horizontally and vertically
v.mindim(256)       # change the minimum dimension of the video
v.framerate(10)     # change the framerate of the video 
```

The transformation is lazy and is incorporated into the FFMPEG complex filter chain so that the transformation is applied when the pixels are needed.  You can always access the current filter chain using `vipy.video.Video.commandline` which will output a commandline string for the ffmpeg executable that you can use to get a deeper underestanding of the transformations that are applied to the video pixels.

Finally, annotated videos can be displayed. 

```python
v.show()
v.show(notebook=True)
v.frame().show()
v.annotate('/path/to/visualization.mp4')
with vipy.video.Video(url='rtmps://youtu.be/...').mindim(512).framerate(5).stream(write=True) as s:
    for im in v.framerate(5):
        s.write(im.annotate().rgb())
```

This will `vipy.video.Scene.show` the video live on your desktop, in a jupyter notebook, show the first `vipy.video.Scene.frame` as a static image, `vipy.video.Scene.annotate` the video so that annotations are in the pixels and save the corresponding video, or live stream a 5Hz video to youtube.  All of the show methods can be configured to customize the colors or captions.

See the [demos](https://github.com/visym/vipy/tree/master/demo) for more examples.


## Parallelization

Vipy includes integration with [Dask Distributed](https://distributed.dask.org) for parallel processing of video and images.   This is useful for video preprocessing of datasets to prepare them for training.  

For example, we can construct a `vipy.dataset.Dataset` object from one or more videos.  This dataset can be transformed in parallel using two processes:

```python
D = vipy.dataset.Dataset(vipy.video.Scene(filename='/path/to/videofile.mp4'))
with vipy.globals.parallel(2):
    R = D.map(lambda v, outdir='/newpath/to/': v.mindim(128).framerate(5).saveas(so.path.join(outdir, vipy.util.filetail(v.filename()))))
```

The result is a transformed dataset which contains transformed videos downsampled to have minimum dimension 128, framerate of 5Hz, with the annotations transformed accordingly.  The `vipy.dataset.Dataset.map` method allows for a lambda function to be applied in parallel to all elements in a dataset.  The fluent design of the VIPY objects allows for easy chaining of video operations to be expressed as a lambda function.  VIPY objects are designed for integration into parallel processing tool chains and can be easily serialized and deserialized for sending to parallel worker tasks.  

VIPY supports integration with distributed schedulers for massively parallel operation.  

```python
D = vipy.dataset.Dataset('/path/to/directory/of/jsonfiles')
with vipy.globals.parallel(scheduler='10.0.0.1:8785'):
    R = D.map(lambda v, outdir='/newpath/to': vipy.util.bz2pkl(os.path.join(outdir, '%s.pkl.bz2' % v.videoid()), v.trackcrop().mindim(128).normalize(mean=(128,128,128)).torch()))
```

This will lazy load a directory of JSON files, where each JSON file corresponds to the annotations of a single video, such as those collected by [Visym Collector](https://visym.github.io/collector).   The `vipy.dataset.Dataset.map` method will communicate with a [scheduler](https://docs.dask.org/en/stable/how-to/deploy-dask/ssh.html) at a given IP address and port and will process the lambda function in parallel to the workers tasked by the scheduler.  In this example, the video will `vipy.video.Scene.trackcrop` the smallest bounding box containing all tracks in the video, resized so this crop is 128 on the smallest side, loaded and normalized to remove the mean, then saved as a torch tensor in a bzipped python pickle file.  This is useful for preprocesssing videos to torch tensors for fast loading of dataset augmentation during training.

## Import

Vipy was designed to define annotated videos and imagery as collections of python objects.  The core objects for images are:

* [vipy.image.Scene](image.html#vipy.image.Scene)
* [vipy.object.Detection](object.html#vipy.object.Detection)
* [vipy.geometry.BoundingBox](geometry.html#vipy.geometry.BoundingBox)

The core objects for videos:

* [vipy.video.Scene](video.html#vipy.video.Scene)
* [vipy.object.Track](object.html#vipy.object.Track)
* [vipy.activity.Activity](activity.html#vipy.activity.Activity)

See the documentation for each object for how to construct them.  Alternatively, see our [open source visual analytics](https://github.com/visym/heyvi) for construction of vipy objects from activity and object detectors.


## Export

All vipy objects can be imported and exported to JSON for interoperatability with other tool chains.  This allows for introspection of the vipy object state in an open format providing transparency

```python
vipy.image.owl().json()
```

## Customization

You can set the following environment variables to customize the output of vipy

* **VIPY_CACHE**='/path/to/directory'.  This directory will contain all of the cached downloaded filenames when downloading URLs.  For example, the following will download all media to '~/.vipy'.

```python
os.environ['VIPY_CACHE'] = vipy.util.remkdir('~/.vipy')
vipy.image.Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').download()
```

This will output an image object:
```python
<vipy.image: filename="~/.vipy/1920px-Bubo_virginianus_06.jpg", filename="~/.vipy/1920px-Bubo_virginianus_06.jpg", url="https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg">
```

This provides control over where large datasets are cached on your local file system.  By default, this will be cached to the system temp directory.

* **VIPY_AWS_ACCESS_KEY_ID**='MYKEY'.  This is the [AWS key](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to download urls of the form "s3://".  
* **VIPY_AWS_SECRET_ACCESS_KEY**='MYKEY'.   This is the [AWS secret key](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to download urls of the form "s3://".


## Versioning

To determine what vipy version you are running you can use:

>>> vipy.__version__
>>> vipy.version.is_at_least('1.11.1') 

# Contact

Visym Labs <info@visym.com>

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
import vipy.dataset

__version__ = vipy.version.VERSION

