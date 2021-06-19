URLS=[
"vipy/index.html",
"vipy/gui/index.html",
"vipy/gui/using_matplotlib.html",
"vipy/dataset/index.html",
"vipy/dataset/aflw.html",
"vipy/dataset/fddb.html",
"vipy/dataset/activitynet.html",
"vipy/dataset/casia.html",
"vipy/dataset/caltech101.html",
"vipy/dataset/lfw.html",
"vipy/dataset/facescrub.html",
"vipy/dataset/msceleb.html",
"vipy/dataset/imagenet.html",
"vipy/dataset/youtubefaces.html",
"vipy/dataset/megaface.html",
"vipy/dataset/mnist.html",
"vipy/dataset/caltech256.html",
"vipy/dataset/ethzshapes.html",
"vipy/dataset/hmdb.html",
"vipy/dataset/vggface.html",
"vipy/dataset/kinetics.html",
"vipy/dataset/vggface2.html",
"vipy/dataset/kthactions.html",
"vipy/dataset/meva.html",
"vipy/dataset/ava.html",
"vipy/dataset/charades.html",
"vipy/dataset/momentsintime.html",
"vipy/dataset/cifar.html",
"vipy/image.html",
"vipy/geometry.html",
"vipy/show.html",
"vipy/torch.html",
"vipy/video.html",
"vipy/math.html",
"vipy/util.html",
"vipy/linalg.html",
"vipy/metrics.html",
"vipy/camera.html",
"vipy/globals.html",
"vipy/flow.html",
"vipy/batch.html",
"vipy/ssim.html",
"vipy/calibration.html",
"vipy/annotation.html",
"vipy/downloader.html",
"vipy/videosearch.html",
"vipy/visualize.html",
"vipy/activity.html",
"vipy/object.html",
"vipy/version.html"
];
INDEX=[
{
"ref":"vipy",
"url":0,
"doc":"VIPY is a python package for representation, transformation and visualization of annotated videos and images. Annotations are the ground truth provided by labelers (e.g. object bounding boxes, face identities, temporal activity clips), suitable for training computer vision systems. VIPY provides tools to easily edit videos and images so that the annotations are transformed along with the pixels. This enables a clean interface for transforming complex datasets for input to your computer vision training and testing pipeline. VIPY provides:  Representation of videos with labeled activities that can be resized, clipped, rotated, scaled and cropped  Representation of images with object bounding boxes that can be manipulated as easily as editing an image  Clean visualization of annotated images and videos  Lazy loading of images and videos suitable for distributed procesing (e.g. dask, spark)  Straightforward integration into machine learning toolchains (e.g. torch, numpy)  Fluent interface for chaining operations on videos and images  Dataset download, unpack and import (e.g. Charades, AVA, ActivityNet, Kinetics, Moments in Time)  Video and image web search tools with URL downloading and caching  Minimum dependencies for easy installation (e.g. AWS Lambda)  Design Goals Vipy was created with three design goals.   Simplicity . Annotated Videos and images should be as easy to manipulate as the pixels. We provide a simple fluent API that enables the transformation of media so that pixels are transformed along with the annotations. We provide a comprehensive unit test suite to validate this pipeline with continuous integration.   Portability . Vipy was designed with the goal of allowing it to be easily retargeted to new platforms. For example, deployment on a serverless architecture such as AWS lambda has restrictions on the allowable code that can be executed in layers. We designed Vipy with minimal dependencies on standard and mature machine learning tool chains (numpy, matplotlib, ffmpeg, pillow) to ensure that it can be ported to new computational environments.   Efficiency . Vipy is written in pure python with the goal of performing in place operations and avoiding copies of media whenever possible. This enables fast video processing by operating on videos as chains of transformations. The documentation describes when an object is changed in place vs. copied. Furthermore, loading of media is delayed until explicitly requested by the user (or the pixels are needed) to enable lazy loading for distributed processing.  Getting started See the [demos](https: github.com/visym/vipy/tree/master/demo) as a starting point.  Import Vipy was designed to define annotated videos and imagery as collections of python objects. The core objects for images are:  [vipy.image.Scene](image.html vipy.image.Scene)  [vipy.object.Detection](object.html vipy.object.Detection)  [vipy.geometry.BoundingBox](geometry.html vipy.geometry.BoundingBox) The core objects for videos:  [vipy.video.Scene](video.html vipy.video.Scene)  [vipy.object.Track](object.html vipy.object.Track)  [vipy.activity.Activity](activity.html vipy.activity.Activity) See the documentation for each object for how to construct them.  Customization You can set the following environment variables to customize the output of vipy   VIPY_CACHE ='/path/to/directory. This directory will contain all of the cached downloaded filenames when downloading URLs. For example, the following will download all media to '~/.vipy'.   os.environ['VIPY_CACHE'] = vipy.util.remkdir('~/.vipy') vipy.image.Image(url='https: upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bubo_virginianus_06.jpg/1920px-Bubo_virginianus_06.jpg').download()   This will output an image object:      This provides control over where large datasets are cached on your local file system. By default, this will be cached to the system temp directory.   VIPY_AWS_ACCESS_KEY_ID ='MYKEY'. This is the [AWS key](https: docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to download urls of the form \"s3: \".   VIPY_AWS_SECRET_ACCESS_KEY ='MYKEY'. This is the [AWS secret key](https: docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html) to download urls of the form \"s3: \".  Parallelization Vipy includes integration with [Dask Distributed](https: distributed.dask.org) for parallel processing of video and images. This is useful for video preprocessing of datasets to export cached tensors for training. For example, to export torch tensors for a list of video objects using four parallel processes:   with vipy.globals.parallel(4): vipy.batch.Batch(my_list_of_vipy_videos).map(lambda v: v.torch( .result()   This supports integration with distributed schedulers for massively parallel operation.  Export All vipy objects can be imported and exported to JSON for interoperatability with other tool chains. This allows for introspection of the vipy object state providing transparency   vipy.video.RandomScene().json()    Versioning To determine what vipy version you are running you can use: >>> vipy.__version__ >>> vipy.version.is_at_least('1.11.1')  Contact  "
},
{
"ref":"vipy.gui",
"url":1,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib",
"url":2,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.escape_to_exit",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.flush",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imflush",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.show",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.noshow",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.savefig",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.figure",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.close",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.closeall",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imshow",
"url":2,
"doc":"Show an image in a figure window (optionally visible), reuse previous figure if it is the same shape",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.text",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.boundingbox",
"url":2,
"doc":"Draw a captioned bounding box on a previously shown image",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imdetection",
"url":2,
"doc":"Show bounding boxes from a list of vipy.object.Detections on the same image, plotted in list order with optional captions",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imframe",
"url":2,
"doc":"Show a scatterplot of fr= x1,y1],[x2,y2] .] 2D points overlayed on an image",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.frame",
"url":2,
"doc":"Show a scatterplot of fr= x1,y1],[x2,y2] .] 2D points",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.colorlist",
"url":2,
"doc":"Return a list of named colors that are higher contrast with a white background",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.edit",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.Annotate",
"url":2,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.Annotate.on_press",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.Annotate.on_release",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle",
"url":2,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.connect",
"url":2,
"doc":"connect to all the events we need",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.on_press",
"url":2,
"doc":"on button press we will see if the mouse is over us and store some data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.on_motion",
"url":2,
"doc":"on motion we will move the rect if the mouse is over us",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.on_release",
"url":2,
"doc":"on release we reset the press data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.disconnect",
"url":2,
"doc":"disconnect all the stored connection ids",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast",
"url":2,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.lock",
"url":2,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.connect",
"url":2,
"doc":"connect to all the events we need",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.on_press",
"url":2,
"doc":"on button press we will see if the mouse is over us and store some data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.on_motion",
"url":2,
"doc":"on motion we will move the rect if the mouse is over us",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.on_release",
"url":2,
"doc":"on release we reset the press data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.disconnect",
"url":2,
"doc":"disconnect all the stored connection ids",
"func":1
},
{
"ref":"vipy.dataset",
"url":3,
"doc":""
},
{
"ref":"vipy.dataset.aflw",
"url":4,
"doc":""
},
{
"ref":"vipy.dataset.aflw.AFLW",
"url":4,
"doc":""
},
{
"ref":"vipy.dataset.aflw.AFLW.dataset",
"url":4,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.aflw.AFLW.export",
"url":4,
"doc":"Export sqlite database file to aflw.csv",
"func":1
},
{
"ref":"vipy.dataset.aflw.landmarks",
"url":4,
"doc":"Return 21x2 frame array of landmark positions in 1-21 order, NaN if occluded",
"func":1
},
{
"ref":"vipy.dataset.aflw.eyes_nose_chin",
"url":4,
"doc":"Return 4x2 frame array of left eye, right eye nose chin",
"func":1
},
{
"ref":"vipy.dataset.fddb",
"url":5,
"doc":""
},
{
"ref":"vipy.dataset.fddb.FDDB",
"url":5,
"doc":"Manages the FDDB dataset: http: vis-www.cs.umass.edu/fddb"
},
{
"ref":"vipy.dataset.fddb.FDDB.fold",
"url":5,
"doc":"Return the foldnum as a list of vipy.image.Scene objects, each containing all vipy.object.Detection faces in the current image",
"func":1
},
{
"ref":"vipy.dataset.activitynet",
"url":6,
"doc":""
},
{
"ref":"vipy.dataset.activitynet.ActivityNet",
"url":6,
"doc":"Activitynet, provide a datadir='/path/to/store/activitynet'"
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.download",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.trainset",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.testset",
"url":6,
"doc":"ActivityNet test set does not include any annotations",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.valset",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.categories",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.analysis",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.casia",
"url":7,
"doc":""
},
{
"ref":"vipy.dataset.casia.WebFace",
"url":7,
"doc":""
},
{
"ref":"vipy.dataset.casia.WebFace.dataset",
"url":7,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.casia.WebFace.subjects",
"url":7,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.casia.WebFace.subjectid",
"url":7,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech101",
"url":8,
"doc":""
},
{
"ref":"vipy.dataset.caltech101.Caltech101",
"url":8,
"doc":"Caltech101, provide a datadir='/path/to/store/caltech101'"
},
{
"ref":"vipy.dataset.caltech101.Caltech101.download_and_unpack",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech101.Caltech101.dataset",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw",
"url":9,
"doc":""
},
{
"ref":"vipy.dataset.lfw.LFW",
"url":9,
"doc":"Datadir contains the unpacked contents of LFW from $URL -> /path/to/lfw"
},
{
"ref":"vipy.dataset.lfw.LFW.download",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.subjects",
"url":9,
"doc":"List of all subject names",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.subject_images",
"url":9,
"doc":"List of Images of a subject",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.dataset",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.dictionary",
"url":9,
"doc":"List of all Images of all subjects",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.list",
"url":9,
"doc":"List of all Images of all subjects",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.take",
"url":9,
"doc":"Return a represenative list of 128 images",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.pairsDevTest",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.pairsDevTrain",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.pairs",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub",
"url":10,
"doc":""
},
{
"ref":"vipy.dataset.facescrub.FaceScrub",
"url":10,
"doc":""
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.parse",
"url":10,
"doc":"Return a list of ImageDetections for all URLs in facescrub",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.download",
"url":10,
"doc":"Download every URL in dataset and store in provided filename",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.validate",
"url":10,
"doc":"Validate downloaded dataset and store cached list of valid bounding boxes and loadable images accessible with dataset()",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.dataset",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.stats",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.subjects",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.split",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.msceleb",
"url":11,
"doc":""
},
{
"ref":"vipy.dataset.msceleb.extract",
"url":11,
"doc":"https: github.com/cmusatyalab/openface/blob/master/data/ms-celeb-1m/extract.py",
"func":1
},
{
"ref":"vipy.dataset.msceleb.export",
"url":11,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.imagenet",
"url":12,
"doc":""
},
{
"ref":"vipy.dataset.imagenet.ImageNet",
"url":12,
"doc":"Provide datadir=/path/to/ILSVRC2012"
},
{
"ref":"vipy.dataset.imagenet.ImageNet.classes",
"url":12,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces",
"url":13,
"doc":""
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces",
"url":13,
"doc":""
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.subjects",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.videos",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.parse",
"url":13,
"doc":"Parse youtubefaces into a list of ImageDetections",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.splits",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.megaface",
"url":14,
"doc":""
},
{
"ref":"vipy.dataset.megaface.MF2",
"url":14,
"doc":""
},
{
"ref":"vipy.dataset.megaface.MF2.tinyset",
"url":14,
"doc":"Return the first (size) image objects in the trainset",
"func":1
},
{
"ref":"vipy.dataset.megaface.Megaface",
"url":14,
"doc":""
},
{
"ref":"vipy.dataset.megaface.Megaface.tinyset",
"url":14,
"doc":"Return the first (size) image objects in the dataset",
"func":1
},
{
"ref":"vipy.dataset.mnist",
"url":15,
"doc":""
},
{
"ref":"vipy.dataset.mnist.MNIST",
"url":15,
"doc":"download URLS above to outdir, then run export()"
},
{
"ref":"vipy.dataset.mnist.MNIST.trainset",
"url":15,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.mnist.MNIST.imtrainset",
"url":15,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.mnist.MNIST.testset",
"url":15,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.mnist.MNIST.imtestset",
"url":15,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech256",
"url":16,
"doc":""
},
{
"ref":"vipy.dataset.caltech256.Caltech256",
"url":16,
"doc":"Caltech256, provide a datadir='/path/to/store/caltech256'"
},
{
"ref":"vipy.dataset.caltech256.Caltech256.download_and_unpack",
"url":16,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech256.Caltech256.dataset",
"url":16,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ethzshapes",
"url":17,
"doc":""
},
{
"ref":"vipy.dataset.ethzshapes.ETHZShapes",
"url":17,
"doc":"ETHZShapes, provide a datadir='/path/to/store/ethzshapes'"
},
{
"ref":"vipy.dataset.ethzshapes.ETHZShapes.download_and_unpack",
"url":17,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ethzshapes.ETHZShapes.dataset",
"url":17,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.hmdb",
"url":18,
"doc":""
},
{
"ref":"vipy.dataset.hmdb.HMDB",
"url":18,
"doc":"Human motion dataset, provide a datadir='/path/to/store/hmdb'"
},
{
"ref":"vipy.dataset.hmdb.HMDB.download",
"url":18,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.hmdb.HMDB.dataset",
"url":18,
"doc":"Return a list of VideoCategory objects",
"func":1
},
{
"ref":"vipy.dataset.vggface",
"url":19,
"doc":""
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL",
"url":19,
"doc":""
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL.subjects",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL.dataset",
"url":19,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL.take",
"url":19,
"doc":"Randomly select n frames from dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace",
"url":19,
"doc":""
},
{
"ref":"vipy.dataset.vggface.VGGFace.subjects",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.wordnetid_to_name",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.dataset",
"url":19,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.fastset",
"url":19,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.take",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.by_subject",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics",
"url":20,
"doc":""
},
{
"ref":"vipy.dataset.kinetics.Kinetics700",
"url":20,
"doc":"Kinetics, provide a datadir='/path/to/store/kinetics'"
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.download",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.isdownloaded",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.trainset",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.testset",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.valset",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.categories",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.analysis",
"url":20,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics600",
"url":20,
"doc":"Kinetics, provide a datadir='/path/to/store/kinetics'"
},
{
"ref":"vipy.dataset.kinetics.Kinetics400",
"url":20,
"doc":"Kinetics, provide a datadir='/path/to/store/kinetics'"
},
{
"ref":"vipy.dataset.vggface2",
"url":21,
"doc":""
},
{
"ref":"vipy.dataset.vggface2.VGGFace2",
"url":21,
"doc":""
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.subjects",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.wordnetid_to_name",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.vggface2_to_vggface1",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.name_to_wordnetid",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.names",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.trainset",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.testset",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.split",
"url":21,
"doc":"Convert absolute path /path/to/subjectid/filename.jpg from training or testing set to (subjectid, filename.jpg)",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.frontalset",
"url":21,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.dataset",
"url":21,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.fastset",
"url":21,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.take",
"url":21,
"doc":"Randomly select n images from the dataset, or n images of a given subjectid",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.take_per_subject",
"url":21,
"doc":"Randomly select n images per subject from the dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.subjectset",
"url":21,
"doc":"Iterator for single subject",
"func":1
},
{
"ref":"vipy.dataset.kthactions",
"url":22,
"doc":""
},
{
"ref":"vipy.dataset.kthactions.KTHActions",
"url":22,
"doc":"KTH ACtions dataset, provide a datadir='/path/to/store/kthactions'"
},
{
"ref":"vipy.dataset.kthactions.KTHActions.split",
"url":22,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kthactions.KTHActions.download_and_unpack",
"url":22,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kthactions.KTHActions.dataset",
"url":22,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.meva",
"url":23,
"doc":""
},
{
"ref":"vipy.dataset.meva.KF1",
"url":23,
"doc":"Parse MEVA annotations (http: mevadata.org) for Known Facility 1 dataset into vipy.video.Scene() objects Kwiver packet format: https: gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/KPF-specification-v4.pdf Args: videodir: [str] path to Directory containing 'drop-01' repodir: [str] path to directory containing clone of https: gitlab.kitware.com/meva/meva-data-repo stride: [int] the integer temporal stride in frames for importing bounding boxes, vipy will do linear interpoluation and boundary handling n_videos: [int] only return an integer number of videos, useful for debugging or for previewing dataset withprefix: [list] only return videos with the filename containing one of the strings in withprefix list, useful for debugging contrib: [bool] include the noisy contrib anntations from DIVA performers d_category_to_shortlabel: [dict] is a dictionary mapping category names to a short displayed label on the video. The standard for visualization is that tracked objects are displayed with their category label (e.g. 'Person', 'Vehicle'), and activities are labeled according to the set of objects that performing the activity. When an activity occurs, the set of objects are labeled with the same color as 'Noun Verbing' (e.g. 'Person Entering', 'Person Reading', 'Vehicle Starting') where 'Verbing' is provided by the shortlabel. This is optional, and will use the default mapping if None verbose: [bool] Parsing verbosity merge: [bool] deduplicate annotations for each video across YAML files by merging them by mean spatial IoU per track (>0.5) and temporal IoU (>0) actor: [bool] Include only those activities that include an associated track for the primary actor: \"Person\" for \"person_ \" and \"hand_ \", else \"Vehicle\" disjoint: [bool]: Enforce that overlapping causal activities (open/close, enter/exit,  .) are disjoint for a track unpad: [bool] remove the arbitrary padding assigned during dataset creation Returns: a list of  vipy.video.Scene objects"
},
{
"ref":"vipy.dataset.meva.KF1.videos",
"url":23,
"doc":"Return list of activity videos",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.tolist",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.instances",
"url":23,
"doc":"Return list of activity instances",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.categories",
"url":23,
"doc":"Return a list of activity categories",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.analysis",
"url":23,
"doc":"Analyze the MEVA dataset to return helpful statistics and plots",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.review",
"url":23,
"doc":"Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from",
"func":1
},
{
"ref":"vipy.dataset.ava",
"url":24,
"doc":""
},
{
"ref":"vipy.dataset.ava.AVA",
"url":24,
"doc":"AVA, provide a datadir='/path/to/store/ava'"
},
{
"ref":"vipy.dataset.ava.AVA.download",
"url":24,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava.AVA.categories",
"url":24,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava.AVA.trainset",
"url":24,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava.AVA.valset",
"url":24,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades",
"url":25,
"doc":""
},
{
"ref":"vipy.dataset.charades.Charades",
"url":25,
"doc":"Charades, provide paths such that datadir contains the contents of 'http: ai2-website.s3.amazonaws.com/data/Charades_v1.zip' and annodir contains 'http: ai2-website.s3.amazonaws.com/data/Charades.zip'"
},
{
"ref":"vipy.dataset.charades.Charades.categories",
"url":25,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades.Charades.trainset",
"url":25,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades.Charades.testset",
"url":25,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades.Charades.review",
"url":25,
"doc":"Generate a standalone HTML file containing quicklooks for each annotated activity in the train set",
"func":1
},
{
"ref":"vipy.dataset.momentsintime",
"url":26,
"doc":""
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments",
"url":26,
"doc":"Multi-Moments in Time: http: moments.csail.mit.edu/ >>> d = MultiMoments('/path/to/dir') >>> valset = d.valset() >>> valset.categories()  return the dictionary mapping integer category to string >>> valset[1].categories()  return set of categories for this clip >>> valset[1].category()  return string encoded category for this clip (comma separated activity indexes) >>> valset[1].play()  Play the original clip >>> valset[1].mindim(224).show()  Resize the clip to have minimum dimension 224, then show the modified clip >>> valset[1].centersquare().mindim(112).saveas('out.mp4')  modify the clip as square crop from the center with mindim=112, and save to new file >>> valset[1].centersquare().mindim(112).normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225 .torch(startframe=0, length=16)  export 16x3x112x112 tensor"
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments.categories",
"url":26,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments.trainset",
"url":26,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments.valset",
"url":26,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.cifar",
"url":27,
"doc":""
},
{
"ref":"vipy.dataset.cifar.CIFAR10",
"url":27,
"doc":"vipy.dataset.cifar.CIFAR10 class >>> D = vipy.dataset.cifar.CIFAR10('/path/to/outdir') >>> (x,y) = D.trainset() >>> im = D[0].mindim(512).show() download URLS above to outdir, then run export()"
},
{
"ref":"vipy.dataset.cifar.CIFAR10.classes",
"url":27,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.cifar.CIFAR10.trainset",
"url":27,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.cifar.CIFAR10.testset",
"url":27,
"doc":"",
"func":1
},
{
"ref":"vipy.image",
"url":28,
"doc":""
},
{
"ref":"vipy.image.Image",
"url":28,
"doc":"vipy.image.Image class The vipy image class provides a fluent, lazy interface for representing, transforming and visualizing images. The following constructors are supported: >>> im = vipy.image.Image(filename=\"/path/to/image.ext\") All image file formats that are readable by PIL are supported here. >>> im = vipy.image.Image(url=\"http: domain.com/path/to/image.ext\") The image will be downloaded from the provided url and saved to a temporary filename. The environment variable VIPY_CACHE controls the location of the directory used for saving images, otherwise this will be saved to the system temp directory. >>> im = vipy.image.Image(url=\"http: domain.com/path/to/image.ext\", filename=\"/path/to/new/image.ext\") The image will be downloaded from the provided url and saved to the provided filename. The url() method provides optional basic authentication set for username and password >>> im = vipy.image.Image(array=img, colorspace='rgb') The image will be constructed from a provided numpy array 'img', with an associated colorspace. The numpy array and colorspace can be one of the following combinations: - 'rgb': uint8, three channel (red, green, blue) - 'rgba': uint8, four channel (rgb + alpha) - 'bgr': uint8, three channel (blue, green, red), such as is returned from cv2.imread() - 'bgra': uint8, four channel - 'hsv': uint8, three channel (hue, saturation, value) - 'lum;: uint8, one channel, luminance (8 bit grey level) - 'grey': float32, one channel in range [0,1] (32 bit intensity) - 'float': float32, any channel in range [-inf, +inf] The most general colorspace is 'float' which is used to manipulate images prior to network encoding, such as applying bias. Args: filename: a path to an image file that is readable by PIL url: a url string to an image file that is readable by PIL array: a numpy array of type uint8 or float32 of shape HxWxC=height x width x channels colorspace: a string in ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum'] attributes: a python dictionary that is passed by reference to the image. This is useful for encoding metadata about the image. Accessible as im.attributes Returns: A  vipy.image.Image object"
},
{
"ref":"vipy.image.Image.cast",
"url":28,
"doc":"Typecast the conformal vipy.image object im as  vipy.image.Image . This is useful for downcasting  vipy.image.Scene or  vipy.image.ImageDetection down to an image. >>> ims = vipy.image.RandomScene() >>> im = vipy.image.Image.cast(im)",
"func":1
},
{
"ref":"vipy.image.Image.from_json",
"url":28,
"doc":"Import the JSON string s as an  vipy.image.Image object. This will perform a round trip such that im1  im2 >>> im1 = vupy.image.RandomImage() >>> im2 = vipy.image.Image.from_json(im1.json( >>> assert im1  im2",
"func":1
},
{
"ref":"vipy.image.Image.print",
"url":28,
"doc":"Print the representation of the image and return self with an optional sleep=n seconds Useful for debugging in long fluent chains.",
"func":1
},
{
"ref":"vipy.image.Image.tile",
"url":28,
"doc":"Generate an image tiling. A tiling is a decomposition of an image into overlapping or non-overlapping rectangular regions. Args: tilewidth: [int] the image width of each tile tileheight: [int] the image height of each tile overlaprows: [int] the number of overlapping rows (height) for each tile overlapcols: [int] the number of overlapping width (width) for each tile Returns: A list of  vipy.image.Image objects such that each image is a single tile and the set of these tiles forms the original image Each image in the returned list contains the 'tile' attribute which encodes the crop used to create the tile.  note -  vipy.image.Image.tile can be undone using  vipy.image.Image.untile - The identity tiling is im.tile(im.widht(), im.height(), overlaprows=0, overlapcols=0) - Ragged tiles outside the image boundary are zero padded - All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.Image.union",
"url":28,
"doc":"No-op for  vipy.image.Image ",
"func":1
},
{
"ref":"vipy.image.Image.untile",
"url":28,
"doc":"Undo an image tiling and recreate the original image. >>> tiles = im.tile(im.width()/2, im.height()/2, 0, 0) >>> imdst = vipy.image.Image.untile(tiles) >>> assert imdst  im Args: imlist: this must be the output of  vipy.image.Image.tile Returns: A new  vipy.image.Image object reconstructed from the tiling, such that this is equivalent to the input to vipy.image.Image.tile  note All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.Image.uncrop",
"url":28,
"doc":"Uncrop using provided bounding box and zeropad to shape=(Height, Width). An uncrop is the inverse operation for a crop, which preserves the cropped portion of the image in the correct location and replaces the rest with zeros out to shape. >>> im = vipy.image.RandomImage(128, 128) >>> bb = vipy.geometry.BoundingBox(xmin=0, ymin=0, width=64, height=64) >>> uncrop = im.crop(bb).uncrop(bb, shape=(128,128 Args: bb: [ vipy.geometry.BoundingBox ] the bounding box used to crop the image in self shape: [tuple] (height, width) of the uncropped image Returns: this  vipy.image.Image object with the pixels uncropped.  note NOT idempotent. This will generate different results if run more than once.",
"func":1
},
{
"ref":"vipy.image.Image.splat",
"url":28,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.Image.store",
"url":28,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.Image.unstore",
"url":28,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.Image.restore",
"url":28,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.Image.abspath",
"url":28,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.Image.relpath",
"url":28,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.Image.canload",
"url":28,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.Image.dict",
"url":28,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.Image.json",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.loader",
"url":28,
"doc":"Lambda function to load an unsupported image filename to a numpy array. This lambda function will be executed during load and the result will be stored in self._array",
"func":1
},
{
"ref":"vipy.image.Image.load",
"url":28,
"doc":"Load image to cached private '_array' attribute. Args: ignoreErrors: [bool] If true, ignore any exceptions thrown during load and print the corresponding error messages. This is useful for loading images distributed without throwing exceptions when some images may be corrupted. In this case, the _array attribute will be None and  vipy.image.Image.isloaded will return false to determine if the image is loaded, which can be used to filter out corrupted images gracefully. verbose: [bool] If true, show additional useful printed output Returns: This  vipy.image.Image object with the pixels loaded in self._array as a numpy array.  note This loader supports any image file format supported by PIL. A custom loader can be added using  vipy.image.Image.loader .",
"func":1
},
{
"ref":"vipy.image.Image.download",
"url":28,
"doc":"Download URL to filename provided by constructor, or to temp filename. Args: ignoreErrors: [bool] If true, do not throw an exception if the download of the URL fails for some reason. Instead, print out a reason and return this image object. The function  vipy.image.Image.hasfilename will return false if the downloaded file does not exist and can be used to filter these failed downloads gracefully. timeout: [int] The timeout in seconds for an http or https connection attempt. See also [urllib.request.urlopen](https: docs.python.org/3/library/urllib.request.html). verbose: [bool] If true, output more helpful message. Returns: This  vipy.image.Image object with the URL downloaded to  vipy.image.Image.filename or to a  vipy.util.tempimage filename which can be retrieved with  vipy.image.Image.filename .",
"func":1
},
{
"ref":"vipy.image.Image.reload",
"url":28,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.Image.isloaded",
"url":28,
"doc":"Return True if  vipy.image.Image.load was successful in reading the image, or if the pixels are present in  vipy.image.Image.array .",
"func":1
},
{
"ref":"vipy.image.Image.channels",
"url":28,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.Image.iscolor",
"url":28,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.Image.istransparent",
"url":28,
"doc":"Transparent images are four channel color images with transparency, float32 or uint8. Return true if this image contains an alpha transparency channel",
"func":1
},
{
"ref":"vipy.image.Image.isgrey",
"url":28,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.Image.isluminance",
"url":28,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.Image.filesize",
"url":28,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.Image.width",
"url":28,
"doc":"Return the width (columns) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Image.height",
"url":28,
"doc":"Return the height (rows) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Image.shape",
"url":28,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image. Returns: A tuple (height=int, width=int) of the image.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Image.aspectratio",
"url":28,
"doc":"Return the aspect ratio of the image as (width/height) ratio. Returns: A float equivalent to ( vipy.image.Image.width /  vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Image.area",
"url":28,
"doc":"Return the area of the image as (width  height). Returns: An integer equivalent to ( vipy.image.Image.width   vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Image.centroid",
"url":28,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y). The centroid is equivalent to half the  vipy.image.Image.shape . Returns: A tuple (column, row) of the floating point center of the image.",
"func":1
},
{
"ref":"vipy.image.Image.centerpixel",
"url":28,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j) The centerpixel is equivalent to half the  vipy.image.Image.shape floored to the nearest integer pixel coordinate. Returns: A tuple (int(column), int(row of the integer center of the image.",
"func":1
},
{
"ref":"vipy.image.Image.array",
"url":28,
"doc":"Replace self._array with provided numpy array Args: np_array: [numpy array] A new array to use as the pixel buffer for this image. copy: [bool] If true, copy the buffer using np.copy(), else use a reference to this buffer. Returns: - If np_array is not None, return the  vipy.image.Image object such that this object points to the provided numpy array as the pixel buffer - If np_array is None, then return the numpy array.  notes - If copy=False, then this  vipy.image.Image object will share the pixel buffer with the owner of np_array. Changes to pixels in this buffer will be shared. - If copy=True, then this will significantly slow down processing for large images. Use referneces wherevery possible.",
"func":1
},
{
"ref":"vipy.image.Image.fromarray",
"url":28,
"doc":"Alias for  vipy.image.Image.array with copy=True. This will set new numpy array as the pixel buffer with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.Image.tonumpy",
"url":28,
"doc":"Alias for  vipy.image.Image.numpy",
"func":1
},
{
"ref":"vipy.image.Image.numpy",
"url":28,
"doc":"Return a mutable numpy array for this  vipy.image.Image .  notes - This will always return a writeable array with the 'WRITEABLE' numpy flag set. This is useful for returning a mutable numpy array as needed while keeping the original non-mutable numpy array (e.g. loaded from a video or PIL) as the underlying pixel buffer for efficiency reasons. - Triggers a  vipy.image.Image.load if the pixel buffer has not been loaded - This will trigger a copy if the ['WRITEABLE' flag](https: numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) is not set.",
"func":1
},
{
"ref":"vipy.image.Image.channel",
"url":28,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None. Iterate over channels as single channel luminance images: >>> for c in self.channel(): >>> print(c) Return the kth channel as a single channel luminance image: >>> c = self.channel(k=0)",
"func":1
},
{
"ref":"vipy.image.Image.red",
"url":28,
"doc":"Return red channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.red()  self.channel(0) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.red()  self.channel(3)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.Image.green",
"url":28,
"doc":"Return green channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.green()  self.channel(1) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.green()  self.channel(1)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.Image.blue",
"url":28,
"doc":"Return blue channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.vlue()  self.channel(2) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.blue()  self.channel(0)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.Image.alpha",
"url":28,
"doc":"Return alpha (transparency) channel as a cloned single channel  vipy.image.Image object",
"func":1
},
{
"ref":"vipy.image.Image.zeros",
"url":28,
"doc":"Set the pixel buffer to all zeros of the same shape and datatype as this  vipy.image.Image object. These are equivalent operations for the resulting buffer shape: >>> import numpy as np >>> np.zeros( (self.width(), self.height(), self.channels( )  self.zeros().array() Returns: This  vipy.image.Image object.  note Triggers load() if the pixel buffer has not been loaded yet.",
"func":1
},
{
"ref":"vipy.image.Image.pil",
"url":28,
"doc":"Convert vipy.image.Image to PIL Image. Returns: A [PIL image](https: pillow.readthedocs.io/en/stable/reference/Image.html) object, that shares the pixel buffer by reference",
"func":1
},
{
"ref":"vipy.image.Image.blur",
"url":28,
"doc":"Apply a Gaussian blur with Gaussian kernel radius=sigma to the pixel buffer. Args: sigma: [float >0] The gaussian blur kernel radius. Returns: This  vipy.image.Image object with the pixel buffer blurred in place.",
"func":1
},
{
"ref":"vipy.image.Image.torch",
"url":28,
"doc":"Convert the batch of 1 HxWxC images to a CxHxW torch tensor. Args: order: ['CHW', 'HWC', 'NCHW', 'NHWC']. The axis order of the torch tensor (channels, height, width) or (height, width, channels) or (1, channels, height, width) or (1, height, width, channels) Returns: A CxHxW or HxWxC or 1xCxHxW or 1xHxWxC [torch tensor](https: pytorch.org/docs/stable/tensors.html) that shares the pixel buffer of this image object by reference.",
"func":1
},
{
"ref":"vipy.image.Image.fromtorch",
"url":28,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.Image.nofilename",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.filename",
"url":28,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.Image.nourl",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.url",
"url":28,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.Image.colorspace",
"url":28,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.Image.uri",
"url":28,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.Image.setattribute",
"url":28,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.Image.setattributes",
"url":28,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.Image.getattribute",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.hasattribute",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.delattribute",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.hasurl",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.hasfilename",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.clone",
"url":28,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.Image.flush",
"url":28,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.Image.resize",
"url":28,
"doc":"Resize the image buffer to (rows x cols) with bilinear interpolation. If rows or cols is provided, rescale image maintaining aspect ratio",
"func":1
},
{
"ref":"vipy.image.Image.resize_like",
"url":28,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.Image.rescale",
"url":28,
"doc":"Scale the image buffer by the given factor - NOT idempotent",
"func":1
},
{
"ref":"vipy.image.Image.maxdim",
"url":28,
"doc":"Resize image preserving aspect ratio so that maximum dimension of image = dim, or return maxdim()",
"func":1
},
{
"ref":"vipy.image.Image.mindim",
"url":28,
"doc":"Resize image preserving aspect ratio so that minimum dimension of image = dim, or return mindim()",
"func":1
},
{
"ref":"vipy.image.Image.zeropad",
"url":28,
"doc":"Pad image using np.pad constant by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Image.zeropadlike",
"url":28,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.Image.meanpad",
"url":28,
"doc":"Pad image using np.pad constant=image mean by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Image.alphapad",
"url":28,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Image.minsquare",
"url":28,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Image.maxsquare",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Image.maxmatte",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.Image.centersquare",
"url":28,
"doc":"Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.Image.centercrop",
"url":28,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.Image.cornercrop",
"url":28,
"doc":"Crop image of size (height x width) from the upper left corner",
"func":1
},
{
"ref":"vipy.image.Image.crop",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.fliplr",
"url":28,
"doc":"Mirror the image buffer about the vertical axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.Image.flipud",
"url":28,
"doc":"Mirror the image buffer about the horizontal axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.Image.imagebox",
"url":28,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.Image.border_mask",
"url":28,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.Image.affine_transform",
"url":28,
"doc":"Apply a 3x3 affine geometric transformation to the image. See also  vipy.geometry.affine_transform  note The image will be loaded and converted to float() prior to applying the affine transformation.",
"func":1
},
{
"ref":"vipy.image.Image.rgb",
"url":28,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.rgba",
"url":28,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.hsv",
"url":28,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.bgr",
"url":28,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.bgra",
"url":28,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.float",
"url":28,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.Image.greyscale",
"url":28,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.Image.grayscale",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Image.grey",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Image.gray",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Image.luminance",
"url":28,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.Image.lum",
"url":28,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.Image.jet",
"url":28,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.Image.rainbow",
"url":28,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Image.hot",
"url":28,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Image.bone",
"url":28,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Image.saturate",
"url":28,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.Image.intensity",
"url":28,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.Image.mat2gray",
"url":28,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.Image.gain",
"url":28,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.Image.bias",
"url":28,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.Image.stats",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.min",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.max",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.mean",
"url":28,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.Image.meanchannel",
"url":28,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.Image.sum",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.closeall",
"url":28,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.Image.close",
"url":28,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.Image.show",
"url":28,
"doc":"Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object",
"func":1
},
{
"ref":"vipy.image.Image.save",
"url":28,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.Image.pkl",
"url":28,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Image.pklif",
"url":28,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Image.saveas",
"url":28,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.Image.saveastmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.Image.savetmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.Image.base64",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.Image.html",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.Image.annotate",
"url":28,
"doc":"Change pixels of this image to include rendered annotation and return an image object",
"func":1
},
{
"ref":"vipy.image.Image.savefig",
"url":28,
"doc":"Save last figure output from self.show() with drawing overlays to provided filename and return filename",
"func":1
},
{
"ref":"vipy.image.Image.map",
"url":28,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.Image.downcast",
"url":28,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.Image.perceptualhash",
"url":28,
"doc":"Perceptual differential hash function This function converts to greyscale, resizes with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs. Args: bits: [int] longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes). asbinary: [bool] If true, return a binary array asbytes: [bool] if true return a byte array Returns: A hash string encoding the perceptual hash such that  vipy.image.Image.perceptualhash_distance can be used to compute a hash distance asbytes: a bytes array asbinary: a numpy binary array  notes - Can be used for near duplicate detection by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing. Equivalently,  vipy.image.Image.perceptualhash_distance . - The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex(h)",
"func":1
},
{
"ref":"vipy.image.Image.perceptualhash_distance",
"url":28,
"doc":"Hamming distance between two perceptual hashes",
"func":1
},
{
"ref":"vipy.image.ImageCategory",
"url":28,
"doc":"vipy ImageCategory class This class provides a representation of a vipy.image.Image with a category. Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label') >>> im = vipy.image.ImageCategory(filename='/path/to/dog_image.ext', category='dog') >>> im = vipy.image.ImageCategory(url='http: path/to/dog_image.ext', category='dog') >>> im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog')"
},
{
"ref":"vipy.image.ImageCategory.cast",
"url":28,
"doc":"Typecast the conformal vipy.image object im as  vipy.image.Image . This is useful for downcasting  vipy.image.Scene or  vipy.image.ImageDetection down to an image. >>> ims = vipy.image.RandomScene() >>> im = vipy.image.Image.cast(im)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.from_json",
"url":28,
"doc":"Import the JSON string s as an  vipy.image.Image object. This will perform a round trip such that im1  im2 >>> im1 = vupy.image.RandomImage() >>> im2 = vipy.image.Image.from_json(im1.json( >>> assert im1  im2",
"func":1
},
{
"ref":"vipy.image.ImageCategory.json",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.is_",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.is_not",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.nocategory",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.category",
"url":28,
"doc":"Return or update the category",
"func":1
},
{
"ref":"vipy.image.ImageCategory.label",
"url":28,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.image.ImageCategory.score",
"url":28,
"doc":"Real valued score for categorization, larger is better",
"func":1
},
{
"ref":"vipy.image.ImageCategory.probability",
"url":28,
"doc":"Real valued probability for categorization, [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageCategory.print",
"url":28,
"doc":"Print the representation of the image and return self with an optional sleep=n seconds Useful for debugging in long fluent chains.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.tile",
"url":28,
"doc":"Generate an image tiling. A tiling is a decomposition of an image into overlapping or non-overlapping rectangular regions. Args: tilewidth: [int] the image width of each tile tileheight: [int] the image height of each tile overlaprows: [int] the number of overlapping rows (height) for each tile overlapcols: [int] the number of overlapping width (width) for each tile Returns: A list of  vipy.image.Image objects such that each image is a single tile and the set of these tiles forms the original image Each image in the returned list contains the 'tile' attribute which encodes the crop used to create the tile.  note -  vipy.image.Image.tile can be undone using  vipy.image.Image.untile - The identity tiling is im.tile(im.widht(), im.height(), overlaprows=0, overlapcols=0) - Ragged tiles outside the image boundary are zero padded - All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.ImageCategory.union",
"url":28,
"doc":"No-op for  vipy.image.Image ",
"func":1
},
{
"ref":"vipy.image.ImageCategory.untile",
"url":28,
"doc":"Undo an image tiling and recreate the original image. >>> tiles = im.tile(im.width()/2, im.height()/2, 0, 0) >>> imdst = vipy.image.Image.untile(tiles) >>> assert imdst  im Args: imlist: this must be the output of  vipy.image.Image.tile Returns: A new  vipy.image.Image object reconstructed from the tiling, such that this is equivalent to the input to vipy.image.Image.tile  note All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.ImageCategory.uncrop",
"url":28,
"doc":"Uncrop using provided bounding box and zeropad to shape=(Height, Width). An uncrop is the inverse operation for a crop, which preserves the cropped portion of the image in the correct location and replaces the rest with zeros out to shape. >>> im = vipy.image.RandomImage(128, 128) >>> bb = vipy.geometry.BoundingBox(xmin=0, ymin=0, width=64, height=64) >>> uncrop = im.crop(bb).uncrop(bb, shape=(128,128 Args: bb: [ vipy.geometry.BoundingBox ] the bounding box used to crop the image in self shape: [tuple] (height, width) of the uncropped image Returns: this  vipy.image.Image object with the pixels uncropped.  note NOT idempotent. This will generate different results if run more than once.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.splat",
"url":28,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.ImageCategory.store",
"url":28,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.ImageCategory.unstore",
"url":28,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.restore",
"url":28,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.abspath",
"url":28,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.relpath",
"url":28,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.canload",
"url":28,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.ImageCategory.dict",
"url":28,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.loader",
"url":28,
"doc":"Lambda function to load an unsupported image filename to a numpy array. This lambda function will be executed during load and the result will be stored in self._array",
"func":1
},
{
"ref":"vipy.image.ImageCategory.load",
"url":28,
"doc":"Load image to cached private '_array' attribute. Args: ignoreErrors: [bool] If true, ignore any exceptions thrown during load and print the corresponding error messages. This is useful for loading images distributed without throwing exceptions when some images may be corrupted. In this case, the _array attribute will be None and  vipy.image.Image.isloaded will return false to determine if the image is loaded, which can be used to filter out corrupted images gracefully. verbose: [bool] If true, show additional useful printed output Returns: This  vipy.image.Image object with the pixels loaded in self._array as a numpy array.  note This loader supports any image file format supported by PIL. A custom loader can be added using  vipy.image.Image.loader .",
"func":1
},
{
"ref":"vipy.image.ImageCategory.download",
"url":28,
"doc":"Download URL to filename provided by constructor, or to temp filename. Args: ignoreErrors: [bool] If true, do not throw an exception if the download of the URL fails for some reason. Instead, print out a reason and return this image object. The function  vipy.image.Image.hasfilename will return false if the downloaded file does not exist and can be used to filter these failed downloads gracefully. timeout: [int] The timeout in seconds for an http or https connection attempt. See also [urllib.request.urlopen](https: docs.python.org/3/library/urllib.request.html). verbose: [bool] If true, output more helpful message. Returns: This  vipy.image.Image object with the URL downloaded to  vipy.image.Image.filename or to a  vipy.util.tempimage filename which can be retrieved with  vipy.image.Image.filename .",
"func":1
},
{
"ref":"vipy.image.ImageCategory.reload",
"url":28,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.ImageCategory.isloaded",
"url":28,
"doc":"Return True if  vipy.image.Image.load was successful in reading the image, or if the pixels are present in  vipy.image.Image.array .",
"func":1
},
{
"ref":"vipy.image.ImageCategory.channels",
"url":28,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.iscolor",
"url":28,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.ImageCategory.istransparent",
"url":28,
"doc":"Transparent images are four channel color images with transparency, float32 or uint8. Return true if this image contains an alpha transparency channel",
"func":1
},
{
"ref":"vipy.image.ImageCategory.isgrey",
"url":28,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.ImageCategory.isluminance",
"url":28,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.ImageCategory.filesize",
"url":28,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.ImageCategory.width",
"url":28,
"doc":"Return the width (columns) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.height",
"url":28,
"doc":"Return the height (rows) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.shape",
"url":28,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image. Returns: A tuple (height=int, width=int) of the image.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.aspectratio",
"url":28,
"doc":"Return the aspect ratio of the image as (width/height) ratio. Returns: A float equivalent to ( vipy.image.Image.width /  vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.area",
"url":28,
"doc":"Return the area of the image as (width  height). Returns: An integer equivalent to ( vipy.image.Image.width   vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centroid",
"url":28,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y). The centroid is equivalent to half the  vipy.image.Image.shape . Returns: A tuple (column, row) of the floating point center of the image.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centerpixel",
"url":28,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j) The centerpixel is equivalent to half the  vipy.image.Image.shape floored to the nearest integer pixel coordinate. Returns: A tuple (int(column), int(row of the integer center of the image.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.array",
"url":28,
"doc":"Replace self._array with provided numpy array Args: np_array: [numpy array] A new array to use as the pixel buffer for this image. copy: [bool] If true, copy the buffer using np.copy(), else use a reference to this buffer. Returns: - If np_array is not None, return the  vipy.image.Image object such that this object points to the provided numpy array as the pixel buffer - If np_array is None, then return the numpy array.  notes - If copy=False, then this  vipy.image.Image object will share the pixel buffer with the owner of np_array. Changes to pixels in this buffer will be shared. - If copy=True, then this will significantly slow down processing for large images. Use referneces wherevery possible.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.fromarray",
"url":28,
"doc":"Alias for  vipy.image.Image.array with copy=True. This will set new numpy array as the pixel buffer with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.ImageCategory.tonumpy",
"url":28,
"doc":"Alias for  vipy.image.Image.numpy",
"func":1
},
{
"ref":"vipy.image.ImageCategory.numpy",
"url":28,
"doc":"Return a mutable numpy array for this  vipy.image.Image .  notes - This will always return a writeable array with the 'WRITEABLE' numpy flag set. This is useful for returning a mutable numpy array as needed while keeping the original non-mutable numpy array (e.g. loaded from a video or PIL) as the underlying pixel buffer for efficiency reasons. - Triggers a  vipy.image.Image.load if the pixel buffer has not been loaded - This will trigger a copy if the ['WRITEABLE' flag](https: numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) is not set.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.channel",
"url":28,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None. Iterate over channels as single channel luminance images: >>> for c in self.channel(): >>> print(c) Return the kth channel as a single channel luminance image: >>> c = self.channel(k=0)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.red",
"url":28,
"doc":"Return red channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.red()  self.channel(0) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.red()  self.channel(3)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.green",
"url":28,
"doc":"Return green channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.green()  self.channel(1) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.green()  self.channel(1)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.blue",
"url":28,
"doc":"Return blue channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.vlue()  self.channel(2) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.blue()  self.channel(0)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.alpha",
"url":28,
"doc":"Return alpha (transparency) channel as a cloned single channel  vipy.image.Image object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.zeros",
"url":28,
"doc":"Set the pixel buffer to all zeros of the same shape and datatype as this  vipy.image.Image object. These are equivalent operations for the resulting buffer shape: >>> import numpy as np >>> np.zeros( (self.width(), self.height(), self.channels( )  self.zeros().array() Returns: This  vipy.image.Image object.  note Triggers load() if the pixel buffer has not been loaded yet.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.pil",
"url":28,
"doc":"Convert vipy.image.Image to PIL Image. Returns: A [PIL image](https: pillow.readthedocs.io/en/stable/reference/Image.html) object, that shares the pixel buffer by reference",
"func":1
},
{
"ref":"vipy.image.ImageCategory.blur",
"url":28,
"doc":"Apply a Gaussian blur with Gaussian kernel radius=sigma to the pixel buffer. Args: sigma: [float >0] The gaussian blur kernel radius. Returns: This  vipy.image.Image object with the pixel buffer blurred in place.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.torch",
"url":28,
"doc":"Convert the batch of 1 HxWxC images to a CxHxW torch tensor. Args: order: ['CHW', 'HWC', 'NCHW', 'NHWC']. The axis order of the torch tensor (channels, height, width) or (height, width, channels) or (1, channels, height, width) or (1, height, width, channels) Returns: A CxHxW or HxWxC or 1xCxHxW or 1xHxWxC [torch tensor](https: pytorch.org/docs/stable/tensors.html) that shares the pixel buffer of this image object by reference.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.fromtorch",
"url":28,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.filename",
"url":28,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.url",
"url":28,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.ImageCategory.colorspace",
"url":28,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.ImageCategory.uri",
"url":28,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.ImageCategory.setattribute",
"url":28,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.ImageCategory.setattributes",
"url":28,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.ImageCategory.clone",
"url":28,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.flush",
"url":28,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.ImageCategory.resize",
"url":28,
"doc":"Resize the image buffer to (rows x cols) with bilinear interpolation. If rows or cols is provided, rescale image maintaining aspect ratio",
"func":1
},
{
"ref":"vipy.image.ImageCategory.resize_like",
"url":28,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rescale",
"url":28,
"doc":"Scale the image buffer by the given factor - NOT idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.maxdim",
"url":28,
"doc":"Resize image preserving aspect ratio so that maximum dimension of image = dim, or return maxdim()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.mindim",
"url":28,
"doc":"Resize image preserving aspect ratio so that minimum dimension of image = dim, or return mindim()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.zeropad",
"url":28,
"doc":"Pad image using np.pad constant by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.zeropadlike",
"url":28,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.meanpad",
"url":28,
"doc":"Pad image using np.pad constant=image mean by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.alphapad",
"url":28,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.minsquare",
"url":28,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.maxsquare",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.maxmatte",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centersquare",
"url":28,
"doc":"Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centercrop",
"url":28,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.cornercrop",
"url":28,
"doc":"Crop image of size (height x width) from the upper left corner",
"func":1
},
{
"ref":"vipy.image.ImageCategory.fliplr",
"url":28,
"doc":"Mirror the image buffer about the vertical axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.flipud",
"url":28,
"doc":"Mirror the image buffer about the horizontal axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.imagebox",
"url":28,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.ImageCategory.border_mask",
"url":28,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.ImageCategory.affine_transform",
"url":28,
"doc":"Apply a 3x3 affine geometric transformation to the image. See also  vipy.geometry.affine_transform  note The image will be loaded and converted to float() prior to applying the affine transformation.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rgb",
"url":28,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rgba",
"url":28,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.hsv",
"url":28,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bgr",
"url":28,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bgra",
"url":28,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.float",
"url":28,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.ImageCategory.greyscale",
"url":28,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageCategory.grayscale",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.grey",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.gray",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.luminance",
"url":28,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.ImageCategory.lum",
"url":28,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.jet",
"url":28,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rainbow",
"url":28,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.hot",
"url":28,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bone",
"url":28,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.saturate",
"url":28,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.ImageCategory.intensity",
"url":28,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.mat2gray",
"url":28,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.gain",
"url":28,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bias",
"url":28,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageCategory.mean",
"url":28,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.meanchannel",
"url":28,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.closeall",
"url":28,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.ImageCategory.close",
"url":28,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.ImageCategory.show",
"url":28,
"doc":"Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.save",
"url":28,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.pkl",
"url":28,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageCategory.pklif",
"url":28,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageCategory.saveas",
"url":28,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.saveastmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.savetmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.base64",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.ImageCategory.html",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.ImageCategory.annotate",
"url":28,
"doc":"Change pixels of this image to include rendered annotation and return an image object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.savefig",
"url":28,
"doc":"Save last figure output from self.show() with drawing overlays to provided filename and return filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.map",
"url":28,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageCategory.downcast",
"url":28,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.perceptualhash",
"url":28,
"doc":"Perceptual differential hash function This function converts to greyscale, resizes with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs. Args: bits: [int] longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes). asbinary: [bool] If true, return a binary array asbytes: [bool] if true return a byte array Returns: A hash string encoding the perceptual hash such that  vipy.image.Image.perceptualhash_distance can be used to compute a hash distance asbytes: a bytes array asbinary: a numpy binary array  notes - Can be used for near duplicate detection by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing. Equivalently,  vipy.image.Image.perceptualhash_distance . - The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex(h)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.perceptualhash_distance",
"url":28,
"doc":"Hamming distance between two perceptual hashes",
"func":1
},
{
"ref":"vipy.image.Scene",
"url":28,
"doc":"vipy.image.Scene class This class provides a representation of a vipy.image.ImageCategory with one or more vipy.object.Detections. The goal of this class is to provide a unified representation for all objects in a scene. Valid constructors include all provided by vipy.image.Image() and vipy.image.ImageCategory() with the additional kwarg 'objects', which is a list of vipy.object.Detections() >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='city', objects=[vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)]) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='city').objects([vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)]) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels='face', xywh=[0,0,100,100]) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels='face', xywh= 0,0,100,100], [100,100,200,200 ) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels=['face', 'desk'] xywh= 0,0,100,100], [200,200,300,300 )"
},
{
"ref":"vipy.image.Scene.cast",
"url":28,
"doc":"Typecast the conformal vipy.image object im as  vipy.image.Image . This is useful for downcasting  vipy.image.Scene or  vipy.image.ImageDetection down to an image. >>> ims = vipy.image.RandomScene() >>> im = vipy.image.Image.cast(im)",
"func":1
},
{
"ref":"vipy.image.Scene.from_json",
"url":28,
"doc":"Import the JSON string s as an  vipy.image.Image object. This will perform a round trip such that im1  im2 >>> im1 = vupy.image.RandomImage() >>> im2 = vipy.image.Image.from_json(im1.json( >>> assert im1  im2",
"func":1
},
{
"ref":"vipy.image.Scene.json",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.append",
"url":28,
"doc":"Append the provided vipy.object.Detection object to the scene object list",
"func":1
},
{
"ref":"vipy.image.Scene.add",
"url":28,
"doc":"Alias for append",
"func":1
},
{
"ref":"vipy.image.Scene.objects",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.objectmap",
"url":28,
"doc":"Apply lambda function f to each object. If f is a list of lambda, apply one to one with the objects",
"func":1
},
{
"ref":"vipy.image.Scene.objectfilter",
"url":28,
"doc":"Apply lambda function f to each object and keep if filter is True",
"func":1
},
{
"ref":"vipy.image.Scene.nms",
"url":28,
"doc":"Non-maximum supporession of objects() by category based on confidence and spatial IoU and cover thresholds",
"func":1
},
{
"ref":"vipy.image.Scene.intersection",
"url":28,
"doc":"Return a Scene() containing the objects in both self and other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.Scene.difference",
"url":28,
"doc":"Return a Scene() containing the objects in self but not other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.Scene.union",
"url":28,
"doc":"Combine the objects of the scene with other and self with no duplicate checking unless miniou is not None",
"func":1
},
{
"ref":"vipy.image.Scene.uncrop",
"url":28,
"doc":"Uncrop a previous crop(bb) called with the supplied bb=BoundingBox(), and zeropad to shape=(H,W)",
"func":1
},
{
"ref":"vipy.image.Scene.clear",
"url":28,
"doc":"Remove all objects from this scene.",
"func":1
},
{
"ref":"vipy.image.Scene.boundingbox",
"url":28,
"doc":"The boundingbox of a scene is the union of all object bounding boxes, or None if there are no objects",
"func":1
},
{
"ref":"vipy.image.Scene.categories",
"url":28,
"doc":"Return list of unique object categories in scene",
"func":1
},
{
"ref":"vipy.image.Scene.imclip",
"url":28,
"doc":"Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image",
"func":1
},
{
"ref":"vipy.image.Scene.rescale",
"url":28,
"doc":"Rescale image buffer and all bounding boxes - Not idempotent",
"func":1
},
{
"ref":"vipy.image.Scene.resize",
"url":28,
"doc":"Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly. If cols or rows is None, then scale isotropically",
"func":1
},
{
"ref":"vipy.image.Scene.centersquare",
"url":28,
"doc":"Crop the image of size (H,W) to be centersquare (min(H,W), min(H,W preserving center, and update bounding boxes",
"func":1
},
{
"ref":"vipy.image.Scene.fliplr",
"url":28,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.Scene.flipud",
"url":28,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.Scene.dilate",
"url":28,
"doc":"Dilate all bounding boxes by scale factor, dilated boxes may be outside image rectangle",
"func":1
},
{
"ref":"vipy.image.Scene.zeropad",
"url":28,
"doc":"Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.Scene.meanpad",
"url":28,
"doc":"Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.Scene.rot90cw",
"url":28,
"doc":"Rotate the scene 90 degrees clockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.Scene.rot90ccw",
"url":28,
"doc":"Rotate the scene 90 degrees counterclockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.Scene.maxdim",
"url":28,
"doc":"Resize scene preserving aspect ratio so that maximum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.Scene.mindim",
"url":28,
"doc":"Resize scene preserving aspect ratio so that minimum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.Scene.crop",
"url":28,
"doc":"Crop the image buffer using the supplied bounding box object (or the only object if bbox=None), clipping the box to the image rectangle, update all scene objects",
"func":1
},
{
"ref":"vipy.image.Scene.centercrop",
"url":28,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.Scene.cornercrop",
"url":28,
"doc":"Crop image of size (height x width) from the upper left corner, returning valid pixels only",
"func":1
},
{
"ref":"vipy.image.Scene.padcrop",
"url":28,
"doc":"Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects",
"func":1
},
{
"ref":"vipy.image.Scene.cornerpadcrop",
"url":28,
"doc":"Crop image of size (height x width) from the upper left corner, returning zero padded result out to (height, width)",
"func":1
},
{
"ref":"vipy.image.Scene.rectangular_mask",
"url":28,
"doc":"Return a binary array of the same size as the image (or using the provided image width and height (W,H) size to avoid an image load), with ones inside the bounding box",
"func":1
},
{
"ref":"vipy.image.Scene.binarymask",
"url":28,
"doc":"Alias for rectangular_mask with in-place update",
"func":1
},
{
"ref":"vipy.image.Scene.bgmask",
"url":28,
"doc":"Set all pixels outside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.Scene.fgmask",
"url":28,
"doc":"Set all pixels inside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.Scene.setzero",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.pixelmask",
"url":28,
"doc":"Replace pixels within all foreground objects with a privacy preserving pixelated foreground with larger pixels",
"func":1
},
{
"ref":"vipy.image.Scene.blurmask",
"url":28,
"doc":"Replace pixels within all foreground objects with a privacy preserving blurred foreground",
"func":1
},
{
"ref":"vipy.image.Scene.replace",
"url":28,
"doc":"Set all image values within the bounding box equal to the provided img, triggers load() and imclip()",
"func":1
},
{
"ref":"vipy.image.Scene.meanmask",
"url":28,
"doc":"Replace pixels within the foreground objects with the mean pixel color",
"func":1
},
{
"ref":"vipy.image.Scene.perceptualhash",
"url":28,
"doc":"Perceptual differential hash function. This function sets foreground objects to mean color, convert to greyscale, resize with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs. Args: bits: [int] longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes). objmask: [bool] if true, replace the foreground object masks with the mean color prior to computing asbinary: [bool] If true, return a binary array asbytes: [bool] if true return a byte array Returns: A hash string encoding the perceptual hash such that  vipy.image.Image.perceptualhash_distance can be used to compute a hash distance asbytes: a bytes array asbinary: a numpy binary array  notes - Can be used for near duplicate detection of background scenes by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing. Equivalently,  vipy.image.Image.perceptualhash_distance . - The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex( bghash()  which is equivalent to perceptualhash(asbinary=True)",
"func":1
},
{
"ref":"vipy.image.Scene.fghash",
"url":28,
"doc":"Perceptual differential hash function, computed for each foreground region independently",
"func":1
},
{
"ref":"vipy.image.Scene.bghash",
"url":28,
"doc":"Percetual differential hash function, masking out foreground regions",
"func":1
},
{
"ref":"vipy.image.Scene.isduplicate",
"url":28,
"doc":"Background hash near duplicate detection, returns true if self and im are near duplicate images using bghash",
"func":1
},
{
"ref":"vipy.image.Scene.show",
"url":28,
"doc":"Show scene detection Args: categories: [list] List of category (or shortlabel) names in the scene to show fontsize: [int] or [str]: Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size figure: [int] Figure number, show the image in the provided figure=int numbered window nocaption: [bool] Show or do not show the text caption in the upper left of the box nocaption_withstring: [list]: Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list  boxalpha (float, [0,1]): Set the text box background to be semi-transparent with an alpha  d_category2color (dict): Define a dictionary of required mapping of specific category() to box colors. Non-specified categories are assigned a random named color from vipy.show.colorlist()  caption_offset (int, int): The relative position of the caption to the upper right corner of the box.  nowindow (bool): Display or not display the image  textfacecolor (str): One of the named colors from vipy.show.colorlist() for the color of the textbox background  textfacealpha (float, [0,1]): The textbox background transparency  shortlabel (bool): Whether to show the shortlabel or the full category name in the caption  mutator (lambda): A lambda function with signature lambda im: f(im) which will modify this image prior to show. Useful for changing labels on the fly",
"func":1
},
{
"ref":"vipy.image.Scene.annotate",
"url":28,
"doc":"Alias for savefig",
"func":1
},
{
"ref":"vipy.image.Scene.savefig",
"url":28,
"doc":"Save show() output to given file or return buffer without popping up a window",
"func":1
},
{
"ref":"vipy.image.Scene.category",
"url":28,
"doc":"Return or update the category",
"func":1
},
{
"ref":"vipy.image.Scene.label",
"url":28,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.image.Scene.score",
"url":28,
"doc":"Real valued score for categorization, larger is better",
"func":1
},
{
"ref":"vipy.image.Scene.probability",
"url":28,
"doc":"Real valued probability for categorization, [0,1]",
"func":1
},
{
"ref":"vipy.image.Scene.print",
"url":28,
"doc":"Print the representation of the image and return self with an optional sleep=n seconds Useful for debugging in long fluent chains.",
"func":1
},
{
"ref":"vipy.image.Scene.tile",
"url":28,
"doc":"Generate an image tiling. A tiling is a decomposition of an image into overlapping or non-overlapping rectangular regions. Args: tilewidth: [int] the image width of each tile tileheight: [int] the image height of each tile overlaprows: [int] the number of overlapping rows (height) for each tile overlapcols: [int] the number of overlapping width (width) for each tile Returns: A list of  vipy.image.Image objects such that each image is a single tile and the set of these tiles forms the original image Each image in the returned list contains the 'tile' attribute which encodes the crop used to create the tile.  note -  vipy.image.Image.tile can be undone using  vipy.image.Image.untile - The identity tiling is im.tile(im.widht(), im.height(), overlaprows=0, overlapcols=0) - Ragged tiles outside the image boundary are zero padded - All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.Scene.untile",
"url":28,
"doc":"Undo an image tiling and recreate the original image. >>> tiles = im.tile(im.width()/2, im.height()/2, 0, 0) >>> imdst = vipy.image.Image.untile(tiles) >>> assert imdst  im Args: imlist: this must be the output of  vipy.image.Image.tile Returns: A new  vipy.image.Image object reconstructed from the tiling, such that this is equivalent to the input to vipy.image.Image.tile  note All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.Scene.splat",
"url":28,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.Scene.store",
"url":28,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.Scene.unstore",
"url":28,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.Scene.restore",
"url":28,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.Scene.abspath",
"url":28,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.Scene.relpath",
"url":28,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.Scene.canload",
"url":28,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.Scene.dict",
"url":28,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.Scene.loader",
"url":28,
"doc":"Lambda function to load an unsupported image filename to a numpy array. This lambda function will be executed during load and the result will be stored in self._array",
"func":1
},
{
"ref":"vipy.image.Scene.load",
"url":28,
"doc":"Load image to cached private '_array' attribute. Args: ignoreErrors: [bool] If true, ignore any exceptions thrown during load and print the corresponding error messages. This is useful for loading images distributed without throwing exceptions when some images may be corrupted. In this case, the _array attribute will be None and  vipy.image.Image.isloaded will return false to determine if the image is loaded, which can be used to filter out corrupted images gracefully. verbose: [bool] If true, show additional useful printed output Returns: This  vipy.image.Image object with the pixels loaded in self._array as a numpy array.  note This loader supports any image file format supported by PIL. A custom loader can be added using  vipy.image.Image.loader .",
"func":1
},
{
"ref":"vipy.image.Scene.download",
"url":28,
"doc":"Download URL to filename provided by constructor, or to temp filename. Args: ignoreErrors: [bool] If true, do not throw an exception if the download of the URL fails for some reason. Instead, print out a reason and return this image object. The function  vipy.image.Image.hasfilename will return false if the downloaded file does not exist and can be used to filter these failed downloads gracefully. timeout: [int] The timeout in seconds for an http or https connection attempt. See also [urllib.request.urlopen](https: docs.python.org/3/library/urllib.request.html). verbose: [bool] If true, output more helpful message. Returns: This  vipy.image.Image object with the URL downloaded to  vipy.image.Image.filename or to a  vipy.util.tempimage filename which can be retrieved with  vipy.image.Image.filename .",
"func":1
},
{
"ref":"vipy.image.Scene.reload",
"url":28,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.Scene.isloaded",
"url":28,
"doc":"Return True if  vipy.image.Image.load was successful in reading the image, or if the pixels are present in  vipy.image.Image.array .",
"func":1
},
{
"ref":"vipy.image.Scene.channels",
"url":28,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.Scene.iscolor",
"url":28,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.Scene.istransparent",
"url":28,
"doc":"Transparent images are four channel color images with transparency, float32 or uint8. Return true if this image contains an alpha transparency channel",
"func":1
},
{
"ref":"vipy.image.Scene.isgrey",
"url":28,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.Scene.isluminance",
"url":28,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.Scene.filesize",
"url":28,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.Scene.width",
"url":28,
"doc":"Return the width (columns) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Scene.height",
"url":28,
"doc":"Return the height (rows) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Scene.shape",
"url":28,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image. Returns: A tuple (height=int, width=int) of the image.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Scene.aspectratio",
"url":28,
"doc":"Return the aspect ratio of the image as (width/height) ratio. Returns: A float equivalent to ( vipy.image.Image.width /  vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Scene.area",
"url":28,
"doc":"Return the area of the image as (width  height). Returns: An integer equivalent to ( vipy.image.Image.width   vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.Scene.centroid",
"url":28,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y). The centroid is equivalent to half the  vipy.image.Image.shape . Returns: A tuple (column, row) of the floating point center of the image.",
"func":1
},
{
"ref":"vipy.image.Scene.centerpixel",
"url":28,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j) The centerpixel is equivalent to half the  vipy.image.Image.shape floored to the nearest integer pixel coordinate. Returns: A tuple (int(column), int(row of the integer center of the image.",
"func":1
},
{
"ref":"vipy.image.Scene.array",
"url":28,
"doc":"Replace self._array with provided numpy array Args: np_array: [numpy array] A new array to use as the pixel buffer for this image. copy: [bool] If true, copy the buffer using np.copy(), else use a reference to this buffer. Returns: - If np_array is not None, return the  vipy.image.Image object such that this object points to the provided numpy array as the pixel buffer - If np_array is None, then return the numpy array.  notes - If copy=False, then this  vipy.image.Image object will share the pixel buffer with the owner of np_array. Changes to pixels in this buffer will be shared. - If copy=True, then this will significantly slow down processing for large images. Use referneces wherevery possible.",
"func":1
},
{
"ref":"vipy.image.Scene.fromarray",
"url":28,
"doc":"Alias for  vipy.image.Image.array with copy=True. This will set new numpy array as the pixel buffer with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.Scene.tonumpy",
"url":28,
"doc":"Alias for  vipy.image.Image.numpy",
"func":1
},
{
"ref":"vipy.image.Scene.numpy",
"url":28,
"doc":"Return a mutable numpy array for this  vipy.image.Image .  notes - This will always return a writeable array with the 'WRITEABLE' numpy flag set. This is useful for returning a mutable numpy array as needed while keeping the original non-mutable numpy array (e.g. loaded from a video or PIL) as the underlying pixel buffer for efficiency reasons. - Triggers a  vipy.image.Image.load if the pixel buffer has not been loaded - This will trigger a copy if the ['WRITEABLE' flag](https: numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) is not set.",
"func":1
},
{
"ref":"vipy.image.Scene.channel",
"url":28,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None. Iterate over channels as single channel luminance images: >>> for c in self.channel(): >>> print(c) Return the kth channel as a single channel luminance image: >>> c = self.channel(k=0)",
"func":1
},
{
"ref":"vipy.image.Scene.red",
"url":28,
"doc":"Return red channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.red()  self.channel(0) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.red()  self.channel(3)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.Scene.green",
"url":28,
"doc":"Return green channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.green()  self.channel(1) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.green()  self.channel(1)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.Scene.blue",
"url":28,
"doc":"Return blue channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.vlue()  self.channel(2) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.blue()  self.channel(0)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.Scene.alpha",
"url":28,
"doc":"Return alpha (transparency) channel as a cloned single channel  vipy.image.Image object",
"func":1
},
{
"ref":"vipy.image.Scene.zeros",
"url":28,
"doc":"Set the pixel buffer to all zeros of the same shape and datatype as this  vipy.image.Image object. These are equivalent operations for the resulting buffer shape: >>> import numpy as np >>> np.zeros( (self.width(), self.height(), self.channels( )  self.zeros().array() Returns: This  vipy.image.Image object.  note Triggers load() if the pixel buffer has not been loaded yet.",
"func":1
},
{
"ref":"vipy.image.Scene.pil",
"url":28,
"doc":"Convert vipy.image.Image to PIL Image. Returns: A [PIL image](https: pillow.readthedocs.io/en/stable/reference/Image.html) object, that shares the pixel buffer by reference",
"func":1
},
{
"ref":"vipy.image.Scene.blur",
"url":28,
"doc":"Apply a Gaussian blur with Gaussian kernel radius=sigma to the pixel buffer. Args: sigma: [float >0] The gaussian blur kernel radius. Returns: This  vipy.image.Image object with the pixel buffer blurred in place.",
"func":1
},
{
"ref":"vipy.image.Scene.torch",
"url":28,
"doc":"Convert the batch of 1 HxWxC images to a CxHxW torch tensor. Args: order: ['CHW', 'HWC', 'NCHW', 'NHWC']. The axis order of the torch tensor (channels, height, width) or (height, width, channels) or (1, channels, height, width) or (1, height, width, channels) Returns: A CxHxW or HxWxC or 1xCxHxW or 1xHxWxC [torch tensor](https: pytorch.org/docs/stable/tensors.html) that shares the pixel buffer of this image object by reference.",
"func":1
},
{
"ref":"vipy.image.Scene.fromtorch",
"url":28,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.filename",
"url":28,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.Scene.url",
"url":28,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.Scene.colorspace",
"url":28,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.Scene.uri",
"url":28,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.Scene.setattribute",
"url":28,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.Scene.setattributes",
"url":28,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.Scene.clone",
"url":28,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.Scene.flush",
"url":28,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.Scene.resize_like",
"url":28,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.Scene.zeropadlike",
"url":28,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.Scene.alphapad",
"url":28,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Scene.minsquare",
"url":28,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Scene.maxsquare",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Scene.maxmatte",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.Scene.imagebox",
"url":28,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.Scene.border_mask",
"url":28,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.Scene.affine_transform",
"url":28,
"doc":"Apply a 3x3 affine geometric transformation to the image. See also  vipy.geometry.affine_transform  note The image will be loaded and converted to float() prior to applying the affine transformation.",
"func":1
},
{
"ref":"vipy.image.Scene.rgb",
"url":28,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.rgba",
"url":28,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.hsv",
"url":28,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.bgr",
"url":28,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.bgra",
"url":28,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.float",
"url":28,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.Scene.greyscale",
"url":28,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.Scene.grayscale",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Scene.grey",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Scene.gray",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Scene.luminance",
"url":28,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.Scene.lum",
"url":28,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.Scene.jet",
"url":28,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.Scene.rainbow",
"url":28,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Scene.hot",
"url":28,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Scene.bone",
"url":28,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Scene.saturate",
"url":28,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.Scene.intensity",
"url":28,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.Scene.mat2gray",
"url":28,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.Scene.gain",
"url":28,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.Scene.bias",
"url":28,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.Scene.mean",
"url":28,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.Scene.meanchannel",
"url":28,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.Scene.closeall",
"url":28,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.Scene.close",
"url":28,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.Scene.save",
"url":28,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.Scene.pkl",
"url":28,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Scene.pklif",
"url":28,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Scene.saveas",
"url":28,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.Scene.saveastmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.Scene.savetmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.Scene.base64",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.Scene.html",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.Scene.map",
"url":28,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.Scene.downcast",
"url":28,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.Scene.perceptualhash_distance",
"url":28,
"doc":"Hamming distance between two perceptual hashes",
"func":1
},
{
"ref":"vipy.image.ImageDetection",
"url":28,
"doc":"vipy.image.ImageDetection class This class provides a representation of a vipy.image.Image with a single object detection with a category and a vipy.geometry.BoundingBox This class inherits all methods of Scene and BoundingBox. Be careful with overloaded methods clone(), width() and height() which will correspond to these methods for Scene() and not BoundingBox(). Use bbclone(), bbwidth() or bbheight() to access the subclass. Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label'), and BoundingBox coordinates >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, width=100, height=100) >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, xmax=100, ymax=100) >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xcentroid=50, ycentroid=50, width=100, height=100) >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', bbox=vipy.geometry.BoundingBox(xmin=0, ymin=0, width=100, height=100 >>> im = vipy.image.ImageCategory(url='http: path/to/dog_image.ext', category='dog').boundingbox(xmin=0, ymin=0, width=100, height=100) >>> im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog', xmin=0, ymin=0, width=100, height=100)"
},
{
"ref":"vipy.image.ImageDetection.cast",
"url":28,
"doc":"Typecast the conformal vipy.image object im as  vipy.image.Image . This is useful for downcasting  vipy.image.Scene or  vipy.image.ImageDetection down to an image. >>> ims = vipy.image.RandomScene() >>> im = vipy.image.Image.cast(im)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.boundingbox",
"url":28,
"doc":"Modify the bounding box using the provided parameters, or return the box if no parameters provided",
"func":1
},
{
"ref":"vipy.image.ImageDetection.asimage",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.asbbox",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.boxmap",
"url":28,
"doc":"Apply the lambda function f to the bounding box, and return the imagedetection",
"func":1
},
{
"ref":"vipy.image.ImageDetection.crop",
"url":28,
"doc":"Crop the image using the bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.append",
"url":28,
"doc":"Append the provided vipy.object.Detection object to the scene object list",
"func":1
},
{
"ref":"vipy.image.ImageDetection.detection",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isinterior",
"url":28,
"doc":"Is the bounding box fully within the image rectangle? Use provided image width and height (W,H) to avoid lots of reloads in some conditions",
"func":1
},
{
"ref":"vipy.image.ImageDetection.from_json",
"url":28,
"doc":"Import the JSON string s as an  vipy.image.Image object. This will perform a round trip such that im1  im2 >>> im1 = vupy.image.RandomImage() >>> im2 = vipy.image.Image.from_json(im1.json( >>> assert im1  im2",
"func":1
},
{
"ref":"vipy.image.ImageDetection.add",
"url":28,
"doc":"Alias for append",
"func":1
},
{
"ref":"vipy.image.ImageDetection.objectmap",
"url":28,
"doc":"Apply lambda function f to each object. If f is a list of lambda, apply one to one with the objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.objectfilter",
"url":28,
"doc":"Apply lambda function f to each object and keep if filter is True",
"func":1
},
{
"ref":"vipy.image.ImageDetection.nms",
"url":28,
"doc":"Non-maximum supporession of objects() by category based on confidence and spatial IoU and cover thresholds",
"func":1
},
{
"ref":"vipy.image.ImageDetection.intersection",
"url":28,
"doc":"Return a Scene() containing the objects in both self and other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.ImageDetection.difference",
"url":28,
"doc":"Return a Scene() containing the objects in self but not other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.ImageDetection.union",
"url":28,
"doc":"Combine the objects of the scene with other and self with no duplicate checking unless miniou is not None",
"func":1
},
{
"ref":"vipy.image.ImageDetection.uncrop",
"url":28,
"doc":"Uncrop a previous crop(bb) called with the supplied bb=BoundingBox(), and zeropad to shape=(H,W)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.clear",
"url":28,
"doc":"Remove all objects from this scene.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.categories",
"url":28,
"doc":"Return list of unique object categories in scene",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imclip",
"url":28,
"doc":"Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rescale",
"url":28,
"doc":"Rescale image buffer and all bounding boxes - Not idempotent",
"func":1
},
{
"ref":"vipy.image.ImageDetection.resize",
"url":28,
"doc":"Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly. If cols or rows is None, then scale isotropically",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centersquare",
"url":28,
"doc":"Crop the image of size (H,W) to be centersquare (min(H,W), min(H,W preserving center, and update bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fliplr",
"url":28,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.ImageDetection.flipud",
"url":28,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilate",
"url":28,
"doc":"Dilate all bounding boxes by scale factor, dilated boxes may be outside image rectangle",
"func":1
},
{
"ref":"vipy.image.ImageDetection.zeropad",
"url":28,
"doc":"Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.ImageDetection.meanpad",
"url":28,
"doc":"Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rot90cw",
"url":28,
"doc":"Rotate the scene 90 degrees clockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rot90ccw",
"url":28,
"doc":"Rotate the scene 90 degrees counterclockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxdim",
"url":28,
"doc":"Resize scene preserving aspect ratio so that maximum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mindim",
"url":28,
"doc":"Resize scene preserving aspect ratio so that minimum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centercrop",
"url":28,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cornercrop",
"url":28,
"doc":"Crop image of size (height x width) from the upper left corner, returning valid pixels only",
"func":1
},
{
"ref":"vipy.image.ImageDetection.padcrop",
"url":28,
"doc":"Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cornerpadcrop",
"url":28,
"doc":"Crop image of size (height x width) from the upper left corner, returning zero padded result out to (height, width)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rectangular_mask",
"url":28,
"doc":"Return a binary array of the same size as the image (or using the provided image width and height (W,H) size to avoid an image load), with ones inside the bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.binarymask",
"url":28,
"doc":"Alias for rectangular_mask with in-place update",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bgmask",
"url":28,
"doc":"Set all pixels outside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fgmask",
"url":28,
"doc":"Set all pixels inside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pixelmask",
"url":28,
"doc":"Replace pixels within all foreground objects with a privacy preserving pixelated foreground with larger pixels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blurmask",
"url":28,
"doc":"Replace pixels within all foreground objects with a privacy preserving blurred foreground",
"func":1
},
{
"ref":"vipy.image.ImageDetection.replace",
"url":28,
"doc":"Set all image values within the bounding box equal to the provided img, triggers load() and imclip()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.meanmask",
"url":28,
"doc":"Replace pixels within the foreground objects with the mean pixel color",
"func":1
},
{
"ref":"vipy.image.ImageDetection.perceptualhash",
"url":28,
"doc":"Perceptual differential hash function. This function sets foreground objects to mean color, convert to greyscale, resize with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs. Args: bits: [int] longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes). objmask: [bool] if true, replace the foreground object masks with the mean color prior to computing asbinary: [bool] If true, return a binary array asbytes: [bool] if true return a byte array Returns: A hash string encoding the perceptual hash such that  vipy.image.Image.perceptualhash_distance can be used to compute a hash distance asbytes: a bytes array asbinary: a numpy binary array  notes - Can be used for near duplicate detection of background scenes by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing. Equivalently,  vipy.image.Image.perceptualhash_distance . - The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex( bghash()  which is equivalent to perceptualhash(asbinary=True)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fghash",
"url":28,
"doc":"Perceptual differential hash function, computed for each foreground region independently",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bghash",
"url":28,
"doc":"Percetual differential hash function, masking out foreground regions",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isduplicate",
"url":28,
"doc":"Background hash near duplicate detection, returns true if self and im are near duplicate images using bghash",
"func":1
},
{
"ref":"vipy.image.ImageDetection.show",
"url":28,
"doc":"Show scene detection Args: categories: [list] List of category (or shortlabel) names in the scene to show fontsize: [int] or [str]: Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size figure: [int] Figure number, show the image in the provided figure=int numbered window nocaption: [bool] Show or do not show the text caption in the upper left of the box nocaption_withstring: [list]: Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list  boxalpha (float, [0,1]): Set the text box background to be semi-transparent with an alpha  d_category2color (dict): Define a dictionary of required mapping of specific category() to box colors. Non-specified categories are assigned a random named color from vipy.show.colorlist()  caption_offset (int, int): The relative position of the caption to the upper right corner of the box.  nowindow (bool): Display or not display the image  textfacecolor (str): One of the named colors from vipy.show.colorlist() for the color of the textbox background  textfacealpha (float, [0,1]): The textbox background transparency  shortlabel (bool): Whether to show the shortlabel or the full category name in the caption  mutator (lambda): A lambda function with signature lambda im: f(im) which will modify this image prior to show. Useful for changing labels on the fly",
"func":1
},
{
"ref":"vipy.image.ImageDetection.annotate",
"url":28,
"doc":"Alias for savefig",
"func":1
},
{
"ref":"vipy.image.ImageDetection.savefig",
"url":28,
"doc":"Save show() output to given file or return buffer without popping up a window",
"func":1
},
{
"ref":"vipy.image.ImageDetection.category",
"url":28,
"doc":"Return or update the category",
"func":1
},
{
"ref":"vipy.image.ImageDetection.label",
"url":28,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.image.ImageDetection.score",
"url":28,
"doc":"Real valued score for categorization, larger is better",
"func":1
},
{
"ref":"vipy.image.ImageDetection.probability",
"url":28,
"doc":"Real valued probability for categorization, [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageDetection.print",
"url":28,
"doc":"Print the representation of the image and return self with an optional sleep=n seconds Useful for debugging in long fluent chains.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.tile",
"url":28,
"doc":"Generate an image tiling. A tiling is a decomposition of an image into overlapping or non-overlapping rectangular regions. Args: tilewidth: [int] the image width of each tile tileheight: [int] the image height of each tile overlaprows: [int] the number of overlapping rows (height) for each tile overlapcols: [int] the number of overlapping width (width) for each tile Returns: A list of  vipy.image.Image objects such that each image is a single tile and the set of these tiles forms the original image Each image in the returned list contains the 'tile' attribute which encodes the crop used to create the tile.  note -  vipy.image.Image.tile can be undone using  vipy.image.Image.untile - The identity tiling is im.tile(im.widht(), im.height(), overlaprows=0, overlapcols=0) - Ragged tiles outside the image boundary are zero padded - All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.ImageDetection.untile",
"url":28,
"doc":"Undo an image tiling and recreate the original image. >>> tiles = im.tile(im.width()/2, im.height()/2, 0, 0) >>> imdst = vipy.image.Image.untile(tiles) >>> assert imdst  im Args: imlist: this must be the output of  vipy.image.Image.tile Returns: A new  vipy.image.Image object reconstructed from the tiling, such that this is equivalent to the input to vipy.image.Image.tile  note All annotations are updated properly for each tile, when the source image is  vipy.image.Scene ",
"func":1
},
{
"ref":"vipy.image.ImageDetection.splat",
"url":28,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.ImageDetection.store",
"url":28,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.ImageDetection.unstore",
"url":28,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.restore",
"url":28,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.abspath",
"url":28,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.relpath",
"url":28,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.canload",
"url":28,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dict",
"url":28,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.ImageDetection.loader",
"url":28,
"doc":"Lambda function to load an unsupported image filename to a numpy array. This lambda function will be executed during load and the result will be stored in self._array",
"func":1
},
{
"ref":"vipy.image.ImageDetection.load",
"url":28,
"doc":"Load image to cached private '_array' attribute. Args: ignoreErrors: [bool] If true, ignore any exceptions thrown during load and print the corresponding error messages. This is useful for loading images distributed without throwing exceptions when some images may be corrupted. In this case, the _array attribute will be None and  vipy.image.Image.isloaded will return false to determine if the image is loaded, which can be used to filter out corrupted images gracefully. verbose: [bool] If true, show additional useful printed output Returns: This  vipy.image.Image object with the pixels loaded in self._array as a numpy array.  note This loader supports any image file format supported by PIL. A custom loader can be added using  vipy.image.Image.loader .",
"func":1
},
{
"ref":"vipy.image.ImageDetection.download",
"url":28,
"doc":"Download URL to filename provided by constructor, or to temp filename. Args: ignoreErrors: [bool] If true, do not throw an exception if the download of the URL fails for some reason. Instead, print out a reason and return this image object. The function  vipy.image.Image.hasfilename will return false if the downloaded file does not exist and can be used to filter these failed downloads gracefully. timeout: [int] The timeout in seconds for an http or https connection attempt. See also [urllib.request.urlopen](https: docs.python.org/3/library/urllib.request.html). verbose: [bool] If true, output more helpful message. Returns: This  vipy.image.Image object with the URL downloaded to  vipy.image.Image.filename or to a  vipy.util.tempimage filename which can be retrieved with  vipy.image.Image.filename .",
"func":1
},
{
"ref":"vipy.image.ImageDetection.reload",
"url":28,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isloaded",
"url":28,
"doc":"Return True if  vipy.image.Image.load was successful in reading the image, or if the pixels are present in  vipy.image.Image.array .",
"func":1
},
{
"ref":"vipy.image.ImageDetection.channels",
"url":28,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iscolor",
"url":28,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.ImageDetection.istransparent",
"url":28,
"doc":"Transparent images are four channel color images with transparency, float32 or uint8. Return true if this image contains an alpha transparency channel",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isgrey",
"url":28,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isluminance",
"url":28,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.ImageDetection.filesize",
"url":28,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.ImageDetection.width",
"url":28,
"doc":"Return the width (columns) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.height",
"url":28,
"doc":"Return the height (rows) of the image in integer pixels.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.shape",
"url":28,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image. Returns: A tuple (height=int, width=int) of the image.  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.aspectratio",
"url":28,
"doc":"Return the aspect ratio of the image as (width/height) ratio. Returns: A float equivalent to ( vipy.image.Image.width /  vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.area",
"url":28,
"doc":"Return the area of the image as (width  height). Returns: An integer equivalent to ( vipy.image.Image.width   vipy.image.Image.height )  note This triggers a  vipy.image.Image.load if the image is not already loaded.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centroid",
"url":28,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y). The centroid is equivalent to half the  vipy.image.Image.shape . Returns: A tuple (column, row) of the floating point center of the image.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centerpixel",
"url":28,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j) The centerpixel is equivalent to half the  vipy.image.Image.shape floored to the nearest integer pixel coordinate. Returns: A tuple (int(column), int(row of the integer center of the image.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.array",
"url":28,
"doc":"Replace self._array with provided numpy array Args: np_array: [numpy array] A new array to use as the pixel buffer for this image. copy: [bool] If true, copy the buffer using np.copy(), else use a reference to this buffer. Returns: - If np_array is not None, return the  vipy.image.Image object such that this object points to the provided numpy array as the pixel buffer - If np_array is None, then return the numpy array.  notes - If copy=False, then this  vipy.image.Image object will share the pixel buffer with the owner of np_array. Changes to pixels in this buffer will be shared. - If copy=True, then this will significantly slow down processing for large images. Use referneces wherevery possible.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fromarray",
"url":28,
"doc":"Alias for  vipy.image.Image.array with copy=True. This will set new numpy array as the pixel buffer with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.tonumpy",
"url":28,
"doc":"Alias for  vipy.image.Image.numpy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.numpy",
"url":28,
"doc":"Return a mutable numpy array for this  vipy.image.Image .  notes - This will always return a writeable array with the 'WRITEABLE' numpy flag set. This is useful for returning a mutable numpy array as needed while keeping the original non-mutable numpy array (e.g. loaded from a video or PIL) as the underlying pixel buffer for efficiency reasons. - Triggers a  vipy.image.Image.load if the pixel buffer has not been loaded - This will trigger a copy if the ['WRITEABLE' flag](https: numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) is not set.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.channel",
"url":28,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None. Iterate over channels as single channel luminance images: >>> for c in self.channel(): >>> print(c) Return the kth channel as a single channel luminance image: >>> c = self.channel(k=0)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.red",
"url":28,
"doc":"Return red channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.red()  self.channel(0) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.red()  self.channel(3)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.green",
"url":28,
"doc":"Return green channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.green()  self.channel(1) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.green()  self.channel(1)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blue",
"url":28,
"doc":"Return blue channel as a cloned single channel  vipy.image.Image object. These are equivalent operations if the colorspace is 'rgb' or 'rgba': >>> self.vlue()  self.channel(2) These are equivalent operations if the colorspace is 'bgr' or 'bgra': >>> self.blue()  self.channel(0)  note OpenCV returns images in BGR colorspace. Use this method to always return the desired channel by color.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.alpha",
"url":28,
"doc":"Return alpha (transparency) channel as a cloned single channel  vipy.image.Image object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.zeros",
"url":28,
"doc":"Set the pixel buffer to all zeros of the same shape and datatype as this  vipy.image.Image object. These are equivalent operations for the resulting buffer shape: >>> import numpy as np >>> np.zeros( (self.width(), self.height(), self.channels( )  self.zeros().array() Returns: This  vipy.image.Image object.  note Triggers load() if the pixel buffer has not been loaded yet.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pil",
"url":28,
"doc":"Convert vipy.image.Image to PIL Image. Returns: A [PIL image](https: pillow.readthedocs.io/en/stable/reference/Image.html) object, that shares the pixel buffer by reference",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blur",
"url":28,
"doc":"Apply a Gaussian blur with Gaussian kernel radius=sigma to the pixel buffer. Args: sigma: [float >0] The gaussian blur kernel radius. Returns: This  vipy.image.Image object with the pixel buffer blurred in place.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.torch",
"url":28,
"doc":"Convert the batch of 1 HxWxC images to a CxHxW torch tensor. Args: order: ['CHW', 'HWC', 'NCHW', 'NHWC']. The axis order of the torch tensor (channels, height, width) or (height, width, channels) or (1, channels, height, width) or (1, height, width, channels) Returns: A CxHxW or HxWxC or 1xCxHxW or 1xHxWxC [torch tensor](https: pytorch.org/docs/stable/tensors.html) that shares the pixel buffer of this image object by reference.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fromtorch",
"url":28,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.filename",
"url":28,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.url",
"url":28,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.ImageDetection.colorspace",
"url":28,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.ImageDetection.uri",
"url":28,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setattribute",
"url":28,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setattributes",
"url":28,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.clone",
"url":28,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.flush",
"url":28,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.ImageDetection.resize_like",
"url":28,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.zeropadlike",
"url":28,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.alphapad",
"url":28,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageDetection.minsquare",
"url":28,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxsquare",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxmatte",
"url":28,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imagebox",
"url":28,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.ImageDetection.border_mask",
"url":28,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.ImageDetection.affine_transform",
"url":28,
"doc":"Apply a 3x3 affine geometric transformation to the image. See also  vipy.geometry.affine_transform  note The image will be loaded and converted to float() prior to applying the affine transformation.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rgb",
"url":28,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rgba",
"url":28,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hsv",
"url":28,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bgr",
"url":28,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bgra",
"url":28,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.float",
"url":28,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.ImageDetection.greyscale",
"url":28,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageDetection.grayscale",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.grey",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.gray",
"url":28,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.luminance",
"url":28,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.ImageDetection.lum",
"url":28,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.jet",
"url":28,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rainbow",
"url":28,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hot",
"url":28,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bone",
"url":28,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.saturate",
"url":28,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.ImageDetection.intensity",
"url":28,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mat2gray",
"url":28,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.gain",
"url":28,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bias",
"url":28,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mean",
"url":28,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.meanchannel",
"url":28,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.closeall",
"url":28,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.ImageDetection.close",
"url":28,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.ImageDetection.save",
"url":28,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pkl",
"url":28,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pklif",
"url":28,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageDetection.saveas",
"url":28,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.saveastmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.savetmp",
"url":28,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.base64",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.ImageDetection.html",
"url":28,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.ImageDetection.map",
"url":28,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageDetection.downcast",
"url":28,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.perceptualhash_distance",
"url":28,
"doc":"Hamming distance between two perceptual hashes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xmin",
"url":29,
"doc":"x coordinate of upper left corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ul",
"url":29,
"doc":"Upper left coordinate (x,y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ulx",
"url":29,
"doc":"Upper left coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.uly",
"url":29,
"doc":"Upper left coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ur",
"url":29,
"doc":"Upper right coordinate (x,y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.urx",
"url":29,
"doc":"Upper right coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ury",
"url":29,
"doc":"Upper right coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ll",
"url":29,
"doc":"Lower left coordinate (x,y), synonym for bl()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bl",
"url":29,
"doc":"Bottom left coordinate (x,y), synonym for ll()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blx",
"url":29,
"doc":"Bottom left coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bly",
"url":29,
"doc":"Bottom left coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.lr",
"url":29,
"doc":"Lower right coordinate (x,y), synonym for br()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.br",
"url":29,
"doc":"Bottom right coordinate (x,y), synonym for lr()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.brx",
"url":29,
"doc":"Bottom right coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bry",
"url":29,
"doc":"Bottom right coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ymin",
"url":29,
"doc":"y coordinate of upper left corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xmax",
"url":29,
"doc":"x coordinate of lower right corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ymax",
"url":29,
"doc":"y coordinate of lower right corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.image.ImageDetection.upperleft",
"url":29,
"doc":"Return the (x,y) upper left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bottomleft",
"url":29,
"doc":"Return the (x,y) lower left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.upperright",
"url":29,
"doc":"Return the (x,y) upper right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bottomright",
"url":29,
"doc":"Return the (x,y) lower right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.int",
"url":29,
"doc":"Convert corners to integer with rounding, in-place update",
"func":1
},
{
"ref":"vipy.image.ImageDetection.significant_digits",
"url":29,
"doc":"Convert corners to have at most n significant digits for efficient JSON storage",
"func":1
},
{
"ref":"vipy.image.ImageDetection.translate",
"url":29,
"doc":"Translate the bounding box by dx in x and dy in y",
"func":1
},
{
"ref":"vipy.image.ImageDetection.to_origin",
"url":29,
"doc":"Translate the bounding box so that (xmin, ymin) = (0,0)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.set_origin",
"url":29,
"doc":"Set the origin of the coordinates of this bounding box to be relative to the upper left of the other bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.offset",
"url":29,
"doc":"Alias for translate",
"func":1
},
{
"ref":"vipy.image.ImageDetection.invalid",
"url":29,
"doc":"Is the box a valid bounding box?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setwidth",
"url":29,
"doc":"Set new width keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setheight",
"url":29,
"doc":"Set new height keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xcentroid",
"url":29,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centroid_x",
"url":29,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ycentroid",
"url":29,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centroid_y",
"url":29,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.to_xywh",
"url":29,
"doc":"Return bounding box corners as (x,y,width,height) tuple",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xywh",
"url":29,
"doc":"Alias for to_xywh",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cxywh",
"url":29,
"doc":"Return or set bounding box corners as (centroidx,centroidy,width,height) tuple",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ulbr",
"url":29,
"doc":"Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.to_ulbr",
"url":29,
"doc":"Alias for ulbr()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dx",
"url":29,
"doc":"Offset bounding box by same xmin as provided box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dy",
"url":29,
"doc":"Offset bounding box by ymin of provided box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.sqdist",
"url":29,
"doc":"Squared Euclidean distance between upper left corners of two bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dist",
"url":29,
"doc":"Distance between centroids of two bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pdist",
"url":29,
"doc":"Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iou",
"url":29,
"doc":"area of intersection / area of union",
"func":1
},
{
"ref":"vipy.image.ImageDetection.intersection_over_union",
"url":29,
"doc":"Alias for iou",
"func":1
},
{
"ref":"vipy.image.ImageDetection.area_of_intersection",
"url":29,
"doc":"area of intersection",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cover",
"url":29,
"doc":"Fraction of this bounding box intersected by other bbox (bb)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxcover",
"url":29,
"doc":"The maximum cover of self to bb and bb to self",
"func":1
},
{
"ref":"vipy.image.ImageDetection.shapeiou",
"url":29,
"doc":"Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hasintersection",
"url":29,
"doc":"Return true if self and bb overlap by any amount, or by the cover threshold (if provided) or the iou threshold (if provided). This is a convenience function that allows for shared computation for fast non-maximum suppression.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isinside",
"url":29,
"doc":"Is this boundingbox fully within the provided bounding box?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ispointinside",
"url":29,
"doc":"Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilatepx",
"url":29,
"doc":"Dilate by a given pixel amount on all sides, keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilate_height",
"url":29,
"doc":"Change scale of bounding box in y direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilate_width",
"url":29,
"doc":"Change scale of bounding box in x direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.top",
"url":29,
"doc":"Make top of box taller (closer to top of image) by an offset dy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bottom",
"url":29,
"doc":"Make bottom of box taller (closer to bottom of image) by an offset dy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.left",
"url":29,
"doc":"Make left of box wider (closer to left side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.image.ImageDetection.right",
"url":29,
"doc":"Make right of box wider (closer to right side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.image.ImageDetection.scalex",
"url":29,
"doc":"Multiply the box corners in the x dimension by a scale factor",
"func":1
},
{
"ref":"vipy.image.ImageDetection.scaley",
"url":29,
"doc":"Multiply the box corners in the y dimension by a scale factor",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imscale",
"url":29,
"doc":"Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iseven",
"url":29,
"doc":"Are all corners even number integers?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.even",
"url":29,
"doc":"Force all corners to be even number integers. This is helpful for FFMPEG crop filters.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hasoverlap",
"url":29,
"doc":"Does the bounding box intersect with the provided image rectangle?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iminterior",
"url":29,
"doc":"Transform bounding box to be interior to the image rectangle with shape (W,H). Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imclipshape",
"url":29,
"doc":"Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.convexhull",
"url":29,
"doc":"Given a set of points  x1,y1],[x2,xy], .], return the bounding rectangle, typecast to float",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mindimension",
"url":29,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ellipse",
"url":29,
"doc":"Convert the boundingbox to a vipy.geometry.Ellipse object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.average",
"url":29,
"doc":"Compute the average bounding box between self and other, and set self to the average. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.averageshape",
"url":29,
"doc":"Compute the average bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.medianshape",
"url":29,
"doc":"Compute the median bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.shapedist",
"url":29,
"doc":"L1 distance between (width,height) of two boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.affine",
"url":29,
"doc":"Apply an 2x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.projective",
"url":29,
"doc":"Apply an 3x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.confidence",
"url":29,
"doc":"Bounding boxes do not have confidences, use vipy.object.Detection()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.grid",
"url":29,
"doc":"Split a bounding box into the smallest grid of non-overlapping bounding boxes such that the union is the original box",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackid",
"url":28,
"doc":"Mutate the image to show track ID with a fixed number of digits appended to the shortlabel as ( )",
"func":1
},
{
"ref":"vipy.image.mutator_show_jointlabel",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackindex",
"url":28,
"doc":"Mutate the image to show track index appended to the shortlabel as ( )",
"func":1
},
{
"ref":"vipy.image.mutator_show_userstring",
"url":28,
"doc":"Mutate the image to show user supplied strings in the shortlabel. The list be the same length oas the number of objects in the image. This is not checked. This is passed to show()",
"func":1
},
{
"ref":"vipy.image.mutator_show_noun_only",
"url":28,
"doc":"Mutate the image to show the noun only",
"func":1
},
{
"ref":"vipy.image.mutator_show_verb_only",
"url":28,
"doc":"Mutate the image to show the verb only",
"func":1
},
{
"ref":"vipy.image.mutator_show_noun_or_verb",
"url":28,
"doc":"Mutate the image to show the verb only if it is non-zero else noun",
"func":1
},
{
"ref":"vipy.image.mutator_capitalize",
"url":28,
"doc":"Mutate the image to show the shortlabel as 'Noun Verb1 Noun Verb2'",
"func":1
},
{
"ref":"vipy.image.mutator_show_activityonly",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackindex_activityonly",
"url":28,
"doc":"Mutate the image to show boxes colored by track index, and only show 'noun verb' captions",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackindex_verbonly",
"url":28,
"doc":"Mutate the image to show boxes colored by track index, and only show 'verb' captions with activity confidence",
"func":1
},
{
"ref":"vipy.image.RandomImage",
"url":28,
"doc":"Return a uniform random color  vipy.image.Image of size (rows, cols)",
"func":1
},
{
"ref":"vipy.image.RandomImageDetection",
"url":28,
"doc":"Return a uniform random color  vipy.image.ImageDetection of size (rows, cols) with a random bounding box",
"func":1
},
{
"ref":"vipy.image.RandomScene",
"url":28,
"doc":"Return a uniform random color  vipy.image.Scene of size (rows, cols) with a specified number of vipy.object.Detection objects",
"func":1
},
{
"ref":"vipy.image.owl",
"url":28,
"doc":"Return a suberb owl image for testing",
"func":1
},
{
"ref":"vipy.image.vehicles",
"url":28,
"doc":"Return a highway scene with the four highest confidence vehicle detections for testing",
"func":1
},
{
"ref":"vipy.image.people",
"url":28,
"doc":"Return a crowd scene with the four highest confidence person detections for testing",
"func":1
},
{
"ref":"vipy.show",
"url":30,
"doc":""
},
{
"ref":"vipy.show.figure",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.close",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.closeall",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.show",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.noshow",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.imshow",
"url":30,
"doc":"Show an image in the provided figure number",
"func":1
},
{
"ref":"vipy.show.imbbox",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.imdetection",
"url":30,
"doc":"Show a list of vipy.object.Detections overlayed on img. Image must be RGB",
"func":1
},
{
"ref":"vipy.show.frame",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.imframe",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.savefig",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.colorlist",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.show.text",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.torch",
"url":31,
"doc":""
},
{
"ref":"vipy.torch.fromtorch",
"url":31,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with an inferred colorspace based on channels and datatype",
"func":1
},
{
"ref":"vipy.torch.GaussianPyramid",
"url":31,
"doc":""
},
{
"ref":"vipy.torch.LaplacianPyramid",
"url":31,
"doc":""
},
{
"ref":"vipy.torch.LaplacianPyramid.reconstruct",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.torch.Foveation",
"url":31,
"doc":""
},
{
"ref":"vipy.torch.Foveation.foveate",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.video",
"url":32,
"doc":""
},
{
"ref":"vipy.video.Video",
"url":32,
"doc":"vipy.video.Video class The vipy.video class provides a fluent, lazy interface for representing, transforming and visualizing videos. The following constructors are supported: >>> vid = vipy.video.Video(filename='/path/to/video.ext') Valid video extensions are those that are supported by ffmpeg ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']. >>> vid = vipy.video.Video(url='https: www.youtube.com/watch?v=MrIN959JuV8') >>> vid = vipy.video.Video(url='http: path/to/video.ext', filename='/path/to/video.ext') Youtube URLs are downloaded to a temporary filename, retrievable as vid.download().filename(). If the environment variable 'VIPY_CACHE' is defined, then videos are saved to this directory rather than the system temporary directory. If a filename is provided to the constructor, then that filename will be used instead of a temp or cached filename. URLs can be defined as an absolute URL to a video file, or to a site supported by 'youtube-dl' (https: ytdl-org.github.io/youtube-dl/supportedsites.html) >>> vid = vipy.video.Video(url='s3: BUCKET.s3.amazonaws.com/PATH/video.ext') If you set the environment variables VIPY_AWS_ACCESS_KEY_ID and VIPY_AWS_SECRET_ACCESS_KEY, then this will download videos directly from S3 using boto3 and store in VIPY_CACHE. Note that the URL protocol should be 's3' and not 'http' to enable keyed downloads. >>> vid = vipy.video.Video(array=array, colorspace='rgb') The input 'array' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video Note that some video transformations are only available prior to load(), and the array() is assumed immutable after load(). >>> frames = [im for im in vipy.video.RandomVideo()] >>> vid = vipy.video.Video(frames=frames) Args: filename: [str] The path to a video file. url: [str] The URL to a video file. If filename is not provided, then a random filename is assigned in VIPY_CACHE on download framerate: [float] The framerate of the video file. This is required. You can introspect this using ffprobe. attributes: [dict] A user supplied dictionary of metadata about this video. colorspace: [str] Must be in ['rgb', 'float'] array: [numpy] An NxHxWxC numpy array for N frames each HxWxC shape startframe: [int] A start frame to clip the video endframe: [int] An end frame to clip the video startsec: [float] A start time in seconds to clip the video (this requires setting framerate) endsec: [float] An end time in seconds to clip the video (this requires setting framerate) frames: [list of  vipy.image.Image ] A list of frames in the video probeshape: [bool] If true, then probe the shape of the video from ffprobe to avoid an explicit preview later. This can speed up loading in some circumstances."
},
{
"ref":"vipy.video.Video.cast",
"url":32,
"doc":"Cast a conformal video object to a  vipy.video.Video object. This is useful for downcasting superclasses. >>> vs = vipy.video.RandomScene() >>> v = vipy.video.Video.cast(vs)",
"func":1
},
{
"ref":"vipy.video.Video.from_json",
"url":32,
"doc":"Import a json string as a  vipy.video.Video object. This will perform a round trip from a video to json and back to a video object. This same operation is used for serialization of all vipy objects to JSON for storage. >>> v = vipy.video.Video.from_json(vipy.video.RandomVideo().json( ",
"func":1
},
{
"ref":"vipy.video.Video.metadata",
"url":32,
"doc":"Return a dictionary of metadata about this video. This is an alias for the 'attributes' dictionary.",
"func":1
},
{
"ref":"vipy.video.Video.videoid",
"url":32,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by SHA1 hash of the  vipy.video.Video.filename and  vipy.video.Video.url . Args: newid: [str] If not None, then update the video_id as newid. Returns: The video ID if newid=None else self  note - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', ${MY_ID}), or pass in newid",
"func":1
},
{
"ref":"vipy.video.Video.frame",
"url":32,
"doc":"Return the kth frame as an  vipy.image Image object",
"func":1
},
{
"ref":"vipy.video.Video.store",
"url":32,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. >>> v  v.store().restore(v.filename(  note -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing.",
"func":1
},
{
"ref":"vipy.video.Video.unstore",
"url":32,
"doc":"Delete the currently stored video from  vipy.video.Video.store",
"func":1
},
{
"ref":"vipy.video.Video.restore",
"url":32,
"doc":"Save the currently stored video as set using  vipy.video.Video.store to filename, and set up filename",
"func":1
},
{
"ref":"vipy.video.Video.stream",
"url":32,
"doc":"Iterator to yield groups of frames streaming from video. A video stream is a real time iterator to read or write from a video. Streams are useful to group together frames into clips that are operated on as a group. The following use cases are supported: >>> v = vipy.video.RandomScene() Stream individual video frames lagged by 10 frames and 20 frames >>> for (im1, im2) in zip(v.stream().frame(n=-10), v.stream().frame(n=-20 : >>> print(im1, im2) Stream overlapping clips such that each clip is a video n=16 frames long and starts at frame i, and the next clip is n=16 frames long and starts at frame i=i+m >>> for vc in v.stream().clip(n=16, m=4): >>> print(vc) Stream non-overlapping batches of frames such that each clip is a video of length n and starts at frame i, and the next clip is length n and starts at frame i+n >>> for vb in v.stream().batch(n=16): >>> print(vb) Create a write stream to incrementally add frames to long video. >>> vi = vipy.video.Video(filename='/path/to/output.mp4') >>> vo = vipy.video.Video(filename='/path/to/input.mp4') >>> with vo.stream(write=True) as s: >>> for im in vi.stream(): >>> s.write(im)  manipulate pixels of im, if desired Args: write: [bool] If true, create a write stream overwrite: [bool] If true, and the video output filename already exists, overwrite it bufsize: [int] The maximum queue size for the pipe thread. Returns: A  vipy.video.Video.stream.Stream object  note Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.",
"func":1
},
{
"ref":"vipy.video.Video.clear",
"url":32,
"doc":"no-op for  vipy.video.Video object, used only for  vipy.video.Scene ",
"func":1
},
{
"ref":"vipy.video.Video.bytes",
"url":32,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.video.Video.frames",
"url":32,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.video.Video.framelist",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.commandline",
"url":32,
"doc":"Return the equivalent ffmpeg command line string that will be used to transcode the video. This is useful for introspecting the complex filter chain that will be used to process the video. You can try to run this command line yourself for debugging purposes, by replacing 'dummyfile' with an appropriately named output file.",
"func":1
},
{
"ref":"vipy.video.Video.probeshape",
"url":32,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe  warning this does not take into account any applied ffmpeg filters. The shape will be the (height, width) of the underlying video file.",
"func":1
},
{
"ref":"vipy.video.Video.duration_in_seconds_of_videofile",
"url":32,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.video.Video.duration_in_frames_of_videofile",
"url":32,
"doc":"Return video duration of the source video file (NOT the filter chain) in frames, requires ffprobe.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.video.Video.framerate_of_videofile",
"url":32,
"doc":"Return video framerate in frames per second of the source video file (NOT the filter chain), requires ffprobe.",
"func":1
},
{
"ref":"vipy.video.Video.probe",
"url":32,
"doc":"Run ffprobe on the filename and return the result as a dictionary",
"func":1
},
{
"ref":"vipy.video.Video.print",
"url":32,
"doc":"Print the representation of the video This is useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing. Args: prefix: prepend a string prefix to the video __repr__ when printing. Useful for logging. verbose: Print out the video __repr__. Set verbose=False to just sleep sleep: Integer number of seconds to sleep[ before returning Returns: The video object after sleeping",
"func":1
},
{
"ref":"vipy.video.Video.dict",
"url":32,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding.",
"func":1
},
{
"ref":"vipy.video.Video.json",
"url":32,
"doc":"Return a json representation of the video. Args: encode: If true, return a JSON encoded string using json.dumps Returns: A JSON encoded string if encode=True, else returns a dictionary object  note If the video is loaded, then the JSON will not include the pixels. Try using  vipy.video.Video.store to serialize videos, or call  vipy.video.Video.flush first.",
"func":1
},
{
"ref":"vipy.video.Video.take",
"url":32,
"doc":"Return n frames from the clip uniformly spaced as numpy array Args: n: Integer number of uniformly spaced frames to return Returns: A numpy array of shape (n,W,H)  warning This assumes that the entire video is loaded into memory (e.g. call  vipy.video.Video.load ). Use with caution.",
"func":1
},
{
"ref":"vipy.video.Video.framerate",
"url":32,
"doc":"Change the input framerate for the video and update frame indexes for all annotations Args: fps: Float frames per second to process the underlying video Returns: If fps is None, return the current framerate, otherwise set the framerate to fps",
"func":1
},
{
"ref":"vipy.video.Video.colorspace",
"url":32,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.video.Video.nourl",
"url":32,
"doc":"Remove the  vipy.video.Video.url from the video",
"func":1
},
{
"ref":"vipy.video.Video.url",
"url":32,
"doc":"Video URL and URL download properties",
"func":1
},
{
"ref":"vipy.video.Video.isloaded",
"url":32,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.video.Video.isloadable",
"url":32,
"doc":"Return True if the video can be loaded successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version. Args: flush: [bool] If true, flush the video after it loads. This will clear the video pixel buffer Returns: True if load() can be called without FFMPEG exception. If flush=False, then self will contain the loaded video, which is helpful to avoid load() twice in some conditions  warning This requires loading and flushing the video. This is an expensive operation when performed on many videos and may result in out of memory conditions with long videos. Use with caution! Try  vipy.video.Video.canload to test if a single frame can be loaded as a less expensive alternative.",
"func":1
},
{
"ref":"vipy.video.Video.canload",
"url":32,
"doc":"Return True if the video can be previewed at frame=k successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version.  notes This will only try to preview a single frame. This will not check if the entire video is loadable. Use  vipy.video.Video.isloadable in this case",
"func":1
},
{
"ref":"vipy.video.Video.iscolor",
"url":32,
"doc":"Is the video a three channel color video as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.video.Video.isgrayscale",
"url":32,
"doc":"Is the video a single channel as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.video.Video.hasfilename",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist?",
"func":1
},
{
"ref":"vipy.video.Video.isdownloaded",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist, meaning that the url has been downloaded to a local file?",
"func":1
},
{
"ref":"vipy.video.Video.hasurl",
"url":32,
"doc":"Is the url returned from  vipy.video.Video.url a well formed url?",
"func":1
},
{
"ref":"vipy.video.Video.array",
"url":32,
"doc":"Set or return the video buffer as a numpy array. Args: array: [np.array] A numpy array of size NxHxWxC = (frames, height, width, channels) of type uint8 or float32. copy: [bool] If true, copy the buffer by value instaed of by reference. Copied buffers do not share pixels. Returns: if array=None, return a reference to the pixel buffer as a numpy array, otherwise return the video object.",
"func":1
},
{
"ref":"vipy.video.Video.fromarray",
"url":32,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.video.Video.fromframes",
"url":32,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.video.Video.tonumpy",
"url":32,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.video.Video.numpy",
"url":32,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.video.Video.zeros",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.reload",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.nofilename",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.filename",
"url":32,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.video.Video.abspath",
"url":32,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.video.Video.relpath",
"url":32,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.video.Video.rename",
"url":32,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.video.Video.filesize",
"url":32,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.video.Video.download",
"url":32,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.video.Video.fetch",
"url":32,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.video.Video.shape",
"url":32,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.video.Video.channels",
"url":32,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.video.Video.width",
"url":32,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Video.height",
"url":32,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Video.aspect_ratio",
"url":32,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.video.Video.preview",
"url":32,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.video.Video.thumbnail",
"url":32,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile. This is functionally equivalent to  vipy.video.Video.frame with an additional outfile argument to easily save an annotated thumbnail image. Args: outfile: [str] an optional outfile to save the annotated frame frame: [int >= 0] The frame to output the thumbnail Returns: A  vipy.image.Image object for frame k.",
"func":1
},
{
"ref":"vipy.video.Video.load",
"url":32,
"doc":"Load a video using ffmpeg, applying the requested filter chain. Args: verbose: [bool] if True. then ffmpeg console output will be displayed. ignoreErrors: [bool] if True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. shape: [tuple (height, width, channels)] If provided, use this shape for reading and reshaping the byte stream from ffmpeg. This is useful for efficient loading in some scenarios. Knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size Returns: this video object, with the pixels loaded in self.array()  warning Loading long videos can result in out of memory conditions. Try to call clip() first to extract a video segment to load().",
"func":1
},
{
"ref":"vipy.video.Video.speed",
"url":32,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.video.Video.clip",
"url":32,
"doc":"Load a video clip betweeen start and end frames",
"func":1
},
{
"ref":"vipy.video.Video.cliprange",
"url":32,
"doc":"Return the planned clip (startframe, endframe) range. This is useful for introspection of the planned clip() before load(), such as for data augmentation purposes without triggering a load. Returns: (startframe, endframe) of the video() such that after load(), the pixel buffer will contain frame=0 equivalent to startframe in the source video, and frame=endframe-startframe-1 equivalent to endframe in the source video. (0, None) If a video does not have a clip() (e.g. clip() was never called, the filter chain does not include a 'trim')  notes The endframe can be retrieved (inefficiently) using: >>> int(round(self.duration_in_frames_of_videofile()  (self.framerate() / self.framerate_of_videofile(  ",
"func":1
},
{
"ref":"vipy.video.Video.rot90cw",
"url":32,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.rot90ccw",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.fliplr",
"url":32,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.video.Video.flipud",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.rescale",
"url":32,
"doc":"Rescale the video by factor s, such that the new dimensions are (s H, s W), can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.resize",
"url":32,
"doc":"Resize the video to be (rows=height, cols=width)",
"func":1
},
{
"ref":"vipy.video.Video.mindim",
"url":32,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Video.maxdim",
"url":32,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Video.randomcrop",
"url":32,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Video.centercrop",
"url":32,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Video.centersquare",
"url":32,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.video.Video.cropeven",
"url":32,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.video.Video.maxsquare",
"url":32,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.video.Video.maxmatte",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.zeropad",
"url":32,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after  notes Older FFMPEG implementations can throw the error \"Input area  : : : not within the padded area  : : : or zero-sized, this is often caused by odd sized padding. Recommend calling self.cropeven().zeropad( .) to avoid this",
"func":1
},
{
"ref":"vipy.video.Video.pad",
"url":32,
"doc":"Alias for zeropad",
"func":1
},
{
"ref":"vipy.video.Video.crop",
"url":32,
"doc":"Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().",
"func":1
},
{
"ref":"vipy.video.Video.pkl",
"url":32,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Video.pklif",
"url":32,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Video.webp",
"url":32,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. Args: strict: If true, assert that the filename must have an .webp extension pause: Integer seconds to pause between loops of the animation smallest: if true, create the smallest possible file but takes much longer to run smaller: If true, create a smaller file, which takes a little longer to run Returns: The filename of the webp file for this video  warning This may be slow for very long or large videos",
"func":1
},
{
"ref":"vipy.video.Video.gif",
"url":32,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. Args: pause: Integer seconds to pause between loops of the animation smallest: If true, create the smallest possible file but takes much longer to run smaller: if trye, create a smaller file, which takes a little longer to run Returns: The filename of the animated GIF of this video  warning This will be very large for big videos, consider using  vipy.video.Video.webp instead.",
"func":1
},
{
"ref":"vipy.video.Video.saveas",
"url":32,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file. Args: outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image) ignoreErrors: if True, then exit gracefully without throwing an exception. Useful for chaining download().saveas() on parallel dataset downloads flush: If true, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video pause: an integer in seconds to pause between loops of animated images if the outfile is webp or animated gif Returns: a new video object with this video filename, and a clean video filter chain  note - If self.array() is loaded, then export the contents of self._array to the video file - If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video - If outfile None or outfile self.filename(), then overwrite the current filename",
"func":1
},
{
"ref":"vipy.video.Video.savetmp",
"url":32,
"doc":"Call  vipy.video.Video.saveas using a new temporary video file, and return the video object with this new filename",
"func":1
},
{
"ref":"vipy.video.Video.savetemp",
"url":32,
"doc":"Alias for  vipy.video.Video.savetmp ",
"func":1
},
{
"ref":"vipy.video.Video.ffplay",
"url":32,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.video.Video.play",
"url":32,
"doc":"Play the saved video filename in self.filename() If there is no filename, try to download it. If the filter chain is dirty or the pixels are loaded, dump to temp video file first then play it. This uses 'ffplay' on the PATH if available, otherwise uses a fallback player by showing a sequence of matplotlib frames. If the output of the ffmpeg filter chain has modified this video, then this will be saved to a temporary video file. To play the original video (indepenedent of the filter chain of this video), use  vipy.video.Video.ffplay . Args: verbose: If true, show more verbose output notebook: If true, play in a jupyter notebook Returns: The unmodified video object",
"func":1
},
{
"ref":"vipy.video.Video.show",
"url":32,
"doc":"Alias for play",
"func":1
},
{
"ref":"vipy.video.Video.quicklook",
"url":32,
"doc":"Generate a montage of n uniformly spaced frames. Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. Input: -n: Number of images in the quicklook -mindim: The minimum dimension of each of the elements in the montage -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.video.Video.torch",
"url":32,
"doc":"Convert the loaded video of shape NxHxWxC frames to an MxCxHxW torch tensor/ Args: startframe: [int >= 0] The start frame of the loaded video to use for constructig the torch tensor endframe: [int >= 0] The end frame of the loaded video to use for constructing the torch tensor length: [int >= 0] The length of the torch tensor if endframe is not provided. stride: [int >= 1] The temporal stride in frames. This is the number of frames to skip. take: [int >= 0] The number of uniformly spaced frames to include in the tensor. boundary: ['repeat', 'cyclic'] The boundary handling for when the requested tensor slice goes beyond the end of the video order: ['nchw', 'nhwc', 'chwn', 'cnhw'] The axis ordering of the returned torch tensor N=number of frames (batchsize), C=channels, H=height, W=width verbose [bool]: Print out the slice used for contructing tensor withslice: [bool] Return a tuple (tensor, slice) that includes the slice used to construct the tensor. Useful for data provenance. scale: [float] An optional scale factor to apply to the tensor. Useful for converting [0,255] -> [0,1] withlabel: [bool] Return a tuple (tensor, labels) that includes the N framewise activity labels. nonelabel: [bool] returns tuple (t, None) if withlabel=False Returns Returns torch float tensor, analogous to torchvision.transforms.ToTensor() Return (tensor, slice) if withslice=True (withslice takes precedence) Returns (tensor, labellist) if withlabel=True  notes - This triggers a load() of the video - The precedence of arguments is (startframe, endframe) or (startframe, startframe+length), then stride and take. - Follows numpy slicing rules. Optionally return the slice used if withslice=True",
"func":1
},
{
"ref":"vipy.video.Video.clone",
"url":32,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected. Args: flushforward: copy the object, and set the cloned object  vipy.video.Video.array to None. This flushes the video buffer for the clone, not the object flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone. flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object. flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos flushfile: Remove the filename and the URL from the video object. Useful for creating new video objects from loaded pixels. rekey: Generate new unique track ID and activity ID keys for this scene shallow: shallow copy everything (copy by reference), except for ffmpeg object. attributes dictionary is shallow copied sharedarray: deep copy of everything, except for pixel buffer which is shared. Changing the pixel buffer on self is reflected in the clone. Returns: A deepcopy of the video object such that changes to self are not reflected in the copy  note Cloning videos is an expensive operation and can slow down real time code. Use sparingly.",
"func":1
},
{
"ref":"vipy.video.Video.flush",
"url":32,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.video.Video.flush_and_return",
"url":32,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.video.Video.map",
"url":32,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.video.Video.gain",
"url":32,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.video.Video.bias",
"url":32,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.video.Video.float",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.channel",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.normalize",
"url":32,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.video.Video.setattribute",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.hasattribute",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.delattribute",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.getattribute",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.VideoCategory",
"url":32,
"doc":"vipy.video.VideoCategory class A VideoCategory is a video with associated category, such as an activity class. This class includes all of the constructors of vipy.video.Video along with the ability to extract a clip based on frames or seconds."
},
{
"ref":"vipy.video.VideoCategory.from_json",
"url":32,
"doc":"Import a json string as a  vipy.video.Video object. This will perform a round trip from a video to json and back to a video object. This same operation is used for serialization of all vipy objects to JSON for storage. >>> v = vipy.video.Video.from_json(vipy.video.RandomVideo().json( ",
"func":1
},
{
"ref":"vipy.video.VideoCategory.json",
"url":32,
"doc":"Return a json representation of the video. Args: encode: If true, return a JSON encoded string using json.dumps Returns: A JSON encoded string if encode=True, else returns a dictionary object  note If the video is loaded, then the JSON will not include the pixels. Try using  vipy.video.Video.store to serialize videos, or call  vipy.video.Video.flush first.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.category",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.VideoCategory.cast",
"url":32,
"doc":"Cast a conformal video object to a  vipy.video.Video object. This is useful for downcasting superclasses. >>> vs = vipy.video.RandomScene() >>> v = vipy.video.Video.cast(vs)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.metadata",
"url":32,
"doc":"Return a dictionary of metadata about this video. This is an alias for the 'attributes' dictionary.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.videoid",
"url":32,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by SHA1 hash of the  vipy.video.Video.filename and  vipy.video.Video.url . Args: newid: [str] If not None, then update the video_id as newid. Returns: The video ID if newid=None else self  note - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', ${MY_ID}), or pass in newid",
"func":1
},
{
"ref":"vipy.video.VideoCategory.frame",
"url":32,
"doc":"Return the kth frame as an  vipy.image Image object",
"func":1
},
{
"ref":"vipy.video.VideoCategory.store",
"url":32,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. >>> v  v.store().restore(v.filename(  note -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.unstore",
"url":32,
"doc":"Delete the currently stored video from  vipy.video.Video.store",
"func":1
},
{
"ref":"vipy.video.VideoCategory.restore",
"url":32,
"doc":"Save the currently stored video as set using  vipy.video.Video.store to filename, and set up filename",
"func":1
},
{
"ref":"vipy.video.VideoCategory.stream",
"url":32,
"doc":"Iterator to yield groups of frames streaming from video. A video stream is a real time iterator to read or write from a video. Streams are useful to group together frames into clips that are operated on as a group. The following use cases are supported: >>> v = vipy.video.RandomScene() Stream individual video frames lagged by 10 frames and 20 frames >>> for (im1, im2) in zip(v.stream().frame(n=-10), v.stream().frame(n=-20 : >>> print(im1, im2) Stream overlapping clips such that each clip is a video n=16 frames long and starts at frame i, and the next clip is n=16 frames long and starts at frame i=i+m >>> for vc in v.stream().clip(n=16, m=4): >>> print(vc) Stream non-overlapping batches of frames such that each clip is a video of length n and starts at frame i, and the next clip is length n and starts at frame i+n >>> for vb in v.stream().batch(n=16): >>> print(vb) Create a write stream to incrementally add frames to long video. >>> vi = vipy.video.Video(filename='/path/to/output.mp4') >>> vo = vipy.video.Video(filename='/path/to/input.mp4') >>> with vo.stream(write=True) as s: >>> for im in vi.stream(): >>> s.write(im)  manipulate pixels of im, if desired Args: write: [bool] If true, create a write stream overwrite: [bool] If true, and the video output filename already exists, overwrite it bufsize: [int] The maximum queue size for the pipe thread. Returns: A  vipy.video.Video.stream.Stream object  note Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.clear",
"url":32,
"doc":"no-op for  vipy.video.Video object, used only for  vipy.video.Scene ",
"func":1
},
{
"ref":"vipy.video.VideoCategory.bytes",
"url":32,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.frames",
"url":32,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.commandline",
"url":32,
"doc":"Return the equivalent ffmpeg command line string that will be used to transcode the video. This is useful for introspecting the complex filter chain that will be used to process the video. You can try to run this command line yourself for debugging purposes, by replacing 'dummyfile' with an appropriately named output file.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.probeshape",
"url":32,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe  warning this does not take into account any applied ffmpeg filters. The shape will be the (height, width) of the underlying video file.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.duration_in_seconds_of_videofile",
"url":32,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.duration_in_frames_of_videofile",
"url":32,
"doc":"Return video duration of the source video file (NOT the filter chain) in frames, requires ffprobe.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.framerate_of_videofile",
"url":32,
"doc":"Return video framerate in frames per second of the source video file (NOT the filter chain), requires ffprobe.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.probe",
"url":32,
"doc":"Run ffprobe on the filename and return the result as a dictionary",
"func":1
},
{
"ref":"vipy.video.VideoCategory.print",
"url":32,
"doc":"Print the representation of the video This is useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing. Args: prefix: prepend a string prefix to the video __repr__ when printing. Useful for logging. verbose: Print out the video __repr__. Set verbose=False to just sleep sleep: Integer number of seconds to sleep[ before returning Returns: The video object after sleeping",
"func":1
},
{
"ref":"vipy.video.VideoCategory.dict",
"url":32,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.take",
"url":32,
"doc":"Return n frames from the clip uniformly spaced as numpy array Args: n: Integer number of uniformly spaced frames to return Returns: A numpy array of shape (n,W,H)  warning This assumes that the entire video is loaded into memory (e.g. call  vipy.video.Video.load ). Use with caution.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.framerate",
"url":32,
"doc":"Change the input framerate for the video and update frame indexes for all annotations Args: fps: Float frames per second to process the underlying video Returns: If fps is None, return the current framerate, otherwise set the framerate to fps",
"func":1
},
{
"ref":"vipy.video.VideoCategory.colorspace",
"url":32,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.video.VideoCategory.nourl",
"url":32,
"doc":"Remove the  vipy.video.Video.url from the video",
"func":1
},
{
"ref":"vipy.video.VideoCategory.url",
"url":32,
"doc":"Video URL and URL download properties",
"func":1
},
{
"ref":"vipy.video.VideoCategory.isloaded",
"url":32,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.video.VideoCategory.isloadable",
"url":32,
"doc":"Return True if the video can be loaded successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version. Args: flush: [bool] If true, flush the video after it loads. This will clear the video pixel buffer Returns: True if load() can be called without FFMPEG exception. If flush=False, then self will contain the loaded video, which is helpful to avoid load() twice in some conditions  warning This requires loading and flushing the video. This is an expensive operation when performed on many videos and may result in out of memory conditions with long videos. Use with caution! Try  vipy.video.Video.canload to test if a single frame can be loaded as a less expensive alternative.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.canload",
"url":32,
"doc":"Return True if the video can be previewed at frame=k successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version.  notes This will only try to preview a single frame. This will not check if the entire video is loadable. Use  vipy.video.Video.isloadable in this case",
"func":1
},
{
"ref":"vipy.video.VideoCategory.iscolor",
"url":32,
"doc":"Is the video a three channel color video as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.video.VideoCategory.isgrayscale",
"url":32,
"doc":"Is the video a single channel as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.video.VideoCategory.hasfilename",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist?",
"func":1
},
{
"ref":"vipy.video.VideoCategory.isdownloaded",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist, meaning that the url has been downloaded to a local file?",
"func":1
},
{
"ref":"vipy.video.VideoCategory.hasurl",
"url":32,
"doc":"Is the url returned from  vipy.video.Video.url a well formed url?",
"func":1
},
{
"ref":"vipy.video.VideoCategory.array",
"url":32,
"doc":"Set or return the video buffer as a numpy array. Args: array: [np.array] A numpy array of size NxHxWxC = (frames, height, width, channels) of type uint8 or float32. copy: [bool] If true, copy the buffer by value instaed of by reference. Copied buffers do not share pixels. Returns: if array=None, return a reference to the pixel buffer as a numpy array, otherwise return the video object.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fromarray",
"url":32,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fromframes",
"url":32,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.video.VideoCategory.tonumpy",
"url":32,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.numpy",
"url":32,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.video.VideoCategory.filename",
"url":32,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.abspath",
"url":32,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.relpath",
"url":32,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rename",
"url":32,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.filesize",
"url":32,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.video.VideoCategory.download",
"url":32,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fetch",
"url":32,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.video.VideoCategory.shape",
"url":32,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.video.VideoCategory.channels",
"url":32,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.video.VideoCategory.width",
"url":32,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.VideoCategory.height",
"url":32,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.VideoCategory.aspect_ratio",
"url":32,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.video.VideoCategory.preview",
"url":32,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.video.VideoCategory.thumbnail",
"url":32,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile. This is functionally equivalent to  vipy.video.Video.frame with an additional outfile argument to easily save an annotated thumbnail image. Args: outfile: [str] an optional outfile to save the annotated frame frame: [int >= 0] The frame to output the thumbnail Returns: A  vipy.image.Image object for frame k.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.load",
"url":32,
"doc":"Load a video using ffmpeg, applying the requested filter chain. Args: verbose: [bool] if True. then ffmpeg console output will be displayed. ignoreErrors: [bool] if True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. shape: [tuple (height, width, channels)] If provided, use this shape for reading and reshaping the byte stream from ffmpeg. This is useful for efficient loading in some scenarios. Knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size Returns: this video object, with the pixels loaded in self.array()  warning Loading long videos can result in out of memory conditions. Try to call clip() first to extract a video segment to load().",
"func":1
},
{
"ref":"vipy.video.VideoCategory.speed",
"url":32,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.clip",
"url":32,
"doc":"Load a video clip betweeen start and end frames",
"func":1
},
{
"ref":"vipy.video.VideoCategory.cliprange",
"url":32,
"doc":"Return the planned clip (startframe, endframe) range. This is useful for introspection of the planned clip() before load(), such as for data augmentation purposes without triggering a load. Returns: (startframe, endframe) of the video() such that after load(), the pixel buffer will contain frame=0 equivalent to startframe in the source video, and frame=endframe-startframe-1 equivalent to endframe in the source video. (0, None) If a video does not have a clip() (e.g. clip() was never called, the filter chain does not include a 'trim')  notes The endframe can be retrieved (inefficiently) using: >>> int(round(self.duration_in_frames_of_videofile()  (self.framerate() / self.framerate_of_videofile(  ",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rot90cw",
"url":32,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rot90ccw",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fliplr",
"url":32,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.video.VideoCategory.flipud",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rescale",
"url":32,
"doc":"Rescale the video by factor s, such that the new dimensions are (s H, s W), can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.resize",
"url":32,
"doc":"Resize the video to be (rows=height, cols=width)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.mindim",
"url":32,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.VideoCategory.maxdim",
"url":32,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.VideoCategory.randomcrop",
"url":32,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.VideoCategory.centercrop",
"url":32,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.VideoCategory.centersquare",
"url":32,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.video.VideoCategory.cropeven",
"url":32,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.maxsquare",
"url":32,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.video.VideoCategory.zeropad",
"url":32,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after  notes Older FFMPEG implementations can throw the error \"Input area  : : : not within the padded area  : : : or zero-sized, this is often caused by odd sized padding. Recommend calling self.cropeven().zeropad( .) to avoid this",
"func":1
},
{
"ref":"vipy.video.VideoCategory.pad",
"url":32,
"doc":"Alias for zeropad",
"func":1
},
{
"ref":"vipy.video.VideoCategory.crop",
"url":32,
"doc":"Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().",
"func":1
},
{
"ref":"vipy.video.VideoCategory.pkl",
"url":32,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.VideoCategory.pklif",
"url":32,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.VideoCategory.webp",
"url":32,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. Args: strict: If true, assert that the filename must have an .webp extension pause: Integer seconds to pause between loops of the animation smallest: if true, create the smallest possible file but takes much longer to run smaller: If true, create a smaller file, which takes a little longer to run Returns: The filename of the webp file for this video  warning This may be slow for very long or large videos",
"func":1
},
{
"ref":"vipy.video.VideoCategory.gif",
"url":32,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. Args: pause: Integer seconds to pause between loops of the animation smallest: If true, create the smallest possible file but takes much longer to run smaller: if trye, create a smaller file, which takes a little longer to run Returns: The filename of the animated GIF of this video  warning This will be very large for big videos, consider using  vipy.video.Video.webp instead.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.saveas",
"url":32,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file. Args: outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image) ignoreErrors: if True, then exit gracefully without throwing an exception. Useful for chaining download().saveas() on parallel dataset downloads flush: If true, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video pause: an integer in seconds to pause between loops of animated images if the outfile is webp or animated gif Returns: a new video object with this video filename, and a clean video filter chain  note - If self.array() is loaded, then export the contents of self._array to the video file - If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video - If outfile None or outfile self.filename(), then overwrite the current filename",
"func":1
},
{
"ref":"vipy.video.VideoCategory.savetmp",
"url":32,
"doc":"Call  vipy.video.Video.saveas using a new temporary video file, and return the video object with this new filename",
"func":1
},
{
"ref":"vipy.video.VideoCategory.savetemp",
"url":32,
"doc":"Alias for  vipy.video.Video.savetmp ",
"func":1
},
{
"ref":"vipy.video.VideoCategory.ffplay",
"url":32,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.video.VideoCategory.play",
"url":32,
"doc":"Play the saved video filename in self.filename() If there is no filename, try to download it. If the filter chain is dirty or the pixels are loaded, dump to temp video file first then play it. This uses 'ffplay' on the PATH if available, otherwise uses a fallback player by showing a sequence of matplotlib frames. If the output of the ffmpeg filter chain has modified this video, then this will be saved to a temporary video file. To play the original video (indepenedent of the filter chain of this video), use  vipy.video.Video.ffplay . Args: verbose: If true, show more verbose output notebook: If true, play in a jupyter notebook Returns: The unmodified video object",
"func":1
},
{
"ref":"vipy.video.VideoCategory.show",
"url":32,
"doc":"Alias for play",
"func":1
},
{
"ref":"vipy.video.VideoCategory.quicklook",
"url":32,
"doc":"Generate a montage of n uniformly spaced frames. Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. Input: -n: Number of images in the quicklook -mindim: The minimum dimension of each of the elements in the montage -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.video.VideoCategory.torch",
"url":32,
"doc":"Convert the loaded video of shape NxHxWxC frames to an MxCxHxW torch tensor/ Args: startframe: [int >= 0] The start frame of the loaded video to use for constructig the torch tensor endframe: [int >= 0] The end frame of the loaded video to use for constructing the torch tensor length: [int >= 0] The length of the torch tensor if endframe is not provided. stride: [int >= 1] The temporal stride in frames. This is the number of frames to skip. take: [int >= 0] The number of uniformly spaced frames to include in the tensor. boundary: ['repeat', 'cyclic'] The boundary handling for when the requested tensor slice goes beyond the end of the video order: ['nchw', 'nhwc', 'chwn', 'cnhw'] The axis ordering of the returned torch tensor N=number of frames (batchsize), C=channels, H=height, W=width verbose [bool]: Print out the slice used for contructing tensor withslice: [bool] Return a tuple (tensor, slice) that includes the slice used to construct the tensor. Useful for data provenance. scale: [float] An optional scale factor to apply to the tensor. Useful for converting [0,255] -> [0,1] withlabel: [bool] Return a tuple (tensor, labels) that includes the N framewise activity labels. nonelabel: [bool] returns tuple (t, None) if withlabel=False Returns Returns torch float tensor, analogous to torchvision.transforms.ToTensor() Return (tensor, slice) if withslice=True (withslice takes precedence) Returns (tensor, labellist) if withlabel=True  notes - This triggers a load() of the video - The precedence of arguments is (startframe, endframe) or (startframe, startframe+length), then stride and take. - Follows numpy slicing rules. Optionally return the slice used if withslice=True",
"func":1
},
{
"ref":"vipy.video.VideoCategory.clone",
"url":32,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected. Args: flushforward: copy the object, and set the cloned object  vipy.video.Video.array to None. This flushes the video buffer for the clone, not the object flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone. flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object. flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos flushfile: Remove the filename and the URL from the video object. Useful for creating new video objects from loaded pixels. rekey: Generate new unique track ID and activity ID keys for this scene shallow: shallow copy everything (copy by reference), except for ffmpeg object. attributes dictionary is shallow copied sharedarray: deep copy of everything, except for pixel buffer which is shared. Changing the pixel buffer on self is reflected in the clone. Returns: A deepcopy of the video object such that changes to self are not reflected in the copy  note Cloning videos is an expensive operation and can slow down real time code. Use sparingly.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.flush",
"url":32,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.video.VideoCategory.flush_and_return",
"url":32,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.video.VideoCategory.map",
"url":32,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.gain",
"url":32,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.video.VideoCategory.bias",
"url":32,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.video.VideoCategory.normalize",
"url":32,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.video.Scene",
"url":32,
"doc":"vipy.video.Scene class The vipy.video.Scene class provides a fluent, lazy interface for representing, transforming and visualizing annotated videos. The following constructors are supported: >>> vid = vipy.video.Scene(filename='/path/to/video.ext') Valid video extensions are those that are supported by ffmpeg ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']. >>> vid = vipy.video.Scene(url='https: www.youtube.com/watch?v=MrIN959JuV8') >>> vid = vipy.video.Scene(url='http: path/to/video.ext', filename='/path/to/video.ext') Youtube URLs are downloaded to a temporary filename, retrievable as vid.download().filename(). If the environment variable 'VIPY_CACHE' is defined, then videos are saved to this directory rather than the system temporary directory. If a filename is provided to the constructor, then that filename will be used instead of a temp or cached filename. URLs can be defined as an absolute URL to a video file, or to a site supported by 'youtube-dl' [https: ytdl-org.github.io/youtube-dl/supportedsites.html] >>> vid = vipy.video.Scene(array=frames, colorspace='rgb') The input 'frames' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video Note that the video transformations (clip, resize, rescale, rotate) are only available prior to load(), and the array() is assumed immutable after load(). >>> vid = vipy.video.Scene(array=greyframes, colorspace='lum') The input 'greyframes' is an NxHxWx1 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video This corresponds to the luminance of an RGB colorspace >>> vid = vipy.video.Scene(array=greyframes, colorspace='lum', tracks=tracks, activities=activities)  tracks = [vipy.object.Track(),  .]  activities = [vipy.object.Activity(),  .] The inputs are lists of tracks and/or activities. An object is a spatial bounding box with a category label. A track is a spatiotemporal bounding box with a category label, such that the box contains the same instance of an object. An activity is one or more tracks with a start and end frame for an activity performed by the object instances. Track and activity timing must be relative to the start frame of the Scene() constructor."
},
{
"ref":"vipy.video.Scene.cast",
"url":32,
"doc":"Cast a conformal vipy object to this class. This is useful for downcast and upcast conversion of video objects.",
"func":1
},
{
"ref":"vipy.video.Scene.from_json",
"url":32,
"doc":"Restore an object serialized with self.json() Usage: >>> vs = vipy.video.Scene.from_json(v.json( ",
"func":1
},
{
"ref":"vipy.video.Scene.pack",
"url":32,
"doc":"Packing a scene returns the scene with the annotations JSON serialized. - This is useful for fast garbage collection when there are many objects in memory - This is useful for distributed processing prior to serializing from a scheduler to a client - This is useful for lazy deserialization of complex attributes when loading many videos into memory - Unpacking is transparent to the end user and is performed on the fly when annotations are accessed. There is no unpack() method. - See the notes in from_json() for why this helps with nested containers and reference cycle tracking with the python garbage collector",
"func":1
},
{
"ref":"vipy.video.Scene.instanceid",
"url":32,
"doc":"Return an annotation instance identifier for this video. An instance ID is a unique identifier for a ground truth annotation within a video, either a track or an activity. More than one instance ID may share the same video ID if they are from the same source videofile. This is useful when calling  vipy.video.Scene.activityclip or  vipy.video.Scene.activitysplit to clip a video into segments such that each clip has a unique identifier, but all share the same underlying  vipy.video.Video.videoid . This is useful when calling  vipy.video.Scene.trackclip or  vipy.video.Scene.tracksplit to clip a video into segments such that each clip has a unique identifier, but all share the same underlying  vipy.video.Video.videoid . Returns: INSTANCEID: if 'instance_id' key is in self.attribute VIDEOID_INSTANCEID: if '_instance_id' key is in self.attribute, as set by activityclip() or trackclip(). This is set using INSTANCE_ID=ACTIVITYID_ACTIVITYINDEX or INSTANCEID=TRACKID_TRACKINDEX, where the index is the temporal order of the annotation in the source video prior to clip(). VIDEOID_ACTIVITYINDEX: if 'activityindex' key is in self.attribute, as set by activityclip(). (fallback for legacy datasets). VIDEOID: otherwise",
"func":1
},
{
"ref":"vipy.video.Scene.frame",
"url":32,
"doc":"Return  vipy.image.Scene object at frame k -The attributes of each of the  vipy.image.Scene.objects in the scene contains helpful metadata for the provenance of the detection, including: - 'trackid' of the track this detection - 'activityid' associated with this detection - 'jointlabel' of this detection, used for visualization - 'noun verb' of this detection, used for visualization Args: k: [int >=- 0] The frame index requested. This is relative to the current frame rate of the video. img: [numpy] An optional image to be used for this frame. This is useful to construct frames efficiently for videos if the pixel buffer is available from a stream rather than a preview. Return: A  vipy.image.Scene object for frame k containing all objects in this frame  notes -Modifying this frame will not affect the source video",
"func":1
},
{
"ref":"vipy.video.Scene.during",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.labeled_frames",
"url":32,
"doc":"Iterate over frames, yielding tuples (activity+object labelset in scene, vipy.image.Scene( ",
"func":1
},
{
"ref":"vipy.video.Scene.framecomposite",
"url":32,
"doc":"Generate a single composite image with minimum dimension mindim as the uniformly blended composite of n frames each separated by dt frames",
"func":1
},
{
"ref":"vipy.video.Scene.isdegenerate",
"url":32,
"doc":"Degenerate scene has empty or malformed tracks",
"func":1
},
{
"ref":"vipy.video.Scene.quicklook",
"url":32,
"doc":"Generate a montage of n uniformly spaced annotated frames centered on the union of the labeled boxes in the current frame to show the activity ocurring in this scene at a glance Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. This quicklook is most useful when len(self.activities() 1) for generating a quicklook from an activityclip(). Input: -n: Number of images in the quicklook -dilate: The dilation factor for the bounding box prior to crop for display -mindim: The minimum dimension of each of the elemnets in the montage -fontsize: The size of the font for the bounding box label -context: If true, replace the first and last frame in the montage with the full frame annotation, to help show the scale of the scene -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.video.Scene.tracks",
"url":32,
"doc":"Return mutable dictionary of tracks",
"func":1
},
{
"ref":"vipy.video.Scene.track",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.trackindex",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.trackidx",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activity",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.next_activity",
"url":32,
"doc":"Return the next activity just after the given activityid",
"func":1
},
{
"ref":"vipy.video.Scene.prev_activity",
"url":32,
"doc":"Return the previous activity just before the given activityid",
"func":1
},
{
"ref":"vipy.video.Scene.tracklist",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.actorid",
"url":32,
"doc":"Return or set the actor ID for the video. - The actor ID is the track ID of the primary actor in the scene. This is useful for assigning a role for activities that are performed by the actor. - The actor ID is the first track is in the tracklist Args: id: [str] if not None, then use this track ID as the actor fluent: [bool] If true, always return self. This is useful for those cases where the actorid being set is None. Returns: [id=None, fluent=False] the actor ID [id is not None] The video with the actor ID set.",
"func":1
},
{
"ref":"vipy.video.Scene.setactorid",
"url":32,
"doc":"Alias for  vipy.video.Scene.actorid ",
"func":1
},
{
"ref":"vipy.video.Scene.actor",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.primary_activity",
"url":32,
"doc":"Return the primary activity of the video. - The primary activity is the first activity in the activitylist. - This is useful for activityclip() videos that are centered on a single activity Returns:  vipy.activity.Activity that is first in the  vipy.video.Scene.activitylist ",
"func":1
},
{
"ref":"vipy.video.Scene.activities",
"url":32,
"doc":"Return mutable dictionary of activities. All temporal alignment is relative to the current clip().",
"func":1
},
{
"ref":"vipy.video.Scene.activityindex",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activitylist",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activityfilter",
"url":32,
"doc":"Apply boolean lambda function f to each activity and keep activity if function is true, remove activity if function is false Filter out all activities longer than 128 frames >>> vid = vid.activityfilter(lambda a: len(a) >> vid = vid.activityfilter(lambda a: a.category() in set(['category1', 'category2'] Args: f: [lambda] a lambda function that takes an activity and returns a boolean Returns: This video with the activities f(a) False removed.",
"func":1
},
{
"ref":"vipy.video.Scene.trackfilter",
"url":32,
"doc":"Apply lambda function f to each object and keep if filter is True. Args: activitytrack: [bool] If true, remove track assignment from activities also, may result in activities with no tracks f: [lambda] The lambda function to apply to each track t, and if f(t) returns True, then keep the track Returns: self, with tracks removed in-place  note Applying track filter with activitytrack=True may result in activities with no associated tracks. You should follow up with self.activityfilter(lambda a: len(a.trackids( > 0).",
"func":1
},
{
"ref":"vipy.video.Scene.trackmap",
"url":32,
"doc":"Apply lambda function f to each activity -strict=True: enforce that lambda function must return non-degenerate Track() objects",
"func":1
},
{
"ref":"vipy.video.Scene.activitymap",
"url":32,
"doc":"Apply lambda function f to each activity",
"func":1
},
{
"ref":"vipy.video.Scene.rekey",
"url":32,
"doc":"Change the track and activity IDs to randomly assigned UUIDs. Useful for cloning unique scenes",
"func":1
},
{
"ref":"vipy.video.Scene.label",
"url":32,
"doc":"Return an iterator over labels in each frame",
"func":1
},
{
"ref":"vipy.video.Scene.labels",
"url":32,
"doc":"Return a set of all object and activity labels in this scene, or at frame int(k)",
"func":1
},
{
"ref":"vipy.video.Scene.activitylabel",
"url":32,
"doc":"Return an iterator over activity labels in each frame, starting from startframe and ending when there are no more activities",
"func":1
},
{
"ref":"vipy.video.Scene.activitylabels",
"url":32,
"doc":"Return a set of all activity categories in this scene, or at startframe, or in range [startframe, endframe]",
"func":1
},
{
"ref":"vipy.video.Scene.objectlabels",
"url":32,
"doc":"Return a python set of all activity categories in this scene, or at frame k. Args: k: [int] The object labels present at frame k. If k=None, then all object labels in the video lower: [bool] If true, return the object labels in alll lower case for case invariant string comparisonsn",
"func":1
},
{
"ref":"vipy.video.Scene.categories",
"url":32,
"doc":"Alias for labels()",
"func":1
},
{
"ref":"vipy.video.Scene.activity_categories",
"url":32,
"doc":"Alias for activitylabels()",
"func":1
},
{
"ref":"vipy.video.Scene.hasactivities",
"url":32,
"doc":"Does this video have any activities?",
"func":1
},
{
"ref":"vipy.video.Scene.hastracks",
"url":32,
"doc":"Does this video have any tracks?",
"func":1
},
{
"ref":"vipy.video.Scene.hastrack",
"url":32,
"doc":"Does the video have this trackid?  note Track IDs are available as vipy.object.Track().id()",
"func":1
},
{
"ref":"vipy.video.Scene.add",
"url":32,
"doc":"Add the object obj to the scene, and return an index to this object for future updates This function is used to incrementally build up a scene frame by frame. Obj can be one of the following types: - obj = vipy.object.Detection(), this must be called from within a frame iterator (e.g. for im in video) to get the current frame index - obj = vipy.object.Track() - obj = vipy.activity.Activity() - obj = [xmin, ymin, width, height], with associated category kwarg, this must be called from within a frame iterator to get the current frame index It is recomended that the objects are added as follows. For a v=vipy.video.Scene(): >>> for im in v: >>>  Do some processing on frame im to detect objects >>> (object_labels, xywh) = object_detection(im) >>> >>>  Add them to the scene, note that each object instance is independent in each frame, use tracks for object correspondence >>> for (lbl,bb) in zip(object_labels, xywh): >>> v.add(bb, lbl) >>> >>>  Do some correspondences to track objects >>> t2 = v.add( vipy.object.Track( .) ) >>> >>>  Update a previous track to add a keyframe >>> v.track(t2).add(  . ) The frame iterator will keep track of the current frame in the video and add the objects in the appropriate place. Alternatively, >>> v.add(vipy.object.Track( ), frame=k) Args: obj: A conformal python object to add to the scene ( vipy.object.Detection ,  vipy.object.Track ,  vipy.activity.Activity , [xmin, ymin, width, height] category: Used if obj is an xywh tuple attributes: Used only if obj is an xywh tuple frame: [int] The frame to add the object rangecheck: [bool] If true, check if the object is within the image rectangle and throw an exception if not. This requires introspecting the video shape using  vipy.video.Video.shape . fluent: [bool] If true, return self instead of the object index",
"func":1
},
{
"ref":"vipy.video.Scene.delete",
"url":32,
"doc":"Delete a given track or activity by id, if present",
"func":1
},
{
"ref":"vipy.video.Scene.addframe",
"url":32,
"doc":"Add im=vipy.image.Scene() into vipy.video.Scene() at given frame. The input image must have been generated using im=self[k] for this to be meaningful, so that trackid can be associated",
"func":1
},
{
"ref":"vipy.video.Scene.clear",
"url":32,
"doc":"Remove all activities and tracks from this object",
"func":1
},
{
"ref":"vipy.video.Scene.cleartracks",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.clearactivities",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.replace",
"url":32,
"doc":"Replace tracks and activities with other if activity/track is during frame",
"func":1
},
{
"ref":"vipy.video.Scene.json",
"url":32,
"doc":"Return JSON encoded string of this object. This may fail if attributes contain non-json encodeable object",
"func":1
},
{
"ref":"vipy.video.Scene.csv",
"url":32,
"doc":"Export scene to CSV file format with header. If there are no tracks, this will be empty.",
"func":1
},
{
"ref":"vipy.video.Scene.framerate",
"url":32,
"doc":"Change the input framerate for the video and update frame indexes for all annotations. >>> fps = self.framerate() >>> self.framerate(fps=15.0)",
"func":1
},
{
"ref":"vipy.video.Scene.activitysplit",
"url":32,
"doc":"Split the scene into k separate scenes, one for each activity. Do not include overlapping activities. This is useful for union()",
"func":1
},
{
"ref":"vipy.video.Scene.tracksplit",
"url":32,
"doc":"Split the scene into k separate scenes, one for each track. Each scene starts at frame 0 and is a shallow copy of self containing exactly one track. - This is useful for visualization by breaking a scene into a list of scenes that contain only one track. - The attribute '_trackindex' is set in the attributes dictionary to provide provenance for the track relative to the source video  notes Use clone() to create a deep copy if needed.",
"func":1
},
{
"ref":"vipy.video.Scene.trackclip",
"url":32,
"doc":"Split the scene into k separate scenes, one for each track. Each scene starts and ends when the track starts and ends",
"func":1
},
{
"ref":"vipy.video.Scene.activityclip",
"url":32,
"doc":"Return a list of  vipy.video.Scene objects each clipped to be temporally centered on a single activity, with an optional padframes before and after. Args: padframes: [int] for symmetric padding same before and after padframes: [tuple] (int, int) for asymmetric padding before and after padframes: [list[tuples [(int, int),  .] for activity specific asymmetric padding multilabel: [bool] include overlapping multilabel secondary activities in each activityclip Returns: A list of  vipy.video.Scene each cloned from the source video and clipped on one activity in the scene  notes - The Scene() category is updated to be the activity category of the clip, and only the objects participating in the activity are included. - Clips are returned ordered in the temporal order they appear in the video. - The returned vipy.video.Scene() objects for each activityclip are clones of the video, with the video buffer flushed. - Each activityclip() is associated with each activity in the scene, and includes all other secondary activities that the objects in the primary activity also perform (if multilabel=True). See activityclip().labels(). - Calling activityclip() on activityclip(multilabel=True) will duplicate activities, due to the overlapping secondary activities being included in each clip with an overlap. Be careful!",
"func":1
},
{
"ref":"vipy.video.Scene.noactivityclip",
"url":32,
"doc":"Return a list of vipy.video.Scene() each clipped on a track segment that has no associated activities. Args: strict: [bool] True means that background can only occur in frames where no tracks are performing any activities. This is useful so that background is not constructed from secondary objects. False means that background can only occur in frames where a given track is not performing any activities. label: [str] The activity label to give the background activities. Defaults to the track category (lowercase) padframes: [int] The amount of temporal padding to apply to the clips before and after in frames. See  vipy.video.Scene.activityclip for options. Returns: A list of  vipy.video.Scene each cloned from the source video and clipped in the temporal region between activities. The union of activityclip() and noactivityclip() should equal the entire video.  notes - Each clip will contain exactly one activity \"Background\" which is the interval for this track where no activities are occurring - Each clip will be at least one frame long",
"func":1
},
{
"ref":"vipy.video.Scene.trackbox",
"url":32,
"doc":"The trackbox is the union of all track bounding boxes in the video, or None if there are no tracks Args: dilate: [float] A dilation factor to apply to the trackbox before returning. See  vipy.geometry.BoundingBox.dilate Returns: A  vipy.geometry.BoundingBox which is the union of all boxes in the track (or None if no boxes exist)",
"func":1
},
{
"ref":"vipy.video.Scene.framebox",
"url":32,
"doc":"Return the bounding box for the image rectangle. Returns: A  vipy.geometry.BoundingBox which defines the image rectangle  notes: This requires calling  vipy.video.Video.preview to get the frame shape from the current filter chain, which touches the video file",
"func":1
},
{
"ref":"vipy.video.Scene.trackcrop",
"url":32,
"doc":"Return the trackcrop() of the scene which is the crop of the video using the  vipy.video.Scene.trackbox . Args: zeropad: [bool] If True, the zero pad the crop if it is outside the image rectangle, otherwise return only valid pixels inside the image rectangle maxsquare: [bool] If True, make the bounding box the maximum square before cropping dilate: [float] The dilation factor to apply to the trackbox prior to cropping Returns: A  vipy.video.Scene object from cropping the video using the trackbox. If there are no tracks, return None.",
"func":1
},
{
"ref":"vipy.video.Scene.activitybox",
"url":32,
"doc":"The activitybox is the union of all activity bounding boxes in the video, which is the union of all tracks contributing to all activities. This is most useful after activityclip(). The activitybox is the smallest bounding box that contains all of the boxes from all of the tracks in all activities in this video.",
"func":1
},
{
"ref":"vipy.video.Scene.activitycuboid",
"url":32,
"doc":"The activitycuboid() is the fixed square spatial crop corresponding to the activitybox (or supplied bounding box), which contains all of the valid activities in the scene. This is most useful after activityclip(). The activitycuboid() is a spatial crop of the video corresponding to the supplied boundingbox or the square activitybox(). This crop must be resized such that the maximum dimension is provided since the crop can be tiny and will not be encodable by ffmpeg",
"func":1
},
{
"ref":"vipy.video.Scene.activitysquare",
"url":32,
"doc":"The activity square is the maxsquare activitybox that contains only valid (non-padded) pixels interior to the image",
"func":1
},
{
"ref":"vipy.video.Scene.activitytube",
"url":32,
"doc":"The activitytube() is a sequence of crops where the spatial box changes on every frame to track the activity. The box in each frame is the square activitybox() for this video which is the union of boxes contributing to this activity in each frame. This function does not perform any temporal clipping. Use activityclip() first to split into individual activities. Crops will be optionally dilated, with zeropadding if the box is outside the image rectangle. All crops will be resized so that the maximum dimension is maxdim (and square by default)",
"func":1
},
{
"ref":"vipy.video.Scene.actortube",
"url":32,
"doc":"The actortube() is a sequence of crops where the spatial box changes on every frame to track the primary actor performing an activity. The box in each frame is the square box centered on the primary actor performing the activity, dilated by a given factor (the original box around the actor is unchanged, this just increases the context, with zero padding) This function does not perform any temporal clipping. Use activityclip() first to split into individual activities. All crops will be resized so that the maximum dimension is maxdim (and square by default)",
"func":1
},
{
"ref":"vipy.video.Scene.speed",
"url":32,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.video.Scene.clip",
"url":32,
"doc":"Clip the video to between (startframe, endframe). This clip is relative to clip() shown by __repr__(). Args: startframe: [int] the start frame relative to the video framerate() for the clip endframe: [int] the end frame relative to the video framerate for the clip, may be none Returns: This video object, clipped so that a load() will result in frame=0 equivalent to startframe. All tracks and activities updated relative to the new startframe.  notes: - This return a clone of the video for idempotence - This does not load the video. This updates the ffmpeg filter chain to temporally trim the video. See self.commandline() for the updated filter chain to run.",
"func":1
},
{
"ref":"vipy.video.Scene.crop",
"url":32,
"doc":"Crop the video using the supplied box, update tracks relative to crop, video is zeropadded if box is outside frame rectangle",
"func":1
},
{
"ref":"vipy.video.Scene.zeropad",
"url":32,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after Update tracks accordingly.",
"func":1
},
{
"ref":"vipy.video.Scene.fliplr",
"url":32,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.video.Scene.flipud",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Scene.rot90ccw",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Scene.rot90cw",
"url":32,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Scene.resize",
"url":32,
"doc":"Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided",
"func":1
},
{
"ref":"vipy.video.Scene.mindim",
"url":32,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Scene.maxdim",
"url":32,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Scene.rescale",
"url":32,
"doc":"Spatially rescale the scene by a constant scale factor. Args: s: [float] Scale factor > 0 to isotropically scale the image.",
"func":1
},
{
"ref":"vipy.video.Scene.startframe",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.extrapolate",
"url":32,
"doc":"Extrapolate the video to frame f and add the extrapolated tracks to the video",
"func":1
},
{
"ref":"vipy.video.Scene.dedupe",
"url":32,
"doc":"Find and delete duplicate tracks by track segmentiou() overlap. Algorithm - For each pair of tracks with the same category, find the larest temporal segment that contains both tracks. - For this segment, compute the IOU for each box interpolated at a stride of dt frames - Compute the mean IOU for this segment. This is the segment IOU. - If the segment IOU is greater than the threshold, merge the shorter of the two tracks with the current track.",
"func":1
},
{
"ref":"vipy.video.Scene.union",
"url":32,
"doc":"Compute the union two scenes as the set of unique activities and tracks. A pair of activities or tracks are non-unique if they overlap spatially and temporally by a given IoU threshold. Merge overlapping tracks. Tracks are merged by considering the mean IoU at the overlapping segment of two tracks with the same category greater than the provided spatial_iou_threshold threshold Activities are merged by considering the temporal IoU of the activities of the same class greater than the provided temporal_iou_threshold threshold Input: -Other: Scene or list of scenes for union. Other may be a clip of self at a different framerate, spatial isotropic scake, clip offset -spatial_iou_threshold: The intersection over union threshold for the mean of the two segments of an overlapping track, Disable by setting to 1.0 -temporal_iou_threshold: The intersection over union threshold for a temporal bounding box for a pair of activities to be declared duplicates. Disable by setting to 1.0 -strict: Require both scenes to share the same underlying video filename -overlap=['average', 'replace', 'keep'] -average: Merge two tracks by averaging the boxes (average=True) if overlapping -replace: merge two tracks by replacing overlapping boxes with other (discard self) -keep: merge two tracks by keeping overlapping boxes with other (discard other) -percentilecover [0,1]: When determining the assignment of two tracks, compute the percentilecover of two tracks by ranking the cover in the overlapping segment and computing the mean of the top-k assignments, where k=len(segment) percentilecover. -percentilesamples [>1]: the number of samples along the overlapping scemgne for computing percentile cover -activity [bool]: union() of activities only -track [bool]: union() of tracks only Output: -Updates this scene to include the non-overlapping activities from other. By default, it takes the strict union of all activities and tracks. Notes: -This is useful for merging scenes computed using a lower resolution/framerate/clipped object or activity detector without running the detector on the high-res scene -This function will preserve the invariance for v  v.clear().union(v.rescale(0.5).framerate(5).activityclip( , to within the quantization error of framerate() downsampling. -percentileiou is a robust method of track assignment when boxes for two tracks (e.g. ground truth and detections) where one track may deform due to occlusion.",
"func":1
},
{
"ref":"vipy.video.Scene.annotate",
"url":32,
"doc":"Generate a video visualization of all annotated objects and activities in the video. The annotation video will be at the resolution and framerate of the underlying video, and pixels in this video will now contain the overlay. This function does not play the video, it only generates an annotation video frames. Use show() which is equivalent to annotate().saveas().play() Args: outfile: [str] An optional file to stream the anntation to without storing the annotated video in memory fontsize: [int] The fontsize of bounding box captions, used by matplotlib captionoffset: (tuple) The (x,y) offset relative to the bounding box to place the caption for each box. textfacecolor: [str] The color of the text in the bounding box caption. Must be in  vipy.gui.using_matplotlib.colorlist . textfacealpha: [float] The transparency of the text in the bounding box caption. Must be in [0,1], where 0=transparent and 1=opaque. shortlabel: [bool] If true, display the shortlabel for each object in the scene, otherwise show the full category boxalpha: [float] The transparency of the box face behind the text. Must be in [0,1], where 0=transparent and 1=opaque. d_category2color: [dict] A dictionary mapping categories of objects in the scene to their box colors. Named colors must be in  vipy.gui.using_matplotlib.colorlist . categories: [list] Only show these categories, or show them all if None nocaption_withstring: [list]: Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list nocaption: [bool] If true, do not show any captions, just boxes mutator: [lambda] A lambda function that will mutate an image to allow for complex visualizations. This should be a mutator like  vipy.image.mutator_show_trackid . timestamp: [bool] If true, show a semitransparent timestamp (when the annotation occurs, not when the video was collected) with frame number in the upper left corner of the video timestampcolor: [str] The color of the timstamp text. Named colors must be in  vipy.gui.using_matplotlib.colorlist . timestampfacecolor: [str] The color of the timestamp background. Named colors must be in  vipy.gui.using_matplotlib.colorlist . verbose: [bool] Show more helpful messages if true Returns: A  vipy.video.Video with annotations in the pixels. If outfile is provided, then the returned video will be flushed.  note In general, this function should not be run on very long videos without the outfile kwarg, as it requires loading the video framewise into memory.",
"func":1
},
{
"ref":"vipy.video.Scene.show",
"url":32,
"doc":"Faster show using interative image show for annotated videos. This can visualize videos before video rendering is complete, but it cannot guarantee frame rates. Large videos with complex scenes will slow this down and will render at lower frame rates.",
"func":1
},
{
"ref":"vipy.video.Scene.thumbnail",
"url":32,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile if provided, otherwise return vipy.image.Scene",
"func":1
},
{
"ref":"vipy.video.Scene.stabilize",
"url":32,
"doc":"Background stablization using flow based stabilization masking foreground region. This will output a video with all frames aligned to the first frame, such that the background is static.",
"func":1
},
{
"ref":"vipy.video.Scene.pixelmask",
"url":32,
"doc":"Replace all pixels in foreground boxes with pixelation",
"func":1
},
{
"ref":"vipy.video.Scene.binarymask",
"url":32,
"doc":"Replace all pixels in foreground boxes with white, zero in background",
"func":1
},
{
"ref":"vipy.video.Scene.asfloatmask",
"url":32,
"doc":"Replace all pixels in foreground boxes with fg, and bg in background, return a copy",
"func":1
},
{
"ref":"vipy.video.Scene.meanmask",
"url":32,
"doc":"Replace all pixels in foreground boxes with mean color",
"func":1
},
{
"ref":"vipy.video.Scene.fgmask",
"url":32,
"doc":"Replace all pixels in foreground boxes with zero",
"func":1
},
{
"ref":"vipy.video.Scene.zeromask",
"url":32,
"doc":"Alias for fgmask",
"func":1
},
{
"ref":"vipy.video.Scene.blurmask",
"url":32,
"doc":"Replace all pixels in foreground boxes with gaussian blurred foreground",
"func":1
},
{
"ref":"vipy.video.Scene.downcast",
"url":32,
"doc":"Cast the object to a  vipy.video.Video class",
"func":1
},
{
"ref":"vipy.video.Scene.merge_tracks",
"url":32,
"doc":"Merge tracks if a track endpoint dilated by a fraction overlaps exactly one track startpoint, and the endpoint and startpoint are close enough together temporally.  note - This is useful for continuing tracking when the detection framerate was too low and the assignment falls outside the measurement gate. - This will not work for complex scenes, as it assumes that there is exactly one possible continuation for a track.",
"func":1
},
{
"ref":"vipy.video.Scene.assign",
"url":32,
"doc":"Assign a list of vipy.object.Detections at frame k to scene by greedy track association. In-place update. Args: miniou: [float] the minimum temporal IOU for activity assignment minconf: [float] the minimum confidence for a detection to be considered as a new track maxhistory: [int] the maximum propagation length of a track with no measurements, the frame history ised for velocity estimates trackconfsamples: [int] the number of uniformly spaced samples along a track to compute a track confidence gate: [int] the gating distance in pixels used for assignment of fast moving detections. Useful for low detection framerates if a detection does not overlap with the track. trackcover: [float] the minimum cover necessary for assignment of a detection to a track activitymerge: [bool] if true, then merge overlapping activity detections of the same track and category, otherwise each activity detection is added as a new detection activitynms: [bool] if true, then perform non-maximum suppression of activity detections of the same actor and category that overlap more than activityiou Returns: This video object with each det assigned to correpsonding track or activity.",
"func":1
},
{
"ref":"vipy.video.Scene.metadata",
"url":32,
"doc":"Return a dictionary of metadata about this video. This is an alias for the 'attributes' dictionary.",
"func":1
},
{
"ref":"vipy.video.Scene.videoid",
"url":32,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by SHA1 hash of the  vipy.video.Video.filename and  vipy.video.Video.url . Args: newid: [str] If not None, then update the video_id as newid. Returns: The video ID if newid=None else self  note - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', ${MY_ID}), or pass in newid",
"func":1
},
{
"ref":"vipy.video.Scene.store",
"url":32,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. >>> v  v.store().restore(v.filename(  note -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing.",
"func":1
},
{
"ref":"vipy.video.Scene.unstore",
"url":32,
"doc":"Delete the currently stored video from  vipy.video.Video.store",
"func":1
},
{
"ref":"vipy.video.Scene.restore",
"url":32,
"doc":"Save the currently stored video as set using  vipy.video.Video.store to filename, and set up filename",
"func":1
},
{
"ref":"vipy.video.Scene.stream",
"url":32,
"doc":"Iterator to yield groups of frames streaming from video. A video stream is a real time iterator to read or write from a video. Streams are useful to group together frames into clips that are operated on as a group. The following use cases are supported: >>> v = vipy.video.RandomScene() Stream individual video frames lagged by 10 frames and 20 frames >>> for (im1, im2) in zip(v.stream().frame(n=-10), v.stream().frame(n=-20 : >>> print(im1, im2) Stream overlapping clips such that each clip is a video n=16 frames long and starts at frame i, and the next clip is n=16 frames long and starts at frame i=i+m >>> for vc in v.stream().clip(n=16, m=4): >>> print(vc) Stream non-overlapping batches of frames such that each clip is a video of length n and starts at frame i, and the next clip is length n and starts at frame i+n >>> for vb in v.stream().batch(n=16): >>> print(vb) Create a write stream to incrementally add frames to long video. >>> vi = vipy.video.Video(filename='/path/to/output.mp4') >>> vo = vipy.video.Video(filename='/path/to/input.mp4') >>> with vo.stream(write=True) as s: >>> for im in vi.stream(): >>> s.write(im)  manipulate pixels of im, if desired Args: write: [bool] If true, create a write stream overwrite: [bool] If true, and the video output filename already exists, overwrite it bufsize: [int] The maximum queue size for the pipe thread. Returns: A  vipy.video.Video.stream.Stream object  note Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.",
"func":1
},
{
"ref":"vipy.video.Scene.bytes",
"url":32,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.video.Scene.frames",
"url":32,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.video.Scene.commandline",
"url":32,
"doc":"Return the equivalent ffmpeg command line string that will be used to transcode the video. This is useful for introspecting the complex filter chain that will be used to process the video. You can try to run this command line yourself for debugging purposes, by replacing 'dummyfile' with an appropriately named output file.",
"func":1
},
{
"ref":"vipy.video.Scene.probeshape",
"url":32,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe  warning this does not take into account any applied ffmpeg filters. The shape will be the (height, width) of the underlying video file.",
"func":1
},
{
"ref":"vipy.video.Scene.duration_in_seconds_of_videofile",
"url":32,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.video.Scene.duration_in_frames_of_videofile",
"url":32,
"doc":"Return video duration of the source video file (NOT the filter chain) in frames, requires ffprobe.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.video.Scene.framerate_of_videofile",
"url":32,
"doc":"Return video framerate in frames per second of the source video file (NOT the filter chain), requires ffprobe.",
"func":1
},
{
"ref":"vipy.video.Scene.probe",
"url":32,
"doc":"Run ffprobe on the filename and return the result as a dictionary",
"func":1
},
{
"ref":"vipy.video.Scene.print",
"url":32,
"doc":"Print the representation of the video This is useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing. Args: prefix: prepend a string prefix to the video __repr__ when printing. Useful for logging. verbose: Print out the video __repr__. Set verbose=False to just sleep sleep: Integer number of seconds to sleep[ before returning Returns: The video object after sleeping",
"func":1
},
{
"ref":"vipy.video.Scene.dict",
"url":32,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding.",
"func":1
},
{
"ref":"vipy.video.Scene.take",
"url":32,
"doc":"Return n frames from the clip uniformly spaced as numpy array Args: n: Integer number of uniformly spaced frames to return Returns: A numpy array of shape (n,W,H)  warning This assumes that the entire video is loaded into memory (e.g. call  vipy.video.Video.load ). Use with caution.",
"func":1
},
{
"ref":"vipy.video.Scene.colorspace",
"url":32,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.video.Scene.nourl",
"url":32,
"doc":"Remove the  vipy.video.Video.url from the video",
"func":1
},
{
"ref":"vipy.video.Scene.url",
"url":32,
"doc":"Video URL and URL download properties",
"func":1
},
{
"ref":"vipy.video.Scene.isloaded",
"url":32,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.video.Scene.isloadable",
"url":32,
"doc":"Return True if the video can be loaded successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version. Args: flush: [bool] If true, flush the video after it loads. This will clear the video pixel buffer Returns: True if load() can be called without FFMPEG exception. If flush=False, then self will contain the loaded video, which is helpful to avoid load() twice in some conditions  warning This requires loading and flushing the video. This is an expensive operation when performed on many videos and may result in out of memory conditions with long videos. Use with caution! Try  vipy.video.Video.canload to test if a single frame can be loaded as a less expensive alternative.",
"func":1
},
{
"ref":"vipy.video.Scene.canload",
"url":32,
"doc":"Return True if the video can be previewed at frame=k successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version.  notes This will only try to preview a single frame. This will not check if the entire video is loadable. Use  vipy.video.Video.isloadable in this case",
"func":1
},
{
"ref":"vipy.video.Scene.iscolor",
"url":32,
"doc":"Is the video a three channel color video as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.video.Scene.isgrayscale",
"url":32,
"doc":"Is the video a single channel as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.video.Scene.hasfilename",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist?",
"func":1
},
{
"ref":"vipy.video.Scene.isdownloaded",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist, meaning that the url has been downloaded to a local file?",
"func":1
},
{
"ref":"vipy.video.Scene.hasurl",
"url":32,
"doc":"Is the url returned from  vipy.video.Video.url a well formed url?",
"func":1
},
{
"ref":"vipy.video.Scene.array",
"url":32,
"doc":"Set or return the video buffer as a numpy array. Args: array: [np.array] A numpy array of size NxHxWxC = (frames, height, width, channels) of type uint8 or float32. copy: [bool] If true, copy the buffer by value instaed of by reference. Copied buffers do not share pixels. Returns: if array=None, return a reference to the pixel buffer as a numpy array, otherwise return the video object.",
"func":1
},
{
"ref":"vipy.video.Scene.fromarray",
"url":32,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.video.Scene.fromframes",
"url":32,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.video.Scene.tonumpy",
"url":32,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.video.Scene.numpy",
"url":32,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.video.Scene.filename",
"url":32,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.video.Scene.abspath",
"url":32,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.video.Scene.relpath",
"url":32,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.video.Scene.rename",
"url":32,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.video.Scene.filesize",
"url":32,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.video.Scene.download",
"url":32,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.video.Scene.fetch",
"url":32,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.video.Scene.shape",
"url":32,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.video.Scene.channels",
"url":32,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.video.Scene.width",
"url":32,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Scene.height",
"url":32,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Scene.aspect_ratio",
"url":32,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.video.Scene.preview",
"url":32,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.video.Scene.load",
"url":32,
"doc":"Load a video using ffmpeg, applying the requested filter chain. Args: verbose: [bool] if True. then ffmpeg console output will be displayed. ignoreErrors: [bool] if True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. shape: [tuple (height, width, channels)] If provided, use this shape for reading and reshaping the byte stream from ffmpeg. This is useful for efficient loading in some scenarios. Knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size Returns: this video object, with the pixels loaded in self.array()  warning Loading long videos can result in out of memory conditions. Try to call clip() first to extract a video segment to load().",
"func":1
},
{
"ref":"vipy.video.Scene.cliprange",
"url":32,
"doc":"Return the planned clip (startframe, endframe) range. This is useful for introspection of the planned clip() before load(), such as for data augmentation purposes without triggering a load. Returns: (startframe, endframe) of the video() such that after load(), the pixel buffer will contain frame=0 equivalent to startframe in the source video, and frame=endframe-startframe-1 equivalent to endframe in the source video. (0, None) If a video does not have a clip() (e.g. clip() was never called, the filter chain does not include a 'trim')  notes The endframe can be retrieved (inefficiently) using: >>> int(round(self.duration_in_frames_of_videofile()  (self.framerate() / self.framerate_of_videofile(  ",
"func":1
},
{
"ref":"vipy.video.Scene.randomcrop",
"url":32,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Scene.centercrop",
"url":32,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Scene.centersquare",
"url":32,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.video.Scene.cropeven",
"url":32,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.video.Scene.maxsquare",
"url":32,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.video.Scene.pad",
"url":32,
"doc":"Alias for zeropad",
"func":1
},
{
"ref":"vipy.video.Scene.pkl",
"url":32,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Scene.pklif",
"url":32,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Scene.webp",
"url":32,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. Args: strict: If true, assert that the filename must have an .webp extension pause: Integer seconds to pause between loops of the animation smallest: if true, create the smallest possible file but takes much longer to run smaller: If true, create a smaller file, which takes a little longer to run Returns: The filename of the webp file for this video  warning This may be slow for very long or large videos",
"func":1
},
{
"ref":"vipy.video.Scene.gif",
"url":32,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. Args: pause: Integer seconds to pause between loops of the animation smallest: If true, create the smallest possible file but takes much longer to run smaller: if trye, create a smaller file, which takes a little longer to run Returns: The filename of the animated GIF of this video  warning This will be very large for big videos, consider using  vipy.video.Video.webp instead.",
"func":1
},
{
"ref":"vipy.video.Scene.saveas",
"url":32,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file. Args: outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image) ignoreErrors: if True, then exit gracefully without throwing an exception. Useful for chaining download().saveas() on parallel dataset downloads flush: If true, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video pause: an integer in seconds to pause between loops of animated images if the outfile is webp or animated gif Returns: a new video object with this video filename, and a clean video filter chain  note - If self.array() is loaded, then export the contents of self._array to the video file - If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video - If outfile None or outfile self.filename(), then overwrite the current filename",
"func":1
},
{
"ref":"vipy.video.Scene.savetmp",
"url":32,
"doc":"Call  vipy.video.Video.saveas using a new temporary video file, and return the video object with this new filename",
"func":1
},
{
"ref":"vipy.video.Scene.savetemp",
"url":32,
"doc":"Alias for  vipy.video.Video.savetmp ",
"func":1
},
{
"ref":"vipy.video.Scene.ffplay",
"url":32,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.video.Scene.play",
"url":32,
"doc":"Play the saved video filename in self.filename() If there is no filename, try to download it. If the filter chain is dirty or the pixels are loaded, dump to temp video file first then play it. This uses 'ffplay' on the PATH if available, otherwise uses a fallback player by showing a sequence of matplotlib frames. If the output of the ffmpeg filter chain has modified this video, then this will be saved to a temporary video file. To play the original video (indepenedent of the filter chain of this video), use  vipy.video.Video.ffplay . Args: verbose: If true, show more verbose output notebook: If true, play in a jupyter notebook Returns: The unmodified video object",
"func":1
},
{
"ref":"vipy.video.Scene.torch",
"url":32,
"doc":"Convert the loaded video of shape NxHxWxC frames to an MxCxHxW torch tensor/ Args: startframe: [int >= 0] The start frame of the loaded video to use for constructig the torch tensor endframe: [int >= 0] The end frame of the loaded video to use for constructing the torch tensor length: [int >= 0] The length of the torch tensor if endframe is not provided. stride: [int >= 1] The temporal stride in frames. This is the number of frames to skip. take: [int >= 0] The number of uniformly spaced frames to include in the tensor. boundary: ['repeat', 'cyclic'] The boundary handling for when the requested tensor slice goes beyond the end of the video order: ['nchw', 'nhwc', 'chwn', 'cnhw'] The axis ordering of the returned torch tensor N=number of frames (batchsize), C=channels, H=height, W=width verbose [bool]: Print out the slice used for contructing tensor withslice: [bool] Return a tuple (tensor, slice) that includes the slice used to construct the tensor. Useful for data provenance. scale: [float] An optional scale factor to apply to the tensor. Useful for converting [0,255] -> [0,1] withlabel: [bool] Return a tuple (tensor, labels) that includes the N framewise activity labels. nonelabel: [bool] returns tuple (t, None) if withlabel=False Returns Returns torch float tensor, analogous to torchvision.transforms.ToTensor() Return (tensor, slice) if withslice=True (withslice takes precedence) Returns (tensor, labellist) if withlabel=True  notes - This triggers a load() of the video - The precedence of arguments is (startframe, endframe) or (startframe, startframe+length), then stride and take. - Follows numpy slicing rules. Optionally return the slice used if withslice=True",
"func":1
},
{
"ref":"vipy.video.Scene.clone",
"url":32,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected. Args: flushforward: copy the object, and set the cloned object  vipy.video.Video.array to None. This flushes the video buffer for the clone, not the object flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone. flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object. flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos flushfile: Remove the filename and the URL from the video object. Useful for creating new video objects from loaded pixels. rekey: Generate new unique track ID and activity ID keys for this scene shallow: shallow copy everything (copy by reference), except for ffmpeg object. attributes dictionary is shallow copied sharedarray: deep copy of everything, except for pixel buffer which is shared. Changing the pixel buffer on self is reflected in the clone. Returns: A deepcopy of the video object such that changes to self are not reflected in the copy  note Cloning videos is an expensive operation and can slow down real time code. Use sparingly.",
"func":1
},
{
"ref":"vipy.video.Scene.flush",
"url":32,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.video.Scene.flush_and_return",
"url":32,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.video.Scene.map",
"url":32,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.video.Scene.gain",
"url":32,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.video.Scene.bias",
"url":32,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.video.Scene.normalize",
"url":32,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.video.RandomVideo",
"url":32,
"doc":"Return a random loaded vipy.video.video. Useful for unit testing, minimum size (32x32x32) for ffmpeg",
"func":1
},
{
"ref":"vipy.video.RandomScene",
"url":32,
"doc":"Return a random loaded vipy.video.Scene. Useful for unit testing.",
"func":1
},
{
"ref":"vipy.video.RandomSceneActivity",
"url":32,
"doc":"Return a random loaded vipy.video.Scene. Useful for unit testing.",
"func":1
},
{
"ref":"vipy.video.EmptyScene",
"url":32,
"doc":"Return an empty scene",
"func":1
},
{
"ref":"vipy.math",
"url":33,
"doc":""
},
{
"ref":"vipy.math.normalize",
"url":33,
"doc":"Whiten the numpy array arr using the provided mean and standard deviation. - Uses numba acceleration since this is a common operation for preparing tensors. - Computes:  scale arr) - mean / std Args: arr: [numpy] A numpy array mean: [numpy] A broadcastable mean vector std: [numpy] A broadcastable std vector scale: [float] A scale factor to apply to arr before whitening (e.g. to scale from [0,255] to [0,1]) Returns  scale arr) - mean / std  notes Does not check that std > 0",
"func":1
},
{
"ref":"vipy.math.iseven",
"url":33,
"doc":"is the number x an even number?",
"func":1
},
{
"ref":"vipy.math.even",
"url":33,
"doc":"Return the largest even integer less than or equal (or greater than if greaterthan=True) to the value",
"func":1
},
{
"ref":"vipy.math.poweroftwo",
"url":33,
"doc":"Return the closest power of two smaller than the scalar value. x=511 -> 256, x=512 -> 512",
"func":1
},
{
"ref":"vipy.math.signsqrt",
"url":33,
"doc":"Return the signed square root of elements in numpy array x",
"func":1
},
{
"ref":"vipy.math.runningmean",
"url":33,
"doc":"Compute the running unweighted mean of X row-wise, with a history of n, reducing the history at the start for column indexes < n",
"func":1
},
{
"ref":"vipy.math.gaussian",
"url":33,
"doc":"1D gaussian window with M points. Replication of scipy.signal.gaussian",
"func":1
},
{
"ref":"vipy.math.gaussian2d",
"url":33,
"doc":"2D float32 gaussian image of size (rows=H, cols=W) with mu=[x, y] and std=[stdx, stdy]",
"func":1
},
{
"ref":"vipy.math.interp1d",
"url":33,
"doc":"Replication of scipy.interpolate.interp1d with assume_sorted=True, and constant replication of boundary handling",
"func":1
},
{
"ref":"vipy.math.find_closest_positive_divisor",
"url":33,
"doc":"Return non-trivial positive integer divisor (bh) of (a) closest to (b) in abs(b-bh) such that a % bh  0.  notes This uses exhaustive search, which is inefficient for large a.",
"func":1
},
{
"ref":"vipy.math.cartesian_to_polar",
"url":33,
"doc":"Cartesian (x,y) coordinates to polar (radius, theta_radians) coordinates, theta in radians in [-pi,pi]",
"func":1
},
{
"ref":"vipy.math.polar_to_cartesian",
"url":33,
"doc":"Polar (r=radius, t=theta_radians) coordinates to cartesian (x=right,y=down) coordinates. (0,0) is upper left of image",
"func":1
},
{
"ref":"vipy.math.rad2deg",
"url":33,
"doc":"Radians to degrees",
"func":1
},
{
"ref":"vipy.util",
"url":34,
"doc":""
},
{
"ref":"vipy.util.bz2pkl",
"url":34,
"doc":"Read/Write a bz2 compressed pickle file",
"func":1
},
{
"ref":"vipy.util.mergedict",
"url":34,
"doc":"Combine keys of two dictionaries and return a dictionary deep copy",
"func":1
},
{
"ref":"vipy.util.hascache",
"url":34,
"doc":"Is the VIPY_CACHE environment variable set?",
"func":1
},
{
"ref":"vipy.util.tocache",
"url":34,
"doc":"If the VIPY_CACHE environment variable is set, then return the filename=/path/to/file.ext in the cache as VIPY_CACHE/file.ext",
"func":1
},
{
"ref":"vipy.util.try_import",
"url":34,
"doc":"Show a helpful error message for missing optional packages",
"func":1
},
{
"ref":"vipy.util.findyaml",
"url":34,
"doc":"Return a list of absolute paths to yaml files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findpkl",
"url":34,
"doc":"Return a list of absolute paths to pkl files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findjson",
"url":34,
"doc":"Return a list of absolute paths to json files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findimage",
"url":34,
"doc":"Return a list of absolute paths to image files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findvideo",
"url":34,
"doc":"Return a list of absolute paths to video files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.readyaml",
"url":34,
"doc":"Read a yaml file and return a parsed dictionary, this is slow for large yaml files",
"func":1
},
{
"ref":"vipy.util.count_images_in_subdirectories",
"url":34,
"doc":"Count the total number of images in indir/subdir1, indir/subdir2, go down only one level and no further .",
"func":1
},
{
"ref":"vipy.util.rowvectorize",
"url":34,
"doc":"Convert a 1D numpy array to a 2D row vector of size (1,N)",
"func":1
},
{
"ref":"vipy.util.columnvectorize",
"url":34,
"doc":"Convert a 1D numpy array to a 2D column vector of size (N,1)",
"func":1
},
{
"ref":"vipy.util.isodd",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.keymax",
"url":34,
"doc":"Return key in dictionary containing maximum value",
"func":1
},
{
"ref":"vipy.util.keymin",
"url":34,
"doc":"Return key in dictionary containing minimum value",
"func":1
},
{
"ref":"vipy.util.isjsonfile",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.writejson",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.readjson",
"url":34,
"doc":"Read jsonfile=/path/to/file.json and return the json parsed object, issue warning if jsonfile does not have .json extension and strict=True",
"func":1
},
{
"ref":"vipy.util.groupby",
"url":34,
"doc":"groupby on unsorted inset",
"func":1
},
{
"ref":"vipy.util.vipy_groupby",
"url":34,
"doc":"groupby on unsorted inset",
"func":1
},
{
"ref":"vipy.util.groupbyasdict",
"url":34,
"doc":"Return dictionary of keys and lists from groupby on unsorted inset, where keyfunc is a lambda function on elements in inset Args: togroup: a list of elements to group keyfunc: a lambda function to operate on elemenets of togroup such that the value returned from the lambda is the equality key for grouping Returns: A dictionary with unique keys returned from keyfunc, and values are lists of elements in togroup with the same key",
"func":1
},
{
"ref":"vipy.util.countby",
"url":34,
"doc":"Return dictionary of keys and group sizes for a grouping of the input list by keyfunc lambda function",
"func":1
},
{
"ref":"vipy.util.most_frequent",
"url":34,
"doc":"Return the most frequent element as determined by element equality",
"func":1
},
{
"ref":"vipy.util.countbyasdict",
"url":34,
"doc":"Alias for  vipy.util.countby ",
"func":1
},
{
"ref":"vipy.util.softmax",
"url":34,
"doc":"Row-wise softmax",
"func":1
},
{
"ref":"vipy.util.permutelist",
"url":34,
"doc":"randomly permute list order",
"func":1
},
{
"ref":"vipy.util.flatlist",
"url":34,
"doc":"Convert list of tuples into a list expanded by concatenating tuples",
"func":1
},
{
"ref":"vipy.util.rmdir",
"url":34,
"doc":"Recursively remove directory and all contents (if the directory exists)",
"func":1
},
{
"ref":"vipy.util.dividelist",
"url":34,
"doc":"Divide inlist into a list of lists such that the size of each sublist is the requseted fraction of the original list. This operation is deterministic and generates the same division in multiple calls. Args: inlist: [list] fractions: [tuple] such as (0.1, 0.7, 0.2) An iterable of fractions that must be non-negative and sum to one",
"func":1
},
{
"ref":"vipy.util.chunklist",
"url":34,
"doc":"Convert list into a list of lists of length num_chunks, such that each element is a list containing a sequential chunk of the original list. >>> (A,B,C) = vipy.util.chunklist(inlist, num_chunks=3) >>> assert len(A)  len(inlist)  3  note The last chunk will be larger for ragged chunks",
"func":1
},
{
"ref":"vipy.util.chunklistbysize",
"url":34,
"doc":"Convert list into a list of lists such that each element is a list containing a sequential chunk of the original list of length size_per_chunk",
"func":1
},
{
"ref":"vipy.util.chunklistWithOverlap",
"url":34,
"doc":"Convert list into a list of lists such that each element is a list containing a sequential chunk of the original list of length size_per_chunk",
"func":1
},
{
"ref":"vipy.util.chunklistwithoverlap",
"url":34,
"doc":"Alias for chunklistWithOverlap",
"func":1
},
{
"ref":"vipy.util.imwritejet",
"url":34,
"doc":"Write a grayscale numpy image as a jet colormapped image to the given file",
"func":1
},
{
"ref":"vipy.util.isuint8",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isnumber",
"url":34,
"doc":"Is the input a python type of a number or a string containing a number?",
"func":1
},
{
"ref":"vipy.util.isfloat",
"url":34,
"doc":"Is the input a float or a string that can be converted to float?",
"func":1
},
{
"ref":"vipy.util.imwritegray",
"url":34,
"doc":"Write a floating point grayscale numpy image in [0,1] as [0,255] grayscale",
"func":1
},
{
"ref":"vipy.util.imwrite",
"url":34,
"doc":"Write a floating point 2D numpy image as jet or gray, 3D numpy as rgb or bgr",
"func":1
},
{
"ref":"vipy.util.print_and_return",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.savetemp",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.gray2jet",
"url":34,
"doc":"[0,1] grayscale to [0.255] RGB",
"func":1
},
{
"ref":"vipy.util.jet",
"url":34,
"doc":"jet colormap",
"func":1
},
{
"ref":"vipy.util.is_hiddenfile",
"url":34,
"doc":"Does the filename start with a period?",
"func":1
},
{
"ref":"vipy.util.seq",
"url":34,
"doc":"Equivalent to matlab [start:step:stop]",
"func":1
},
{
"ref":"vipy.util.loadh5",
"url":34,
"doc":"Load an HDF5 file",
"func":1
},
{
"ref":"vipy.util.loadmat73",
"url":34,
"doc":"Matlab 7.3 format, keys should be a list of keys to access HDF5 file as f[key1][key2] . Returned as numpy array",
"func":1
},
{
"ref":"vipy.util.saveas",
"url":34,
"doc":"Save variables as a dill pickled file",
"func":1
},
{
"ref":"vipy.util.loadas",
"url":34,
"doc":"Load variables from a dill pickled file",
"func":1
},
{
"ref":"vipy.util.load",
"url":34,
"doc":"Load variables from a relocatable archive file format, either Dill Pickle or JSON. Loading is performed by attemping the following: 1. load the pickle or json file 2. if abspath=true, then convert relative paths to absolute paths for object when loaded 3. If the loaded object is a vipy object (or iterable) and the relocatable path /$PATH is present, try to repath it to the directory containing this archive (this has been deprecated) 4. If the resulting files are not found, throw a warning 5. If a large number of objects are loaded, disable garbage collection.",
"func":1
},
{
"ref":"vipy.util.canload",
"url":34,
"doc":"Attempt to load a pkl file, and return true if it can be successfully loaded, otherwise False",
"func":1
},
{
"ref":"vipy.util.save",
"url":34,
"doc":"Save variables to an archive file",
"func":1
},
{
"ref":"vipy.util.distload",
"url":34,
"doc":"Load a redistributable pickle file that replaces absolute paths in datapath with srcpath. See also vipy.util.distsave(), This function has been deprecated, all archives should be distributed with relative paths",
"func":1
},
{
"ref":"vipy.util.distsave",
"url":34,
"doc":"Save a archive file for redistribution, where datapath is replaced by dstpath. Useful for redistribuing pickle files with absolute paths. See also vipy.util.distload(). This function has been deprecated, all archives should be distributed with relative paths",
"func":1
},
{
"ref":"vipy.util.repath",
"url":34,
"doc":"Change the filename with prefix srcpath to dstpath, for any element in v that supports the filename() api",
"func":1
},
{
"ref":"vipy.util.scpsave",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.scpload",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.load_opencv_yaml",
"url":34,
"doc":"Load a numpy array from YAML file exported from OpenCV",
"func":1
},
{
"ref":"vipy.util.matrix_to_opencv_yaml",
"url":34,
"doc":"Write list of matrices to OpenCV yaml file format with given variable names",
"func":1
},
{
"ref":"vipy.util.save_opencv_yaml",
"url":34,
"doc":"Save a numpy array to YAML file importable by OpenCV",
"func":1
},
{
"ref":"vipy.util.tofilename",
"url":34,
"doc":"Convert arbitrary string to valid filename with underscores replacing invalid chars",
"func":1
},
{
"ref":"vipy.util.isexe",
"url":34,
"doc":"Is the file an executable binary?",
"func":1
},
{
"ref":"vipy.util.isinstalled",
"url":34,
"doc":"Is the command is available on the path",
"func":1
},
{
"ref":"vipy.util.isextension",
"url":34,
"doc":"Does the filename end with the extension ext? >>> isextension('/path/to/myfile.json', 'json')  True >>> isextension('/path/to/myfile.json', '.json')  True >>> isextension('/path/to/myfile.json', '.pkl')  False",
"func":1
},
{
"ref":"vipy.util.ispkl",
"url":34,
"doc":"Is the file a pickle archive file",
"func":1
},
{
"ref":"vipy.util.ispklfile",
"url":34,
"doc":"Is the file a pickle archive file",
"func":1
},
{
"ref":"vipy.util.ishtml",
"url":34,
"doc":"Is the file an HTMLfile",
"func":1
},
{
"ref":"vipy.util.ispickle",
"url":34,
"doc":"Is the file a pickle archive file",
"func":1
},
{
"ref":"vipy.util.ishdf5",
"url":34,
"doc":"Is the file an HDF5 file?",
"func":1
},
{
"ref":"vipy.util.filebase",
"url":34,
"doc":"Return c for filename /a/b/c.ext  warning Will return /a/b/c.d for multidot filenames like /a/b/c.d.e (e.g. /a/b/filename.tar.gz)",
"func":1
},
{
"ref":"vipy.util.filepath",
"url":34,
"doc":"Return /a/b/c for filename /a/b/c/d.ext, /a/b for filename /a/b/c/d.ext if depth=1, etc",
"func":1
},
{
"ref":"vipy.util.delpath",
"url":34,
"doc":"Return c/d.ext for filename /a/b/c/d.ext and indir /a/b",
"func":1
},
{
"ref":"vipy.util.newpath",
"url":34,
"doc":"Return /d/e/c.ext for filename /a/b/c.ext and newdir /d/e/",
"func":1
},
{
"ref":"vipy.util.newprefix",
"url":34,
"doc":"Return /a/b/c/h/i.ext for filename /f/g/h/i.ext and prefix /a/b/c and depth=1",
"func":1
},
{
"ref":"vipy.util.newpathdir",
"url":34,
"doc":"Return /a/b/n/d/e.ext for filename=/a/b/c/d/e.ext, olddir=c, newdir=n",
"func":1
},
{
"ref":"vipy.util.newpathroot",
"url":34,
"doc":"Return /r/b/c.ext for filename /a/b/c.ext and new root directory r",
"func":1
},
{
"ref":"vipy.util.topath",
"url":34,
"doc":"Alias for newpath",
"func":1
},
{
"ref":"vipy.util.filefull",
"url":34,
"doc":"Return /a/b/c for filename /a/b/c.ext",
"func":1
},
{
"ref":"vipy.util.filetail",
"url":34,
"doc":"Return c.ext for filename /a/b/c.ext",
"func":1
},
{
"ref":"vipy.util.matread",
"url":34,
"doc":"Whitespace separated values defining columns, lines define rows. Return numpy array",
"func":1
},
{
"ref":"vipy.util.imlist",
"url":34,
"doc":"return list of images with absolute path in a directory",
"func":1
},
{
"ref":"vipy.util.videolist",
"url":34,
"doc":"return list of videos with absolute path in a directory",
"func":1
},
{
"ref":"vipy.util.dirlist",
"url":34,
"doc":"return list of directories in a directory",
"func":1
},
{
"ref":"vipy.util.dirlist_sorted_bycreation",
"url":34,
"doc":"Sort the directory list from newest first to oldest last by creation date",
"func":1
},
{
"ref":"vipy.util.extlist",
"url":34,
"doc":"return list of files with absolute path in a directory that have the provided extension (with the prepended dot, ext='.mp4')",
"func":1
},
{
"ref":"vipy.util.listext",
"url":34,
"doc":"Alias for extlist",
"func":1
},
{
"ref":"vipy.util.jsonlist",
"url":34,
"doc":"return list of fJSON iles with absolute path in a directory",
"func":1
},
{
"ref":"vipy.util.listjson",
"url":34,
"doc":"Alias for jsonlist",
"func":1
},
{
"ref":"vipy.util.writelist",
"url":34,
"doc":"Write list of strings to an output file with each row an element of the list",
"func":1
},
{
"ref":"vipy.util.readlist",
"url":34,
"doc":"Read each row of file as an element of the list",
"func":1
},
{
"ref":"vipy.util.readtxt",
"url":34,
"doc":"Read a text file one string per row",
"func":1
},
{
"ref":"vipy.util.writecsv",
"url":34,
"doc":"Write list of tuples to an output csv file with each list element on a row and tuple elements separated by comma",
"func":1
},
{
"ref":"vipy.util.readcsv",
"url":34,
"doc":"Read a csv file into a list of lists",
"func":1
},
{
"ref":"vipy.util.readcsvwithheader",
"url":34,
"doc":"Read a csv file into a list of lists",
"func":1
},
{
"ref":"vipy.util.imsavelist",
"url":34,
"doc":"Write out all images in a directory to a provided file with each line containing absolute path to image",
"func":1
},
{
"ref":"vipy.util.csvlist",
"url":34,
"doc":"Return a list of absolute paths of  .csv files in current directory",
"func":1
},
{
"ref":"vipy.util.pklist",
"url":34,
"doc":"Return a list of absolute paths of  .pk files in current directory",
"func":1
},
{
"ref":"vipy.util.listpkl",
"url":34,
"doc":"Return a list of absolute paths of  .pk files in current directory",
"func":1
},
{
"ref":"vipy.util.txtlist",
"url":34,
"doc":"Return a list of absolute paths of  .txt files in current directory",
"func":1
},
{
"ref":"vipy.util.imlistidx",
"url":34,
"doc":"Return index in list of filename containing index number",
"func":1
},
{
"ref":"vipy.util.mat2gray",
"url":34,
"doc":"Convert numpy array to float32 with 1.0=max and 0=min",
"func":1
},
{
"ref":"vipy.util.mdlist",
"url":34,
"doc":"Preallocate 2D list of size MxN",
"func":1
},
{
"ref":"vipy.util.isurl",
"url":34,
"doc":"Is a path a URL?",
"func":1
},
{
"ref":"vipy.util.shortuuid",
"url":34,
"doc":"Generate a short UUID with n hex digits",
"func":1
},
{
"ref":"vipy.util.isimageurl",
"url":34,
"doc":"Is a path a URL with image extension?",
"func":1
},
{
"ref":"vipy.util.isvideourl",
"url":34,
"doc":"Is a path a URL with video extension?",
"func":1
},
{
"ref":"vipy.util.isS3url",
"url":34,
"doc":"Is a path a URL for an S3 object?",
"func":1
},
{
"ref":"vipy.util.isyoutubeurl",
"url":34,
"doc":"Is a path a youtube URL?",
"func":1
},
{
"ref":"vipy.util.checkerboard",
"url":34,
"doc":"m=number of square by column, n=size of final image",
"func":1
},
{
"ref":"vipy.util.islist",
"url":34,
"doc":"Is an object a python list",
"func":1
},
{
"ref":"vipy.util.islistoflists",
"url":34,
"doc":"Is an object a python list of lists x= 1,2], [3,4 ",
"func":1
},
{
"ref":"vipy.util.istupleoftuples",
"url":34,
"doc":"Is an object a python list of lists x= 1,2], [3,4 ",
"func":1
},
{
"ref":"vipy.util.isimageobject",
"url":34,
"doc":"Is an object a vipy.image class Image, ImageCategory, ImageDetection?",
"func":1
},
{
"ref":"vipy.util.isvideotype",
"url":34,
"doc":"Is an object a vipy.video class Video, VideoCategory, Scene?",
"func":1
},
{
"ref":"vipy.util.isvideoobject",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isvipyobject",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.istuple",
"url":34,
"doc":"Is an object a python tuple?",
"func":1
},
{
"ref":"vipy.util.tolist",
"url":34,
"doc":"Convert a python tuple or singleton object to a list if not already a list",
"func":1
},
{
"ref":"vipy.util.isimg",
"url":34,
"doc":"Is an object an image with a supported image extension ['.jpg','.jpeg','.png','.tif','.tiff','.pgm','.ppm','.gif','.bmp']?",
"func":1
},
{
"ref":"vipy.util.isimage",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isvideofile",
"url":34,
"doc":"Equivalent to isvideo()",
"func":1
},
{
"ref":"vipy.util.isimgfile",
"url":34,
"doc":"Convenience function for isimg",
"func":1
},
{
"ref":"vipy.util.isimagefile",
"url":34,
"doc":"Convenience function for isimg",
"func":1
},
{
"ref":"vipy.util.isgif",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isjpeg",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.iswebp",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.ispng",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isjpg",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.iscsv",
"url":34,
"doc":"Is a file a CSV file extension?",
"func":1
},
{
"ref":"vipy.util.isvideo",
"url":34,
"doc":"Is a filename in path a video with a known video extension ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']?",
"func":1
},
{
"ref":"vipy.util.isnumpy",
"url":34,
"doc":"Is a python object a numpy object?",
"func":1
},
{
"ref":"vipy.util.isnumpyarray",
"url":34,
"doc":"Is a python object a numpy array?",
"func":1
},
{
"ref":"vipy.util.istextfile",
"url":34,
"doc":"Is the given file a text file?",
"func":1
},
{
"ref":"vipy.util.isxml",
"url":34,
"doc":"Is the given file an xml file?",
"func":1
},
{
"ref":"vipy.util.bgr2gray",
"url":34,
"doc":"Wrapper for numpy uint8 BGR image to uint8 numpy grayscale",
"func":1
},
{
"ref":"vipy.util.gray2bgr",
"url":34,
"doc":"Wrapper for numpy float32 gray image to uint8 numpy BGR",
"func":1
},
{
"ref":"vipy.util.gray2rgb",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.bgr2rgb",
"url":34,
"doc":"Wrapper for numpy BGR uint8 to numpy RGB uint8",
"func":1
},
{
"ref":"vipy.util.rgb2bgr",
"url":34,
"doc":"same as bgr2rgb",
"func":1
},
{
"ref":"vipy.util.bgr2hsv",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.gray2hsv",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isarchive",
"url":34,
"doc":"Is filename a zip or gzip compressed tar archive?",
"func":1
},
{
"ref":"vipy.util.istgz",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isbz2",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.tempfilename",
"url":34,
"doc":"Create a temporary filename $TEMPDIR/$UUID.suffix, suffix should include the dot such as suffix='.jpg',",
"func":1
},
{
"ref":"vipy.util.totempdir",
"url":34,
"doc":"Convert a filename '/patj/to/filename.ext' to '/tempdir/filename.ext'",
"func":1
},
{
"ref":"vipy.util.templike",
"url":34,
"doc":"Create a new temporary filename with the same extension as filename",
"func":1
},
{
"ref":"vipy.util.cached",
"url":34,
"doc":"Create a new filename in the cache, or tempdir if not found",
"func":1
},
{
"ref":"vipy.util.tempimage",
"url":34,
"doc":"Create a temporary image with the given extension",
"func":1
},
{
"ref":"vipy.util.temppng",
"url":34,
"doc":"Create a temporay PNG file",
"func":1
},
{
"ref":"vipy.util.temppickle",
"url":34,
"doc":"Create a temporary pickle file",
"func":1
},
{
"ref":"vipy.util.tempjpg",
"url":34,
"doc":"Create a temporary JPG file in system temp directory",
"func":1
},
{
"ref":"vipy.util.tempMP4",
"url":34,
"doc":"Create a temporary MP4 file in system temp directory",
"func":1
},
{
"ref":"vipy.util.tmpjpg",
"url":34,
"doc":"Create a temporary JPG file in /tmp",
"func":1
},
{
"ref":"vipy.util.tempcsv",
"url":34,
"doc":"Create a temporary CSV file",
"func":1
},
{
"ref":"vipy.util.temphtml",
"url":34,
"doc":"Create a temporary HTMLfile",
"func":1
},
{
"ref":"vipy.util.temppkl",
"url":34,
"doc":"Create a temporary pickle file",
"func":1
},
{
"ref":"vipy.util.tempyaml",
"url":34,
"doc":"Create a temporary YAML file",
"func":1
},
{
"ref":"vipy.util.tempjson",
"url":34,
"doc":"Create a temporary JSON file",
"func":1
},
{
"ref":"vipy.util.temppdf",
"url":34,
"doc":"Create a temporary PDF file",
"func":1
},
{
"ref":"vipy.util.mktemp",
"url":34,
"doc":"Create a temporary file with extension .ext",
"func":1
},
{
"ref":"vipy.util.tempdir",
"url":34,
"doc":"Wrapper around tempfile, because I can never remember the syntax",
"func":1
},
{
"ref":"vipy.util.imread",
"url":34,
"doc":"Wrapper for opencv imread. Note that color images are imported as BGR!",
"func":1
},
{
"ref":"vipy.util.imrescale",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.imresize",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.touch",
"url":34,
"doc":"Create an empty file containing mystr",
"func":1
},
{
"ref":"vipy.util.isboundingbox",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.Stopwatch",
"url":34,
"doc":"Return elapsed system time in seconds between calls to enter and exit"
},
{
"ref":"vipy.util.Stopwatch.since",
"url":34,
"doc":"Return seconds since start or last call to this method",
"func":1
},
{
"ref":"vipy.util.Stopwatch.reset",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.util.Stopwatch.duration",
"url":34,
"doc":"Time in seconds since last reset",
"func":1
},
{
"ref":"vipy.util.Timer",
"url":34,
"doc":"Pretty print elapsed system time in seconds between calls to enter and exit >>> t = Timer(): >>> [some code] >>> print(t) >>> [some more code] >>> print(t) >>> with Timer(): >>> [some code]"
},
{
"ref":"vipy.util.isfile",
"url":34,
"doc":"Wrapper for os.path.isfile",
"func":1
},
{
"ref":"vipy.util.isstring",
"url":34,
"doc":"Is an object a python string or unicode string?",
"func":1
},
{
"ref":"vipy.util.timestamp",
"url":34,
"doc":"Return date and time string in form DDMMMYY_HHMMSS",
"func":1
},
{
"ref":"vipy.util.clockstamp",
"url":34,
"doc":"Datetime stamp in local timezone with second resolution with format Year-Month-Day Hour:Minute:Second",
"func":1
},
{
"ref":"vipy.util.minutestamp",
"url":34,
"doc":"Return date and time string in form DDMMMYY_HHMM",
"func":1
},
{
"ref":"vipy.util.datestamp",
"url":34,
"doc":"Return date and time string in form DDMMMYY",
"func":1
},
{
"ref":"vipy.util.remkdir",
"url":34,
"doc":"Create a given directory if not already exists",
"func":1
},
{
"ref":"vipy.util.rermdir",
"url":34,
"doc":"Recursively delete a given directory (if exists), and remake it",
"func":1
},
{
"ref":"vipy.util.premkdir",
"url":34,
"doc":"pre-create directory /path/to/subdir using  vipy.util.remkdir if it does not exist for outfile=/path/to/subdir/file.ext, and return filename",
"func":1
},
{
"ref":"vipy.util.newbase",
"url":34,
"doc":"Convert filename=/a/b/c.ext base=d -> /a/b/d.ext",
"func":1
},
{
"ref":"vipy.util.toextension",
"url":34,
"doc":"Convert filename='/path/to/myfile.ext' to /path/to/myfile.xyz, such that newext='xyz' or newext='.xyz'",
"func":1
},
{
"ref":"vipy.util.topkl",
"url":34,
"doc":"Convert filename='/path/to/myfile.ext' to /path/to/myfile.pkl",
"func":1
},
{
"ref":"vipy.util.splitextension",
"url":34,
"doc":"Given /a/b/c.ext return tuple of strings ('/a/b/c', '.ext')",
"func":1
},
{
"ref":"vipy.util.hasextension",
"url":34,
"doc":"Does the provided filename have a file extension (e.g. /path/to/file.ext) or not (e.g. /path/to/file)",
"func":1
},
{
"ref":"vipy.util.fileext",
"url":34,
"doc":"Given filename /a/b/c.ext return '.ext', or /a/b/c.tar.gz return '.tar.gz'. If multidot=False, then return '.gz'. If withdot=False, return 'ext'",
"func":1
},
{
"ref":"vipy.util.mediaextension",
"url":34,
"doc":"Return '.mp4' for filename='/a/b/c.mp4'",
"func":1
},
{
"ref":"vipy.util.ismacosx",
"url":34,
"doc":"Is the current platform MacOSX?",
"func":1
},
{
"ref":"vipy.util.islinux",
"url":34,
"doc":"is the current platform Linux?",
"func":1
},
{
"ref":"vipy.util.linuxversion",
"url":34,
"doc":"Return linux version",
"func":1
},
{
"ref":"vipy.util.imcrop",
"url":34,
"doc":"Crop a 2D or 3D numpy image given a vipy.geometry.BoundingBox",
"func":1
},
{
"ref":"vipy.util.Failed",
"url":34,
"doc":"Raised when unit test fails to throw an exception"
},
{
"ref":"vipy.util.string_to_pil_interpolation",
"url":34,
"doc":"Internal function to convert interp string to interp object",
"func":1
},
{
"ref":"vipy.linalg",
"url":35,
"doc":""
},
{
"ref":"vipy.linalg.random_positive_semidefinite_matrix",
"url":35,
"doc":"Return a randomly generated numpy float64 positive semidefinite matrix of size NxN",
"func":1
},
{
"ref":"vipy.linalg.column_stochastic",
"url":35,
"doc":"Given a numpy array X of size MxN, return column stochastic matrix such that each of N columns sum to one. Args: X: [numpy] A 2D array eps: [float] a small floating point value to avoid divide by zero Returns: Matrix X such that columns sum to one.",
"func":1
},
{
"ref":"vipy.linalg.row_stochastic",
"url":35,
"doc":"Given a numpy array X of size MxN, return row stochastic matrix such that each of M rows sum to one. Args: X: [numpy] A 2D array eps: [float] a small floating point value to avoid divide by zero Returns: Matrix X such that rows sum to one.",
"func":1
},
{
"ref":"vipy.linalg.rowstochastic",
"url":35,
"doc":"Alias for  vipy.linalg.row_stochastic ",
"func":1
},
{
"ref":"vipy.linalg.bistochastic",
"url":35,
"doc":"Given a square numpy array X of size NxN, return bistochastic matrix such that each of N rows and N columns sum to one. Bistochastic matrix (doubly stochastic matrix) using Sinkhorn normalization. Args: X: [numpy] A square 2D array eps: [float] a small floating point value to avoid divide by zero numIterations: [int] The number of sinkhorn normalization iterations to apply Returns: Bistochastic matrix X",
"func":1
},
{
"ref":"vipy.linalg.rectangular_bistochastic",
"url":35,
"doc":"Given a rectangular numpy array X of size MxN, return bistochastic matrix such that each of M rows sum to N/M and each if N columns sum to 1. Bistochastic matrix using Sinkhorn normalization on rectangular matrices Args: X: [numpy] A 2D array eps: [float] a small floating point value to avoid divide by zero numIterations: [int] The number of sinkhorn normalization iterations to apply Returns: Rectangular bistochastic matrix X",
"func":1
},
{
"ref":"vipy.linalg.row_normalized",
"url":35,
"doc":"Given a rectangular numpy array X of size MxN, return a matrix such that each row has unit L2 norm. Args: X: [numpy] A 2D array Returns: Row normalized matrix X such that np.linalg.norm(X[i])  1, for all rows i",
"func":1
},
{
"ref":"vipy.linalg.row_ssqrt",
"url":35,
"doc":"Given a rectangular numpy array X of size MxN, return a matrix such that each element is the signed square root of the element in X. Args: X: [numpy] A rectangular 2D array Returns: Matrix M such that elements M[i,j] preserve the sign of corresponding element in X, but the value is M[i,j] = sign(X[i,j])  sqrt(abs(X[i,j] ",
"func":1
},
{
"ref":"vipy.linalg.normalize",
"url":35,
"doc":"Given a numpy vector X of size N, return a vector with unit norm. Args: X: [numpy] A 1D array or a 2D array with one dim  1 Returns: Unit L2 norm of x, flattened to 1D",
"func":1
},
{
"ref":"vipy.linalg.vectorize",
"url":35,
"doc":"Convert a tuple X=([1], [2,3], [4,5,6]) to a numpy vector [1,2,3,4,5,6]. Args: X: [list of lists, tuple of tuples] Returns: 1D numpy array with all elements in X stacked horizontally",
"func":1
},
{
"ref":"vipy.linalg.columnvector",
"url":35,
"doc":"Convert a tuple with N elements to an Nx1 column vector",
"func":1
},
{
"ref":"vipy.linalg.columnize",
"url":35,
"doc":"Convert a numpy array into a flattened Nx1 column vector",
"func":1
},
{
"ref":"vipy.linalg.rowvector",
"url":35,
"doc":"Convert a tuple with N elements to an 1xN row vector",
"func":1
},
{
"ref":"vipy.linalg.is_poweroftwo",
"url":35,
"doc":"Is the number x a power of two? >>> assert vipy.linalg.is_poweroftwo(4)  True >>> assert vipy.linalg.is_poweroftwo(3)  False",
"func":1
},
{
"ref":"vipy.linalg.ndmax",
"url":35,
"doc":"Return the (i,j, .)=(row, col, .) entry corresponding to the maximum element in the nd numpy matrix A >>> A = np.array( 1,2,3],[4,100,6 ) >>> assert vipy.linalg.ndmax(A)  (1,2)",
"func":1
},
{
"ref":"vipy.linalg.ndmin",
"url":35,
"doc":"Return the (i,j, .)=(row,col, .) entry corresponding to the minimum element in the nd numpy matrix A >>> A = np.array( 1,2,3],[4,100,6 ) >>> assert vipy.linalg.ndmin(A)  (0,0)",
"func":1
},
{
"ref":"vipy.metrics",
"url":36,
"doc":""
},
{
"ref":"vipy.metrics.cumulative_match_characteristic",
"url":36,
"doc":"CMC curve for probe x gallery similarity matrix (larger is more similar) and ground truth match matrix (one +1 per row, rest zeros)",
"func":1
},
{
"ref":"vipy.metrics.plot_cmc",
"url":36,
"doc":"Generate cumulative match characteristic (CMC) plot",
"func":1
},
{
"ref":"vipy.metrics.tdr_at_rank",
"url":36,
"doc":"Janus metric for correct retrieval (true detection rate) within a specific rank",
"func":1
},
{
"ref":"vipy.metrics.auroc",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.roc",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.roc_per_image",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.roc_eer",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.tpr_at_fpr",
"url":36,
"doc":"Janus metric for true positive rate at a specific false positive rate",
"func":1
},
{
"ref":"vipy.metrics.fpr_at_tpr",
"url":36,
"doc":"Janus metric for false positive rate at a specific true positive rate",
"func":1
},
{
"ref":"vipy.metrics.plot_roc",
"url":36,
"doc":"http: scikit-learn.org/stable/auto_examples/plot_roc.html",
"func":1
},
{
"ref":"vipy.metrics.mean_average_precision",
"url":36,
"doc":"numpy wrapper for mean",
"func":1
},
{
"ref":"vipy.metrics.average_precision",
"url":36,
"doc":"sklearn wrapper",
"func":1
},
{
"ref":"vipy.metrics.f1_score",
"url":36,
"doc":"sklearn wrapper",
"func":1
},
{
"ref":"vipy.metrics.confusion_matrix",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.categorization_report",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.precision_recall",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.plot_pr",
"url":36,
"doc":"Plot precision recall curve using matplotlib, with optional figure save",
"func":1
},
{
"ref":"vipy.metrics.plot_ap",
"url":36,
"doc":"Plot Average-Precision bar chart using matplotlib, with optional figure save",
"func":1
},
{
"ref":"vipy.metrics.histogram",
"url":36,
"doc":"Plot histogram bar chart using matplotlib with vertical axis labels on x-axis with optional figure save. Inputs: -freq: the output of (freq, categories) = np.histogram( ., bins=n) -categories [list]: a list of category names that must be length n, or the output of (f,c) = np.histogram( .) and categories=c[:-1] -xrot ['vertical'|None]: rotate the xticks -barcolors [list]: list of named colors equal to the length of categories",
"func":1
},
{
"ref":"vipy.metrics.pie",
"url":36,
"doc":"Generate a matplotlib style pie chart with wedges with specified size and labels, with an optional outfile",
"func":1
},
{
"ref":"vipy.camera",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.CAM",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.FRAMERATE",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.TIC",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.TOC",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.RESIZE",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.GREY",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Camera.PROCESS",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Webcam",
"url":37,
"doc":"Create a webcam object that will yield  vipy.image.Image frames. This is a light wrapper to OpenCV webcam object (cv2.VideoCapture) that yields vipy objects. >>> cam = vipy.cmaera.Webcam() >>> cam.frame().show() Or as an iterator: >>> for im in vipy.camera.Webcam(): >>> im.show() To capture a video: >>> Args: framerate: [float] The framerate to grab from the camera url: [int] The camera index to open"
},
{
"ref":"vipy.camera.Webcam.current",
"url":37,
"doc":"Alias for  vipy.camera.Webcam.next ",
"func":1
},
{
"ref":"vipy.camera.Webcam.next",
"url":37,
"doc":"Return a  vipy.image.Image from the camera",
"func":1
},
{
"ref":"vipy.camera.Webcam.frame",
"url":37,
"doc":"Alias for  vipy.camera.Webcam.next ",
"func":1
},
{
"ref":"vipy.camera.Webcam.video",
"url":37,
"doc":"Return a  vipy.video.Video with n frames, constructed using the provided framerate (defaults to 30Hz)",
"func":1
},
{
"ref":"vipy.camera.Ipcam",
"url":37,
"doc":"Create a IPcam object that will yield  vipy.image.Image frames. >>> cam = vipy.cmaera.IPcam() >>> cam.frame().show() Or as an iterator: >>> for im in vipy.camera.IPcam(): >>> im.show()"
},
{
"ref":"vipy.camera.Ipcam.TMPFILE",
"url":37,
"doc":""
},
{
"ref":"vipy.camera.Ipcam.next",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.globals",
"url":38,
"doc":""
},
{
"ref":"vipy.globals.logging",
"url":38,
"doc":"Single entry point for enabling/disabling logging vs. printing All vipy functions overload \"from vipy.globals import print\" for simplified readability of code. This global function redirects print or warn to using the standard logging module. If format is provided, this will create a basicConfig handler, but this should be configured by the end-user.",
"func":1
},
{
"ref":"vipy.globals.warn",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.print",
"url":38,
"doc":"Main entry point for all print statements in the vipy package. All vipy code calls this to print helpful messages.  notes -Printing can be disabled by calling vipy.globals.silent() -Printing can be redirected to logging by calling vipy.globals.logging(True) -All print() statements in vipy. are overloaded to call vipy.globals.print() so that it can be redirected to logging",
"func":1
},
{
"ref":"vipy.globals.verbose",
"url":38,
"doc":"The global verbosity level, only really used right now for FFMPEG messages",
"func":1
},
{
"ref":"vipy.globals.isverbose",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.silent",
"url":38,
"doc":"Silence the global verbosity level, only really used right now for FFMPEG messages",
"func":1
},
{
"ref":"vipy.globals.issilent",
"url":38,
"doc":"Is the global verbosity silent?",
"func":1
},
{
"ref":"vipy.globals.verbosity",
"url":38,
"doc":"Set the global verbosity level [0,1,2]=debug, warn, info",
"func":1
},
{
"ref":"vipy.globals.debug",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.isdebug",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.cache",
"url":38,
"doc":"The cache is the location that URLs are downloaded to on your system. This can be set here, or with the environment variable VIPY_CACHE >>> vipy.globals.cache('/path/to/.vipy') >>> cachedir = vipy.globals.cache() Args: cachedir: the location to store cached files when downloaded. Can also be set using the VIPY_CACHE environment variable. if none, return the current cachedir Returns: The current cachedir if cachedir=None else None",
"func":1
},
{
"ref":"vipy.globals.cpuonly",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.gpuindex",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.dask",
"url":38,
"doc":"Return the current Dask client, can be accessed globally for parallel processing. Args: pct: float in [0,1] the percentage of the current machine to use address: the dask scheduler of the form 'HOSTNAME:PORT' num_processes: the number of prpcesses to use on the current machine num_gpus: the number of GPUs to use on the current machine dashboard: [bool] whether to inialize the dask client with a web dashboard Returns: The  vipy.batch.Dask object pointing to the Dask Distrbuted object",
"func":1
},
{
"ref":"vipy.globals.parallel",
"url":38,
"doc":"Enable parallel processing with n>=1 processes or a percentage of system core (pct \\in [0,1]) or a dask scheduler . This can be be used as a context manager >>> with vipy.globals.parallel(n=4): >>> vipy.batch.Batch( .) or using the global variables: >>> vipy.globals.parallel(n=4): >>> vipy.batch.Batch( .) >>> vipy.globals.noparallel() To check the current parallelism level: >>> num_processes = vipy.globals.parallel() To run with a dask scheduler: >>> with vipy.globals.parallel(scheduler='10.0.1.1:8585') >>> vipy.batch.Batch( .) Args: n: [int] number of parallel processes pct: [float] the percentage [0,1] of system cores to dedicate to parallel processing scheduler: [str] the dask scheduler of the form 'HOSTNAME:PORT' like '128.0.0.1:8785'. See  ",
"func":1
},
{
"ref":"vipy.globals.noparallel",
"url":38,
"doc":"Disable all parallel processing",
"func":1
},
{
"ref":"vipy.globals.nodask",
"url":38,
"doc":"Alias for  vipy.globals.noparallel ",
"func":1
},
{
"ref":"vipy.flow",
"url":39,
"doc":""
},
{
"ref":"vipy.flow.Image",
"url":39,
"doc":"vipy.flow.Image() class"
},
{
"ref":"vipy.flow.Image.min",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.max",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.scale",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.threshold",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.width",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.height",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.shape",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.flow",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.colorflow",
"url":39,
"doc":"Flow visualization image (HSV: H=flow angle, V=flow magnitude), returns vipy.image.Image()",
"func":1
},
{
"ref":"vipy.flow.Image.warp",
"url":39,
"doc":"Warp image imfrom=vipy.image.Image() to imto=vipy.image.Image() using flow computed as imfrom->imto, updating objects",
"func":1
},
{
"ref":"vipy.flow.Image.alphapad",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.zeropad",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.dx",
"url":39,
"doc":"Return dx (horizontal) component of flow",
"func":1
},
{
"ref":"vipy.flow.Image.dy",
"url":39,
"doc":"Return dy (vertical) component of flow",
"func":1
},
{
"ref":"vipy.flow.Image.shift",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.show",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.rescale",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.resize_like",
"url":39,
"doc":"Resize flow buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.flow.Image.resize",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.magnitude",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.angle",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.clone",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.print",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video",
"url":39,
"doc":"vipy.flow.Video() class"
},
{
"ref":"vipy.flow.Video.min",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.max",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.width",
"url":39,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.flow.Video.height",
"url":39,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.flow.Video.flow",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.colorflow",
"url":39,
"doc":"Flow visualization video",
"func":1
},
{
"ref":"vipy.flow.Video.magnitude",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.show",
"url":39,
"doc":"Alias for play",
"func":1
},
{
"ref":"vipy.flow.Video.print",
"url":39,
"doc":"Print the representation of the video This is useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing. Args: prefix: prepend a string prefix to the video __repr__ when printing. Useful for logging. verbose: Print out the video __repr__. Set verbose=False to just sleep sleep: Integer number of seconds to sleep[ before returning Returns: The video object after sleeping",
"func":1
},
{
"ref":"vipy.flow.Video.cast",
"url":32,
"doc":"Cast a conformal video object to a  vipy.video.Video object. This is useful for downcasting superclasses. >>> vs = vipy.video.RandomScene() >>> v = vipy.video.Video.cast(vs)",
"func":1
},
{
"ref":"vipy.flow.Video.from_json",
"url":32,
"doc":"Import a json string as a  vipy.video.Video object. This will perform a round trip from a video to json and back to a video object. This same operation is used for serialization of all vipy objects to JSON for storage. >>> v = vipy.video.Video.from_json(vipy.video.RandomVideo().json( ",
"func":1
},
{
"ref":"vipy.flow.Video.metadata",
"url":32,
"doc":"Return a dictionary of metadata about this video. This is an alias for the 'attributes' dictionary.",
"func":1
},
{
"ref":"vipy.flow.Video.videoid",
"url":32,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by SHA1 hash of the  vipy.video.Video.filename and  vipy.video.Video.url . Args: newid: [str] If not None, then update the video_id as newid. Returns: The video ID if newid=None else self  note - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', ${MY_ID}), or pass in newid",
"func":1
},
{
"ref":"vipy.flow.Video.frame",
"url":32,
"doc":"Return the kth frame as an  vipy.image Image object",
"func":1
},
{
"ref":"vipy.flow.Video.store",
"url":32,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. >>> v  v.store().restore(v.filename(  note -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing.",
"func":1
},
{
"ref":"vipy.flow.Video.unstore",
"url":32,
"doc":"Delete the currently stored video from  vipy.video.Video.store",
"func":1
},
{
"ref":"vipy.flow.Video.restore",
"url":32,
"doc":"Save the currently stored video as set using  vipy.video.Video.store to filename, and set up filename",
"func":1
},
{
"ref":"vipy.flow.Video.stream",
"url":32,
"doc":"Iterator to yield groups of frames streaming from video. A video stream is a real time iterator to read or write from a video. Streams are useful to group together frames into clips that are operated on as a group. The following use cases are supported: >>> v = vipy.video.RandomScene() Stream individual video frames lagged by 10 frames and 20 frames >>> for (im1, im2) in zip(v.stream().frame(n=-10), v.stream().frame(n=-20 : >>> print(im1, im2) Stream overlapping clips such that each clip is a video n=16 frames long and starts at frame i, and the next clip is n=16 frames long and starts at frame i=i+m >>> for vc in v.stream().clip(n=16, m=4): >>> print(vc) Stream non-overlapping batches of frames such that each clip is a video of length n and starts at frame i, and the next clip is length n and starts at frame i+n >>> for vb in v.stream().batch(n=16): >>> print(vb) Create a write stream to incrementally add frames to long video. >>> vi = vipy.video.Video(filename='/path/to/output.mp4') >>> vo = vipy.video.Video(filename='/path/to/input.mp4') >>> with vo.stream(write=True) as s: >>> for im in vi.stream(): >>> s.write(im)  manipulate pixels of im, if desired Args: write: [bool] If true, create a write stream overwrite: [bool] If true, and the video output filename already exists, overwrite it bufsize: [int] The maximum queue size for the pipe thread. Returns: A  vipy.video.Video.stream.Stream object  note Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.",
"func":1
},
{
"ref":"vipy.flow.Video.clear",
"url":32,
"doc":"no-op for  vipy.video.Video object, used only for  vipy.video.Scene ",
"func":1
},
{
"ref":"vipy.flow.Video.bytes",
"url":32,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.flow.Video.frames",
"url":32,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.flow.Video.commandline",
"url":32,
"doc":"Return the equivalent ffmpeg command line string that will be used to transcode the video. This is useful for introspecting the complex filter chain that will be used to process the video. You can try to run this command line yourself for debugging purposes, by replacing 'dummyfile' with an appropriately named output file.",
"func":1
},
{
"ref":"vipy.flow.Video.probeshape",
"url":32,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe  warning this does not take into account any applied ffmpeg filters. The shape will be the (height, width) of the underlying video file.",
"func":1
},
{
"ref":"vipy.flow.Video.duration_in_seconds_of_videofile",
"url":32,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.flow.Video.duration_in_frames_of_videofile",
"url":32,
"doc":"Return video duration of the source video file (NOT the filter chain) in frames, requires ffprobe.  notes This is the duration of the source video and NOT the duration of the filter chain. If you load(), this may be different duration depending on clip() or framerate() directives.",
"func":1
},
{
"ref":"vipy.flow.Video.framerate_of_videofile",
"url":32,
"doc":"Return video framerate in frames per second of the source video file (NOT the filter chain), requires ffprobe.",
"func":1
},
{
"ref":"vipy.flow.Video.probe",
"url":32,
"doc":"Run ffprobe on the filename and return the result as a dictionary",
"func":1
},
{
"ref":"vipy.flow.Video.dict",
"url":32,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding.",
"func":1
},
{
"ref":"vipy.flow.Video.json",
"url":32,
"doc":"Return a json representation of the video. Args: encode: If true, return a JSON encoded string using json.dumps Returns: A JSON encoded string if encode=True, else returns a dictionary object  note If the video is loaded, then the JSON will not include the pixels. Try using  vipy.video.Video.store to serialize videos, or call  vipy.video.Video.flush first.",
"func":1
},
{
"ref":"vipy.flow.Video.take",
"url":32,
"doc":"Return n frames from the clip uniformly spaced as numpy array Args: n: Integer number of uniformly spaced frames to return Returns: A numpy array of shape (n,W,H)  warning This assumes that the entire video is loaded into memory (e.g. call  vipy.video.Video.load ). Use with caution.",
"func":1
},
{
"ref":"vipy.flow.Video.framerate",
"url":32,
"doc":"Change the input framerate for the video and update frame indexes for all annotations Args: fps: Float frames per second to process the underlying video Returns: If fps is None, return the current framerate, otherwise set the framerate to fps",
"func":1
},
{
"ref":"vipy.flow.Video.colorspace",
"url":32,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.flow.Video.nourl",
"url":32,
"doc":"Remove the  vipy.video.Video.url from the video",
"func":1
},
{
"ref":"vipy.flow.Video.url",
"url":32,
"doc":"Video URL and URL download properties",
"func":1
},
{
"ref":"vipy.flow.Video.isloaded",
"url":32,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.flow.Video.isloadable",
"url":32,
"doc":"Return True if the video can be loaded successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version. Args: flush: [bool] If true, flush the video after it loads. This will clear the video pixel buffer Returns: True if load() can be called without FFMPEG exception. If flush=False, then self will contain the loaded video, which is helpful to avoid load() twice in some conditions  warning This requires loading and flushing the video. This is an expensive operation when performed on many videos and may result in out of memory conditions with long videos. Use with caution! Try  vipy.video.Video.canload to test if a single frame can be loaded as a less expensive alternative.",
"func":1
},
{
"ref":"vipy.flow.Video.canload",
"url":32,
"doc":"Return True if the video can be previewed at frame=k successfully. This is useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version.  notes This will only try to preview a single frame. This will not check if the entire video is loadable. Use  vipy.video.Video.isloadable in this case",
"func":1
},
{
"ref":"vipy.flow.Video.iscolor",
"url":32,
"doc":"Is the video a three channel color video as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.flow.Video.isgrayscale",
"url":32,
"doc":"Is the video a single channel as returned from  vipy.video.Video.channels ?",
"func":1
},
{
"ref":"vipy.flow.Video.hasfilename",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist?",
"func":1
},
{
"ref":"vipy.flow.Video.isdownloaded",
"url":32,
"doc":"Does the filename returned from  vipy.video.Video.filename exist, meaning that the url has been downloaded to a local file?",
"func":1
},
{
"ref":"vipy.flow.Video.hasurl",
"url":32,
"doc":"Is the url returned from  vipy.video.Video.url a well formed url?",
"func":1
},
{
"ref":"vipy.flow.Video.array",
"url":32,
"doc":"Set or return the video buffer as a numpy array. Args: array: [np.array] A numpy array of size NxHxWxC = (frames, height, width, channels) of type uint8 or float32. copy: [bool] If true, copy the buffer by value instaed of by reference. Copied buffers do not share pixels. Returns: if array=None, return a reference to the pixel buffer as a numpy array, otherwise return the video object.",
"func":1
},
{
"ref":"vipy.flow.Video.fromarray",
"url":32,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.flow.Video.fromframes",
"url":32,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.flow.Video.tonumpy",
"url":32,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.flow.Video.numpy",
"url":32,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.flow.Video.filename",
"url":32,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.flow.Video.abspath",
"url":32,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.flow.Video.relpath",
"url":32,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.flow.Video.rename",
"url":32,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.flow.Video.filesize",
"url":32,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.flow.Video.download",
"url":32,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.flow.Video.fetch",
"url":32,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.flow.Video.shape",
"url":32,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.flow.Video.channels",
"url":32,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.flow.Video.aspect_ratio",
"url":32,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.flow.Video.preview",
"url":32,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.flow.Video.thumbnail",
"url":32,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile. This is functionally equivalent to  vipy.video.Video.frame with an additional outfile argument to easily save an annotated thumbnail image. Args: outfile: [str] an optional outfile to save the annotated frame frame: [int >= 0] The frame to output the thumbnail Returns: A  vipy.image.Image object for frame k.",
"func":1
},
{
"ref":"vipy.flow.Video.load",
"url":32,
"doc":"Load a video using ffmpeg, applying the requested filter chain. Args: verbose: [bool] if True. then ffmpeg console output will be displayed. ignoreErrors: [bool] if True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. shape: [tuple (height, width, channels)] If provided, use this shape for reading and reshaping the byte stream from ffmpeg. This is useful for efficient loading in some scenarios. Knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size Returns: this video object, with the pixels loaded in self.array()  warning Loading long videos can result in out of memory conditions. Try to call clip() first to extract a video segment to load().",
"func":1
},
{
"ref":"vipy.flow.Video.speed",
"url":32,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.flow.Video.clip",
"url":32,
"doc":"Load a video clip betweeen start and end frames",
"func":1
},
{
"ref":"vipy.flow.Video.cliprange",
"url":32,
"doc":"Return the planned clip (startframe, endframe) range. This is useful for introspection of the planned clip() before load(), such as for data augmentation purposes without triggering a load. Returns: (startframe, endframe) of the video() such that after load(), the pixel buffer will contain frame=0 equivalent to startframe in the source video, and frame=endframe-startframe-1 equivalent to endframe in the source video. (0, None) If a video does not have a clip() (e.g. clip() was never called, the filter chain does not include a 'trim')  notes The endframe can be retrieved (inefficiently) using: >>> int(round(self.duration_in_frames_of_videofile()  (self.framerate() / self.framerate_of_videofile(  ",
"func":1
},
{
"ref":"vipy.flow.Video.rot90cw",
"url":32,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.rot90ccw",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.fliplr",
"url":32,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.flow.Video.flipud",
"url":32,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.rescale",
"url":32,
"doc":"Rescale the video by factor s, such that the new dimensions are (s H, s W), can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.resize",
"url":32,
"doc":"Resize the video to be (rows=height, cols=width)",
"func":1
},
{
"ref":"vipy.flow.Video.mindim",
"url":32,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.flow.Video.maxdim",
"url":32,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.flow.Video.randomcrop",
"url":32,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.flow.Video.centercrop",
"url":32,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.flow.Video.centersquare",
"url":32,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.flow.Video.cropeven",
"url":32,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.flow.Video.maxsquare",
"url":32,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.flow.Video.zeropad",
"url":32,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after  notes Older FFMPEG implementations can throw the error \"Input area  : : : not within the padded area  : : : or zero-sized, this is often caused by odd sized padding. Recommend calling self.cropeven().zeropad( .) to avoid this",
"func":1
},
{
"ref":"vipy.flow.Video.pad",
"url":32,
"doc":"Alias for zeropad",
"func":1
},
{
"ref":"vipy.flow.Video.crop",
"url":32,
"doc":"Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().",
"func":1
},
{
"ref":"vipy.flow.Video.pkl",
"url":32,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.flow.Video.pklif",
"url":32,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.flow.Video.webp",
"url":32,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. Args: strict: If true, assert that the filename must have an .webp extension pause: Integer seconds to pause between loops of the animation smallest: if true, create the smallest possible file but takes much longer to run smaller: If true, create a smaller file, which takes a little longer to run Returns: The filename of the webp file for this video  warning This may be slow for very long or large videos",
"func":1
},
{
"ref":"vipy.flow.Video.gif",
"url":32,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. Args: pause: Integer seconds to pause between loops of the animation smallest: If true, create the smallest possible file but takes much longer to run smaller: if trye, create a smaller file, which takes a little longer to run Returns: The filename of the animated GIF of this video  warning This will be very large for big videos, consider using  vipy.video.Video.webp instead.",
"func":1
},
{
"ref":"vipy.flow.Video.saveas",
"url":32,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file. Args: outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image) ignoreErrors: if True, then exit gracefully without throwing an exception. Useful for chaining download().saveas() on parallel dataset downloads flush: If true, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video pause: an integer in seconds to pause between loops of animated images if the outfile is webp or animated gif Returns: a new video object with this video filename, and a clean video filter chain  note - If self.array() is loaded, then export the contents of self._array to the video file - If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video - If outfile None or outfile self.filename(), then overwrite the current filename",
"func":1
},
{
"ref":"vipy.flow.Video.savetmp",
"url":32,
"doc":"Call  vipy.video.Video.saveas using a new temporary video file, and return the video object with this new filename",
"func":1
},
{
"ref":"vipy.flow.Video.savetemp",
"url":32,
"doc":"Alias for  vipy.video.Video.savetmp ",
"func":1
},
{
"ref":"vipy.flow.Video.ffplay",
"url":32,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.flow.Video.play",
"url":32,
"doc":"Play the saved video filename in self.filename() If there is no filename, try to download it. If the filter chain is dirty or the pixels are loaded, dump to temp video file first then play it. This uses 'ffplay' on the PATH if available, otherwise uses a fallback player by showing a sequence of matplotlib frames. If the output of the ffmpeg filter chain has modified this video, then this will be saved to a temporary video file. To play the original video (indepenedent of the filter chain of this video), use  vipy.video.Video.ffplay . Args: verbose: If true, show more verbose output notebook: If true, play in a jupyter notebook Returns: The unmodified video object",
"func":1
},
{
"ref":"vipy.flow.Video.quicklook",
"url":32,
"doc":"Generate a montage of n uniformly spaced frames. Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. Input: -n: Number of images in the quicklook -mindim: The minimum dimension of each of the elements in the montage -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.flow.Video.torch",
"url":32,
"doc":"Convert the loaded video of shape NxHxWxC frames to an MxCxHxW torch tensor/ Args: startframe: [int >= 0] The start frame of the loaded video to use for constructig the torch tensor endframe: [int >= 0] The end frame of the loaded video to use for constructing the torch tensor length: [int >= 0] The length of the torch tensor if endframe is not provided. stride: [int >= 1] The temporal stride in frames. This is the number of frames to skip. take: [int >= 0] The number of uniformly spaced frames to include in the tensor. boundary: ['repeat', 'cyclic'] The boundary handling for when the requested tensor slice goes beyond the end of the video order: ['nchw', 'nhwc', 'chwn', 'cnhw'] The axis ordering of the returned torch tensor N=number of frames (batchsize), C=channels, H=height, W=width verbose [bool]: Print out the slice used for contructing tensor withslice: [bool] Return a tuple (tensor, slice) that includes the slice used to construct the tensor. Useful for data provenance. scale: [float] An optional scale factor to apply to the tensor. Useful for converting [0,255] -> [0,1] withlabel: [bool] Return a tuple (tensor, labels) that includes the N framewise activity labels. nonelabel: [bool] returns tuple (t, None) if withlabel=False Returns Returns torch float tensor, analogous to torchvision.transforms.ToTensor() Return (tensor, slice) if withslice=True (withslice takes precedence) Returns (tensor, labellist) if withlabel=True  notes - This triggers a load() of the video - The precedence of arguments is (startframe, endframe) or (startframe, startframe+length), then stride and take. - Follows numpy slicing rules. Optionally return the slice used if withslice=True",
"func":1
},
{
"ref":"vipy.flow.Video.clone",
"url":32,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected. Args: flushforward: copy the object, and set the cloned object  vipy.video.Video.array to None. This flushes the video buffer for the clone, not the object flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone. flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object. flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos flushfile: Remove the filename and the URL from the video object. Useful for creating new video objects from loaded pixels. rekey: Generate new unique track ID and activity ID keys for this scene shallow: shallow copy everything (copy by reference), except for ffmpeg object. attributes dictionary is shallow copied sharedarray: deep copy of everything, except for pixel buffer which is shared. Changing the pixel buffer on self is reflected in the clone. Returns: A deepcopy of the video object such that changes to self are not reflected in the copy  note Cloning videos is an expensive operation and can slow down real time code. Use sparingly.",
"func":1
},
{
"ref":"vipy.flow.Video.flush",
"url":32,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.flow.Video.flush_and_return",
"url":32,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.flow.Video.map",
"url":32,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.flow.Video.gain",
"url":32,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.flow.Video.bias",
"url":32,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.flow.Video.normalize",
"url":32,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.flow.Flow",
"url":39,
"doc":"vipy.flow.Flow() class"
},
{
"ref":"vipy.flow.Flow.imageflow",
"url":39,
"doc":"Default opencv dense flow, from im to imprev. This should be overloaded",
"func":1
},
{
"ref":"vipy.flow.Flow.videoflow",
"url":39,
"doc":"Compute optical flow for a video framewise skipping framestep frames, compute optical flow acrsos flowstep frames,",
"func":1
},
{
"ref":"vipy.flow.Flow.videoflowframe",
"url":39,
"doc":"Computer the videoflow for a single frame",
"func":1
},
{
"ref":"vipy.flow.Flow.keyflow",
"url":39,
"doc":"Compute optical flow for a video framewise relative to keyframes separated by keystep",
"func":1
},
{
"ref":"vipy.flow.Flow.keyflowframe",
"url":39,
"doc":"Compute the keyflow for a single frame",
"func":1
},
{
"ref":"vipy.flow.Flow.affineflow",
"url":39,
"doc":"Return a flow field of size (height=H, width=W) consistent with a 2x3 affine transformation A",
"func":1
},
{
"ref":"vipy.flow.Flow.euclideanflow",
"url":39,
"doc":"Return a flow field of size (height=H, width=W) consistent with an Euclidean transform parameterized by a 2x2 Rotation and 2x1 translation",
"func":1
},
{
"ref":"vipy.flow.Flow.stabilize",
"url":39,
"doc":"Affine stabilization to frame zero using multi-scale optical flow correspondence with foreground object keepouts. Recommended usage: >>> v = vipy.video.Scene(filename='/path/to/my/video.mp4').stabilize() Args: v: [ vipy.video.Scene ]: The input video to stabilize, should be resized to mindim=256 keystep: [int] The local stabilization step between keyframes (should be <= 30) padheightfrac: [float] The height padding (relative to video height) to be applied to output video to allow for vertical stabilization padwidthfrac: [float] The width padding (relative to video width) to be applied to output video to allow for horizontal stabilization padheightpx: [int] The height padding to be applied to output video to allow for vertical stabilization. Overrides padheight. padwidthpx: [int] The width padding to be applied to output video to allow for horizontal stabilization. Overrides padwidth. border: [float] The border keepout fraction to ignore during flow correspondence. This should be proportional to the maximum frame to frame flow dilate: [float] The dilation to apply to the foreground object boxes to define a foregroun keepout for flow computation contrast: [float] The minimum gradient necessary for flow correspondence, to avoid flow on low contrast regions rigid: [bool] Euclidean stabilization affine: [bool] Affine stabilization verbose: [bool] This takes a while to run so show some progress  . strict: [bool] If true, throw an exception on error, otherwise return the original video and set v.hasattribute('unstabilized'), useful for large scale stabilization outfile: [str] the file path to the stabilized output video Returns: A cloned  vipy.video.Scene with filename=outfile, such that pixels and tracks are background stabilized.  notes - The remaining distortion after stabilization is due to: rolling shutter distortion, perspective distortion and non-keepout moving objects in background - If the video contains objects, the object boxes will be transformed along with the stabilization - This requires loading videos entirely into memory. Be careful with stabilizing long videos. - The returned video has the attribute 'stabilize' which contains the mean and median residual of the flow field relative to the motion model. This can be used for stabilization quality filtering.",
"func":1
},
{
"ref":"vipy.batch",
"url":40,
"doc":""
},
{
"ref":"vipy.batch.Dask",
"url":40,
"doc":""
},
{
"ref":"vipy.batch.Dask.num_gpus",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.has_dashboard",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.dashboard",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.num_processes",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.shutdown",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.client",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Checkpoint",
"url":40,
"doc":"Batch checkpoints for long running jobs"
},
{
"ref":"vipy.batch.Checkpoint.checkpoint",
"url":40,
"doc":"Return the last checkpointed result. Useful for recovering from dask crashes for long jobs.",
"func":1
},
{
"ref":"vipy.batch.Checkpoint.last_archive",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Batch",
"url":40,
"doc":"vipy.batch.Batch class This class provides a representation of a set of vipy objects. All of the object types must be the same. If so, then an operation on the batch is performed on each of the elements in the batch in parallel. Examples: >>> b = vipy.batch.Batch([Image(filename='img_%06d.png' % k) for k in range(0,100)]) >>> b.map(lambda im: im.bgr( >>> b.map(lambda im: np.sum(im.array( ) >>> b.map(lambda im, f: im.saveas(f), args=['out%d.jpg' % k for k in range(0,100)]) >>> v = vipy.video.RandomSceneActivity() >>> b = vipy.batch.Batch(v, n_processes=16) >>> b.map(lambda v,k: v[k], args=[(k,) for k in range(0, len(v ])  paralle interpolation >>> d = vipy.dataset.kinetics.Kinetics700('/path/to/kinetics').download().trainset() >>> b = vipy.batch.Batch(d, n_processes=32) >>> b.map(lambda v: v.download().save(  will download and clip dataset in parallel >>> b.result()  retrieve results after a sequence of map or filter chains Args: strict: [bool] if distributed processing fails, return None for that element and print the exception rather than raise as_completed: [bool] Return the objects to the scheduler as they complete, this can introduce instabilities for large complex objects, use with caution Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call"
},
{
"ref":"vipy.batch.Batch.restore",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Batch.result",
"url":40,
"doc":"Return the result of the batch processing, ordered",
"func":1
},
{
"ref":"vipy.batch.Batch.map",
"url":40,
"doc":"Run the lambda function on each of the elements of the batch and return the batch object. >>> iml = [vipy.image.RandomScene(512,512) for k in range(0,1000)] >>> imb = vipy.image.Batch(iml) >>> imb.map(lambda im: im.rgb( The lambda function f_lambda should not include closures. If it does, construct the lambda with default parameter capture: >>> f = lambda x, prm1=42: x+prm1 instead of: >>> prm1 = 42 >>> f = lambda x: x+prm1",
"func":1
},
{
"ref":"vipy.batch.Batch.filter",
"url":40,
"doc":"Run the lambda function on each of the elements of the batch and filter based on the provided lambda keeping those elements that return true",
"func":1
},
{
"ref":"vipy.batch.Batch.scattermap",
"url":40,
"doc":"Scatter obj to all workers, and apply lambda function f(obj, im) to each element in batch Usage: >>> Batch(mylist, ngpu=8).scattermap(lambda net, im: net(im), net).result() This will scatter the large object net to all workers, and pin it to a specific GPU. Within the net object, you can call vipy.global.gpuindex() to retrieve your assigned GPU index, which can be used by torch.cuda.device(). Then, the net object processes each element in the batch using net according to the lambda, and returns the results. This function includes ngpu processes, and assumes there are ngpu available on the target machine. Each net is replicated in a different process, so it is the callers responsibility for getting vipy.global.gpuindex() from within the process and setting net to take advantage of this GPU rather than using the default cuda:0.",
"func":1
},
{
"ref":"vipy.batch.Batch.checkpoint",
"url":40,
"doc":"Return the last checkpointed result. Useful for recovering from dask crashes for long jobs.",
"func":1
},
{
"ref":"vipy.ssim",
"url":41,
"doc":""
},
{
"ref":"vipy.ssim.SSIM",
"url":41,
"doc":"Structural similarity (SSIM) index"
},
{
"ref":"vipy.ssim.SSIM.match",
"url":41,
"doc":"Return a set of matching points in img1 and img2 in the form suitable for homography estimation",
"func":1
},
{
"ref":"vipy.ssim.SSIM.warp",
"url":41,
"doc":"Warp an image im_src with points src_pts to align with dst_pts",
"func":1
},
{
"ref":"vipy.ssim.SSIM.align",
"url":41,
"doc":"Return an image which is the warped version of img1 that aligns with img2",
"func":1
},
{
"ref":"vipy.ssim.SSIM.rgb2gray",
"url":41,
"doc":"Convert RGB image to grayscale; accesory function",
"func":1
},
{
"ref":"vipy.ssim.SSIM.similarity",
"url":41,
"doc":"Compute the Structural Similarity Index (SSIM) score of two images Inputs: 1) I1, image array 2) I2, image array 3) K1, float (optional, default=0.01) - constant 4) K2, float (optional, default=0.03) - constant Outputs: 1) out; float - SSIM score 2) ssim_map; 2-D image array - SSIM map",
"func":1
},
{
"ref":"vipy.ssim.SSIM.ssim",
"url":41,
"doc":"Return structural similarity score when aligning im_degraded to im_reference",
"func":1
},
{
"ref":"vipy.ssim.demo",
"url":41,
"doc":"Synthetically rotate an image by 10 degrees, and compute structural similarity with and without alignment, return images",
"func":1
},
{
"ref":"vipy.calibration",
"url":42,
"doc":""
},
{
"ref":"vipy.calibration.checkerboard",
"url":42,
"doc":"Create a 2D checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with black and white colors with black in upper left and bottom right. Returns: 2D numpy array np.array() float32 in [0,1]",
"func":1
},
{
"ref":"vipy.calibration.red_checkerboard_image",
"url":42,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with red colors. Returns: vipy.image.Image",
"func":1
},
{
"ref":"vipy.calibration.blue_checkerboard_image",
"url":42,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with blue colors. Returns: vipy.image.Image",
"func":1
},
{
"ref":"vipy.calibration.color_checkerboard_image",
"url":42,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with random colors Returns: vipy.image.Image",
"func":1
},
{
"ref":"vipy.calibration.color_checkerboard",
"url":42,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with random colors. Returns: 3D numpy array with three channels, uint8",
"func":1
},
{
"ref":"vipy.calibration.testimage",
"url":42,
"doc":"Return a  vipy.image.Image object of a superb owl from wikipedia",
"func":1
},
{
"ref":"vipy.calibration.owl",
"url":42,
"doc":"Return a  vipy.image.Image object of a superb owl from wikipedia",
"func":1
},
{
"ref":"vipy.calibration.randomimage",
"url":42,
"doc":"Return a uniform random RGB image as uint8 numpy array of size (m,n,3)",
"func":1
},
{
"ref":"vipy.calibration.testimg",
"url":42,
"doc":"Return a numpy array for  vipy.calibration.testimage of a superb owl",
"func":1
},
{
"ref":"vipy.calibration.tile",
"url":42,
"doc":"Create a 2D tile pattern with texture T repeated (nrows, ncols) times. Returns: float32 numpy array of size (T.shape[0] nrows, T.shape[1] ncols)",
"func":1
},
{
"ref":"vipy.calibration.greenblock",
"url":42,
"doc":"Return an (dx, dy, 3) numpy array float64 RGB channel image with green channel=1.0",
"func":1
},
{
"ref":"vipy.calibration.redblock",
"url":42,
"doc":"Return an (dx, dy, 3) numpy array float64 RGB channel image with red channel=1.0",
"func":1
},
{
"ref":"vipy.calibration.blueblock",
"url":42,
"doc":"Return an (dx, dy, 3) numpy array float64 RGB channel image with blue channel=1.0",
"func":1
},
{
"ref":"vipy.calibration.bayer",
"url":42,
"doc":"Return an (M,N) tiled texture pattern of [blue, green, blue, green; green red green red; blue green blue green, green red green red] such that each subblock element is (dx,dy) and the total repeated subblock size is (4 dx, 4 dy)",
"func":1
},
{
"ref":"vipy.calibration.bayer_image",
"url":42,
"doc":"Return  vipy.calibration.bayer as  vipy.image.Image ",
"func":1
},
{
"ref":"vipy.calibration.dots",
"url":42,
"doc":"Create a sequence of dots (e.g. single pixels on black background) separated by strides (dx, dy) with image of size (dx ncols,dy nrows) Returns: float32 numpy array in range [0,1]",
"func":1
},
{
"ref":"vipy.calibration.vertical_gradient",
"url":42,
"doc":"Create 2D linear ramp image with the ramp increasing from top to bottom. Returns: uint8 numpy array of size (nrows, ncols) with veritical gradient increasing over rows",
"func":1
},
{
"ref":"vipy.calibration.centersquare",
"url":42,
"doc":"Create a white square on a black background of an image of shape (width, height). Returns: numpy array of appropriate channels of float64 in [0,1]",
"func":1
},
{
"ref":"vipy.calibration.centersquare_image",
"url":42,
"doc":"Returns  vipy.image.Image for  vipy.calibration.centersquare numpy array",
"func":1
},
{
"ref":"vipy.calibration.circle",
"url":42,
"doc":"Create a white circle on a black background centered at (x,y) with radius r pixels, of shape (width, height). Returns: numpy array of approproate channels of float32 in [0,1]",
"func":1
},
{
"ref":"vipy.calibration.square",
"url":42,
"doc":"",
"func":1
},
{
"ref":"vipy.annotation",
"url":43,
"doc":""
},
{
"ref":"vipy.annotation.googlesearch",
"url":43,
"doc":"Return a list of image URLs from google image search associated with the provided tag",
"func":1
},
{
"ref":"vipy.annotation.basic_level_categories",
"url":43,
"doc":"Return a list of nouns from wordnet that can be used as an initial list of basic level object categories",
"func":1
},
{
"ref":"vipy.annotation.verbs",
"url":43,
"doc":"Return a list of verbs from verbnet that can be used to define a set of activities",
"func":1
},
{
"ref":"vipy.annotation.facebookprofilerange",
"url":43,
"doc":"",
"func":1
},
{
"ref":"vipy.annotation.facebookprofile",
"url":43,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader",
"url":44,
"doc":""
},
{
"ref":"vipy.downloader.generate_sha1",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.verify_sha1",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.verify_md5",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.generate_md5",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.scp",
"url":44,
"doc":"Download using pre-installed SSH keys where hostname is formatted 'scp: hostname.com:/path/to/file.jpg'",
"func":1
},
{
"ref":"vipy.downloader.s3",
"url":44,
"doc":"Thin wrapper for boto3",
"func":1
},
{
"ref":"vipy.downloader.s3_bucket",
"url":44,
"doc":"Thin wrapper for boto3",
"func":1
},
{
"ref":"vipy.downloader.download",
"url":44,
"doc":"Downloads file at  url and write it in  output_filename ",
"func":1
},
{
"ref":"vipy.downloader.unpack",
"url":44,
"doc":"Extracts  archive_filename in  output_dirname . Supported archives:          -  Zip formats and equivalents: .zip, .egg, .jar  Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2  gzip compressed files  non-tar .bz2",
"func":1
},
{
"ref":"vipy.downloader.download_and_unpack",
"url":44,
"doc":"Downloads and extracts archive in  url into  output_dirname . Note that  output_dirname has to exist and won't be created by this function.",
"func":1
},
{
"ref":"vipy.downloader.download_unpack_and_cleanup",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.unpack_and_cleanup",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ArchiveException",
"url":44,
"doc":"Base exception class for all archive errors."
},
{
"ref":"vipy.downloader.UnrecognizedArchiveFormat",
"url":44,
"doc":"Error raised when passed file is not a recognized archive format."
},
{
"ref":"vipy.downloader.extract",
"url":44,
"doc":"Unpack the tar or zip file at the specified  archive_filename to the directory specified by  output_dirname .",
"func":1
},
{
"ref":"vipy.downloader.Archive",
"url":44,
"doc":"The external API class that encapsulates an archive implementation."
},
{
"ref":"vipy.downloader.Archive.extract",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.Archive.list",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.BaseArchive",
"url":44,
"doc":"Base Archive class. Implementations should inherit this class."
},
{
"ref":"vipy.downloader.BaseArchive.extract",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.BaseArchive.list",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ExtractInterface",
"url":44,
"doc":"Interface class exposing common extract functionalities for standard-library-based Archive classes (e.g. based on modules like tarfile, zipfile)."
},
{
"ref":"vipy.downloader.ExtractInterface.extract",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.TarArchive",
"url":44,
"doc":"Interface class exposing common extract functionalities for standard-library-based Archive classes (e.g. based on modules like tarfile, zipfile)."
},
{
"ref":"vipy.downloader.TarArchive.list",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.TarArchive.get_members",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ZipArchive",
"url":44,
"doc":"Interface class exposing common extract functionalities for standard-library-based Archive classes (e.g. based on modules like tarfile, zipfile)."
},
{
"ref":"vipy.downloader.ZipArchive.list",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ZipArchive.get_members",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.videosearch",
"url":45,
"doc":""
},
{
"ref":"vipy.videosearch.isactiveyoutuber",
"url":45,
"doc":"Does the youtube user have any uploaded videos?",
"func":1
},
{
"ref":"vipy.videosearch.youtubeuser",
"url":45,
"doc":"return all unique /user/ urls returned for a search for a given query tag",
"func":1
},
{
"ref":"vipy.videosearch.is_downloadable_url",
"url":45,
"doc":"Check to see if youtube-dl can download the path, this requires exeecuting 'youtube-dl $URL -q -j' to see if the returncode is non-zero",
"func":1
},
{
"ref":"vipy.videosearch.youtube",
"url":45,
"doc":"Return a list of YouTube URLs for the given tag and optional channel",
"func":1
},
{
"ref":"vipy.videosearch.liveleak",
"url":45,
"doc":"",
"func":1
},
{
"ref":"vipy.videosearch.download",
"url":45,
"doc":"Use youtube-dl to download a video URL to a video file",
"func":1
},
{
"ref":"vipy.videosearch.bulkdownload",
"url":45,
"doc":"Use youtube-dl to download a list of video URLs to video files using the provided sprintf outpattern=/path/to/out_%d.mp4 where the index is provided by the URL list index",
"func":1
},
{
"ref":"vipy.visualize",
"url":46,
"doc":""
},
{
"ref":"vipy.visualize.montage",
"url":46,
"doc":"Create a montage image from the of provided list of vipy.image.Image objects. Args: imlist: [list, tuple] iterable of vipy.image.Image objects which is used to montage rowwise imgheight: [int] The height of each individual image in the grid imgwidth: [int] the width of each individual image in the grid gridrows: [int] The number of images per row, and number of images per column. This defines the montage shape. gridcols: [int] The number of images per row, and number of images per column. This defines the montage shape. aspectratio: [float]. This is an optional parameter which defines the shape of the montage as (gridcols/gridrows) without specifying the gridrows, gridcols input crop: [bool] If true, the vipy.image.Image objects should call crop(), which will trigger a load skip: [bool] Whether images should be skipped on failure to load(), useful for lazy downloading border: [int] a border of size in pixels surrounding each image in the grid border_bgr [tuple (r,g,b)]: the border color in a bgr color tuple (b, g, r) in [0,255], uint8 do_flush: [bool] flush the loaded images as garbage collection for large montages verbose: [bool] display optional verbose messages Returns: Return a vipy.image.Image montage which is of size (gridrows (imgheight + 2 border), gridcols (imgwidth+2 border ",
"func":1
},
{
"ref":"vipy.visualize.videomontage",
"url":46,
"doc":"Generate a video montage for the provided videos by creating a image montage for every frame. Args:  vipy.visualize.montage : See the args framerate: [float] the framerate of the montage video. All of the input videos are resampled to this common frame rate max_duration: [float] If not None, the maximum diuration of any element in the montage before it cycles Returns: An video file in outfile that shows each video tiled into a montage.   warning - This loads every video into memory, so be careful with large montages! - If max_duration is not set, then this can result in loading very long video elements in the montage, which will make for long videos",
"func":1
},
{
"ref":"vipy.visualize.urls",
"url":46,
"doc":"Given a list of public image URLs, create a stand-alone HTML page to show them all. Args: urllist: [list] A list of urls to display title: [str] The title of the html file imagewidth: [int] The size of the images in the page outfile: [str] The path to the output html file display: [bool] open the html file in the default system viewer when complete",
"func":1
},
{
"ref":"vipy.visualize.tohtml",
"url":46,
"doc":"Given a list of vipy.image.Image objects, show the images along with the dictionary contents of imdict (one per image) in a single standalone HTML file Args: imlist: [list  vipy.image.Image ] imdict: [list of dict] An optional list of dictionaries, such that each dictionary is visualized per image title: [str] The title of the html file imagewidth: [int] The size of the images in the page outfile: [str] The path to the output html file display: [bool] open the html file in the default system viewer when complete Returns: An html file in outfile that contains all the images as a standalone embedded file (no links or external files).",
"func":1
},
{
"ref":"vipy.visualize.imagelist",
"url":46,
"doc":"Given a list of image filenames wth absolute paths, copy to outdir, and create an index.html file that visualizes each.",
"func":1
},
{
"ref":"vipy.visualize.imagetuplelist",
"url":46,
"doc":"Imagelist but put tuples on same row",
"func":1
},
{
"ref":"vipy.activity",
"url":47,
"doc":""
},
{
"ref":"vipy.activity.Activity",
"url":47,
"doc":"vipy.object.Activity class An activity is a grouping of one or more tracks involved in an activity within a given startframe and endframe. The activity occurs at a given (startframe, endframe), where these frame indexes are extracted at the provided framerate. All objects are passed by reference with a globally unique track ID, for the tracks involved with the activity. This is done since tracks can exist after an activity completes, and that tracks should update the spatial transformation of boxes. The shortlabel defines the string shown on the visualization video. Valid constructors >>> t = vipy.object.Track(category='Person').add( . >>> a = vipy.object.Activity(startframe=0, endframe=10, category='Walking', tracks={t.id():t})"
},
{
"ref":"vipy.activity.Activity.hasattribute",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.confidence",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.from_json",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.duration",
"url":47,
"doc":"The length of the activity in seconds. Args: s: [float] The number of seconds for this activity, starting at the startframe centered: [bool] If true, then set the duration centered on the middle frame Returns: The duration in seconds of this activity object (if s=None) This activity object with the requested duration (if s!=None)",
"func":1
},
{
"ref":"vipy.activity.Activity.dict",
"url":47,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.activity.Activity.json",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.actorid",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.startframe",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.endframe",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.middleframe",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.framerate",
"url":47,
"doc":"Resample (startframe, endframe) from known original framerate set by constructor to be new framerate fps",
"func":1
},
{
"ref":"vipy.activity.Activity.category",
"url":47,
"doc":"Change the label (and shortlabel) to the new label (and shortlabel)",
"func":1
},
{
"ref":"vipy.activity.Activity.label",
"url":47,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.activity.Activity.shortlabel",
"url":47,
"doc":"A optional shorter label string to show in the visualizations",
"func":1
},
{
"ref":"vipy.activity.Activity.add",
"url":47,
"doc":"Add the track id for the track to this activity, so that if the track is changed externally it is reflected here",
"func":1
},
{
"ref":"vipy.activity.Activity.tracks",
"url":47,
"doc":"alias for trackids",
"func":1
},
{
"ref":"vipy.activity.Activity.cleartracks",
"url":47,
"doc":"Remove all track IDs from this activity",
"func":1
},
{
"ref":"vipy.activity.Activity.trackids",
"url":47,
"doc":"Return a set of track IDs associated with this activity",
"func":1
},
{
"ref":"vipy.activity.Activity.hasoverlap",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.isneighbor",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.hastrack",
"url":47,
"doc":"Is the track part of the activity?",
"func":1
},
{
"ref":"vipy.activity.Activity.append",
"url":47,
"doc":"Append newtrack to this activity and set as actorid()",
"func":1
},
{
"ref":"vipy.activity.Activity.trackfilter",
"url":47,
"doc":"Remove all tracks such that the lambda function f(trackid) resolves to False",
"func":1
},
{
"ref":"vipy.activity.Activity.replace",
"url":47,
"doc":"Replace oldtrack with newtrack if present in self._tracks. Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change",
"func":1
},
{
"ref":"vipy.activity.Activity.replaceid",
"url":47,
"doc":"Replace oldtrack with newtrack if present in self._tracks. Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change",
"func":1
},
{
"ref":"vipy.activity.Activity.during",
"url":47,
"doc":"Is frame during the time interval (startframe, endframe) inclusive?",
"func":1
},
{
"ref":"vipy.activity.Activity.during_interval",
"url":47,
"doc":"Is the activity occurring for any frames within the interval [startframe, endframe) (non-inclusive of endframe)?",
"func":1
},
{
"ref":"vipy.activity.Activity.union",
"url":47,
"doc":"Compute the union of the new activity other to this activity by updating the start and end times and computing the mean confidence. -Note: other must have the same category and track IDs as self -confweight [0,1]: the convex combinatiopn weight applied to the new activity",
"func":1
},
{
"ref":"vipy.activity.Activity.temporal_iou",
"url":47,
"doc":"Return the temporal intersection over union of two activities",
"func":1
},
{
"ref":"vipy.activity.Activity.offset",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.truncate",
"url":47,
"doc":"Truncate the activity so that it is between startframe and endframe",
"func":1
},
{
"ref":"vipy.activity.Activity.id",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.clone",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.temporalpad",
"url":47,
"doc":"Add a temporal pad of df=(before frames, after frames) or df=pad frames to the start and end of the activity. The padded start frame may be negative.",
"func":1
},
{
"ref":"vipy.activity.Activity.padto",
"url":47,
"doc":"Add a symmetric temporal pad so that the activity is at least t seconds long",
"func":1
},
{
"ref":"vipy.activity.Activity.disjoint",
"url":47,
"doc":"Enforce disjoint activities with other by shifting the endframe or startframe of self to not overlap if they share the same tracks. Other may be an Activity() or list of Activity() if strict=True, then throw an exception if other or self is fully contained with the other, resulting in degenerate activity after disjoint",
"func":1
},
{
"ref":"vipy.activity.Activity.temporal_distance",
"url":47,
"doc":"Return the temporal distance in frames between self and other which is the minimum frame difference between the end of one to the start of the other, or zero if they overlap",
"func":1
},
{
"ref":"vipy.geometry",
"url":29,
"doc":""
},
{
"ref":"vipy.geometry.covariance_to_ellipse",
"url":29,
"doc":"2x2 covariance matrix to ellipse (major_axis_length, minor_axis_length, angle_in_radians)",
"func":1
},
{
"ref":"vipy.geometry.dehomogenize",
"url":29,
"doc":"Convert 3x1 homogenous point (x,y,h) to 2x1 non-homogenous point (x/h, y/h)",
"func":1
},
{
"ref":"vipy.geometry.homogenize",
"url":29,
"doc":"Convert 2xN non-homogenous points (x,y) to 3xN non-homogenous point (x, y, 1)",
"func":1
},
{
"ref":"vipy.geometry.apply_homography",
"url":29,
"doc":"Apply a 3x3 homography H to non-homogenous point p and return a transformed point",
"func":1
},
{
"ref":"vipy.geometry.similarity_transform_2x3",
"url":29,
"doc":"Return a 2x3 similarity transform with rotation r (radians), scale s and origin c=(x,y)",
"func":1
},
{
"ref":"vipy.geometry.similarity_transform",
"url":29,
"doc":"Return a 3x3 similarity transformation with translation tuple txy=(x,y), rotation r (radians, scale=s",
"func":1
},
{
"ref":"vipy.geometry.affine_transform",
"url":29,
"doc":"Compose and return a 3x3 affine transformation for translation txy=(0,0), rotation r (radians), scalex=sx, scaley=sy, shearx=kx, sheary=ky. Usage: >>> A = vipy.geometry.affine_transform(r=np.pi/4) >>> vipy.image.Image(array=vipy.geometry.imtransform(im.array(), A), colorspace='float') Equivalently: >>> im = vipy.image.RandomImage().affine_transform(A)",
"func":1
},
{
"ref":"vipy.geometry.random_affine_transform",
"url":29,
"doc":"Return a random 3x3 affine transformation matrix for the provided ranges, inputs must be tuples",
"func":1
},
{
"ref":"vipy.geometry.imtransform",
"url":29,
"doc":"Transform an numpy array image (MxNx3) following the affine or similiarity transformation A",
"func":1
},
{
"ref":"vipy.geometry.normalize",
"url":29,
"doc":"Given a vector x, return the vector unit normalized as float64",
"func":1
},
{
"ref":"vipy.geometry.imagebox",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox",
"url":29,
"doc":"Core bounding box class with flexible constructors in this priority order: (xmin,ymin,xmax,ymax) (xmin,ymin,width,height) (centroid[0],centroid[1],width,height) (xcentroid,ycentroid,width,height) xywh=(xmin,ymin,width,height) ulbr=(xmin,ymin,xmax,ymax) bounding rectangle of binary mask image"
},
{
"ref":"vipy.geometry.BoundingBox.cast",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.from_json",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dict",
"url":29,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.json",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.clone",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bbclone",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xmin",
"url":29,
"doc":"x coordinate of upper left corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ul",
"url":29,
"doc":"Upper left coordinate (x,y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ulx",
"url":29,
"doc":"Upper left coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.uly",
"url":29,
"doc":"Upper left coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ur",
"url":29,
"doc":"Upper right coordinate (x,y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.urx",
"url":29,
"doc":"Upper right coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ury",
"url":29,
"doc":"Upper right coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ll",
"url":29,
"doc":"Lower left coordinate (x,y), synonym for bl()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bl",
"url":29,
"doc":"Bottom left coordinate (x,y), synonym for ll()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.blx",
"url":29,
"doc":"Bottom left coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bly",
"url":29,
"doc":"Bottom left coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.lr",
"url":29,
"doc":"Lower right coordinate (x,y), synonym for br()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.br",
"url":29,
"doc":"Bottom right coordinate (x,y), synonym for lr()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.brx",
"url":29,
"doc":"Bottom right coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bry",
"url":29,
"doc":"Bottom right coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ymin",
"url":29,
"doc":"y coordinate of upper left corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xmax",
"url":29,
"doc":"x coordinate of lower right corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ymax",
"url":29,
"doc":"y coordinate of lower right corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.upperleft",
"url":29,
"doc":"Return the (x,y) upper left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bottomleft",
"url":29,
"doc":"Return the (x,y) lower left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.upperright",
"url":29,
"doc":"Return the (x,y) upper right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bottomright",
"url":29,
"doc":"Return the (x,y) lower right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isinteger",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.int",
"url":29,
"doc":"Convert corners to integer with rounding, in-place update",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.float",
"url":29,
"doc":"Convert corners to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.significant_digits",
"url":29,
"doc":"Convert corners to have at most n significant digits for efficient JSON storage",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.translate",
"url":29,
"doc":"Translate the bounding box by dx in x and dy in y",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.to_origin",
"url":29,
"doc":"Translate the bounding box so that (xmin, ymin) = (0,0)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.set_origin",
"url":29,
"doc":"Set the origin of the coordinates of this bounding box to be relative to the upper left of the other bounding box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.offset",
"url":29,
"doc":"Alias for translate",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.invalid",
"url":29,
"doc":"Is the box a valid bounding box?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.valid",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isvalid",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isdegenerate",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isnonnegative",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.width",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bbwidth",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.setwidth",
"url":29,
"doc":"Set new width keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.setheight",
"url":29,
"doc":"Set new height keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.height",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bbheight",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.centroid",
"url":29,
"doc":"(x,y) tuple of centroid position of bounding box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.x_centroid",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xcentroid",
"url":29,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.centroid_x",
"url":29,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.y_centroid",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ycentroid",
"url":29,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.centroid_y",
"url":29,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.area",
"url":29,
"doc":"Return the area=width height of the bounding box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.to_xywh",
"url":29,
"doc":"Return bounding box corners as (x,y,width,height) tuple",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xywh",
"url":29,
"doc":"Alias for to_xywh",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.cxywh",
"url":29,
"doc":"Return or set bounding box corners as (centroidx,centroidy,width,height) tuple",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ulbr",
"url":29,
"doc":"Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.to_ulbr",
"url":29,
"doc":"Alias for ulbr()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dx",
"url":29,
"doc":"Offset bounding box by same xmin as provided box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dy",
"url":29,
"doc":"Offset bounding box by ymin of provided box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.sqdist",
"url":29,
"doc":"Squared Euclidean distance between upper left corners of two bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dist",
"url":29,
"doc":"Distance between centroids of two bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.pdist",
"url":29,
"doc":"Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.iou",
"url":29,
"doc":"area of intersection / area of union",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.intersection_over_union",
"url":29,
"doc":"Alias for iou",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.area_of_intersection",
"url":29,
"doc":"area of intersection",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.area_of_union",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.cover",
"url":29,
"doc":"Fraction of this bounding box intersected by other bbox (bb)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxcover",
"url":29,
"doc":"The maximum cover of self to bb and bb to self",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.shapeiou",
"url":29,
"doc":"Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.intersection",
"url":29,
"doc":"Intersection of two bounding boxes, throw an error on degeneracy of intersection result (if strict=True)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.hasintersection",
"url":29,
"doc":"Return true if self and bb overlap by any amount, or by the cover threshold (if provided) or the iou threshold (if provided). This is a convenience function that allows for shared computation for fast non-maximum suppression.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.union",
"url":29,
"doc":"Union of one or more bounding boxes with this box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isinside",
"url":29,
"doc":"Is this boundingbox fully within the provided bounding box?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ispointinside",
"url":29,
"doc":"Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilate",
"url":29,
"doc":"Change scale of bounding box keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilatepx",
"url":29,
"doc":"Dilate by a given pixel amount on all sides, keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilate_height",
"url":29,
"doc":"Change scale of bounding box in y direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilate_width",
"url":29,
"doc":"Change scale of bounding box in x direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.top",
"url":29,
"doc":"Make top of box taller (closer to top of image) by an offset dy",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bottom",
"url":29,
"doc":"Make bottom of box taller (closer to bottom of image) by an offset dy",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.left",
"url":29,
"doc":"Make left of box wider (closer to left side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.right",
"url":29,
"doc":"Make right of box wider (closer to right side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.rescale",
"url":29,
"doc":"Multiply the box corners by a scale factor",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.scalex",
"url":29,
"doc":"Multiply the box corners in the x dimension by a scale factor",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.scaley",
"url":29,
"doc":"Multiply the box corners in the y dimension by a scale factor",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.resize",
"url":29,
"doc":"Change the aspect ratio width and height of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.rot90cw",
"url":29,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.rot90ccw",
"url":29,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.fliplr",
"url":29,
"doc":"Flip the box left/right consistent with fliplr of the provided img (or consistent with the image width)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.flipud",
"url":29,
"doc":"Flip the box up/down consistent with flipud of the provided img (or consistent with the image height)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.imscale",
"url":29,
"doc":"Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxsquare",
"url":29,
"doc":"Set the bounding box to be square by setting width and height to the maximum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxsquareif",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.issquare",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.iseven",
"url":29,
"doc":"Are all corners even number integers?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.even",
"url":29,
"doc":"Force all corners to be even number integers. This is helpful for FFMPEG crop filters.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.minsquare",
"url":29,
"doc":"Set the bounding box to be square by setting width and height to the minimum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.hasoverlap",
"url":29,
"doc":"Does the bounding box intersect with the provided image rectangle?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isinterior",
"url":29,
"doc":"Is this boundingbox fully within the provided image rectangle?  If border in [0,1], then the image is dilated by a border percentage prior to computing interior, useful to check if self is near the image edge  If border=0.8, then the image rectangle is dilated by 80% (smaller) keeping the centroid constant.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.iminterior",
"url":29,
"doc":"Transform bounding box to be interior to the image rectangle with shape (W,H). Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.imclip",
"url":29,
"doc":"Clip bounding box to image rectangle [0,0,width,height] or img.shape=(width, height) and, throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.imclipshape",
"url":29,
"doc":"Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.convexhull",
"url":29,
"doc":"Given a set of points  x1,y1],[x2,xy], .], return the bounding rectangle, typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.aspectratio",
"url":29,
"doc":"Return the aspect ratio (width/height) of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.shape",
"url":29,
"doc":"Return the (height, width) tuple for the box shape",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.mindimension",
"url":29,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.mindim",
"url":29,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxdim",
"url":29,
"doc":"Return max(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ellipse",
"url":29,
"doc":"Convert the boundingbox to a vipy.geometry.Ellipse object",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.average",
"url":29,
"doc":"Compute the average bounding box between self and other, and set self to the average. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.averageshape",
"url":29,
"doc":"Compute the average bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.medianshape",
"url":29,
"doc":"Compute the median bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.shapedist",
"url":29,
"doc":"L1 distance between (width,height) of two boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.affine",
"url":29,
"doc":"Apply an 2x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.projective",
"url":29,
"doc":"Apply an 3x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.crop",
"url":29,
"doc":"Crop an HxW 2D numpy image, HxWxC 3D numpy image, or NxHxWxC 4D numpy image array using this bounding box applied to HxW dimensions. Crop is performed in-place.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.confidence",
"url":29,
"doc":"Bounding boxes do not have confidences, use vipy.object.Detection()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.grid",
"url":29,
"doc":"Split a bounding box into the smallest grid of non-overlapping bounding boxes such that the union is the original box",
"func":1
},
{
"ref":"vipy.geometry.Ellipse",
"url":29,
"doc":"Ellipse parameterization, for length of semimajor (half width of ellipse) and semiminor axis (half height), center point and angle phi in radians"
},
{
"ref":"vipy.geometry.Ellipse.dict",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.area",
"url":29,
"doc":"Area of ellipse",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.center",
"url":29,
"doc":"Return centroid",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.centroid",
"url":29,
"doc":"Alias for center",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.axes",
"url":29,
"doc":"Return the (major,minor) axis lengths",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.angle",
"url":29,
"doc":"Return the angle phi (in degrees)",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.rescale",
"url":29,
"doc":"Scale ellipse by scale factor",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.boundingbox",
"url":29,
"doc":"Estimate an equivalent bounding box based on scaling to a common area. Note, this does not factor in rotation. (c l) (c w) = a_e  > c = sqrt(a_e / a_r)",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.inside",
"url":29,
"doc":"Return true if a point p=(x,y) is inside the ellipse",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.mask",
"url":29,
"doc":"Return a binary mask of size equal to the bounding box such that the pixels correspond to the interior of the ellipse",
"func":1
},
{
"ref":"vipy.geometry.union",
"url":29,
"doc":"Return the union of a list of vipy.geometry.BoundingBox",
"func":1
},
{
"ref":"vipy.object",
"url":48,
"doc":""
},
{
"ref":"vipy.object.Detection",
"url":48,
"doc":"vipy.object.Detection class This class represent a single object detection in the form a bounding box with a label and confidence. The constructor of this class follows a subset of the constructor patterns of vipy.geometry.BoundingBox >>> d = vipy.object.Detection(category='Person', xmin=0, ymin=0, width=50, height=100) >>> d = vipy.object.Detection(label='Person', xmin=0, ymin=0, width=50, height=100)  \"label\" is an alias for \"category\" >>> d = vipy.object.Detection(label='John Doe', shortlabel='Person', xmin=0, ymin=0, width=50, height=100)  shortlabel is displayed >>> d = vipy.object.Detection(label='Person', xywh=[0,0,50,100]) >>> d = vupy.object.Detection( ., id=True)  generate a unique UUID for this detection retrievable with d.id()"
},
{
"ref":"vipy.object.Detection.cast",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.from_json",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.dict",
"url":48,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.object.Detection.json",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.nocategory",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.noshortlabel",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.category",
"url":48,
"doc":"Update the category and shortlabel (optional) of the detection",
"func":1
},
{
"ref":"vipy.object.Detection.shortlabel",
"url":48,
"doc":"A optional shorter label string to show in the visualizations, defaults to category()",
"func":1
},
{
"ref":"vipy.object.Detection.label",
"url":48,
"doc":"Alias for category to update both category and shortlabel",
"func":1
},
{
"ref":"vipy.object.Detection.id",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.clone",
"url":48,
"doc":"Copy the object, if deep=True, then include a deep copy of the attribute dictionary, else a shallow copy",
"func":1
},
{
"ref":"vipy.object.Detection.confidence",
"url":48,
"doc":"Bounding boxes do not have confidences, use vipy.object.Detection()",
"func":1
},
{
"ref":"vipy.object.Detection.hasattribute",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.getattribute",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.setattribute",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.delattribute",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.noattributes",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.xmin",
"url":29,
"doc":"x coordinate of upper left corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.object.Detection.ul",
"url":29,
"doc":"Upper left coordinate (x,y)",
"func":1
},
{
"ref":"vipy.object.Detection.ulx",
"url":29,
"doc":"Upper left coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.uly",
"url":29,
"doc":"Upper left coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.ur",
"url":29,
"doc":"Upper right coordinate (x,y)",
"func":1
},
{
"ref":"vipy.object.Detection.urx",
"url":29,
"doc":"Upper right coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.ury",
"url":29,
"doc":"Upper right coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.ll",
"url":29,
"doc":"Lower left coordinate (x,y), synonym for bl()",
"func":1
},
{
"ref":"vipy.object.Detection.bl",
"url":29,
"doc":"Bottom left coordinate (x,y), synonym for ll()",
"func":1
},
{
"ref":"vipy.object.Detection.blx",
"url":29,
"doc":"Bottom left coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.bly",
"url":29,
"doc":"Bottom left coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.lr",
"url":29,
"doc":"Lower right coordinate (x,y), synonym for br()",
"func":1
},
{
"ref":"vipy.object.Detection.br",
"url":29,
"doc":"Bottom right coordinate (x,y), synonym for lr()",
"func":1
},
{
"ref":"vipy.object.Detection.brx",
"url":29,
"doc":"Bottom right coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.bry",
"url":29,
"doc":"Bottom right coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.ymin",
"url":29,
"doc":"y coordinate of upper left corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.object.Detection.xmax",
"url":29,
"doc":"x coordinate of lower right corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.object.Detection.ymax",
"url":29,
"doc":"y coordinate of lower right corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.object.Detection.upperleft",
"url":29,
"doc":"Return the (x,y) upper left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.bottomleft",
"url":29,
"doc":"Return the (x,y) lower left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.upperright",
"url":29,
"doc":"Return the (x,y) upper right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.bottomright",
"url":29,
"doc":"Return the (x,y) lower right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.int",
"url":29,
"doc":"Convert corners to integer with rounding, in-place update",
"func":1
},
{
"ref":"vipy.object.Detection.float",
"url":29,
"doc":"Convert corners to float",
"func":1
},
{
"ref":"vipy.object.Detection.significant_digits",
"url":29,
"doc":"Convert corners to have at most n significant digits for efficient JSON storage",
"func":1
},
{
"ref":"vipy.object.Detection.translate",
"url":29,
"doc":"Translate the bounding box by dx in x and dy in y",
"func":1
},
{
"ref":"vipy.object.Detection.to_origin",
"url":29,
"doc":"Translate the bounding box so that (xmin, ymin) = (0,0)",
"func":1
},
{
"ref":"vipy.object.Detection.set_origin",
"url":29,
"doc":"Set the origin of the coordinates of this bounding box to be relative to the upper left of the other bounding box",
"func":1
},
{
"ref":"vipy.object.Detection.offset",
"url":29,
"doc":"Alias for translate",
"func":1
},
{
"ref":"vipy.object.Detection.invalid",
"url":29,
"doc":"Is the box a valid bounding box?",
"func":1
},
{
"ref":"vipy.object.Detection.setwidth",
"url":29,
"doc":"Set new width keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.setheight",
"url":29,
"doc":"Set new height keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.centroid",
"url":29,
"doc":"(x,y) tuple of centroid position of bounding box",
"func":1
},
{
"ref":"vipy.object.Detection.xcentroid",
"url":29,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.centroid_x",
"url":29,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.ycentroid",
"url":29,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.centroid_y",
"url":29,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.area",
"url":29,
"doc":"Return the area=width height of the bounding box",
"func":1
},
{
"ref":"vipy.object.Detection.to_xywh",
"url":29,
"doc":"Return bounding box corners as (x,y,width,height) tuple",
"func":1
},
{
"ref":"vipy.object.Detection.xywh",
"url":29,
"doc":"Alias for to_xywh",
"func":1
},
{
"ref":"vipy.object.Detection.cxywh",
"url":29,
"doc":"Return or set bounding box corners as (centroidx,centroidy,width,height) tuple",
"func":1
},
{
"ref":"vipy.object.Detection.ulbr",
"url":29,
"doc":"Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)",
"func":1
},
{
"ref":"vipy.object.Detection.to_ulbr",
"url":29,
"doc":"Alias for ulbr()",
"func":1
},
{
"ref":"vipy.object.Detection.dx",
"url":29,
"doc":"Offset bounding box by same xmin as provided box",
"func":1
},
{
"ref":"vipy.object.Detection.dy",
"url":29,
"doc":"Offset bounding box by ymin of provided box",
"func":1
},
{
"ref":"vipy.object.Detection.sqdist",
"url":29,
"doc":"Squared Euclidean distance between upper left corners of two bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.dist",
"url":29,
"doc":"Distance between centroids of two bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.pdist",
"url":29,
"doc":"Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box",
"func":1
},
{
"ref":"vipy.object.Detection.iou",
"url":29,
"doc":"area of intersection / area of union",
"func":1
},
{
"ref":"vipy.object.Detection.intersection_over_union",
"url":29,
"doc":"Alias for iou",
"func":1
},
{
"ref":"vipy.object.Detection.area_of_intersection",
"url":29,
"doc":"area of intersection",
"func":1
},
{
"ref":"vipy.object.Detection.cover",
"url":29,
"doc":"Fraction of this bounding box intersected by other bbox (bb)",
"func":1
},
{
"ref":"vipy.object.Detection.maxcover",
"url":29,
"doc":"The maximum cover of self to bb and bb to self",
"func":1
},
{
"ref":"vipy.object.Detection.shapeiou",
"url":29,
"doc":"Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation",
"func":1
},
{
"ref":"vipy.object.Detection.intersection",
"url":29,
"doc":"Intersection of two bounding boxes, throw an error on degeneracy of intersection result (if strict=True)",
"func":1
},
{
"ref":"vipy.object.Detection.hasintersection",
"url":29,
"doc":"Return true if self and bb overlap by any amount, or by the cover threshold (if provided) or the iou threshold (if provided). This is a convenience function that allows for shared computation for fast non-maximum suppression.",
"func":1
},
{
"ref":"vipy.object.Detection.union",
"url":29,
"doc":"Union of one or more bounding boxes with this box",
"func":1
},
{
"ref":"vipy.object.Detection.isinside",
"url":29,
"doc":"Is this boundingbox fully within the provided bounding box?",
"func":1
},
{
"ref":"vipy.object.Detection.ispointinside",
"url":29,
"doc":"Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?",
"func":1
},
{
"ref":"vipy.object.Detection.dilate",
"url":29,
"doc":"Change scale of bounding box keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.dilatepx",
"url":29,
"doc":"Dilate by a given pixel amount on all sides, keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.dilate_height",
"url":29,
"doc":"Change scale of bounding box in y direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.dilate_width",
"url":29,
"doc":"Change scale of bounding box in x direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.top",
"url":29,
"doc":"Make top of box taller (closer to top of image) by an offset dy",
"func":1
},
{
"ref":"vipy.object.Detection.bottom",
"url":29,
"doc":"Make bottom of box taller (closer to bottom of image) by an offset dy",
"func":1
},
{
"ref":"vipy.object.Detection.left",
"url":29,
"doc":"Make left of box wider (closer to left side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.object.Detection.right",
"url":29,
"doc":"Make right of box wider (closer to right side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.object.Detection.rescale",
"url":29,
"doc":"Multiply the box corners by a scale factor",
"func":1
},
{
"ref":"vipy.object.Detection.scalex",
"url":29,
"doc":"Multiply the box corners in the x dimension by a scale factor",
"func":1
},
{
"ref":"vipy.object.Detection.scaley",
"url":29,
"doc":"Multiply the box corners in the y dimension by a scale factor",
"func":1
},
{
"ref":"vipy.object.Detection.resize",
"url":29,
"doc":"Change the aspect ratio width and height of the box",
"func":1
},
{
"ref":"vipy.object.Detection.rot90cw",
"url":29,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.object.Detection.rot90ccw",
"url":29,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.object.Detection.fliplr",
"url":29,
"doc":"Flip the box left/right consistent with fliplr of the provided img (or consistent with the image width)",
"func":1
},
{
"ref":"vipy.object.Detection.flipud",
"url":29,
"doc":"Flip the box up/down consistent with flipud of the provided img (or consistent with the image height)",
"func":1
},
{
"ref":"vipy.object.Detection.imscale",
"url":29,
"doc":"Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image",
"func":1
},
{
"ref":"vipy.object.Detection.maxsquare",
"url":29,
"doc":"Set the bounding box to be square by setting width and height to the maximum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.iseven",
"url":29,
"doc":"Are all corners even number integers?",
"func":1
},
{
"ref":"vipy.object.Detection.even",
"url":29,
"doc":"Force all corners to be even number integers. This is helpful for FFMPEG crop filters.",
"func":1
},
{
"ref":"vipy.object.Detection.minsquare",
"url":29,
"doc":"Set the bounding box to be square by setting width and height to the minimum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.hasoverlap",
"url":29,
"doc":"Does the bounding box intersect with the provided image rectangle?",
"func":1
},
{
"ref":"vipy.object.Detection.isinterior",
"url":29,
"doc":"Is this boundingbox fully within the provided image rectangle?  If border in [0,1], then the image is dilated by a border percentage prior to computing interior, useful to check if self is near the image edge  If border=0.8, then the image rectangle is dilated by 80% (smaller) keeping the centroid constant.",
"func":1
},
{
"ref":"vipy.object.Detection.iminterior",
"url":29,
"doc":"Transform bounding box to be interior to the image rectangle with shape (W,H). Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit",
"func":1
},
{
"ref":"vipy.object.Detection.imclip",
"url":29,
"doc":"Clip bounding box to image rectangle [0,0,width,height] or img.shape=(width, height) and, throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.object.Detection.imclipshape",
"url":29,
"doc":"Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.object.Detection.convexhull",
"url":29,
"doc":"Given a set of points  x1,y1],[x2,xy], .], return the bounding rectangle, typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.aspectratio",
"url":29,
"doc":"Return the aspect ratio (width/height) of the box",
"func":1
},
{
"ref":"vipy.object.Detection.shape",
"url":29,
"doc":"Return the (height, width) tuple for the box shape",
"func":1
},
{
"ref":"vipy.object.Detection.mindimension",
"url":29,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.mindim",
"url":29,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.maxdim",
"url":29,
"doc":"Return max(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.ellipse",
"url":29,
"doc":"Convert the boundingbox to a vipy.geometry.Ellipse object",
"func":1
},
{
"ref":"vipy.object.Detection.average",
"url":29,
"doc":"Compute the average bounding box between self and other, and set self to the average. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.averageshape",
"url":29,
"doc":"Compute the average bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.medianshape",
"url":29,
"doc":"Compute the median bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.shapedist",
"url":29,
"doc":"L1 distance between (width,height) of two boxes",
"func":1
},
{
"ref":"vipy.object.Detection.affine",
"url":29,
"doc":"Apply an 2x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.object.Detection.projective",
"url":29,
"doc":"Apply an 3x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.object.Detection.crop",
"url":29,
"doc":"Crop an HxW 2D numpy image, HxWxC 3D numpy image, or NxHxWxC 4D numpy image array using this bounding box applied to HxW dimensions. Crop is performed in-place.",
"func":1
},
{
"ref":"vipy.object.Detection.grid",
"url":29,
"doc":"Split a bounding box into the smallest grid of non-overlapping bounding boxes such that the union is the original box",
"func":1
},
{
"ref":"vipy.object.Track",
"url":48,
"doc":"vipy.object.Track class A track represents one or more labeled bounding boxes of an object instance through time. A track is defined as a finite set of labeled boxes observed at keyframes, which are discrete observations of this instance. Each keyframe has an associated vipy.geometry.BoundingBox() which defines the spatial bounding box of the instance in this keyframe. The kwarg \"interpolation\" defines how the track is interpolated between keyframes, and the kwarg \"boundary\" defines how the track is interpolated outside the (min,max) of the keyframes. Valid constructors are: >>> t = vipy.object.Track(keyframes=[0,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person') >>> t = vipy.object.Track(keyframes=[0,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person', interpolation='linear') >>> t = vipy.object.Track(keyframes=[10,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person', boundary='strict') Tracks can be constructed incrementally: >>> t = vipy.object.Track('Person') >>> t.add(0, vipy.geometry.BoundingBox(0,0,10,10 >>> t.add(100, vipy.geometry.BoundingBox(0,0,20,20 Tracks can be resampled at a new framerate, as long as the framerate is known when the keyframes are extracted >>> t.framerate(newfps)"
},
{
"ref":"vipy.object.Track.from_json",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.json",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.isempty",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.confidence",
"url":48,
"doc":"The confidence of a track is the mean confidence of all (or just last=last frames, or samples=samples uniformly spaced) keyboxes (if confidences are available) else 0",
"func":1
},
{
"ref":"vipy.object.Track.isdegenerate",
"url":48,
"doc":"Is the track degenerate? A degenerate track has: - Unequal length keyboxes and keyframes - length zero track - Non increasing keyframes - Invalid keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.dict",
"url":48,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.object.Track.add",
"url":48,
"doc":"Add a new keyframe and associated box to track, preserve sorted order of keyframes. If keyframe is already in track, throw an exception. In this case use update() instead -strict [bool]: If box is degenerate, throw an exception if strict=True, otherwise just don't add it  note The BoundingBox is added by reference. If you want to this to be a copy, pass in bbox.clone()",
"func":1
},
{
"ref":"vipy.object.Track.update",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.replace",
"url":48,
"doc":"Replace the keyframe and associated box(es), preserve sorted order of keyframes",
"func":1
},
{
"ref":"vipy.object.Track.delete",
"url":48,
"doc":"Replace a keyframe and associated box to track, preserve sorted order of keyframes",
"func":1
},
{
"ref":"vipy.object.Track.keyframes",
"url":48,
"doc":"Return keyframe frame indexes where there are track observations",
"func":1
},
{
"ref":"vipy.object.Track.num_keyframes",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.keyboxes",
"url":48,
"doc":"Return keyboxes where there are track observations",
"func":1
},
{
"ref":"vipy.object.Track.meanshape",
"url":48,
"doc":"Return the mean (width,height) of the box during the track, or None if the track is degenerate",
"func":1
},
{
"ref":"vipy.object.Track.meanbox",
"url":48,
"doc":"Return the mean bounding box during the track, or None if the track is degenerate",
"func":1
},
{
"ref":"vipy.object.Track.shapevariance",
"url":48,
"doc":"Return the variance (width, height) of the box shape relative to  vipy.object.Track.meanbox during the track or None if the track is degenerate. This is useful for filtering spurious tracks where the aspect ratio changes rapidly and randomly Returns: (width_variance, height_variance) of the box shape during the track (or None)",
"func":1
},
{
"ref":"vipy.object.Track.framerate",
"url":48,
"doc":"Resample keyframes from known original framerate set by constructor to be new framerate fps. Args: fps: [float] The new frame rate in frames per second speed: [float] An optional speed factor which will multiply the current framerate by this factor (e.g. speed=2  > fps=self.framerate() 2) Returns: This track object with the keyframes resampled to the new framerate",
"func":1
},
{
"ref":"vipy.object.Track.startframe",
"url":48,
"doc":"Return the startframe of the track or None if there are no keyframes. The frame index is relative to the framerate set in the constructor.",
"func":1
},
{
"ref":"vipy.object.Track.endframe",
"url":48,
"doc":"Return the endframe of the track or None if there are no keyframes. The frame index is relative to the framerate set in the constructor.",
"func":1
},
{
"ref":"vipy.object.Track.linear_interpolation",
"url":48,
"doc":"Linear bounding box interpolation at frame=k given observed boxes (x,y,w,h) at keyframes. This returns a  vipy.object.Detection which is the interpolation of the  vipy.object.Track at frame k - If self._boundary='extend', then boxes are repeated if the interpolation is outside the keyframes - If self._boundary='strict', then interpolation returns None if the interpolation is outside the keyframes  note The returned object is not cloned when possible for speed purposes, be careful when modifying this object. clone() if necessary",
"func":1
},
{
"ref":"vipy.object.Track.category",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.label",
"url":48,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.object.Track.shortlabel",
"url":48,
"doc":"A optional shorter label string to show as a caption in visualizations",
"func":1
},
{
"ref":"vipy.object.Track.during",
"url":48,
"doc":"Does the track contain a keyframe during the time interval (startframe, endframe) inclusive?",
"func":1
},
{
"ref":"vipy.object.Track.during_interval",
"url":48,
"doc":"Does the track contain a keyframe during the inclusive frame interval (startframe, endframe)?  note The start and end frames are inclusive",
"func":1
},
{
"ref":"vipy.object.Track.offset",
"url":48,
"doc":"Apply a temporal shift of dt frames, and a spatial shift of (dx, dy) pixels. Args: dt: [int] frame offset dx: [float] horizontal spatial offset dy: [float] vertical spatial offset Returns: This box updated in place",
"func":1
},
{
"ref":"vipy.object.Track.uncrop",
"url":48,
"doc":"Apply a transformation to the track that will undo a crop of a bounding box with an optional scale factor. A typical operation is as follows. A video is cropped and zommed in order to run a detector on a region of interest. However, we want to align the resulting tracks on the original video before the crop and zoom. Args: bb: [ vipy.geometry.BoundingBox ]. A bounding box which was used to crop this track s: [float] A scale factor applied after the bounding box crop Returns: This track after undoing the scale and crop",
"func":1
},
{
"ref":"vipy.object.Track.frameoffset",
"url":48,
"doc":"Offset boxes by (dx,dy) in each frame. This is used to apply a different offset for each frame. To apply one offset to all frames, use  vipy.object.offset . Args: dx: [list] This should be a list of frame offsets at each keyframe the same length as the number of keyboxes dy: [list] This should be a list of frame offsets at each keyframe the same length as the number of keyboxes Returns: This track updated in place",
"func":1
},
{
"ref":"vipy.object.Track.truncate",
"url":48,
"doc":"Truncate a track so that any keyframes less than startframe or greater than endframe (inclusive) are removed. Interpolate keyboxes at (startframe, endframe) endpoints. Args: startframe: [int] The startframe of the truncation relative to the track framerate. All keyframes less than or equal to startframe are included. If the keyframe does not exist at startframe, one is interpolated and added. endframe: [int] The endframe of the truncation relative to the track framerate. All keyframes greater than or equal to the endframe are included. If the keyfrmae does not exist at endframe, one is interpolated and added. Returns: This track such that all keyboxes  = endframe are removed.  note The startframe and endframe for truncation are inclusive.",
"func":1
},
{
"ref":"vipy.object.Track.rescale",
"url":48,
"doc":"Rescale track boxes by scale factor s",
"func":1
},
{
"ref":"vipy.object.Track.scale",
"url":48,
"doc":"Alias for rescale",
"func":1
},
{
"ref":"vipy.object.Track.scalex",
"url":48,
"doc":"Rescale track boxes by scale factor sx",
"func":1
},
{
"ref":"vipy.object.Track.scaley",
"url":48,
"doc":"Rescale track boxes by scale factor sx",
"func":1
},
{
"ref":"vipy.object.Track.dilate",
"url":48,
"doc":"Dilate track boxes by scale factor s",
"func":1
},
{
"ref":"vipy.object.Track.rot90cw",
"url":48,
"doc":"Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent",
"func":1
},
{
"ref":"vipy.object.Track.rot90ccw",
"url":48,
"doc":"Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent",
"func":1
},
{
"ref":"vipy.object.Track.fliplr",
"url":48,
"doc":"Flip an image left and right (mirror about vertical axis)",
"func":1
},
{
"ref":"vipy.object.Track.flipud",
"url":48,
"doc":"Flip an image left and right (mirror about vertical axis)",
"func":1
},
{
"ref":"vipy.object.Track.id",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.clone",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.clone_during",
"url":48,
"doc":"Clone a track during a specific interval (startframe, endframe) relative to the framerate of the track. - This is useful for copying a small segment of a long track without the expense of copying the whole track. - All keyframes and keyboxes not in (startframe, endframe) are not copied. - Boundary keyframes are copied to enable proper interpolation.",
"func":1
},
{
"ref":"vipy.object.Track.boundingbox",
"url":48,
"doc":"The bounding box of a track is the smallest spatial box that contains all of the BoundingBoxes of the track within startframe and endframe, or None if there are no detections. Args: startframe: [int] the startframe of the track to compute the bounding box. endframe: [int] the endframe of the track to compute the bounding box. Returns:  vipy.geometry.BoundingBox which is the smallest box that contains all boxes of the track from (startframe, endframe)",
"func":1
},
{
"ref":"vipy.object.Track.smallestbox",
"url":48,
"doc":"The smallest box of a track is the smallest spatial box in area along the track",
"func":1
},
{
"ref":"vipy.object.Track.biggestbox",
"url":48,
"doc":"The biggest box of a track is the largest spatial box in area along the track",
"func":1
},
{
"ref":"vipy.object.Track.pathlength",
"url":48,
"doc":"The path length of a track is the cumulative Euclidean distance in pixels that the box travels",
"func":1
},
{
"ref":"vipy.object.Track.startbox",
"url":48,
"doc":"The startbox is the first bounding box in the track",
"func":1
},
{
"ref":"vipy.object.Track.endbox",
"url":48,
"doc":"The endbox is the last box in the track",
"func":1
},
{
"ref":"vipy.object.Track.loop_closure_distance",
"url":48,
"doc":"The loop closure track distance is the Euclidean distance in pixels between the start frame bounding box and end frame bounding box",
"func":1
},
{
"ref":"vipy.object.Track.boundary",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.clip",
"url":48,
"doc":"Clip a track to be within (startframe,endframe) with strict boundary handling",
"func":1
},
{
"ref":"vipy.object.Track.iou",
"url":48,
"doc":"Compute the spatial IoU between two tracks as the mean IoU per frame in the range (self.startframe(), self.endframe( ",
"func":1
},
{
"ref":"vipy.object.Track.segment_maxiou",
"url":48,
"doc":"Return the maximum framewise bounding box IOU between self and other in the range (startframe, endframe)",
"func":1
},
{
"ref":"vipy.object.Track.maxiou",
"url":48,
"doc":"Compute the maximum spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe( ",
"func":1
},
{
"ref":"vipy.object.Track.fragmentiou",
"url":48,
"doc":"A fragment is a track that is fully contained within self",
"func":1
},
{
"ref":"vipy.object.Track.endpointiou",
"url":48,
"doc":"Compute the mean spatial IoU between two tracks at the two overlapping endpoints. useful for track continuation",
"func":1
},
{
"ref":"vipy.object.Track.segmentiou",
"url":48,
"doc":"Compute the mean spatial IoU between two tracks at the overlapping segment, sampling by dt. Useful for track continuation for densely overlapping tracks",
"func":1
},
{
"ref":"vipy.object.Track.segmentcover",
"url":48,
"doc":"Compute the mean spatial cover between two tracks at the overlapping segment, sampling by dt. Useful for track continuation for densely overlapping tracks",
"func":1
},
{
"ref":"vipy.object.Track.rankiou",
"url":48,
"doc":"Compute the mean spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe( using only the top-k (rank) frame overlaps Sample tracks at endpoints and n uniformly spaced frames or a stride of dt frames. - rank [>1]: The top-k best IOU overlaps to average when computing the rank IOU - This is useful for track continuation where the box deforms in the overlapping segment at the end due to occlusion. - This is useful for track correspondence where a ground truth box does not match an estimated box precisely (e.g. loose box, non-visually grounded box) - This is the robust version of segmentiou. - Use percentileiou to determine the rank based a fraction of the length of the overlap, which will be more efficient for long tracks",
"func":1
},
{
"ref":"vipy.object.Track.percentileiou",
"url":48,
"doc":"Percentile iou returns rankiou for rank=percentile len(overlap(self, other -other [Track] -percentile [0,1]: The top-k best overlaps to average when computing rankiou -samples: The number of uniformly spaced samples to take along the track for computing the rankiou",
"func":1
},
{
"ref":"vipy.object.Track.segment_percentileiou",
"url":48,
"doc":"percentiliou on the overlapping segment with other",
"func":1
},
{
"ref":"vipy.object.Track.segment_percentilecover",
"url":48,
"doc":"percentile cover on the overlapping segment with other",
"func":1
},
{
"ref":"vipy.object.Track.union",
"url":48,
"doc":"Compute the union of two tracks. Overlapping boxes between self and other: Inputs - average [bool]: average framewise interpolated boxes at overlapping keyframes - replace [bool]: replace the box with other if other and self overlap at a keyframe - keep [bool]: keep the box from self (discard other) at a keyframe",
"func":1
},
{
"ref":"vipy.object.Track.average",
"url":48,
"doc":"Compute the average of two tracks by the framewise interpolated boxes at the keyframes of this track",
"func":1
},
{
"ref":"vipy.object.Track.temporal_distance",
"url":48,
"doc":"The temporal distance between two tracks is the minimum number of frames separating them",
"func":1
},
{
"ref":"vipy.object.Track.smooth",
"url":48,
"doc":"Track smoothing by averaging neighboring keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.smoothshape",
"url":48,
"doc":"Track smoothing by averaging width and height of neighboring keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.medianshape",
"url":48,
"doc":"Track smoothing by median width and height of neighboring keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.spline",
"url":48,
"doc":"Track smoothing by cubic spline fit, will return resampled dt=1 track. Smoothing factor will increase with smoothing > 1 and decrease with 0 < smoothing < 1 This function requires optional package scipy",
"func":1
},
{
"ref":"vipy.object.Track.linear_extrapolation",
"url":48,
"doc":"Track extrapolation by linear fit.  Requires at least 2 keyboxes.  Returned boxes may be degenerate.  shape=True then both the position and shape (width, height) of the box is extrapolated",
"func":1
},
{
"ref":"vipy.object.Track.imclip",
"url":48,
"doc":"Clip the track to the image rectangle (width, height). If a keybox is outside the image rectangle, remove it otherwise clip to the image rectangle. This operation can change the length of the track and the size of the keyboxes. The result may be an empty track if the track is completely outside the image rectangle, which results in an exception.",
"func":1
},
{
"ref":"vipy.object.Track.resample",
"url":48,
"doc":"Resample the track using a stride of dt frames. This reduces the density of keyframes by interpolating new keyframes as a uniform stride of dt. This is useful for track compression",
"func":1
},
{
"ref":"vipy.object.Track.significant_digits",
"url":48,
"doc":"Round the coordinates of all boxes so that they have n significant digits for efficient serialization",
"func":1
},
{
"ref":"vipy.object.Track.bearing",
"url":48,
"doc":"The bearing of a track at frame f is the angle of the velocity vector relative to the (x,y) image coordinate frame, in radians [-pi, pi]",
"func":1
},
{
"ref":"vipy.object.Track.bearing_change",
"url":48,
"doc":"The bearing change of a track from frame f1 (or start) and frame f2 (or end) is the relative angle of the velocity vectors in radians [-pi,pi]. Args: f1: [int] the start frame for computing the bearing change. If None, then use self.startframe() f2: [int] the end frame for computing the bearing change. if None, then use self.endframe() dt: [int] The number of frames between computations of the velocity vector for bearing minspeed: [float] The minimum speed in frames per second used to threshold bearing computations if there is no motion samples: [int] The number of samples to average for computing the bearing change Returns: The floating point bearing change in radians in [-pi, pi] from (f1,f2) where bearing is computed at samples=n points, and each bearing is computed with a velocity stride of dt frames.",
"func":1
},
{
"ref":"vipy.object.Track.acceleration",
"url":48,
"doc":"Return the (x,y) track acceleration magnitude at frame f computed using central finite differences of velocity. Returns: acceleration in (pixels / seconds^2) using velocity computed at (f-2 dt, f-dt), (f+dt, f+2 dt)",
"func":1
},
{
"ref":"vipy.object.Track.velocity",
"url":48,
"doc":"Return the (x,y) track velocity at frame f in units of pixels per frame computed by mean finite difference of the box centroid",
"func":1
},
{
"ref":"vipy.object.Track.speed",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.boxmap",
"url":48,
"doc":"Apply the lambda function to each keybox",
"func":1
},
{
"ref":"vipy.object.Track.shape_invariant_velocity",
"url":48,
"doc":"Return the (x,y) track velocity at frame f in units of pixels per frame computed by minimum mean finite differences of any box corner independent of changes in shape, over a finite time window of [f-dt, f]",
"func":1
},
{
"ref":"vipy.object.Track.velocity_x",
"url":48,
"doc":"Return the left/right velocity at frame f in units of pixels per frame computed by mean finite difference over a fixed time window (dt, frames) of the box centroid",
"func":1
},
{
"ref":"vipy.object.Track.velocity_y",
"url":48,
"doc":"Return the up/down velocity at frame f in units of pixels per frame computed by mean finite difference over a fixed time window (dt, frames) of the box centroid",
"func":1
},
{
"ref":"vipy.object.Track.velocity_w",
"url":48,
"doc":"Return the width velocity at frame f in units of pixels per frame computed by finite difference",
"func":1
},
{
"ref":"vipy.object.Track.velocity_h",
"url":48,
"doc":"Return the height velocity at frame f in units of pixels per frame computed by finite difference",
"func":1
},
{
"ref":"vipy.object.Track.nearest_keyframe",
"url":48,
"doc":"Nearest keyframe to frame f",
"func":1
},
{
"ref":"vipy.object.Track.nearest_keybox",
"url":48,
"doc":"Nearest keybox to frame f",
"func":1
},
{
"ref":"vipy.object.Track.ismoving",
"url":48,
"doc":"Is the track moving in the frame range (startframe,endframe)?",
"func":1
},
{
"ref":"vipy.object.non_maximum_suppression",
"url":48,
"doc":"Compute greedy non-maximum suppression of a list of vipy.object.Detection() based on spatial IOU threshold (iou) and cover threhsold (cover) sorted by confidence (conf). Args: detlist: [list  vipy.object.Detection ] conf: [float] minimum confidence for non-maximum suppression iou: [float] minimum iou for non-maximum suporession bycategory: [bool] NMS only within the same category cover: [float, None] A minimum cover for NMS (stricter than iou) gridsize: [tuple, (rows, cols)] An optional grid for fast intersection lookups Returns: List of  vipy.object.Detection non-maximum suppressed, sorted by increasing confidence",
"func":1
},
{
"ref":"vipy.object.greedy_assignment",
"url":48,
"doc":"Compute a greedy one-to-one assignment of each vipy.object.Detection() in srclist to a unique element in dstlist with the largest IoU greater than miniou, else None Args: srclist: [list,  vipy.object.Detection ] dstlist: [list,  vipy.object.Detection ] miniou: [float, >=0,  dstlist[j]",
"func":1
},
{
"ref":"vipy.object.greedy_track_assignment",
"url":48,
"doc":"Compute a greedy one-to-ine assignment of each  vipy.object.Track in srclist to a unique element in dstlist with the largest assignment score. - Assignment score:  vipy.object.Track.segment_percentileiou   vipy.object.Track.confidence , if maxiou() > miniou else 0 - Assigment order: longest to shortest src track Args: srclist: [list,  vipy.object.Track ] dstlist: [list,  vipy.object.Track ] miniou: [float, >=0,  dstlist[j]",
"func":1
},
{
"ref":"vipy.object.RandomDetection",
"url":48,
"doc":"Return a random  vipy.object.Detection in the range (0 < xmin < W, 0 < ymin < H, height < 100, width < 100). Useful for unit testing.",
"func":1
},
{
"ref":"vipy.version",
"url":49,
"doc":""
},
{
"ref":"vipy.version.num",
"url":49,
"doc":"Convert the version string of the form 'X.Y.Z' to an integer 100000 X + 100 Y + Z for version comparison",
"func":1
},
{
"ref":"vipy.version.split",
"url":49,
"doc":"Split the version string 'X.Y.Z' and return tuple (int(X), int(Y), int(Z ",
"func":1
},
{
"ref":"vipy.version.major",
"url":49,
"doc":"Return the major version number int(X) for versionstring 'X.Y.Z'",
"func":1
},
{
"ref":"vipy.version.minor",
"url":49,
"doc":"Return the minor version number int(Y) for versionstring 'X.Y.Z'",
"func":1
},
{
"ref":"vipy.version.release",
"url":49,
"doc":"Return the release version number int(Z) for versionstring 'X.Y.Z'",
"func":1
},
{
"ref":"vipy.version.at_least_version",
"url":49,
"doc":"Is versionstring='X.Y.Z' at least the current version?",
"func":1
},
{
"ref":"vipy.version.is_at_least",
"url":49,
"doc":"Synonym for at_least_version",
"func":1
},
{
"ref":"vipy.version.is_exactly",
"url":49,
"doc":"Is the versionstring = 'X,Y.Z' exactly equal to  vipy.version.VERSION ",
"func":1
},
{
"ref":"vipy.version.at_least_major_version",
"url":49,
"doc":"is the major version (e.g. X, for version X.Y.Z) greater than or equal to the major version integer supplied?",
"func":1
}
]