URLS=[
"vipy/index.html",
"vipy/metrics.html",
"vipy/show.html",
"vipy/annotation.html",
"vipy/version.html",
"vipy/useragent.html",
"vipy/downloader.html",
"vipy/ssim.html",
"vipy/object.html",
"vipy/geometry.html",
"vipy/util.html",
"vipy/dataset/index.html",
"vipy/dataset/msceleb.html",
"vipy/dataset/vggface.html",
"vipy/dataset/ava.html",
"vipy/dataset/aflw.html",
"vipy/dataset/facescrub.html",
"vipy/dataset/imagenet.html",
"vipy/dataset/lfw.html",
"vipy/dataset/ethzshapes.html",
"vipy/dataset/fddb.html",
"vipy/dataset/megaface.html",
"vipy/dataset/kthactions.html",
"vipy/dataset/vggface2.html",
"vipy/dataset/caltech101.html",
"vipy/dataset/youtubefaces.html",
"vipy/dataset/hmdb.html",
"vipy/dataset/casia.html",
"vipy/dataset/meva.html",
"vipy/dataset/charades.html",
"vipy/dataset/momentsintime.html",
"vipy/dataset/kinetics.html",
"vipy/dataset/mnist.html",
"vipy/dataset/caltech256.html",
"vipy/dataset/activitynet.html",
"vipy/globals.html",
"vipy/batch.html",
"vipy/activity.html",
"vipy/flow.html",
"vipy/video.html",
"vipy/videosearch.html",
"vipy/visualize.html",
"vipy/dropbox.html",
"vipy/camera.html",
"vipy/linalg.html",
"vipy/gui/index.html",
"vipy/gui/using_matplotlib.html",
"vipy/math.html",
"vipy/torch.html",
"vipy/image.html",
"vipy/calibration.html"
];
INDEX=[
{
"ref":"vipy",
"url":0,
"doc":"VIPY is a python package for representation, transformation and visualization of annotated videos and images. Annotations are the ground truth provided by labelers (e.g. object bounding boxes, face identities, temporal activity clips), suitable for training computer vision systems. VIPY provides tools to easily edit videos and images so that the annotations are transformed along with the pixels. This enables a clean interface for transforming complex datasets for input to your computer vision training and testing pipeline. VIPY provides:  Representation of videos with labeled activities that can be resized, clipped, rotated, scaled and cropped  Representation of images with object bounding boxes that can be manipulated as easily as editing an image  Clean visualization of annotated images and videos  Lazy loading of images and videos suitable for distributed procesing (e.g. dask, spark)  Straightforward integration into machine learning toolchains (e.g. torch, numpy)  Fluent interface for chaining operations on videos and images  Dataset download, unpack and import (e.g. Charades, AVA, ActivityNet, Kinetics, Moments in Time)  Video and image web search tools with URL downloading and caching  Minimum dependencies for easy installation (e.g. AWS Lambda)"
},
{
"ref":"vipy.metrics",
"url":1,
"doc":""
},
{
"ref":"vipy.metrics.cumulative_match_characteristic",
"url":1,
"doc":"CMC curve for probe x gallery similarity matrix (larger is more similar) and ground truth match matrix (one +1 per row, rest zeros)",
"func":1
},
{
"ref":"vipy.metrics.plot_cmc",
"url":1,
"doc":"Generate cumulative match characteristic (CMC) plot",
"func":1
},
{
"ref":"vipy.metrics.tdr_at_rank",
"url":1,
"doc":"Janus metric for correct retrieval (true detection rate) within a specific rank",
"func":1
},
{
"ref":"vipy.metrics.auroc",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.roc",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.roc_per_image",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.roc_eer",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.tpr_at_fpr",
"url":1,
"doc":"Janus metric for true positive rate at a specific false positive rate",
"func":1
},
{
"ref":"vipy.metrics.fpr_at_tpr",
"url":1,
"doc":"Janus metric for false positive rate at a specific true positive rate",
"func":1
},
{
"ref":"vipy.metrics.plot_roc",
"url":1,
"doc":"http: scikit-learn.org/stable/auto_examples/plot_roc.html",
"func":1
},
{
"ref":"vipy.metrics.mean_average_precision",
"url":1,
"doc":"numpy wrapper for mean",
"func":1
},
{
"ref":"vipy.metrics.average_precision",
"url":1,
"doc":"sklearn wrapper",
"func":1
},
{
"ref":"vipy.metrics.f1_score",
"url":1,
"doc":"sklearn wrapper",
"func":1
},
{
"ref":"vipy.metrics.confusion_matrix",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.categorization_report",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.precision_recall",
"url":1,
"doc":"",
"func":1
},
{
"ref":"vipy.metrics.plot_pr",
"url":1,
"doc":"Plot precision recall curve using matplotlib, with optional figure save",
"func":1
},
{
"ref":"vipy.metrics.plot_ap",
"url":1,
"doc":"Plot Average-Precision bar chart using matplotlib, with optional figure save",
"func":1
},
{
"ref":"vipy.metrics.histogram",
"url":1,
"doc":"Plot histogram bar chart using matplotlib with vertical axis labels on x-axis with optional figure save. Inputs: -freq: the output of (freq, categories) = np.histogram( ., bins=n) -categories [list]: a list of category names that must be length n, or the output of (f,c) = np.histogram( .) and categories=c[:-1] -xrot ['vertical'|None]: rotate the xticks -barcolors [list]: list of named colors equal to the length of categories",
"func":1
},
{
"ref":"vipy.metrics.pie",
"url":1,
"doc":"Generate a matplotlib style pie chart with wedges with specified size and labels, with an optional outfile",
"func":1
},
{
"ref":"vipy.show",
"url":2,
"doc":""
},
{
"ref":"vipy.show.figure",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.close",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.closeall",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.show",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.noshow",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.imshow",
"url":2,
"doc":"Show an image in the provided figure number",
"func":1
},
{
"ref":"vipy.show.imbbox",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.imdetection",
"url":2,
"doc":"Show a list of vipy.object.Detections overlayed on img. Image must be RGB",
"func":1
},
{
"ref":"vipy.show.frame",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.imframe",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.savefig",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.colorlist",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.show.text",
"url":2,
"doc":"",
"func":1
},
{
"ref":"vipy.annotation",
"url":3,
"doc":""
},
{
"ref":"vipy.annotation.googlesearch",
"url":3,
"doc":"Return a list of image URLs from google image search associated with the provided tag",
"func":1
},
{
"ref":"vipy.annotation.basic_level_categories",
"url":3,
"doc":"Return a list of nouns from wordnet that can be used as an initial list of basic level object categories",
"func":1
},
{
"ref":"vipy.annotation.verbs",
"url":3,
"doc":"Return a list of verbs from verbnet that can be used to define a set of activities",
"func":1
},
{
"ref":"vipy.annotation.facebookprofilerange",
"url":3,
"doc":"",
"func":1
},
{
"ref":"vipy.annotation.facebookprofile",
"url":3,
"doc":"",
"func":1
},
{
"ref":"vipy.version",
"url":4,
"doc":""
},
{
"ref":"vipy.version.num",
"url":4,
"doc":"",
"func":1
},
{
"ref":"vipy.version.at_least_version",
"url":4,
"doc":"Is versionstring='X.Y.Z' at least the current version?",
"func":1
},
{
"ref":"vipy.version.is_at_least",
"url":4,
"doc":"Synonym for at_least_version",
"func":1
},
{
"ref":"vipy.version.is_exactly",
"url":4,
"doc":"",
"func":1
},
{
"ref":"vipy.version.at_least_major_version",
"url":4,
"doc":"",
"func":1
},
{
"ref":"vipy.useragent",
"url":5,
"doc":""
},
{
"ref":"vipy.downloader",
"url":6,
"doc":""
},
{
"ref":"vipy.downloader.generate_sha1",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.verify_sha1",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.verify_md5",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.generate_md5",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.scp",
"url":6,
"doc":"Download using pre-installed SSH keys where hostname is formatted 'scp: hostname.com:/path/to/file.jpg'",
"func":1
},
{
"ref":"vipy.downloader.s3",
"url":6,
"doc":"Thin wrapper for boto3",
"func":1
},
{
"ref":"vipy.downloader.s3_bucket",
"url":6,
"doc":"Thin wrapper for boto3",
"func":1
},
{
"ref":"vipy.downloader.download",
"url":6,
"doc":"Downloads file at  url and write it in  output_filename ",
"func":1
},
{
"ref":"vipy.downloader.unpack",
"url":6,
"doc":"Extracts  archive_filename in  output_dirname . Supported archives:          -  Zip formats and equivalents: .zip, .egg, .jar  Tar and compressed tar formats: .tar, .tar.gz, .tgz, .tar.bz2, .tz2  gzip compressed files  non-tar .bz2",
"func":1
},
{
"ref":"vipy.downloader.download_and_unpack",
"url":6,
"doc":"Downloads and extracts archive in  url into  output_dirname . Note that  output_dirname has to exist and won't be created by this function.",
"func":1
},
{
"ref":"vipy.downloader.download_unpack_and_cleanup",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.unpack_and_cleanup",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ArchiveException",
"url":6,
"doc":"Base exception class for all archive errors."
},
{
"ref":"vipy.downloader.UnrecognizedArchiveFormat",
"url":6,
"doc":"Error raised when passed file is not a recognized archive format."
},
{
"ref":"vipy.downloader.extract",
"url":6,
"doc":"Unpack the tar or zip file at the specified  archive_filename to the directory specified by  output_dirname .",
"func":1
},
{
"ref":"vipy.downloader.Archive",
"url":6,
"doc":"The external API class that encapsulates an archive implementation."
},
{
"ref":"vipy.downloader.Archive.extract",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.Archive.list",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.BaseArchive",
"url":6,
"doc":"Base Archive class. Implementations should inherit this class."
},
{
"ref":"vipy.downloader.BaseArchive.extract",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.BaseArchive.list",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ExtractInterface",
"url":6,
"doc":"Interface class exposing common extract functionalities for standard-library-based Archive classes (e.g. based on modules like tarfile, zipfile)."
},
{
"ref":"vipy.downloader.ExtractInterface.extract",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.TarArchive",
"url":6,
"doc":"Interface class exposing common extract functionalities for standard-library-based Archive classes (e.g. based on modules like tarfile, zipfile)."
},
{
"ref":"vipy.downloader.TarArchive.list",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.TarArchive.get_members",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ZipArchive",
"url":6,
"doc":"Interface class exposing common extract functionalities for standard-library-based Archive classes (e.g. based on modules like tarfile, zipfile)."
},
{
"ref":"vipy.downloader.ZipArchive.list",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.downloader.ZipArchive.get_members",
"url":6,
"doc":"",
"func":1
},
{
"ref":"vipy.ssim",
"url":7,
"doc":""
},
{
"ref":"vipy.ssim.SSIM",
"url":7,
"doc":"Structural similarity (SSIM) index"
},
{
"ref":"vipy.ssim.SSIM.match",
"url":7,
"doc":"Return a set of matching points in img1 and img2 in the form suitable for homography estimation",
"func":1
},
{
"ref":"vipy.ssim.SSIM.warp",
"url":7,
"doc":"Warp an image im_src with points src_pts to align with dst_pts",
"func":1
},
{
"ref":"vipy.ssim.SSIM.align",
"url":7,
"doc":"Return an image which is the warped version of img1 that aligns with img2",
"func":1
},
{
"ref":"vipy.ssim.SSIM.rgb2gray",
"url":7,
"doc":"Convert RGB image to grayscale; accesory function",
"func":1
},
{
"ref":"vipy.ssim.SSIM.similarity",
"url":7,
"doc":"Compute the Structural Similarity Index (SSIM) score of two images Inputs: 1) I1, image array 2) I2, image array 3) K1, float (optional, default=0.01) - constant 4) K2, float (optional, default=0.03) - constant Outputs: 1) out; float - SSIM score 2) ssim_map; 2-D image array - SSIM map",
"func":1
},
{
"ref":"vipy.ssim.SSIM.ssim",
"url":7,
"doc":"Return structural similarity score when aligning im_degraded to im_reference",
"func":1
},
{
"ref":"vipy.ssim.demo",
"url":7,
"doc":"Synthetically rotate an image by 10 degrees, and compute structural similarity with and without alignment, return images",
"func":1
},
{
"ref":"vipy.object",
"url":8,
"doc":""
},
{
"ref":"vipy.object.Detection",
"url":8,
"doc":"vipy.object.Detection class This class represent a single object detection in the form a bounding box with a label and confidence. The constructor of this class follows a subset of the constructor patterns of vipy.geometry.BoundingBox >>> d = vipy.object.Detection(category='Person', xmin=0, ymin=0, width=50, height=100) >>> d = vipy.object.Detection(label='Person', xmin=0, ymin=0, width=50, height=100)  \"label\" is an alias for \"category\" >>> d = vipy.object.Detection(label='John Doe', shortlabel='Person', xmin=0, ymin=0, width=50, height=100)  shortlabel is displayed >>> d = vipy.object.Detection(label='Person', xywh=[0,0,50,100]) >>> d = vupy.object.Detection( ., id=True)  generate a unique UUID for this detection retrievable with d.id()"
},
{
"ref":"vipy.object.Detection.cast",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.from_json",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.dict",
"url":8,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.object.Detection.json",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.nocategory",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.noshortlabel",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.category",
"url":8,
"doc":"Update the category and shortlabel (optional) of the detection",
"func":1
},
{
"ref":"vipy.object.Detection.shortlabel",
"url":8,
"doc":"A optional shorter label string to show in the visualizations, defaults to category()",
"func":1
},
{
"ref":"vipy.object.Detection.label",
"url":8,
"doc":"Alias for category to update both category and shortlabel",
"func":1
},
{
"ref":"vipy.object.Detection.id",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.clone",
"url":8,
"doc":"Copy the object, if deep=True, then include a deep copy of the attribute dictionary, else a shallow copy",
"func":1
},
{
"ref":"vipy.object.Detection.confidence",
"url":8,
"doc":"Bounding boxes do not have confidences, use vipy.object.Detection()",
"func":1
},
{
"ref":"vipy.object.Detection.hasattribute",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.getattribute",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.setattribute",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.delattribute",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.noattributes",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Detection.xmin",
"url":9,
"doc":"x coordinate of upper left corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.object.Detection.ul",
"url":9,
"doc":"Upper left coordinate (x,y)",
"func":1
},
{
"ref":"vipy.object.Detection.ulx",
"url":9,
"doc":"Upper left coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.uly",
"url":9,
"doc":"Upper left coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.ur",
"url":9,
"doc":"Upper right coordinate (x,y)",
"func":1
},
{
"ref":"vipy.object.Detection.urx",
"url":9,
"doc":"Upper right coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.ury",
"url":9,
"doc":"Upper right coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.ll",
"url":9,
"doc":"Lower left coordinate (x,y), synonym for bl()",
"func":1
},
{
"ref":"vipy.object.Detection.bl",
"url":9,
"doc":"Bottom left coordinate (x,y), synonym for ll()",
"func":1
},
{
"ref":"vipy.object.Detection.blx",
"url":9,
"doc":"Bottom left coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.bly",
"url":9,
"doc":"Bottom left coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.lr",
"url":9,
"doc":"Lower right coordinate (x,y), synonym for br()",
"func":1
},
{
"ref":"vipy.object.Detection.br",
"url":9,
"doc":"Bottom right coordinate (x,y), synonym for lr()",
"func":1
},
{
"ref":"vipy.object.Detection.brx",
"url":9,
"doc":"Bottom right coordinate (x)",
"func":1
},
{
"ref":"vipy.object.Detection.bry",
"url":9,
"doc":"Bottom right coordinate (y)",
"func":1
},
{
"ref":"vipy.object.Detection.ymin",
"url":9,
"doc":"y coordinate of upper left corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.object.Detection.xmax",
"url":9,
"doc":"x coordinate of lower right corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.object.Detection.ymax",
"url":9,
"doc":"y coordinate of lower right corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.object.Detection.upperleft",
"url":9,
"doc":"Return the (x,y) upper left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.bottomleft",
"url":9,
"doc":"Return the (x,y) lower left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.upperright",
"url":9,
"doc":"Return the (x,y) upper right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.bottomright",
"url":9,
"doc":"Return the (x,y) lower right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.object.Detection.int",
"url":9,
"doc":"Convert corners to integer with rounding, in-place update",
"func":1
},
{
"ref":"vipy.object.Detection.float",
"url":9,
"doc":"Convert corners to float",
"func":1
},
{
"ref":"vipy.object.Detection.significant_digits",
"url":9,
"doc":"Convert corners to have at most n significant digits for efficient JSON storage",
"func":1
},
{
"ref":"vipy.object.Detection.translate",
"url":9,
"doc":"Translate the bounding box by dx in x and dy in y",
"func":1
},
{
"ref":"vipy.object.Detection.set_origin",
"url":9,
"doc":"Set the origin of the coordinates of this bounding box to be relative to the upper left of the other bounding box",
"func":1
},
{
"ref":"vipy.object.Detection.offset",
"url":9,
"doc":"Alias for translate",
"func":1
},
{
"ref":"vipy.object.Detection.invalid",
"url":9,
"doc":"Is the box a valid bounding box?",
"func":1
},
{
"ref":"vipy.object.Detection.setwidth",
"url":9,
"doc":"Set new width keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.setheight",
"url":9,
"doc":"Set new height keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.centroid",
"url":9,
"doc":"(x,y) tuple of centroid position of bounding box",
"func":1
},
{
"ref":"vipy.object.Detection.xcentroid",
"url":9,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.centroid_x",
"url":9,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.ycentroid",
"url":9,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.centroid_y",
"url":9,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.object.Detection.area",
"url":9,
"doc":"Return the area=width height of the bounding box",
"func":1
},
{
"ref":"vipy.object.Detection.to_xywh",
"url":9,
"doc":"Return bounding box corners as (x,y,width,height) tuple",
"func":1
},
{
"ref":"vipy.object.Detection.xywh",
"url":9,
"doc":"Alias for to_xywh",
"func":1
},
{
"ref":"vipy.object.Detection.cxywh",
"url":9,
"doc":"Return or set bounding box corners as (centroidx,centroidy,width,height) tuple",
"func":1
},
{
"ref":"vipy.object.Detection.ulbr",
"url":9,
"doc":"Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)",
"func":1
},
{
"ref":"vipy.object.Detection.to_ulbr",
"url":9,
"doc":"Alias for ulbr()",
"func":1
},
{
"ref":"vipy.object.Detection.dx",
"url":9,
"doc":"Offset bounding box by same xmin as provided box",
"func":1
},
{
"ref":"vipy.object.Detection.dy",
"url":9,
"doc":"Offset bounding box by ymin of provided box",
"func":1
},
{
"ref":"vipy.object.Detection.sqdist",
"url":9,
"doc":"Squared Euclidean distance between upper left corners of two bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.dist",
"url":9,
"doc":"Distance between centroids of two bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.pdist",
"url":9,
"doc":"Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box",
"func":1
},
{
"ref":"vipy.object.Detection.iou",
"url":9,
"doc":"area of intersection / area of union",
"func":1
},
{
"ref":"vipy.object.Detection.intersection_over_union",
"url":9,
"doc":"Alias for iou",
"func":1
},
{
"ref":"vipy.object.Detection.area_of_intersection",
"url":9,
"doc":"area of intersection",
"func":1
},
{
"ref":"vipy.object.Detection.cover",
"url":9,
"doc":"Fraction of this bounding box intersected by other bbox (bb)",
"func":1
},
{
"ref":"vipy.object.Detection.maxcover",
"url":9,
"doc":"The maximum cover of self to bb and bb to self",
"func":1
},
{
"ref":"vipy.object.Detection.shapeiou",
"url":9,
"doc":"Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation",
"func":1
},
{
"ref":"vipy.object.Detection.intersection",
"url":9,
"doc":"Intersection of two bounding boxes, throw an error on degeneracy of intersection result (if strict=True)",
"func":1
},
{
"ref":"vipy.object.Detection.hasintersection",
"url":9,
"doc":"Return true if self and bb overlap by any amount, or by the cover threshold (if provided) or the iou threshold (if provided). This is a convenience function that allows for shared computation for fast non-maximum suppression.",
"func":1
},
{
"ref":"vipy.object.Detection.union",
"url":9,
"doc":"Union of one or more bounding boxes with this box",
"func":1
},
{
"ref":"vipy.object.Detection.isinside",
"url":9,
"doc":"Is this boundingbox fully within the provided bounding box?",
"func":1
},
{
"ref":"vipy.object.Detection.ispointinside",
"url":9,
"doc":"Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?",
"func":1
},
{
"ref":"vipy.object.Detection.dilate",
"url":9,
"doc":"Change scale of bounding box keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.dilatepx",
"url":9,
"doc":"Dilate by a given pixel amount on all sides, keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.dilate_height",
"url":9,
"doc":"Change scale of bounding box in y direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.dilate_width",
"url":9,
"doc":"Change scale of bounding box in x direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.top",
"url":9,
"doc":"Make top of box taller (closer to top of image) by an offset dy",
"func":1
},
{
"ref":"vipy.object.Detection.bottom",
"url":9,
"doc":"Make bottom of box taller (closer to bottom of image) by an offset dy",
"func":1
},
{
"ref":"vipy.object.Detection.left",
"url":9,
"doc":"Make left of box wider (closer to left side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.object.Detection.right",
"url":9,
"doc":"Make right of box wider (closer to right side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.object.Detection.rescale",
"url":9,
"doc":"Multiply the box corners by a scale factor",
"func":1
},
{
"ref":"vipy.object.Detection.scalex",
"url":9,
"doc":"Multiply the box corners in the x dimension by a scale factor",
"func":1
},
{
"ref":"vipy.object.Detection.scaley",
"url":9,
"doc":"Multiply the box corners in the y dimension by a scale factor",
"func":1
},
{
"ref":"vipy.object.Detection.resize",
"url":9,
"doc":"Change the aspect ratio width and height of the box",
"func":1
},
{
"ref":"vipy.object.Detection.rot90cw",
"url":9,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.object.Detection.rot90ccw",
"url":9,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.object.Detection.fliplr",
"url":9,
"doc":"Flip the box left/right consistent with fliplr of the provided img (or consistent with the image width)",
"func":1
},
{
"ref":"vipy.object.Detection.flipud",
"url":9,
"doc":"Flip the box up/down consistent with flipud of the provided img (or consistent with the image height)",
"func":1
},
{
"ref":"vipy.object.Detection.imscale",
"url":9,
"doc":"Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image",
"func":1
},
{
"ref":"vipy.object.Detection.maxsquare",
"url":9,
"doc":"Set the bounding box to be square by setting width and height to the maximum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.iseven",
"url":9,
"doc":"Are all corners even number integers?",
"func":1
},
{
"ref":"vipy.object.Detection.even",
"url":9,
"doc":"Force all corners to be even number integers. This is helpful for FFMPEG crop filters.",
"func":1
},
{
"ref":"vipy.object.Detection.minsquare",
"url":9,
"doc":"Set the bounding box to be square by setting width and height to the minimum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.object.Detection.hasoverlap",
"url":9,
"doc":"Does the bounding box intersect with the provided image rectangle?",
"func":1
},
{
"ref":"vipy.object.Detection.isinterior",
"url":9,
"doc":"Is this boundingbox fully within the provided image rectangle?  If border in [0,1], then the image is dilated by a border percentage prior to computing interior, useful to check if self is near the image edge  If border=0.8, then the image rectangle is dilated by 80% (smaller) keeping the centroid constant.",
"func":1
},
{
"ref":"vipy.object.Detection.iminterior",
"url":9,
"doc":"Transform bounding box to be interior to the image rectangle with shape (W,H). Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit",
"func":1
},
{
"ref":"vipy.object.Detection.imclip",
"url":9,
"doc":"Clip bounding box to image rectangle [0,0,width,height] or img.shape=(width, height) and, throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.object.Detection.imclipshape",
"url":9,
"doc":"Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.object.Detection.convexhull",
"url":9,
"doc":"Given a set of points  x1,y1],[x2,xy], .], return the bounding rectangle, typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.aspectratio",
"url":9,
"doc":"Return the aspect ratio (width/height) of the box",
"func":1
},
{
"ref":"vipy.object.Detection.shape",
"url":9,
"doc":"Return the (height, width) tuple for the box shape",
"func":1
},
{
"ref":"vipy.object.Detection.mindimension",
"url":9,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.mindim",
"url":9,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.maxdim",
"url":9,
"doc":"Return max(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.object.Detection.ellipse",
"url":9,
"doc":"Convert the boundingbox to a vipy.geometry.Ellipse object",
"func":1
},
{
"ref":"vipy.object.Detection.average",
"url":9,
"doc":"Compute the average bounding box between self and other, and set self to the average. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.averageshape",
"url":9,
"doc":"Compute the average bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.medianshape",
"url":9,
"doc":"Compute the median bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.object.Detection.shapedist",
"url":9,
"doc":"L1 distance between (width,height) of two boxes",
"func":1
},
{
"ref":"vipy.object.Detection.affine",
"url":9,
"doc":"Apply an 2x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.object.Detection.projective",
"url":9,
"doc":"Apply an 3x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.object.Detection.crop",
"url":9,
"doc":"Crop an HxW 2D numpy image, HxWxC 3D numpy image, or NxHxWxC 4D numpy image array using this bounding box applied to HxW dimensions. Crop is performed in-place.",
"func":1
},
{
"ref":"vipy.object.Detection.grid",
"url":9,
"doc":"Split a bounding box into the smallest grid of non-overlapping bounding boxes such that the union is the original box",
"func":1
},
{
"ref":"vipy.object.Track",
"url":8,
"doc":"vipy.object.Track class A track represents one or more labeled bounding boxes of an object instance through time. A track is defined as a finite set of labeled boxes observed at keyframes, which are discrete observations of this instance. Each keyframe has an associated vipy.geometry.BoundingBox() which defines the spatial bounding box of the instance in this keyframe. The kwarg \"interpolation\" defines how the track is interpolated between keyframes, and the kwarg \"boundary\" defines how the track is interpolated outside the (min,max) of the keyframes. Valid constructors are: >>> t = vipy.object.Track(keyframes=[0,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person') >>> t = vipy.object.Track(keyframes=[0,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person', interpolation='linear') >>> t = vipy.object.Track(keyframes=[10,100], boxes=[vipy.geometry.BoundingBox(0,0,10,10), vipy.geometry.BoundingBox(0,0,20,20)], label='Person', boundary='strict') Tracks can be constructed incrementally: >>> t = vipy.object.Track('Person') >>> t.add(0, vipy.geometry.BoundingBox(0,0,10,10 >>> t.add(100, vipy.geometry.BoundingBox(0,0,20,20 Tracks can be resampled at a new framerate, as long as the framerate is known when the keyframes are extracted >>> t.framerate(newfps)"
},
{
"ref":"vipy.object.Track.from_json",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.json",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.isempty",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.confidence",
"url":8,
"doc":"The confidence of a track is the mean confidence of all (or just last=last frames, or samples=samples uniformly spaced) keyboxes (if confidences are available) else 0",
"func":1
},
{
"ref":"vipy.object.Track.isdegenerate",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.dict",
"url":8,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.object.Track.add",
"url":8,
"doc":"Add a new keyframe and associated box to track, preserve sorted order of keyframes. If keyframe is already in track, throw an exception. In this case use update() instead -strict [bool]: If box is degenerate, throw an exception if strict=True, otherwise just don't add it",
"func":1
},
{
"ref":"vipy.object.Track.update",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.replace",
"url":8,
"doc":"Replace the keyframe and associated box(es), preserve sorted order of keyframes",
"func":1
},
{
"ref":"vipy.object.Track.delete",
"url":8,
"doc":"Replace a keyframe and associated box to track, preserve sorted order of keyframes",
"func":1
},
{
"ref":"vipy.object.Track.keyframes",
"url":8,
"doc":"Return keyframe frame indexes where there are track observations",
"func":1
},
{
"ref":"vipy.object.Track.num_keyframes",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.keyboxes",
"url":8,
"doc":"Return keyboxes where there are track observations",
"func":1
},
{
"ref":"vipy.object.Track.meanshape",
"url":8,
"doc":"Return the mean (width,height) of the box during the track, or None if the track is degenerate",
"func":1
},
{
"ref":"vipy.object.Track.meanbox",
"url":8,
"doc":"Return the mean bounding box during the track, or None if the track is degenerate",
"func":1
},
{
"ref":"vipy.object.Track.shapevariance",
"url":8,
"doc":"Return the variance (width, height) of the box shape during the track or None if the track is degenerate. This is useful for filtering spurious tracks where the aspect ratio changes rapidly and randomly",
"func":1
},
{
"ref":"vipy.object.Track.framerate",
"url":8,
"doc":"Resample keyframes from known original framerate set by constructor to be new framerate fps",
"func":1
},
{
"ref":"vipy.object.Track.startframe",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.endframe",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.linear_interpolation",
"url":8,
"doc":"Linear bounding box interpolation at frame=k given observed boxes (x,y,w,h) at keyframes. This returns a vipy.object.Detection() which is the interpolation of the Track() at frame k If self._boundary='extend', then boxes are repeated if the interpolation is outside the keyframes If self._boundary='strict', then interpolation returns None if the interpolation is outside the keyframes - The returned object is not cloned when possible for speed purposes, be careful when modifying this object. clone() if necessary",
"func":1
},
{
"ref":"vipy.object.Track.category",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.label",
"url":8,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.object.Track.shortlabel",
"url":8,
"doc":"A optional shorter label string to show as a caption in visualizations",
"func":1
},
{
"ref":"vipy.object.Track.during",
"url":8,
"doc":"Is frame during the time interval (startframe, endframe) inclusive?",
"func":1
},
{
"ref":"vipy.object.Track.during_interval",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.offset",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.frameoffset",
"url":8,
"doc":"Offset boxes by (dx,dy) in each frame",
"func":1
},
{
"ref":"vipy.object.Track.truncate",
"url":8,
"doc":"Truncate a track so that any keyframes less than startframe or greater than endframe are removed",
"func":1
},
{
"ref":"vipy.object.Track.rescale",
"url":8,
"doc":"Rescale track boxes by scale factor s",
"func":1
},
{
"ref":"vipy.object.Track.scale",
"url":8,
"doc":"Alias for rescale",
"func":1
},
{
"ref":"vipy.object.Track.scalex",
"url":8,
"doc":"Rescale track boxes by scale factor sx",
"func":1
},
{
"ref":"vipy.object.Track.scaley",
"url":8,
"doc":"Rescale track boxes by scale factor sx",
"func":1
},
{
"ref":"vipy.object.Track.dilate",
"url":8,
"doc":"Dilate track boxes by scale factor s",
"func":1
},
{
"ref":"vipy.object.Track.rot90cw",
"url":8,
"doc":"Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent",
"func":1
},
{
"ref":"vipy.object.Track.rot90ccw",
"url":8,
"doc":"Rotate an image with (H,W)=shape 90 degrees clockwise and update all boxes to be consistent",
"func":1
},
{
"ref":"vipy.object.Track.fliplr",
"url":8,
"doc":"Flip an image left and right (mirror about vertical axis)",
"func":1
},
{
"ref":"vipy.object.Track.flipud",
"url":8,
"doc":"Flip an image left and right (mirror about vertical axis)",
"func":1
},
{
"ref":"vipy.object.Track.id",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.clone",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.clone_during",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.boundingbox",
"url":8,
"doc":"The bounding box of a track is the smallest spatial box that contains all of the detections within startframe and endframe, or None if there are no detections",
"func":1
},
{
"ref":"vipy.object.Track.smallestbox",
"url":8,
"doc":"The smallest box of a track is the smallest spatial box in area along the track",
"func":1
},
{
"ref":"vipy.object.Track.biggestbox",
"url":8,
"doc":"The biggest box of a track is the largest spatial box in area along the track",
"func":1
},
{
"ref":"vipy.object.Track.pathlength",
"url":8,
"doc":"The path length of a track is the cumulative Euclidean distance in pixels that the box travels",
"func":1
},
{
"ref":"vipy.object.Track.startbox",
"url":8,
"doc":"The startbox is the first bounding box in the track",
"func":1
},
{
"ref":"vipy.object.Track.endbox",
"url":8,
"doc":"The endbox is the last box in the track",
"func":1
},
{
"ref":"vipy.object.Track.loop_closure_distance",
"url":8,
"doc":"The loop closure track distance is the Euclidean distance in pixels between the start frame bounding box and end frame bounding box",
"func":1
},
{
"ref":"vipy.object.Track.boundary",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.clip",
"url":8,
"doc":"Clip a track to be within (startframe,endframe) with strict boundary handling",
"func":1
},
{
"ref":"vipy.object.Track.iou",
"url":8,
"doc":"Compute the spatial IoU between two tracks as the mean IoU per frame in the range (self.startframe(), self.endframe( ",
"func":1
},
{
"ref":"vipy.object.Track.segment_maxiou",
"url":8,
"doc":"Return the maximum framewise bounding box IOU between self and other in the range (startframe, endframe)",
"func":1
},
{
"ref":"vipy.object.Track.maxiou",
"url":8,
"doc":"Compute the maximum spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe( ",
"func":1
},
{
"ref":"vipy.object.Track.fragmentiou",
"url":8,
"doc":"A fragment is a track that is fully contained within self",
"func":1
},
{
"ref":"vipy.object.Track.endpointiou",
"url":8,
"doc":"Compute the mean spatial IoU between two tracks at the two overlapping endpoints. useful for track continuation",
"func":1
},
{
"ref":"vipy.object.Track.segmentiou",
"url":8,
"doc":"Compute the mean spatial IoU between two tracks at the overlapping segment, sampling by dt. Useful for track continuation for densely overlapping tracks",
"func":1
},
{
"ref":"vipy.object.Track.segmentcover",
"url":8,
"doc":"Compute the mean spatial cover between two tracks at the overlapping segment, sampling by dt. Useful for track continuation for densely overlapping tracks",
"func":1
},
{
"ref":"vipy.object.Track.rankiou",
"url":8,
"doc":"Compute the mean spatial IoU between two tracks per frame in the range (self.startframe(), self.endframe( using only the top-k (rank) frame overlaps Sample tracks at endpoints and n uniformly spaced frames or a stride of dt frames. - rank [>1]: The top-k best IOU overlaps to average when computing the rank IOU - This is useful for track continuation where the box deforms in the overlapping segment at the end due to occlusion. - This is useful for track correspondence where a ground truth box does not match an estimated box precisely (e.g. loose box, non-visually grounded box) - This is the robust version of segmentiou. - Use percentileiou to determine the rank based a fraction of the length of the overlap, which will be more efficient for long tracks",
"func":1
},
{
"ref":"vipy.object.Track.percentileiou",
"url":8,
"doc":"Percentile iou returns rankiou for rank=percentile len(overlap(self, other -other [Track] -percentile [0,1]: The top-k best overlaps to average when computing rankiou -samples: The number of uniformly spaced samples to take along the track for computing the rankiou",
"func":1
},
{
"ref":"vipy.object.Track.segment_percentileiou",
"url":8,
"doc":"percentiliou on the overlapping segment with other",
"func":1
},
{
"ref":"vipy.object.Track.segment_percentilecover",
"url":8,
"doc":"percentile cover on the overlapping segment with other",
"func":1
},
{
"ref":"vipy.object.Track.union",
"url":8,
"doc":"Compute the union of two tracks. Overlapping boxes between self and other: Inputs - average [bool]: average framewise interpolated boxes at overlapping keyframes - replace [bool]: replace the box with other if other and self overlap at a keyframe - keep [bool]: keep the box from self (discard other) at a keyframe",
"func":1
},
{
"ref":"vipy.object.Track.average",
"url":8,
"doc":"Compute the average of two tracks by the framewise interpolated boxes at the keyframes of this track",
"func":1
},
{
"ref":"vipy.object.Track.temporal_distance",
"url":8,
"doc":"The temporal distance between two tracks is the minimum number of frames separating them",
"func":1
},
{
"ref":"vipy.object.Track.smooth",
"url":8,
"doc":"Track smoothing by averaging neighboring keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.smoothshape",
"url":8,
"doc":"Track smoothing by averaging width and height of neighboring keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.medianshape",
"url":8,
"doc":"Track smoothing by median width and height of neighboring keyboxes",
"func":1
},
{
"ref":"vipy.object.Track.spline",
"url":8,
"doc":"Track smoothing by cubic spline fit, will return resampled dt=1 track. Smoothing factor will increase with smoothing > 1 and decrease with 0 < smoothing < 1 This function requires optional package scipy",
"func":1
},
{
"ref":"vipy.object.Track.linear_extrapolation",
"url":8,
"doc":"Track extrapolation by linear fit.  Requires at least 2 keyboxes.  Returned boxes may be degenerate.  shape=True then both the position and shape (width, height) of the box is extrapolated",
"func":1
},
{
"ref":"vipy.object.Track.imclip",
"url":8,
"doc":"Clip the track to the image rectangle (width, height). If a keybox is outside the image rectangle, remove it otherwise clip to the image rectangle. This operation can change the length of the track and the size of the keyboxes. The result may be an empty track if the track is completely outside the image rectangle, which results in an exception.",
"func":1
},
{
"ref":"vipy.object.Track.resample",
"url":8,
"doc":"Resample the track using a stride of dt frames. This reduces the density of keyframes by interpolating new keyframes as a uniform stride of dt. This is useful for track compression",
"func":1
},
{
"ref":"vipy.object.Track.significant_digits",
"url":8,
"doc":"Round the coordinates of all boxes so that they have n significant digits for efficient serialization",
"func":1
},
{
"ref":"vipy.object.Track.bearing",
"url":8,
"doc":"The bearing of a track at frame f is the angle of the velocity vector relative to the (x,y) image coordinate frame, in radians [-pi, pi]",
"func":1
},
{
"ref":"vipy.object.Track.bearing_change",
"url":8,
"doc":"The bearing change of a track from frame f1 (or start) and frame f2 (or end) is the relative angle of the velocity vectors in radians [-pi,pi]",
"func":1
},
{
"ref":"vipy.object.Track.acceleration",
"url":8,
"doc":"Return the (x,y) track acceleration magnitude at frame f computed using central finite differences of velocity",
"func":1
},
{
"ref":"vipy.object.Track.velocity",
"url":8,
"doc":"Return the (x,y) track velocity at frame f in units of pixels per frame computed by mean finite difference of the box centroid",
"func":1
},
{
"ref":"vipy.object.Track.speed",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.object.Track.boxmap",
"url":8,
"doc":"Apply the lambda function to each keybox",
"func":1
},
{
"ref":"vipy.object.Track.shape_invariant_velocity",
"url":8,
"doc":"Return the (x,y) track velocity at frame f in units of pixels per frame computed by minimum mean finite differences of any box corner independent of changes in shape, over a finite time window of [f-dt, f]",
"func":1
},
{
"ref":"vipy.object.Track.velocity_x",
"url":8,
"doc":"Return the left/right velocity at frame f in units of pixels per frame computed by mean finite difference over a fixed time window (dt, frames) of the box centroid",
"func":1
},
{
"ref":"vipy.object.Track.velocity_y",
"url":8,
"doc":"Return the up/down velocity at frame f in units of pixels per frame computed by mean finite difference over a fixed time window (dt, frames) of the box centroid",
"func":1
},
{
"ref":"vipy.object.Track.velocity_w",
"url":8,
"doc":"Return the width velocity at frame f in units of pixels per frame computed by finite difference",
"func":1
},
{
"ref":"vipy.object.Track.velocity_h",
"url":8,
"doc":"Return the height velocity at frame f in units of pixels per frame computed by finite difference",
"func":1
},
{
"ref":"vipy.object.Track.nearest_keyframe",
"url":8,
"doc":"Nearest keyframe to frame f",
"func":1
},
{
"ref":"vipy.object.Track.nearest_keybox",
"url":8,
"doc":"Nearest keybox to frame f",
"func":1
},
{
"ref":"vipy.object.Track.ismoving",
"url":8,
"doc":"Is the track moving in the frame range (startframe,endframe)?",
"func":1
},
{
"ref":"vipy.object.non_maximum_suppression",
"url":8,
"doc":"Compute greedy non-maximum suppression of a list of vipy.object.Detection() based on spatial IOU threshold (iou) and cover threhsold (cover) sorted by confidence (conf)",
"func":1
},
{
"ref":"vipy.object.greedy_assignment",
"url":8,
"doc":"Compute a greedy one-to-one assignment of each vipy.object.Detection() in srclist to a unique element in dstlist with the largest IoU greater than miniou, else None returns: assignlist [list]: same length as srclist, where j=assignlist[i] is the index of the assignment such that srclist[i]  dstlist[j]",
"func":1
},
{
"ref":"vipy.object.RandomDetection",
"url":8,
"doc":"",
"func":1
},
{
"ref":"vipy.util",
"url":10,
"doc":""
},
{
"ref":"vipy.util.bz2pkl",
"url":10,
"doc":"Read/Write a bz2 compressed pickle file",
"func":1
},
{
"ref":"vipy.util.mergedict",
"url":10,
"doc":"Combine keys of two dictionaries and return a dictionary deep copy",
"func":1
},
{
"ref":"vipy.util.hascache",
"url":10,
"doc":"Is the VIPY_CACHE environment variable set?",
"func":1
},
{
"ref":"vipy.util.tocache",
"url":10,
"doc":"If the VIPY_CACHE environment variable is set, then return the filename in the cache",
"func":1
},
{
"ref":"vipy.util.try_import",
"url":10,
"doc":"Show a helpful error message for missing optional packages",
"func":1
},
{
"ref":"vipy.util.findyaml",
"url":10,
"doc":"Return a list of absolute paths to yaml files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findpkl",
"url":10,
"doc":"Return a list of absolute paths to pkl files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findjson",
"url":10,
"doc":"Return a list of absolute paths to json files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findimage",
"url":10,
"doc":"Return a list of absolute paths to image files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.findvideo",
"url":10,
"doc":"Return a list of absolute paths to video files recursively discovered by walking the directory tree rooted at basedir",
"func":1
},
{
"ref":"vipy.util.readyaml",
"url":10,
"doc":"Read a yaml file and return a parsed dictionary, this is slow for large yaml files",
"func":1
},
{
"ref":"vipy.util.count_images_in_subdirectories",
"url":10,
"doc":"Count the total number of images in indir/subdir1, indir/subdir2, go down only one level and no further .",
"func":1
},
{
"ref":"vipy.util.rowvectorize",
"url":10,
"doc":"Convert a 1D numpy array to a 2D row vector of size (1,N)",
"func":1
},
{
"ref":"vipy.util.columnvectorize",
"url":10,
"doc":"Convert a 1D numpy array to a 2D column vector of size (N,1)",
"func":1
},
{
"ref":"vipy.util.isodd",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.keymax",
"url":10,
"doc":"Return key in dictionary containing maximum value",
"func":1
},
{
"ref":"vipy.util.keymin",
"url":10,
"doc":"Return key in dictionary containing minimum value",
"func":1
},
{
"ref":"vipy.util.isjsonfile",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.writejson",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.readjson",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.groupby",
"url":10,
"doc":"groupby on unsorted inset",
"func":1
},
{
"ref":"vipy.util.vipy_groupby",
"url":10,
"doc":"groupby on unsorted inset",
"func":1
},
{
"ref":"vipy.util.groupbyasdict",
"url":10,
"doc":"Return dictionary of keys and lists from groupby on unsorted inset, where keyfunc is a lambda function on elements in inset",
"func":1
},
{
"ref":"vipy.util.countby",
"url":10,
"doc":"Return dictionary of keys and group sizes for a grouping of the input list by keyfunc lambda function",
"func":1
},
{
"ref":"vipy.util.most_frequent",
"url":10,
"doc":"Return the most frequent element as determined by element equality",
"func":1
},
{
"ref":"vipy.util.countbyasdict",
"url":10,
"doc":"Alias for countby",
"func":1
},
{
"ref":"vipy.util.softmax",
"url":10,
"doc":"Row-wise softmax",
"func":1
},
{
"ref":"vipy.util.permutelist",
"url":10,
"doc":"randomly permute list order",
"func":1
},
{
"ref":"vipy.util.flatlist",
"url":10,
"doc":"Convert list of tuples into a list expanded by concatenating tuples",
"func":1
},
{
"ref":"vipy.util.rmdir",
"url":10,
"doc":"Recursively remove directory and all contents (if the directory exists)",
"func":1
},
{
"ref":"vipy.util.dividelist",
"url":10,
"doc":"Divide inlist into a list of lists such that the size of each sublist is the requseted fraction of the original list. This operation is deterministic and generates the same division in multiple calls. Input: -inlist=list -fractions=(0.1, 0.7, 0.2) An iterable of fractions that must be non-negative and sum to one",
"func":1
},
{
"ref":"vipy.util.chunklist",
"url":10,
"doc":"Convert list into a list of lists of length num_chunks each element is a list containing a sequential chunk of the original list",
"func":1
},
{
"ref":"vipy.util.chunklistbysize",
"url":10,
"doc":"Convert list into a list of lists such that each element is a list containing a sequential chunk of the original list of length size_per_chunk",
"func":1
},
{
"ref":"vipy.util.chunklistWithOverlap",
"url":10,
"doc":"Convert list into a list of lists such that each element is a list containing a sequential chunk of the original list of length size_per_chunk",
"func":1
},
{
"ref":"vipy.util.chunklistwithoverlap",
"url":10,
"doc":"Alias for chunklistWithOverlap",
"func":1
},
{
"ref":"vipy.util.imwritejet",
"url":10,
"doc":"Write a grayscale numpy image as a jet colormapped image to the given file",
"func":1
},
{
"ref":"vipy.util.isuint8",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isnumber",
"url":10,
"doc":"Is the input a python type of a number or a string containing a number?",
"func":1
},
{
"ref":"vipy.util.isfloat",
"url":10,
"doc":"Is the input a float or a string that can be converted to float?",
"func":1
},
{
"ref":"vipy.util.imwritegray",
"url":10,
"doc":"Write a floating point grayscale numpy image in [0,1] as [0,255] grayscale",
"func":1
},
{
"ref":"vipy.util.imwrite",
"url":10,
"doc":"Write a floating point 2D numpy image as jet or gray, 3D numpy as rgb or bgr",
"func":1
},
{
"ref":"vipy.util.print_and_return",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.savetemp",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.gray2jet",
"url":10,
"doc":"[0,1] grayscale to [0.255] RGB",
"func":1
},
{
"ref":"vipy.util.jet",
"url":10,
"doc":"jet colormap",
"func":1
},
{
"ref":"vipy.util.is_hiddenfile",
"url":10,
"doc":"Does the filename start with a period?",
"func":1
},
{
"ref":"vipy.util.seq",
"url":10,
"doc":"Equivalent to matlab [start:step:stop]",
"func":1
},
{
"ref":"vipy.util.loadh5",
"url":10,
"doc":"Load an HDF5 file",
"func":1
},
{
"ref":"vipy.util.loadmat73",
"url":10,
"doc":"Matlab 7.3 format, keys should be a list of keys to access HDF5 file as f[key1][key2] . Returned as numpy array",
"func":1
},
{
"ref":"vipy.util.saveas",
"url":10,
"doc":"Save variables as a dill pickled file",
"func":1
},
{
"ref":"vipy.util.loadas",
"url":10,
"doc":"Load variables from a dill pickled file",
"func":1
},
{
"ref":"vipy.util.load",
"url":10,
"doc":"Load variables from a relocatable archive file format, either Dill Pickle or JSON. Loading is performed by attemping the following: 1. load the pickle or json file 2. if abspath=true, then convert relative paths to absolute paths for object when loaded 3. If the loaded object is a vipy object (or iterable) and the relocatable path /$PATH is present, try to repath it to the directory containing this archive (this has been deprecated) 4. If the resulting files are not found, throw a warning 5. If a large number of objects are loaded, disable garbage collection.",
"func":1
},
{
"ref":"vipy.util.canload",
"url":10,
"doc":"Attempt to load a pkl file, and return true if it can be successfully loaded, otherwise False",
"func":1
},
{
"ref":"vipy.util.save",
"url":10,
"doc":"Save variables to an archive file",
"func":1
},
{
"ref":"vipy.util.distload",
"url":10,
"doc":"Load a redistributable pickle file that replaces absolute paths in datapath with srcpath. See also vipy.util.distsave(), This function has been deprecated, all archives should be distributed with relative paths",
"func":1
},
{
"ref":"vipy.util.distsave",
"url":10,
"doc":"Save a archive file for redistribution, where datapath is replaced by dstpath. Useful for redistribuing pickle files with absolute paths. See also vipy.util.distload(). This function has been deprecated, all archives should be distributed with relative paths",
"func":1
},
{
"ref":"vipy.util.repath",
"url":10,
"doc":"Change the filename with prefix srcpath to dstpath, for any element in v that supports the filename() api",
"func":1
},
{
"ref":"vipy.util.scpsave",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.scpload",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.load_opencv_yaml",
"url":10,
"doc":"Load a numpy array from YAML file exported from OpenCV",
"func":1
},
{
"ref":"vipy.util.matrix_to_opencv_yaml",
"url":10,
"doc":"Write list of matrices to OpenCV yaml file format with given variable names",
"func":1
},
{
"ref":"vipy.util.save_opencv_yaml",
"url":10,
"doc":"Save a numpy array to YAML file importable by OpenCV",
"func":1
},
{
"ref":"vipy.util.tofilename",
"url":10,
"doc":"Convert arbitrary string to valid filename with underscores replacing invalid chars",
"func":1
},
{
"ref":"vipy.util.isexe",
"url":10,
"doc":"Is the file an executable binary?",
"func":1
},
{
"ref":"vipy.util.isinstalled",
"url":10,
"doc":"Is the command is available on the path",
"func":1
},
{
"ref":"vipy.util.ispkl",
"url":10,
"doc":"Is the file a pickle archive file",
"func":1
},
{
"ref":"vipy.util.ispklfile",
"url":10,
"doc":"Is the file a pickle archive file",
"func":1
},
{
"ref":"vipy.util.ishtml",
"url":10,
"doc":"Is the file an HTMLfile",
"func":1
},
{
"ref":"vipy.util.ispickle",
"url":10,
"doc":"Is the file a pickle archive file",
"func":1
},
{
"ref":"vipy.util.ishdf5",
"url":10,
"doc":"Is the file an HDF5 file?",
"func":1
},
{
"ref":"vipy.util.filebase",
"url":10,
"doc":"Return c for filename /a/b/c.ext",
"func":1
},
{
"ref":"vipy.util.filepath",
"url":10,
"doc":"Return /a/b/c for filename /a/b/c/d.ext, /a/b for filename /a/b/c/d.ext if depth=1, etc",
"func":1
},
{
"ref":"vipy.util.delpath",
"url":10,
"doc":"Return c/d.ext for filename /a/b/c/d.ext and indir /a/b",
"func":1
},
{
"ref":"vipy.util.newpath",
"url":10,
"doc":"Return /d/e/c.ext for filename /a/b/c.ext and newdir /d/e/",
"func":1
},
{
"ref":"vipy.util.newprefix",
"url":10,
"doc":"Return /a/b/c/h/i.ext for filename /f/g/h/i.ext and prefix /a/b/c and depth=1",
"func":1
},
{
"ref":"vipy.util.newpathdir",
"url":10,
"doc":"Return /a/b/n/d/e.ext for filename=/a/b/c/d/e.ext, olddir=c, newdir=n",
"func":1
},
{
"ref":"vipy.util.newpathroot",
"url":10,
"doc":"Return /r/b/c.ext for filename /a/b/c.ext and new root directory r",
"func":1
},
{
"ref":"vipy.util.topath",
"url":10,
"doc":"Alias for newpath",
"func":1
},
{
"ref":"vipy.util.filefull",
"url":10,
"doc":"Return /a/b/c for filename /a/b/c.ext",
"func":1
},
{
"ref":"vipy.util.filetail",
"url":10,
"doc":"Return c.ext for filename /a/b/c.ext",
"func":1
},
{
"ref":"vipy.util.matread",
"url":10,
"doc":"Whitespace separated values defining columns, lines define rows. Return numpy array",
"func":1
},
{
"ref":"vipy.util.imlist",
"url":10,
"doc":"return list of images with absolute path in a directory",
"func":1
},
{
"ref":"vipy.util.videolist",
"url":10,
"doc":"return list of videos with absolute path in a directory",
"func":1
},
{
"ref":"vipy.util.dirlist",
"url":10,
"doc":"return list of directories in a directory",
"func":1
},
{
"ref":"vipy.util.dirlist_sorted_bycreation",
"url":10,
"doc":"Sort the directory list from newest first to oldest last by creation date",
"func":1
},
{
"ref":"vipy.util.extlist",
"url":10,
"doc":"return list of files with absolute path in a directory that have the provided extension (with the prepended dot, ext='.mp4')",
"func":1
},
{
"ref":"vipy.util.listext",
"url":10,
"doc":"Alias for extlist",
"func":1
},
{
"ref":"vipy.util.jsonlist",
"url":10,
"doc":"return list of fJSON iles with absolute path in a directory",
"func":1
},
{
"ref":"vipy.util.listjson",
"url":10,
"doc":"Alias for jsonlist",
"func":1
},
{
"ref":"vipy.util.writelist",
"url":10,
"doc":"Write list of strings to an output file with each row an element of the list",
"func":1
},
{
"ref":"vipy.util.readlist",
"url":10,
"doc":"Read each row of file as an element of the list",
"func":1
},
{
"ref":"vipy.util.readtxt",
"url":10,
"doc":"Read a text file one string per row",
"func":1
},
{
"ref":"vipy.util.writecsv",
"url":10,
"doc":"Write list of tuples to an output csv file with each list element on a row and tuple elements separated by comma",
"func":1
},
{
"ref":"vipy.util.readcsv",
"url":10,
"doc":"Read a csv file into a list of lists",
"func":1
},
{
"ref":"vipy.util.readcsvwithheader",
"url":10,
"doc":"Read a csv file into a list of lists",
"func":1
},
{
"ref":"vipy.util.imsavelist",
"url":10,
"doc":"Write out all images in a directory to a provided file with each line containing absolute path to image",
"func":1
},
{
"ref":"vipy.util.csvlist",
"url":10,
"doc":"Return a list of absolute paths of  .csv files in current directory",
"func":1
},
{
"ref":"vipy.util.pklist",
"url":10,
"doc":"Return a list of absolute paths of  .pk files in current directory",
"func":1
},
{
"ref":"vipy.util.listpkl",
"url":10,
"doc":"Return a list of absolute paths of  .pk files in current directory",
"func":1
},
{
"ref":"vipy.util.txtlist",
"url":10,
"doc":"Return a list of absolute paths of  .txt files in current directory",
"func":1
},
{
"ref":"vipy.util.imlistidx",
"url":10,
"doc":"Return index in list of filename containing index number",
"func":1
},
{
"ref":"vipy.util.mat2gray",
"url":10,
"doc":"Convert numpy array to float32 with 1.0=max and 0=min",
"func":1
},
{
"ref":"vipy.util.mdlist",
"url":10,
"doc":"Preallocate 2D list of size MxN",
"func":1
},
{
"ref":"vipy.util.isurl",
"url":10,
"doc":"Is a path a URL?",
"func":1
},
{
"ref":"vipy.util.shortuuid",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isimageurl",
"url":10,
"doc":"Is a path a URL with image extension?",
"func":1
},
{
"ref":"vipy.util.isvideourl",
"url":10,
"doc":"Is a path a URL with video extension?",
"func":1
},
{
"ref":"vipy.util.isS3url",
"url":10,
"doc":"Is a path a URL for an S3 object?",
"func":1
},
{
"ref":"vipy.util.isyoutubeurl",
"url":10,
"doc":"Is a path a youtube URL?",
"func":1
},
{
"ref":"vipy.util.checkerboard",
"url":10,
"doc":"m=number of square by column, n=size of final image",
"func":1
},
{
"ref":"vipy.util.islist",
"url":10,
"doc":"Is an object a python list",
"func":1
},
{
"ref":"vipy.util.islistoflists",
"url":10,
"doc":"Is an object a python list of lists x= 1,2], [3,4 ",
"func":1
},
{
"ref":"vipy.util.istupleoftuples",
"url":10,
"doc":"Is an object a python list of lists x= 1,2], [3,4 ",
"func":1
},
{
"ref":"vipy.util.isimageobject",
"url":10,
"doc":"Is an object a vipy.image class Image, ImageCategory, ImageDetection?",
"func":1
},
{
"ref":"vipy.util.isvideotype",
"url":10,
"doc":"Is an object a vipy.video class Video, VideoCategory, Scene?",
"func":1
},
{
"ref":"vipy.util.isvideoobject",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isvipyobject",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.istuple",
"url":10,
"doc":"Is an object a python tuple?",
"func":1
},
{
"ref":"vipy.util.tolist",
"url":10,
"doc":"Convert a python tuple or singleton object to a list if not already a list",
"func":1
},
{
"ref":"vipy.util.isimg",
"url":10,
"doc":"Is an object an image with a supported image extension ['.jpg','.jpeg','.png','.tif','.tiff','.pgm','.ppm','.gif','.bmp']?",
"func":1
},
{
"ref":"vipy.util.isimage",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isvideofile",
"url":10,
"doc":"Equivalent to isvideo()",
"func":1
},
{
"ref":"vipy.util.isimgfile",
"url":10,
"doc":"Convenience function for isimg",
"func":1
},
{
"ref":"vipy.util.isimagefile",
"url":10,
"doc":"Convenience function for isimg",
"func":1
},
{
"ref":"vipy.util.isgif",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isjpeg",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.iswebp",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.ispng",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isjpg",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.iscsv",
"url":10,
"doc":"Is a file a CSV file extension?",
"func":1
},
{
"ref":"vipy.util.isvideo",
"url":10,
"doc":"Is a filename in path a video with a known video extension ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']?",
"func":1
},
{
"ref":"vipy.util.isnumpy",
"url":10,
"doc":"Is a python object a numpy object?",
"func":1
},
{
"ref":"vipy.util.isnumpyarray",
"url":10,
"doc":"Is a python object a numpy array?",
"func":1
},
{
"ref":"vipy.util.istextfile",
"url":10,
"doc":"Is the given file a text file?",
"func":1
},
{
"ref":"vipy.util.isxml",
"url":10,
"doc":"Is the given file an xml file?",
"func":1
},
{
"ref":"vipy.util.bgr2gray",
"url":10,
"doc":"Wrapper for numpy uint8 BGR image to uint8 numpy grayscale",
"func":1
},
{
"ref":"vipy.util.gray2bgr",
"url":10,
"doc":"Wrapper for numpy float32 gray image to uint8 numpy BGR",
"func":1
},
{
"ref":"vipy.util.gray2rgb",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.bgr2rgb",
"url":10,
"doc":"Wrapper for numpy BGR uint8 to numpy RGB uint8",
"func":1
},
{
"ref":"vipy.util.rgb2bgr",
"url":10,
"doc":"same as bgr2rgb",
"func":1
},
{
"ref":"vipy.util.bgr2hsv",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.gray2hsv",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isarchive",
"url":10,
"doc":"Is filename a zip or gzip compressed tar archive?",
"func":1
},
{
"ref":"vipy.util.istgz",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.isbz2",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.tempfilename",
"url":10,
"doc":"Create a temporary filename $TEMPDIR/$UUID.suffix, suffix should include the dot such as suffix='.jpg',",
"func":1
},
{
"ref":"vipy.util.totempdir",
"url":10,
"doc":"Convert a filename '/patj/to/filename.ext' to '/tempdir/filename.ext'",
"func":1
},
{
"ref":"vipy.util.templike",
"url":10,
"doc":"Create a new temporary filename with the same extension as filename",
"func":1
},
{
"ref":"vipy.util.cached",
"url":10,
"doc":"Create a new filename in the cache, or tempdir if not found",
"func":1
},
{
"ref":"vipy.util.tempimage",
"url":10,
"doc":"Create a temporary image with the given extension",
"func":1
},
{
"ref":"vipy.util.temppng",
"url":10,
"doc":"Create a temporay PNG file",
"func":1
},
{
"ref":"vipy.util.temppickle",
"url":10,
"doc":"Create a temporary pickle file",
"func":1
},
{
"ref":"vipy.util.tempjpg",
"url":10,
"doc":"Create a temporary JPG file in system temp directory",
"func":1
},
{
"ref":"vipy.util.tempMP4",
"url":10,
"doc":"Create a temporary MP4 file in system temp directory",
"func":1
},
{
"ref":"vipy.util.tmpjpg",
"url":10,
"doc":"Create a temporary JPG file in /tmp",
"func":1
},
{
"ref":"vipy.util.tempcsv",
"url":10,
"doc":"Create a temporary CSV file",
"func":1
},
{
"ref":"vipy.util.temphtml",
"url":10,
"doc":"Create a temporary HTMLfile",
"func":1
},
{
"ref":"vipy.util.temppkl",
"url":10,
"doc":"Create a temporary pickle file",
"func":1
},
{
"ref":"vipy.util.tempyaml",
"url":10,
"doc":"Create a temporary YAML file",
"func":1
},
{
"ref":"vipy.util.tempjson",
"url":10,
"doc":"Create a temporary JSON file",
"func":1
},
{
"ref":"vipy.util.temppdf",
"url":10,
"doc":"Create a temporary PDF file",
"func":1
},
{
"ref":"vipy.util.mktemp",
"url":10,
"doc":"Create a temporary file with extension .ext",
"func":1
},
{
"ref":"vipy.util.tempdir",
"url":10,
"doc":"Wrapper around tempfile, because I can never remember the syntax",
"func":1
},
{
"ref":"vipy.util.imread",
"url":10,
"doc":"Wrapper for opencv imread. Note that color images are imported as BGR!",
"func":1
},
{
"ref":"vipy.util.imrescale",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.imresize",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.touch",
"url":10,
"doc":"Create an empty file containing mystr",
"func":1
},
{
"ref":"vipy.util.isboundingbox",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.Stopwatch",
"url":10,
"doc":"Return elapsed system time in seconds between calls to enter and exit"
},
{
"ref":"vipy.util.Stopwatch.since",
"url":10,
"doc":"Return seconds since start or last call to this method",
"func":1
},
{
"ref":"vipy.util.Stopwatch.reset",
"url":10,
"doc":"",
"func":1
},
{
"ref":"vipy.util.Stopwatch.duration",
"url":10,
"doc":"Time in seconds since last reset",
"func":1
},
{
"ref":"vipy.util.Timer",
"url":10,
"doc":"Pretty print elapsed system time in seconds between calls to enter and exit >>> t = Timer(): >>> [some code] >>> print(t) >>> [some more code] >>> print(t) >>> with Timer(): >>> [some code]"
},
{
"ref":"vipy.util.isfile",
"url":10,
"doc":"Wrapper for os.path.isfile",
"func":1
},
{
"ref":"vipy.util.isstring",
"url":10,
"doc":"Is an object a python string or unicode string?",
"func":1
},
{
"ref":"vipy.util.timestamp",
"url":10,
"doc":"Return date and time string in form DDMMMYY_HHMMSS",
"func":1
},
{
"ref":"vipy.util.clockstamp",
"url":10,
"doc":"Datetime stamp in local timezone with second resolution with format Year-Month-Day Hour:Minute:Second",
"func":1
},
{
"ref":"vipy.util.minutestamp",
"url":10,
"doc":"Return date and time string in form DDMMMYY_HHMM",
"func":1
},
{
"ref":"vipy.util.datestamp",
"url":10,
"doc":"Return date and time string in form DDMMMYY",
"func":1
},
{
"ref":"vipy.util.remkdir",
"url":10,
"doc":"Create a given directory if not already exists",
"func":1
},
{
"ref":"vipy.util.rermdir",
"url":10,
"doc":"Recursively delete a given directory (if exists), and remake it",
"func":1
},
{
"ref":"vipy.util.premkdir",
"url":10,
"doc":"create directory /path/to/subdir if not exist for outfile=/path/to/subdir/file.ext, and return filename",
"func":1
},
{
"ref":"vipy.util.newbase",
"url":10,
"doc":"Convert filename=/a/b/c.ext base=d -> /a/b/d.ext",
"func":1
},
{
"ref":"vipy.util.toextension",
"url":10,
"doc":"Convert filename='/path/to/myfile.ext' to /path/to/myfile.xyz, such that newext='xyz' or newext='.xyz'",
"func":1
},
{
"ref":"vipy.util.topkl",
"url":10,
"doc":"Convert filename='/path/to/myfile.ext' to /path/to/myfile.pkl",
"func":1
},
{
"ref":"vipy.util.splitextension",
"url":10,
"doc":"Given /a/b/c.ext return tuple of strings ('/a/b/c', '.ext')",
"func":1
},
{
"ref":"vipy.util.hasextension",
"url":10,
"doc":"Does the provided filename have a file extension (e.g. /path/to/file.ext) or not (e.g. /path/to/file)",
"func":1
},
{
"ref":"vipy.util.fileext",
"url":10,
"doc":"Given filename /a/b/c.ext return '.ext', or /a/b/c.tar.gz return '.tar.gz'. If multidot=False, then return '.gz'. If withdot=False, return 'ext'",
"func":1
},
{
"ref":"vipy.util.mediaextension",
"url":10,
"doc":"Return '.mp4' for filename='/a/b/c.mp4'",
"func":1
},
{
"ref":"vipy.util.ismacosx",
"url":10,
"doc":"Is the current platform MacOSX?",
"func":1
},
{
"ref":"vipy.util.islinux",
"url":10,
"doc":"is the current platform Linux?",
"func":1
},
{
"ref":"vipy.util.linuxversion",
"url":10,
"doc":"Return linux version",
"func":1
},
{
"ref":"vipy.util.imcrop",
"url":10,
"doc":"Crop a 2D or 3D numpy image given a vipy.geometry.BoundingBox",
"func":1
},
{
"ref":"vipy.util.Failed",
"url":10,
"doc":"Raised when unit test fails to throw an exception"
},
{
"ref":"vipy.util.string_to_pil_interpolation",
"url":10,
"doc":"Internal function to convert interp string to interp object",
"func":1
},
{
"ref":"vipy.dataset",
"url":11,
"doc":""
},
{
"ref":"vipy.dataset.msceleb",
"url":12,
"doc":""
},
{
"ref":"vipy.dataset.msceleb.extract",
"url":12,
"doc":"https: github.com/cmusatyalab/openface/blob/master/data/ms-celeb-1m/extract.py",
"func":1
},
{
"ref":"vipy.dataset.msceleb.export",
"url":12,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface",
"url":13,
"doc":""
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL",
"url":13,
"doc":""
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL.subjects",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL.dataset",
"url":13,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFaceURL.take",
"url":13,
"doc":"Randomly select n frames from dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace",
"url":13,
"doc":""
},
{
"ref":"vipy.dataset.vggface.VGGFace.subjects",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.wordnetid_to_name",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.dataset",
"url":13,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.fastset",
"url":13,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.take",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface.VGGFace.by_subject",
"url":13,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava",
"url":14,
"doc":""
},
{
"ref":"vipy.dataset.ava.AVA",
"url":14,
"doc":"AVA, provide a datadir='/path/to/store/ava'"
},
{
"ref":"vipy.dataset.ava.AVA.download",
"url":14,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava.AVA.categories",
"url":14,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava.AVA.trainset",
"url":14,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ava.AVA.valset",
"url":14,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.aflw",
"url":15,
"doc":""
},
{
"ref":"vipy.dataset.aflw.AFLW",
"url":15,
"doc":""
},
{
"ref":"vipy.dataset.aflw.AFLW.dataset",
"url":15,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.aflw.AFLW.export",
"url":15,
"doc":"Export sqlite database file to aflw.csv",
"func":1
},
{
"ref":"vipy.dataset.aflw.landmarks",
"url":15,
"doc":"Return 21x2 frame array of landmark positions in 1-21 order, NaN if occluded",
"func":1
},
{
"ref":"vipy.dataset.aflw.eyes_nose_chin",
"url":15,
"doc":"Return 4x2 frame array of left eye, right eye nose chin",
"func":1
},
{
"ref":"vipy.dataset.facescrub",
"url":16,
"doc":""
},
{
"ref":"vipy.dataset.facescrub.FaceScrub",
"url":16,
"doc":""
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.parse",
"url":16,
"doc":"Return a list of ImageDetections for all URLs in facescrub",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.download",
"url":16,
"doc":"Download every URL in dataset and store in provided filename",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.validate",
"url":16,
"doc":"Validate downloaded dataset and store cached list of valid bounding boxes and loadable images accessible with dataset()",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.dataset",
"url":16,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.stats",
"url":16,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.subjects",
"url":16,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.facescrub.FaceScrub.split",
"url":16,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.imagenet",
"url":17,
"doc":""
},
{
"ref":"vipy.dataset.imagenet.ImageNet",
"url":17,
"doc":"Provide datadir=/path/to/ILSVRC2012"
},
{
"ref":"vipy.dataset.imagenet.ImageNet.classes",
"url":17,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw",
"url":18,
"doc":""
},
{
"ref":"vipy.dataset.lfw.LFW",
"url":18,
"doc":"Datadir contains the unpacked contents of LFW from $URL -> /path/to/lfw"
},
{
"ref":"vipy.dataset.lfw.LFW.download",
"url":18,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.subjects",
"url":18,
"doc":"List of all subject names",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.subject_images",
"url":18,
"doc":"List of Images of a subject",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.dataset",
"url":18,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.dictionary",
"url":18,
"doc":"List of all Images of all subjects",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.list",
"url":18,
"doc":"List of all Images of all subjects",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.take",
"url":18,
"doc":"Return a represenative list of 128 images",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.pairsDevTest",
"url":18,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.pairsDevTrain",
"url":18,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.lfw.LFW.pairs",
"url":18,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ethzshapes",
"url":19,
"doc":""
},
{
"ref":"vipy.dataset.ethzshapes.ETHZShapes",
"url":19,
"doc":"ETHZShapes, provide a datadir='/path/to/store/ethzshapes'"
},
{
"ref":"vipy.dataset.ethzshapes.ETHZShapes.download_and_unpack",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.ethzshapes.ETHZShapes.dataset",
"url":19,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.fddb",
"url":20,
"doc":""
},
{
"ref":"vipy.dataset.fddb.FDDB",
"url":20,
"doc":"Manages the FDDB dataset: http: vis-www.cs.umass.edu/fddb"
},
{
"ref":"vipy.dataset.fddb.FDDB.fold",
"url":20,
"doc":"Return the foldnum as a list of vipy.image.Scene objects, each containing all vipy.object.Detection faces in the current image",
"func":1
},
{
"ref":"vipy.dataset.megaface",
"url":21,
"doc":""
},
{
"ref":"vipy.dataset.megaface.MF2",
"url":21,
"doc":""
},
{
"ref":"vipy.dataset.megaface.MF2.tinyset",
"url":21,
"doc":"Return the first (size) image objects in the trainset",
"func":1
},
{
"ref":"vipy.dataset.megaface.Megaface",
"url":21,
"doc":""
},
{
"ref":"vipy.dataset.megaface.Megaface.tinyset",
"url":21,
"doc":"Return the first (size) image objects in the dataset",
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
"ref":"vipy.dataset.vggface2",
"url":23,
"doc":""
},
{
"ref":"vipy.dataset.vggface2.VGGFace2",
"url":23,
"doc":""
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.subjects",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.wordnetid_to_name",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.vggface2_to_vggface1",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.name_to_wordnetid",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.names",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.trainset",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.testset",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.split",
"url":23,
"doc":"Convert absolute path /path/to/subjectid/filename.jpg from training or testing set to (subjectid, filename.jpg)",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.frontalset",
"url":23,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.dataset",
"url":23,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.fastset",
"url":23,
"doc":"Return a generator to iterate over dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.take",
"url":23,
"doc":"Randomly select n images from the dataset, or n images of a given subjectid",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.take_per_subject",
"url":23,
"doc":"Randomly select n images per subject from the dataset",
"func":1
},
{
"ref":"vipy.dataset.vggface2.VGGFace2.subjectset",
"url":23,
"doc":"Iterator for single subject",
"func":1
},
{
"ref":"vipy.dataset.caltech101",
"url":24,
"doc":""
},
{
"ref":"vipy.dataset.caltech101.Caltech101",
"url":24,
"doc":"Caltech101, provide a datadir='/path/to/store/caltech101'"
},
{
"ref":"vipy.dataset.caltech101.Caltech101.download_and_unpack",
"url":24,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech101.Caltech101.dataset",
"url":24,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces",
"url":25,
"doc":""
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces",
"url":25,
"doc":""
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.subjects",
"url":25,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.videos",
"url":25,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.parse",
"url":25,
"doc":"Parse youtubefaces into a list of ImageDetections",
"func":1
},
{
"ref":"vipy.dataset.youtubefaces.YouTubeFaces.splits",
"url":25,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.hmdb",
"url":26,
"doc":""
},
{
"ref":"vipy.dataset.hmdb.HMDB",
"url":26,
"doc":"Human motion dataset, provide a datadir='/path/to/store/hmdb'"
},
{
"ref":"vipy.dataset.hmdb.HMDB.download",
"url":26,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.hmdb.HMDB.dataset",
"url":26,
"doc":"Return a list of VideoCategory objects",
"func":1
},
{
"ref":"vipy.dataset.casia",
"url":27,
"doc":""
},
{
"ref":"vipy.dataset.casia.WebFace",
"url":27,
"doc":""
},
{
"ref":"vipy.dataset.casia.WebFace.dataset",
"url":27,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.casia.WebFace.subjects",
"url":27,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.casia.WebFace.subjectid",
"url":27,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.meva",
"url":28,
"doc":""
},
{
"ref":"vipy.dataset.meva.KF1",
"url":28,
"doc":"Parse MEVA annotations (http: mevadata.org) for KNown Facility 1 dataset into vipy.video.Scene() objects Kwiver packet format: https: gitlab.kitware.com/meva/meva-data-repo/blob/master/documents/KPF-specification-v4.pdf Inputs: -videodir=str: path to Directory containing 'drop-01' -repodir=str: path to directory containing clone of https: gitlab.kitware.com/meva/meva-data-repo -stride=int: the temporal stride in frames for importing bounding boxes, vipy will do linear interpoluation and boundary handling -n_videos=int: only return an integer number of videos, useful for debugging or for previewing dataset -withprefix=list: only return videos with the filename containing one of the strings in withprefix list, useful for debugging -contrib=bool: include the noisy contrib anntations from DIVA performers -d_category_to_shortlabel is a dictionary mapping category names to a short displayed label on the video. The standard for visualization is that tracked objects are displayed with their category label (e.g. 'Person', 'Vehicle'), and activities are labeled according to the set of objects that performing the activity. When an activity occurs, the set of objects are labeled with the same color as 'Noun Verbing' (e.g. 'Person Entering', 'Person Reading', 'Vehicle Starting') where 'Verbing' is provided by the shortlabel. This is optional, and will use the default mapping if None -verbose=bool: Parsing verbosity -merge: deduplicate annotations for each video across YAML files by merging them by mean spatial IoU per track (>0.5) and temporal IoU (>0) -actor [bool]: Include only those activities that include an associated track for the primary actor: \"Person\" for \"person_ \" and \"hand_ \", else \"Vehicle\" -disjoint [bool]: Enforce that overlapping causal activities (open/close, enter/exit,  .) are disjoint for a track -unpad [bool]: remove the arbitrary padding assigned during dataset creation"
},
{
"ref":"vipy.dataset.meva.KF1.videos",
"url":28,
"doc":"Return list of activity videos",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.tolist",
"url":28,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.instances",
"url":28,
"doc":"Return list of activity instances",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.categories",
"url":28,
"doc":"Return a list of activity categories",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.analysis",
"url":28,
"doc":"Analyze the MEVA dataset to return helpful statistics and plots",
"func":1
},
{
"ref":"vipy.dataset.meva.KF1.review",
"url":28,
"doc":"Generate a standalone HTML file containing quicklooks for each annotated activity in dataset, along with some helpful provenance information for where the annotation came from",
"func":1
},
{
"ref":"vipy.dataset.charades",
"url":29,
"doc":""
},
{
"ref":"vipy.dataset.charades.Charades",
"url":29,
"doc":"Charades, provide paths such that datadir contains the contents of 'http: ai2-website.s3.amazonaws.com/data/Charades_v1.zip' and annodir contains 'http: ai2-website.s3.amazonaws.com/data/Charades.zip'"
},
{
"ref":"vipy.dataset.charades.Charades.categories",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades.Charades.trainset",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades.Charades.testset",
"url":29,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.charades.Charades.review",
"url":29,
"doc":"Generate a standalone HTML file containing quicklooks for each annotated activity in the train set",
"func":1
},
{
"ref":"vipy.dataset.momentsintime",
"url":30,
"doc":""
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments",
"url":30,
"doc":"Multi-Moments in Time: http: moments.csail.mit.edu/ >>> d = MultiMoments('/path/to/dir') >>> valset = d.valset() >>> valset.categories()  return the dictionary mapping integer category to string >>> valset[1].categories()  return set of categories for this clip >>> valset[1].category()  return string encoded category for this clip (comma separated activity indexes) >>> valset[1].play()  Play the original clip >>> valset[1].mindim(224).show()  Resize the clip to have minimum dimension 224, then show the modified clip >>> valset[1].centersquare().mindim(112).saveas('out.mp4')  modify the clip as square crop from the center with mindim=112, and save to new file >>> valset[1].centersquare().mindim(112).normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225 .torch(startframe=0, length=16)  export 16x3x112x112 tensor"
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments.categories",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments.trainset",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.momentsintime.MultiMoments.valset",
"url":30,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics",
"url":31,
"doc":""
},
{
"ref":"vipy.dataset.kinetics.Kinetics700",
"url":31,
"doc":"Kinetics, provide a datadir='/path/to/store/kinetics'"
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.download",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.isdownloaded",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.trainset",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.testset",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.valset",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.categories",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics700.analysis",
"url":31,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.kinetics.Kinetics600",
"url":31,
"doc":"Kinetics, provide a datadir='/path/to/store/kinetics'"
},
{
"ref":"vipy.dataset.kinetics.Kinetics400",
"url":31,
"doc":"Kinetics, provide a datadir='/path/to/store/kinetics'"
},
{
"ref":"vipy.dataset.mnist",
"url":32,
"doc":""
},
{
"ref":"vipy.dataset.mnist.MNIST",
"url":32,
"doc":"download URLS above to outdir, then run export()"
},
{
"ref":"vipy.dataset.mnist.MNIST.trainset",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.mnist.MNIST.testset",
"url":32,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech256",
"url":33,
"doc":""
},
{
"ref":"vipy.dataset.caltech256.Caltech256",
"url":33,
"doc":"Caltech256, provide a datadir='/path/to/store/caltech256'"
},
{
"ref":"vipy.dataset.caltech256.Caltech256.download_and_unpack",
"url":33,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.caltech256.Caltech256.dataset",
"url":33,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet",
"url":34,
"doc":""
},
{
"ref":"vipy.dataset.activitynet.ActivityNet",
"url":34,
"doc":"Activitynet, provide a datadir='/path/to/store/activitynet'"
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.download",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.trainset",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.testset",
"url":34,
"doc":"ActivityNet test set does not include any annotations",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.valset",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.categories",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.dataset.activitynet.ActivityNet.analysis",
"url":34,
"doc":"",
"func":1
},
{
"ref":"vipy.globals",
"url":35,
"doc":""
},
{
"ref":"vipy.globals.logging",
"url":35,
"doc":"Single entry point for enabling/disabling logging vs. printing All vipy functions overload \"from vipy.globals import print\" for simplified readability of code. This global function redirects print or warn to using the standard logging module. If format is provided, this will create a basicConfig handler, but this should be configured by the end-user.",
"func":1
},
{
"ref":"vipy.globals.warn",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.print",
"url":35,
"doc":"Main entry point for all print statements in the vipy package. All vipy code calls this to print helpful messages. -Printing can be disabled by calling vipy.globals.silent() -Printing can be redirected to logging by calling vipy.globals.logging(True) -All print() statements in vipy. are overloaded to call vipy.globals.print() so that it can be redirected to logging",
"func":1
},
{
"ref":"vipy.globals.verbose",
"url":35,
"doc":"The global verbosity level, only really used right now for FFMPEG messages",
"func":1
},
{
"ref":"vipy.globals.isverbose",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.silent",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.issilent",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.verbosity",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.debug",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.isdebug",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.cache",
"url":35,
"doc":"The cache is the location that URLs are downloaded to on your system. This can be set here, or with the environment variable VIPY_CACHE",
"func":1
},
{
"ref":"vipy.globals.user_hit_escape",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.cpuonly",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.gpuindex",
"url":35,
"doc":"",
"func":1
},
{
"ref":"vipy.globals.dask",
"url":35,
"doc":"Return the current Dask client, can be accessed globally for parallel processing. -pct: [0,1] the percentage of the current machine to use -address: the dask scheduler of the form 'HOSTNAME:PORT' -num_processes: the number of prpcesses to use on the current machine -num_gpus: the number of GPUs to use on the current machine",
"func":1
},
{
"ref":"vipy.globals.parallel",
"url":35,
"doc":"Enable parallel processing with n>=1 processes. >>> with vipy.globals.parallel(n=4): vipy.batch.Batch( .)",
"func":1
},
{
"ref":"vipy.globals.noparallel",
"url":35,
"doc":"Disable all parallel processing",
"func":1
},
{
"ref":"vipy.globals.nodask",
"url":35,
"doc":"Alias for noparallel()",
"func":1
},
{
"ref":"vipy.batch",
"url":36,
"doc":""
},
{
"ref":"vipy.batch.Dask",
"url":36,
"doc":""
},
{
"ref":"vipy.batch.Dask.num_gpus",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.has_dashboard",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.dashboard",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.num_processes",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.shutdown",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Dask.client",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Checkpoint",
"url":36,
"doc":"Batch checkpoints for long running jobs"
},
{
"ref":"vipy.batch.Checkpoint.checkpoint",
"url":36,
"doc":"Return the last checkpointed result. Useful for recovering from dask crashes for long jobs.",
"func":1
},
{
"ref":"vipy.batch.Checkpoint.last_archive",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Batch",
"url":36,
"doc":"vipy.batch.Batch class This class provides a representation of a set of vipy objects. All of the object types must be the same. If so, then an operation on the batch is performed on each of the elements in the batch in parallel. Examples: >>> b = vipy.batch.Batch([Image(filename='img_%06d.png' % k) for k in range(0,100)]) >>> b.map(lambda im: im.bgr( >>> b.map(lambda im: np.sum(im.array( ) >>> b.map(lambda im, f: im.saveas(f), args=['out%d.jpg' % k for k in range(0,100)]) >>> v = vipy.video.RandomSceneActivity() >>> b = vipy.batch.Batch(v, n_processes=16) >>> b.map(lambda v,k: v[k], args=[(k,) for k in range(0, len(v ])  paralle interpolation >>> d = vipy.dataset.kinetics.Kinetics700('/path/to/kinetics').download().trainset() >>> b = vipy.batch.Batch(d, n_processes=32) >>> b.map(lambda v: v.download().save(  will download and clip dataset in parallel >>> b.result()  retrieve results after a sequence of map or filter chains Parameters: -strict=False: if distributed processing fails, return None for that element and print the exception rather than raise -as_completed=True: Return the objects to the scheduler as they complete, this can introduce instabilities for large complex objects, use with caution Create a batch of homogeneous vipy.image objects from an iterable that can be operated on with a single parallel function call"
},
{
"ref":"vipy.batch.Batch.restore",
"url":36,
"doc":"",
"func":1
},
{
"ref":"vipy.batch.Batch.result",
"url":36,
"doc":"Return the result of the batch processing, ordered",
"func":1
},
{
"ref":"vipy.batch.Batch.map",
"url":36,
"doc":"Run the lambda function on each of the elements of the batch and return the batch object. >>> iml = [vipy.image.RandomScene(512,512) for k in range(0,1000)] >>> imb = vipy.image.Batch(iml) >>> imb.map(lambda im: im.rgb( The lambda function f_lambda must not include closures. If it does, construct the batch with tuples (obj,prms) or with default parameter capture: >>> f = lambda x, prm1=1, prm2=2: x+prm1+prm2",
"func":1
},
{
"ref":"vipy.batch.Batch.filter",
"url":36,
"doc":"Run the lambda function on each of the elements of the batch and filter based on the provided lambda keeping those elements that return true",
"func":1
},
{
"ref":"vipy.batch.Batch.scattermap",
"url":36,
"doc":"Scatter obj to all workers, and apply lambda function f(obj, im) to each element in batch Usage: >>> Batch(mylist, ngpu=8).scattermap(lambda net, im: net(im), net).result() This will scatter the large object net to all workers, and pin it to a specific GPU. Within the net object, you can call vipy.global.gpuindex() to retrieve your assigned GPU index, which can be used by torch.cuda.device(). Then, the net object processes each element in the batch using net according to the lambda, and returns the results. This function includes ngpu processes, and assumes there are ngpu available on the target machine. Each net is replicated in a different process, so it is the callers responsibility for getting vipy.global.gpuindex() from within the process and setting net to take advantage of this GPU rather than using the default cuda:0.",
"func":1
},
{
"ref":"vipy.batch.Batch.checkpoint",
"url":36,
"doc":"Return the last checkpointed result. Useful for recovering from dask crashes for long jobs.",
"func":1
},
{
"ref":"vipy.activity",
"url":37,
"doc":""
},
{
"ref":"vipy.activity.Activity",
"url":37,
"doc":"vipy.object.Activity class An activity is a grouping of one or more tracks involved in an activity within a given startframe and endframe. The activity occurs at a given (startframe, endframe), where these frame indexes are extracted at the provided framerate. All objects are passed by reference with a globally unique track ID, for the tracks involved with the activity. This is done since tracks can exist after an activity completes, and that tracks should update the spatial transformation of boxes. The shortlabel defines the string shown on the visualization video. Valid constructors >>> t = vipy.object.Track(category='Person').add( . >>> a = vipy.object.Activity(startframe=0, endframe=10, category='Walking', tracks={t.id():t})"
},
{
"ref":"vipy.activity.Activity.hasattribute",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.confidence",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.from_json",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.dict",
"url":37,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.activity.Activity.json",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.actorid",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.startframe",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.endframe",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.middleframe",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.framerate",
"url":37,
"doc":"Resample (startframe, endframe) from known original framerate set by constructor to be new framerate fps",
"func":1
},
{
"ref":"vipy.activity.Activity.category",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.label",
"url":37,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.activity.Activity.shortlabel",
"url":37,
"doc":"A optional shorter label string to show in the visualizations",
"func":1
},
{
"ref":"vipy.activity.Activity.add",
"url":37,
"doc":"Add the track id for the track to this activity, so that if the track is changed externally it is reflected here",
"func":1
},
{
"ref":"vipy.activity.Activity.tracks",
"url":37,
"doc":"alias for trackids",
"func":1
},
{
"ref":"vipy.activity.Activity.trackids",
"url":37,
"doc":"Return a set of track IDs associated with this activity",
"func":1
},
{
"ref":"vipy.activity.Activity.hasoverlap",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.isneighbor",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.hastrack",
"url":37,
"doc":"Is the track part of the activity?",
"func":1
},
{
"ref":"vipy.activity.Activity.append",
"url":37,
"doc":"Append newtrack to this activity and set as actorid()",
"func":1
},
{
"ref":"vipy.activity.Activity.trackfilter",
"url":37,
"doc":"Remove all tracks such that the lambda function f(trackid) resolves to False",
"func":1
},
{
"ref":"vipy.activity.Activity.replace",
"url":37,
"doc":"Replace oldtrack with newtrack if present in self._tracks. Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change",
"func":1
},
{
"ref":"vipy.activity.Activity.replaceid",
"url":37,
"doc":"Replace oldtrack with newtrack if present in self._tracks. Pass in a trackdict to share reference to track, so that track owner can modify the track and this object observes the change",
"func":1
},
{
"ref":"vipy.activity.Activity.during",
"url":37,
"doc":"Is frame during the time interval (startframe, endframe) inclusive?",
"func":1
},
{
"ref":"vipy.activity.Activity.during_interval",
"url":37,
"doc":"Is the activity occurring for any frames within the interval [startframe, endframe) (non-inclusive of endframe)?",
"func":1
},
{
"ref":"vipy.activity.Activity.union",
"url":37,
"doc":"Compute the union of the new activity other to this activity by updating the start and end times and computing the mean confidence. -Note: other must have the same category and track IDs as self -confweight [0,1]: the convex combinatiopn weight applied to the new activity",
"func":1
},
{
"ref":"vipy.activity.Activity.temporal_iou",
"url":37,
"doc":"Return the temporal intersection over union of two activities",
"func":1
},
{
"ref":"vipy.activity.Activity.offset",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.truncate",
"url":37,
"doc":"Truncate the activity so that it is between startframe and endframe",
"func":1
},
{
"ref":"vipy.activity.Activity.id",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.clone",
"url":37,
"doc":"",
"func":1
},
{
"ref":"vipy.activity.Activity.temporalpad",
"url":37,
"doc":"Add a temporal pad of df=(before,after) or df=pad frames to the start and end of the activity. The padded start frame may be negative.",
"func":1
},
{
"ref":"vipy.activity.Activity.padto",
"url":37,
"doc":"Add a symmetric temporal pad so that the activity is at least t seconds long",
"func":1
},
{
"ref":"vipy.activity.Activity.disjoint",
"url":37,
"doc":"Enforce disjoint activities with other by shifting the endframe or startframe of self to not overlap if they share the same tracks. Other may be an Activity() or list of Activity() if strict=True, then throw an exception if other or self is fully contained with the other, resulting in degenerate activity after disjoint",
"func":1
},
{
"ref":"vipy.activity.Activity.temporal_distance",
"url":37,
"doc":"Return the temporal distance in frames between self and other which is the minimum frame difference between the end of one to the start of the other, or zero if they overlap",
"func":1
},
{
"ref":"vipy.flow",
"url":38,
"doc":""
},
{
"ref":"vipy.flow.Image",
"url":38,
"doc":"vipy.flow.Image() class"
},
{
"ref":"vipy.flow.Image.min",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.max",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.scale",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.threshold",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.width",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.height",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.shape",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.flow",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.colorflow",
"url":38,
"doc":"Flow visualization image (HSV: H=flow angle, V=flow magnitude), returns vipy.image.Image()",
"func":1
},
{
"ref":"vipy.flow.Image.warp",
"url":38,
"doc":"Warp image imfrom=vipy.image.Image() to imto=vipy.image.Image() using flow computed as imfrom->imto, updating objects",
"func":1
},
{
"ref":"vipy.flow.Image.alphapad",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.zeropad",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.dx",
"url":38,
"doc":"Return dx (horizontal) component of flow",
"func":1
},
{
"ref":"vipy.flow.Image.dy",
"url":38,
"doc":"Return dy (vertical) component of flow",
"func":1
},
{
"ref":"vipy.flow.Image.shift",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.show",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.rescale",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.resize_like",
"url":38,
"doc":"Resize flow buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.flow.Image.resize",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.magnitude",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.angle",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.clone",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Image.print",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video",
"url":38,
"doc":"vipy.flow.Video() class"
},
{
"ref":"vipy.flow.Video.min",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.max",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.width",
"url":38,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.flow.Video.height",
"url":38,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.flow.Video.flow",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.colorflow",
"url":38,
"doc":"Flow visualization video",
"func":1
},
{
"ref":"vipy.flow.Video.magnitude",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.show",
"url":38,
"doc":"",
"func":1
},
{
"ref":"vipy.flow.Video.print",
"url":38,
"doc":"Print the representation of the video - useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing",
"func":1
},
{
"ref":"vipy.flow.Video.metadata",
"url":39,
"doc":"Return a dictionary of metadata about this video",
"func":1
},
{
"ref":"vipy.flow.Video.videoid",
"url":39,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by hashing the filename() and url(). Notes: - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', $MY_ID)",
"func":1
},
{
"ref":"vipy.flow.Video.frame",
"url":39,
"doc":"Alias for self.__getitem__[k]",
"func":1
},
{
"ref":"vipy.flow.Video.store",
"url":39,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.flow.Video.unstore",
"url":39,
"doc":"Delete the currently stored video from store()",
"func":1
},
{
"ref":"vipy.flow.Video.restore",
"url":39,
"doc":"Save the currently stored video to filename, and set up filename",
"func":1
},
{
"ref":"vipy.flow.Video.stream",
"url":39,
"doc":"Iterator to yield frames streaming from video  Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.  FFMPEG stdout pipe may screw up bash shell newlines, requiring issuing command \"reset\"",
"func":1
},
{
"ref":"vipy.flow.Video.clear",
"url":39,
"doc":"no-op for Video()",
"func":1
},
{
"ref":"vipy.flow.Video.bytes",
"url":39,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.flow.Video.frames",
"url":39,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.flow.Video.isdirty",
"url":39,
"doc":"Has the FFMPEG filter chain been modified from the default? If so, then ffplay() on the video file will be different from self.load().play()",
"func":1
},
{
"ref":"vipy.flow.Video.probeshape",
"url":39,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe, this does not take into account any applied ffmpeg filters",
"func":1
},
{
"ref":"vipy.flow.Video.duration_in_seconds_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache",
"func":1
},
{
"ref":"vipy.flow.Video.duration_in_frames_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in frames, requires ffprobe",
"func":1
},
{
"ref":"vipy.flow.Video.probe",
"url":39,
"doc":"Run ffprobe on the filename and return the result as a JSON file",
"func":1
},
{
"ref":"vipy.flow.Video.dict",
"url":39,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.flow.Video.take",
"url":39,
"doc":"Return n frames from the clip uniformly spaced as numpy array",
"func":1
},
{
"ref":"vipy.flow.Video.framerate",
"url":39,
"doc":"Change the input framerate for the video and update frame indexes for all annotations  NOTE: do not call framerate() after calling clip() as this introduces extra repeated final frames during load()",
"func":1
},
{
"ref":"vipy.flow.Video.colorspace",
"url":39,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.flow.Video.url",
"url":39,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.flow.Video.isloaded",
"url":39,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.flow.Video.canload",
"url":39,
"doc":"Return True if the video can be loaded successfully, useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version",
"func":1
},
{
"ref":"vipy.flow.Video.fromarray",
"url":39,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.flow.Video.fromframes",
"url":39,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.flow.Video.tonumpy",
"url":39,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.flow.Video.numpy",
"url":39,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.flow.Video.filename",
"url":39,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.flow.Video.abspath",
"url":39,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.flow.Video.relpath",
"url":39,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.flow.Video.rename",
"url":39,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.flow.Video.filesize",
"url":39,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.flow.Video.download",
"url":39,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.flow.Video.fetch",
"url":39,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.flow.Video.shape",
"url":39,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.flow.Video.channels",
"url":39,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.flow.Video.aspect_ratio",
"url":39,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.flow.Video.preview",
"url":39,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.flow.Video.thumbnail",
"url":39,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile",
"func":1
},
{
"ref":"vipy.flow.Video.load",
"url":39,
"doc":"Load a video using ffmpeg, applying the requested filter chain. - If verbose=True. then ffmpeg console output will be displayed. - If ignoreErrors=True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. - shape tuple(height, width, channels): If provided, use this shape for reading and reshaping the byte stream from ffmpeg - knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size",
"func":1
},
{
"ref":"vipy.flow.Video.speed",
"url":39,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.flow.Video.clip",
"url":39,
"doc":"Load a video clip betweeen start and end frames",
"func":1
},
{
"ref":"vipy.flow.Video.cliptime",
"url":39,
"doc":"Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly",
"func":1
},
{
"ref":"vipy.flow.Video.rot90cw",
"url":39,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.rot90ccw",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.fliplr",
"url":39,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.flow.Video.flipud",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.rescale",
"url":39,
"doc":"Rescale the video by factor s, such that the new dimensions are (s H, s W), can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.flow.Video.resize",
"url":39,
"doc":"Resize the video to be (rows=height, cols=width)",
"func":1
},
{
"ref":"vipy.flow.Video.mindim",
"url":39,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.flow.Video.maxdim",
"url":39,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.flow.Video.randomcrop",
"url":39,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.flow.Video.centercrop",
"url":39,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.flow.Video.centersquare",
"url":39,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.flow.Video.cropeven",
"url":39,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.flow.Video.maxsquare",
"url":39,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.flow.Video.zeropad",
"url":39,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after  NOTE: Older FFMPEG implementations can throw the error \"Input area  : : : not within the padded area  : : : or zero-sized, this is often caused by odd sized padding. Recommend calling self.cropeven().zeropad( .) to avoid this",
"func":1
},
{
"ref":"vipy.flow.Video.crop",
"url":39,
"doc":"Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().",
"func":1
},
{
"ref":"vipy.flow.Video.pkl",
"url":39,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.flow.Video.pklif",
"url":39,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.flow.Video.webp",
"url":39,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. -strict=[bool]: assert that the filename must have an .webp extension -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.flow.Video.gif",
"url":39,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. WARNING: this will be very large for big videos, consider using webp instead. -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.flow.Video.saveas",
"url":39,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file.  outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image)  If self.array() is loaded, then export the contents of self._array to the video file  If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video  If outfile None or outfile self.filename(), then overwrite the current filename  If ignoreErrors=True, then exit gracefully. Useful for chaining download().saveas() on parallel dataset downloads  Returns a new video object with this video filename, and a clean video filter chain  if flush=True, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel  framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video  pause: an integer in seconds to pause between loops of animated images",
"func":1
},
{
"ref":"vipy.flow.Video.ffplay",
"url":39,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.flow.Video.play",
"url":39,
"doc":"Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it, if the filter chain is dirty, dump to temp file first",
"func":1
},
{
"ref":"vipy.flow.Video.quicklook",
"url":39,
"doc":"Generate a montage of n uniformly spaced frames. Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. Input: -n: Number of images in the quicklook -mindim: The minimum dimension of each of the elements in the montage -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.flow.Video.torch",
"url":39,
"doc":"Convert the loaded video of shape N HxWxC frames to an MxCxHxW torch tensor, forces a load().  Order of arguments is (startframe, endframe) or (startframe, startframe+length) or (random_startframe, random_starframe+takelength), then stride or take.  Follows numpy slicing rules. Optionally return the slice used if withslice=True  Returns float tensor in the range [0,1] following torchvision.transforms.ToTensor()  order can be ['nchw', 'nhwc', 'cnhw'] for batchsize=n, channels=c, height=h, width=w  boundary can be ['repeat', 'strict', 'cyclic']  withlabel=True, returns tuple (t, labellist), where labellist is a list of tuples of activity labels occurring at the corresponding frame in the tensor  withslice=Trye, returnss tuple (t, (startframe, endframe, stride  nonelabel=True, returns tuple (t, None) if withlabel=False",
"func":1
},
{
"ref":"vipy.flow.Video.clone",
"url":39,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.  flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos  rekey: Generate new unique track ID and activity ID keys for this scene  shallow: shallow copy everything (copy by reference), except for ffmpeg object  sharedarray: deep copy of everything, except for pixel buffer which is shared",
"func":1
},
{
"ref":"vipy.flow.Video.flush",
"url":39,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.flow.Video.flush_and_return",
"url":39,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.flow.Video.map",
"url":39,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.flow.Video.gain",
"url":39,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.flow.Video.bias",
"url":39,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.flow.Video.normalize",
"url":39,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.flow.Flow",
"url":38,
"doc":"vipy.flow.Flow() class"
},
{
"ref":"vipy.flow.Flow.imageflow",
"url":38,
"doc":"Default opencv dense flow, from im to imprev. This should be overloaded",
"func":1
},
{
"ref":"vipy.flow.Flow.videoflow",
"url":38,
"doc":"Compute optical flow for a video framewise skipping framestep frames, compute optical flow acrsos flowstep frames,",
"func":1
},
{
"ref":"vipy.flow.Flow.videoflowframe",
"url":38,
"doc":"Computer the videoflow for a single frame",
"func":1
},
{
"ref":"vipy.flow.Flow.keyflow",
"url":38,
"doc":"Compute optical flow for a video framewise relative to keyframes separated by keystep",
"func":1
},
{
"ref":"vipy.flow.Flow.keyflowframe",
"url":38,
"doc":"Compute the keyflow for a single frame",
"func":1
},
{
"ref":"vipy.flow.Flow.affineflow",
"url":38,
"doc":"Return a flow field of size (height=H, width=W) consistent with a 2x3 affine transformation A",
"func":1
},
{
"ref":"vipy.flow.Flow.euclideanflow",
"url":38,
"doc":"Return a flow field of size (height=H, width=W) consistent with an Euclidean transform parameterized by a 2x2 Rotation and 2x1 translation",
"func":1
},
{
"ref":"vipy.flow.Flow.stabilize",
"url":38,
"doc":"Affine stabilization to frame zero using multi-scale optical flow correspondence with foreground object keepouts.  v [vipy.video.Scene]: The input video to stabilize, should be resized to mindim=256  keystep [int]: The local stabilization step between keyframes (should be <= 30)  padheightfrac [float]: The height padding (relative to video height) to be applied to output video to allow for vertical stabilization  padwidthfrac [float]: The width padding (relative to video width) to be applied to output video to allow for horizontal stabilization  padheightpx [int]: The height padding to be applied to output video to allow for vertical stabilization. Overrides padheight.  padwidthpx [int]: The width padding to be applied to output video to allow for horizontal stabilization. Overrides padwidth.  border [float]: The border keepout fraction to ignore during flow correspondence. This should be proportional to the maximum frame to frame flow  dilate [float]: The dilation to apply to the foreground object boxes to define a foregroun keepout for flow computation  contrast [float]: The minimum gradient necessary for flow correspondence, to avoid flow on low contrast regions  rigid [bool]: Euclidean stabilization  affine [bool]: Affine stabilization  verbose [bool]: This takes a while to run  .  strict [bool]: If true, throw an exception on error, otherwise return the original video and set v.hasattribute('unstabilized'), useful for large scale stabilization  outfile [str]: the file path to the stabilized output video  NOTE: The remaining distortion after stabilization is due to: rolling shutter distortion, perspective distortion and non-keepout moving objects in background  NOTE: If the video contains objects, the object boxes will be transformed along with the stabilization  NOTE: This requires loading videos entirely into memory. Be careful with stabilizing long videos.",
"func":1
},
{
"ref":"vipy.videosearch",
"url":40,
"doc":""
},
{
"ref":"vipy.videosearch.isactiveyoutuber",
"url":40,
"doc":"Does the youtube user have any uploaded videos?",
"func":1
},
{
"ref":"vipy.videosearch.youtubeuser",
"url":40,
"doc":"return all unique /user/ urls returned for a search for a given query tag",
"func":1
},
{
"ref":"vipy.videosearch.is_downloadable_url",
"url":40,
"doc":"Check to see if youtube-dl can download the path, this requires exeecuting 'youtube-dl $URL -q -j' to see if the returncode is non-zero",
"func":1
},
{
"ref":"vipy.videosearch.youtube",
"url":40,
"doc":"Return a list of YouTube URLs for the given tag and optional channel",
"func":1
},
{
"ref":"vipy.videosearch.liveleak",
"url":40,
"doc":"",
"func":1
},
{
"ref":"vipy.videosearch.download",
"url":40,
"doc":"Use youtube-dl to download a video URL to a video file",
"func":1
},
{
"ref":"vipy.videosearch.bulkdownload",
"url":40,
"doc":"Use youtube-dl to download a list of video URLs to video files using the provided sprintf outpattern=/path/to/out_%d.mp4 where the index is provided by the URL list index",
"func":1
},
{
"ref":"vipy.visualize",
"url":41,
"doc":""
},
{
"ref":"vipy.visualize.montage",
"url":41,
"doc":"Create a montage image from the of provided list of vipy.image.Image objects. Inputs:  imlist: iterable of vipy.image.Image objects which is used to montage rowwise  (imgheight, imgwidth): the size of each individual image in the grid  (gridrows, gridcols): The number of images per row, and number of images per column. This defines the montage shape.  aspectratio. This is an optional parameter which defines the shape of the montage as (gridcols/gridrows) without specifying the gridrows, gridcols input  crop=[True|False]: Whether the vipy.image.Image objects should call crop(), which will trigger a load  skip=[True|False]: Whether images should be skipped on failure to load(), useful for lazy downloading  border: a border of size in pixels surrounding each image in the grid  border_bgr: the border color in a bgr color tuple (b, g, r) in [0,255], uint8  do_flush=[True|False]: flush the loaded images as garbage collection for large montages  verbose=[True|False]: display optional verbose messages Outputs:  Return a vipy.image.Image montage which is of size (gridrows (imgheight + 2 border), gridcols (imgwidth+2 border ",
"func":1
},
{
"ref":"vipy.visualize.videomontage",
"url":41,
"doc":"Generate a video montage for the provided videos by creating a image montage for every frame. This loads every video into memory, so be careful with large montages!",
"func":1
},
{
"ref":"vipy.visualize.urls",
"url":41,
"doc":"Given a list of public image URLs, create a stand-alone HTML page to show them all",
"func":1
},
{
"ref":"vipy.visualize.tohtml",
"url":41,
"doc":"Given a list of vipy.image.Image objects, show the images along with the dictionary contents of imdict (one per image) in a single standalone HTML file",
"func":1
},
{
"ref":"vipy.visualize.imagelist",
"url":41,
"doc":"Given a list of image filenames wth absolute paths, copy to outdir, and create an index.html file that visualizes each",
"func":1
},
{
"ref":"vipy.visualize.imagetuplelist",
"url":41,
"doc":"Imageset but put tuples on same row",
"func":1
},
{
"ref":"vipy.dropbox",
"url":42,
"doc":""
},
{
"ref":"vipy.dropbox.Dropbox",
"url":42,
"doc":""
},
{
"ref":"vipy.dropbox.Dropbox.link",
"url":42,
"doc":"",
"func":1
},
{
"ref":"vipy.dropbox.Dropbox.put",
"url":42,
"doc":"",
"func":1
},
{
"ref":"vipy.dropbox.Dropbox.get",
"url":42,
"doc":"",
"func":1
},
{
"ref":"vipy.camera",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.CAM",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.FRAMERATE",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.TIC",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.TOC",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.RESIZE",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.GREY",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Camera.PROCESS",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Webcam",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Webcam.current",
"url":43,
"doc":"",
"func":1
},
{
"ref":"vipy.camera.Webcam.next",
"url":43,
"doc":"",
"func":1
},
{
"ref":"vipy.camera.Flow",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Flow.IMCURRENT",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Flow.IMPREV",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Flow.next",
"url":43,
"doc":"",
"func":1
},
{
"ref":"vipy.camera.Ipcam",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Ipcam.TMPFILE",
"url":43,
"doc":""
},
{
"ref":"vipy.camera.Ipcam.next",
"url":43,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg",
"url":44,
"doc":""
},
{
"ref":"vipy.linalg.random_positive_semidefinite_matrix",
"url":44,
"doc":"Return a randomly generated float64 positive semidefinite matrix of size NxN",
"func":1
},
{
"ref":"vipy.linalg.column_stochastic",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.row_stochastic",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.rowstochastic",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.bistochastic",
"url":44,
"doc":"Sinkhorn normalization",
"func":1
},
{
"ref":"vipy.linalg.rectangular_bistochastic",
"url":44,
"doc":"Sinkhorn normalization for rectangular matrices",
"func":1
},
{
"ref":"vipy.linalg.row_normalized",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.row_ssqrt",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.normalize",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.vectorize",
"url":44,
"doc":"Convert a tuple X=([1], [2,3], [4,5,6]) to a vector [1,2,3,4,5,6]",
"func":1
},
{
"ref":"vipy.linalg.columnvector",
"url":44,
"doc":"Convert a tuple with N elements to an Nx1 column vector",
"func":1
},
{
"ref":"vipy.linalg.columnize",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.rowvector",
"url":44,
"doc":"Convert a tuple with N elements to an 1xN row vector",
"func":1
},
{
"ref":"vipy.linalg.poweroftwo",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.ndmax",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.linalg.ndmin",
"url":44,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry",
"url":9,
"doc":""
},
{
"ref":"vipy.geometry.covariance_to_ellipse",
"url":9,
"doc":"2x2 covariance matrix to ellipse (major_axis_length, minor_axis_length, angle_in_radians)",
"func":1
},
{
"ref":"vipy.geometry.dehomogenize",
"url":9,
"doc":"Convert 3x1 homogenous point (x,y,h) to 2x1 non-homogenous point (x/h, y/h)",
"func":1
},
{
"ref":"vipy.geometry.homogenize",
"url":9,
"doc":"Convert 2xN non-homogenous points (x,y) to 3xN non-homogenous point (x, y, 1)",
"func":1
},
{
"ref":"vipy.geometry.apply_homography",
"url":9,
"doc":"Apply a 3x3 homography H to non-homogenous point p and return a transformed point",
"func":1
},
{
"ref":"vipy.geometry.similarity_transform_2x3",
"url":9,
"doc":"Return a 2x3 similarity transform with rotation r (radians), scale s and origin c=(x,y)",
"func":1
},
{
"ref":"vipy.geometry.similarity_transform",
"url":9,
"doc":"Return a 3x3 similarity transformation with translation tuple txy=(x,y), rotation r (radians, scale=s",
"func":1
},
{
"ref":"vipy.geometry.affine_transform",
"url":9,
"doc":"Compose and return a 3x3 affine transformation for translation txy=(0,0), rotation r (radians), scalex=sx, scaley=sy, shearx=kx, sheary=ky",
"func":1
},
{
"ref":"vipy.geometry.random_affine_transform",
"url":9,
"doc":"Return a random 3x3 affine transformation matrix for the provided ranges, inputs must be tuples",
"func":1
},
{
"ref":"vipy.geometry.imtransform",
"url":9,
"doc":"Transform an numpy array image (MxNx3) following the affine or similiarity transformation A",
"func":1
},
{
"ref":"vipy.geometry.normalize",
"url":9,
"doc":"Given a vector x, return the vector unit normalized as float64",
"func":1
},
{
"ref":"vipy.geometry.imagebox",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox",
"url":9,
"doc":"Core bounding box class with flexible constructors in this priority order: (xmin,ymin,xmax,ymax) (xmin,ymin,width,height) (centroid[0],centroid[1],width,height) (xcentroid,ycentroid,width,height) xywh=(xmin,ymin,width,height) ulbr=(xmin,ymin,xmax,ymax) bounding rectangle of binary mask image"
},
{
"ref":"vipy.geometry.BoundingBox.cast",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.from_json",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dict",
"url":9,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.json",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.clone",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bbclone",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xmin",
"url":9,
"doc":"x coordinate of upper left corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ul",
"url":9,
"doc":"Upper left coordinate (x,y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ulx",
"url":9,
"doc":"Upper left coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.uly",
"url":9,
"doc":"Upper left coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ur",
"url":9,
"doc":"Upper right coordinate (x,y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.urx",
"url":9,
"doc":"Upper right coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ury",
"url":9,
"doc":"Upper right coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ll",
"url":9,
"doc":"Lower left coordinate (x,y), synonym for bl()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bl",
"url":9,
"doc":"Bottom left coordinate (x,y), synonym for ll()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.blx",
"url":9,
"doc":"Bottom left coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bly",
"url":9,
"doc":"Bottom left coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.lr",
"url":9,
"doc":"Lower right coordinate (x,y), synonym for br()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.br",
"url":9,
"doc":"Bottom right coordinate (x,y), synonym for lr()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.brx",
"url":9,
"doc":"Bottom right coordinate (x)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bry",
"url":9,
"doc":"Bottom right coordinate (y)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ymin",
"url":9,
"doc":"y coordinate of upper left corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xmax",
"url":9,
"doc":"x coordinate of lower right corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ymax",
"url":9,
"doc":"y coordinate of lower right corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.upperleft",
"url":9,
"doc":"Return the (x,y) upper left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bottomleft",
"url":9,
"doc":"Return the (x,y) lower left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.upperright",
"url":9,
"doc":"Return the (x,y) upper right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bottomright",
"url":9,
"doc":"Return the (x,y) lower right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isinteger",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.int",
"url":9,
"doc":"Convert corners to integer with rounding, in-place update",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.float",
"url":9,
"doc":"Convert corners to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.significant_digits",
"url":9,
"doc":"Convert corners to have at most n significant digits for efficient JSON storage",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.translate",
"url":9,
"doc":"Translate the bounding box by dx in x and dy in y",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.to_origin",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.set_origin",
"url":9,
"doc":"Set the origin of the coordinates of this bounding box to be relative to the upper left of the other bounding box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.offset",
"url":9,
"doc":"Alias for translate",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.invalid",
"url":9,
"doc":"Is the box a valid bounding box?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.valid",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isvalid",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isdegenerate",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isnonnegative",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.width",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bbwidth",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.setwidth",
"url":9,
"doc":"Set new width keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.setheight",
"url":9,
"doc":"Set new height keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.height",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bbheight",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.centroid",
"url":9,
"doc":"(x,y) tuple of centroid position of bounding box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.x_centroid",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xcentroid",
"url":9,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.centroid_x",
"url":9,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.y_centroid",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ycentroid",
"url":9,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.centroid_y",
"url":9,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.area",
"url":9,
"doc":"Return the area=width height of the bounding box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.to_xywh",
"url":9,
"doc":"Return bounding box corners as (x,y,width,height) tuple",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.xywh",
"url":9,
"doc":"Alias for to_xywh",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.cxywh",
"url":9,
"doc":"Return or set bounding box corners as (centroidx,centroidy,width,height) tuple",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ulbr",
"url":9,
"doc":"Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.to_ulbr",
"url":9,
"doc":"Alias for ulbr()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dx",
"url":9,
"doc":"Offset bounding box by same xmin as provided box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dy",
"url":9,
"doc":"Offset bounding box by ymin of provided box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.sqdist",
"url":9,
"doc":"Squared Euclidean distance between upper left corners of two bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dist",
"url":9,
"doc":"Distance between centroids of two bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.pdist",
"url":9,
"doc":"Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.iou",
"url":9,
"doc":"area of intersection / area of union",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.intersection_over_union",
"url":9,
"doc":"Alias for iou",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.area_of_intersection",
"url":9,
"doc":"area of intersection",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.area_of_union",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.cover",
"url":9,
"doc":"Fraction of this bounding box intersected by other bbox (bb)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxcover",
"url":9,
"doc":"The maximum cover of self to bb and bb to self",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.shapeiou",
"url":9,
"doc":"Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.intersection",
"url":9,
"doc":"Intersection of two bounding boxes, throw an error on degeneracy of intersection result (if strict=True)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.hasintersection",
"url":9,
"doc":"Return true if self and bb overlap by any amount, or by the cover threshold (if provided) or the iou threshold (if provided). This is a convenience function that allows for shared computation for fast non-maximum suppression.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.union",
"url":9,
"doc":"Union of one or more bounding boxes with this box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isinside",
"url":9,
"doc":"Is this boundingbox fully within the provided bounding box?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ispointinside",
"url":9,
"doc":"Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilate",
"url":9,
"doc":"Change scale of bounding box keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilatepx",
"url":9,
"doc":"Dilate by a given pixel amount on all sides, keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilate_height",
"url":9,
"doc":"Change scale of bounding box in y direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.dilate_width",
"url":9,
"doc":"Change scale of bounding box in x direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.top",
"url":9,
"doc":"Make top of box taller (closer to top of image) by an offset dy",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.bottom",
"url":9,
"doc":"Make bottom of box taller (closer to bottom of image) by an offset dy",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.left",
"url":9,
"doc":"Make left of box wider (closer to left side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.right",
"url":9,
"doc":"Make right of box wider (closer to right side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.rescale",
"url":9,
"doc":"Multiply the box corners by a scale factor",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.scalex",
"url":9,
"doc":"Multiply the box corners in the x dimension by a scale factor",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.scaley",
"url":9,
"doc":"Multiply the box corners in the y dimension by a scale factor",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.resize",
"url":9,
"doc":"Change the aspect ratio width and height of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.rot90cw",
"url":9,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.rot90ccw",
"url":9,
"doc":"Rotate a bounding box such that if an image of size (H,W) is rotated 90 deg clockwise, the boxes align",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.fliplr",
"url":9,
"doc":"Flip the box left/right consistent with fliplr of the provided img (or consistent with the image width)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.flipud",
"url":9,
"doc":"Flip the box up/down consistent with flipud of the provided img (or consistent with the image height)",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.imscale",
"url":9,
"doc":"Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxsquare",
"url":9,
"doc":"Set the bounding box to be square by setting width and height to the maximum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxsquareif",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.issquare",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.iseven",
"url":9,
"doc":"Are all corners even number integers?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.even",
"url":9,
"doc":"Force all corners to be even number integers. This is helpful for FFMPEG crop filters.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.minsquare",
"url":9,
"doc":"Set the bounding box to be square by setting width and height to the minimum dimension of the box, keeping centroid constant",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.hasoverlap",
"url":9,
"doc":"Does the bounding box intersect with the provided image rectangle?",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.isinterior",
"url":9,
"doc":"Is this boundingbox fully within the provided image rectangle?  If border in [0,1], then the image is dilated by a border percentage prior to computing interior, useful to check if self is near the image edge  If border=0.8, then the image rectangle is dilated by 80% (smaller) keeping the centroid constant.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.iminterior",
"url":9,
"doc":"Transform bounding box to be interior to the image rectangle with shape (W,H). Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.imclip",
"url":9,
"doc":"Clip bounding box to image rectangle [0,0,width,height] or img.shape=(width, height) and, throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.imclipshape",
"url":9,
"doc":"Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.convexhull",
"url":9,
"doc":"Given a set of points  x1,y1],[x2,xy], .], return the bounding rectangle, typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.aspectratio",
"url":9,
"doc":"Return the aspect ratio (width/height) of the box",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.shape",
"url":9,
"doc":"Return the (height, width) tuple for the box shape",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.mindimension",
"url":9,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.mindim",
"url":9,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.maxdim",
"url":9,
"doc":"Return max(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.ellipse",
"url":9,
"doc":"Convert the boundingbox to a vipy.geometry.Ellipse object",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.average",
"url":9,
"doc":"Compute the average bounding box between self and other, and set self to the average. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.averageshape",
"url":9,
"doc":"Compute the average bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.medianshape",
"url":9,
"doc":"Compute the median bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.shapedist",
"url":9,
"doc":"L1 distance between (width,height) of two boxes",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.affine",
"url":9,
"doc":"Apply an 2x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.projective",
"url":9,
"doc":"Apply an 3x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.crop",
"url":9,
"doc":"Crop an HxW 2D numpy image, HxWxC 3D numpy image, or NxHxWxC 4D numpy image array using this bounding box applied to HxW dimensions. Crop is performed in-place.",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.confidence",
"url":9,
"doc":"Bounding boxes do not have confidences, use vipy.object.Detection()",
"func":1
},
{
"ref":"vipy.geometry.BoundingBox.grid",
"url":9,
"doc":"Split a bounding box into the smallest grid of non-overlapping bounding boxes such that the union is the original box",
"func":1
},
{
"ref":"vipy.geometry.Ellipse",
"url":9,
"doc":"Ellipse parameterization, for length of semimajor (half width of ellipse) and semiminor axis (half height), center point and angle phi in radians"
},
{
"ref":"vipy.geometry.Ellipse.dict",
"url":9,
"doc":"",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.area",
"url":9,
"doc":"Area of ellipse",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.center",
"url":9,
"doc":"Return centroid",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.centroid",
"url":9,
"doc":"Alias for center",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.axes",
"url":9,
"doc":"Return the (major,minor) axis lengths",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.angle",
"url":9,
"doc":"Return the angle phi (in degrees)",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.rescale",
"url":9,
"doc":"Scale ellipse by scale factor",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.boundingbox",
"url":9,
"doc":"Estimate an equivalent bounding box based on scaling to a common area. Note, this does not factor in rotation. (c l) (c w) = a_e  > c = sqrt(a_e / a_r)",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.inside",
"url":9,
"doc":"Return true if a point p=(x,y) is inside the ellipse",
"func":1
},
{
"ref":"vipy.geometry.Ellipse.mask",
"url":9,
"doc":"Return a binary mask of size equal to the bounding box such that the pixels correspond to the interior of the ellipse",
"func":1
},
{
"ref":"vipy.geometry.union",
"url":9,
"doc":"Return the union of a list of vipy.geometry.BoundingBox",
"func":1
},
{
"ref":"vipy.gui",
"url":45,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib",
"url":46,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.escape_to_exit",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.flush",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imflush",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.show",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.noshow",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.savefig",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.figure",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.close",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.closeall",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imshow",
"url":46,
"doc":"Show an image in a figure window (optionally visible), reuse previous figure if it is the same shape",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.text",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.boundingbox",
"url":46,
"doc":"Draw a captioned bounding box on a previously shown image",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imdetection",
"url":46,
"doc":"Show bounding boxes from a list of vipy.object.Detections on the same image, plotted in list order with optional captions",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.imframe",
"url":46,
"doc":"Show a scatterplot of fr= x1,y1],[x2,y2] .] 2D points overlayed on an image",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.frame",
"url":46,
"doc":"Show a scatterplot of fr= x1,y1],[x2,y2] .] 2D points",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.colorlist",
"url":46,
"doc":"Return a list of named colors that are higher contrast with a white background",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.edit",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.Annotate",
"url":46,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.Annotate.on_press",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.Annotate.on_release",
"url":46,
"doc":"",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle",
"url":46,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.connect",
"url":46,
"doc":"connect to all the events we need",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.on_press",
"url":46,
"doc":"on button press we will see if the mouse is over us and store some data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.on_motion",
"url":46,
"doc":"on motion we will move the rect if the mouse is over us",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.on_release",
"url":46,
"doc":"on release we reset the press data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangle.disconnect",
"url":46,
"doc":"disconnect all the stored connection ids",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast",
"url":46,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.lock",
"url":46,
"doc":""
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.connect",
"url":46,
"doc":"connect to all the events we need",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.on_press",
"url":46,
"doc":"on button press we will see if the mouse is over us and store some data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.on_motion",
"url":46,
"doc":"on motion we will move the rect if the mouse is over us",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.on_release",
"url":46,
"doc":"on release we reset the press data",
"func":1
},
{
"ref":"vipy.gui.using_matplotlib.DraggableRectangleFast.disconnect",
"url":46,
"doc":"disconnect all the stored connection ids",
"func":1
},
{
"ref":"vipy.math",
"url":47,
"doc":""
},
{
"ref":"vipy.math.normalize",
"url":47,
"doc":"Whiten the numpy array arr",
"func":1
},
{
"ref":"vipy.math.iseven",
"url":47,
"doc":"",
"func":1
},
{
"ref":"vipy.math.even",
"url":47,
"doc":"Return the largest even integer less than or equal (or greater than if greaterthan=True) to the value",
"func":1
},
{
"ref":"vipy.math.poweroftwo",
"url":47,
"doc":"Return the closest power of two smaller than the value. x=511 -> 256, x=512 -> 512",
"func":1
},
{
"ref":"vipy.math.signsqrt",
"url":47,
"doc":"Return the signed square root of elements in x",
"func":1
},
{
"ref":"vipy.math.runningmean",
"url":47,
"doc":"Compute the running unweighted mean of X row-wise, with a history of n, reducing the history at the start",
"func":1
},
{
"ref":"vipy.math.gaussian",
"url":47,
"doc":"1D gaussian window with M points, Replication of scipy.signal.gaussian",
"func":1
},
{
"ref":"vipy.math.gaussian2d",
"url":47,
"doc":"2D gaussian image of size (rows=H, cols=W) with mu=[x,y] and std=[stdx, stdy]",
"func":1
},
{
"ref":"vipy.math.interp1d",
"url":47,
"doc":"Replication of scipy.interpolate.interp1d with assume_sorted=True, and constant replication of boundary handling",
"func":1
},
{
"ref":"vipy.math.find_closest_positive_divisor",
"url":47,
"doc":"Return non-trivial positive integer divisor (bh) of (a) closest to (b) in abs(b-bh) such that a % bh  0. This uses exhaustive search, which is inefficient for large a.",
"func":1
},
{
"ref":"vipy.math.cartesian_to_polar",
"url":47,
"doc":"Cartesian (x,y) coordinates to polar (radius, theta) coordinates, theta in radians in [-pi,pi]",
"func":1
},
{
"ref":"vipy.math.polar_to_cartesian",
"url":47,
"doc":"Polar (radius, theta) coordinates to cartesian (x=right,y=down) coordinates. (0,0) is upper left of image",
"func":1
},
{
"ref":"vipy.math.rad2deg",
"url":47,
"doc":"Radians to degrees",
"func":1
},
{
"ref":"vipy.torch",
"url":48,
"doc":""
},
{
"ref":"vipy.torch.fromtorch",
"url":48,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with an inferred colorspace based on channels and datatype",
"func":1
},
{
"ref":"vipy.torch.GaussianPyramid",
"url":48,
"doc":""
},
{
"ref":"vipy.torch.LaplacianPyramid",
"url":48,
"doc":""
},
{
"ref":"vipy.torch.LaplacianPyramid.reconstruct",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.torch.Foveation",
"url":48,
"doc":""
},
{
"ref":"vipy.torch.Foveation.foveate",
"url":48,
"doc":"",
"func":1
},
{
"ref":"vipy.video",
"url":39,
"doc":""
},
{
"ref":"vipy.video.Video",
"url":39,
"doc":"vipy.video.Video class The vipy.video class provides a fluent, lazy interface for representing, transforming and visualizing videos. The following constructors are supported: >>> vid = vipy.video.Video(filename='/path/to/video.ext') Valid video extensions are those that are supported by ffmpeg ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']. >>> vid = vipy.video.Video(url='https: www.youtube.com/watch?v=MrIN959JuV8') >>> vid = vipy.video.Video(url='http: path/to/video.ext', filename='/path/to/video.ext') Youtube URLs are downloaded to a temporary filename, retrievable as vid.download().filename(). If the environment variable 'VIPY_CACHE' is defined, then videos are saved to this directory rather than the system temporary directory. If a filename is provided to the constructor, then that filename will be used instead of a temp or cached filename. URLs can be defined as an absolute URL to a video file, or to a site supported by 'youtube-dl' (https: ytdl-org.github.io/youtube-dl/supportedsites.html) >>> vid = vipy.video.Video(array=frames, colorspace='rgb') The input 'frames' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video Note that the video transformations (clip, resize, rescale, rotate) are only available prior to load(), and the array() is assumed immutable after load()."
},
{
"ref":"vipy.video.Video.cast",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.from_json",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.metadata",
"url":39,
"doc":"Return a dictionary of metadata about this video",
"func":1
},
{
"ref":"vipy.video.Video.videoid",
"url":39,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by hashing the filename() and url(). Notes: - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', $MY_ID)",
"func":1
},
{
"ref":"vipy.video.Video.frame",
"url":39,
"doc":"Alias for self.__getitem__[k]",
"func":1
},
{
"ref":"vipy.video.Video.store",
"url":39,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.video.Video.unstore",
"url":39,
"doc":"Delete the currently stored video from store()",
"func":1
},
{
"ref":"vipy.video.Video.restore",
"url":39,
"doc":"Save the currently stored video to filename, and set up filename",
"func":1
},
{
"ref":"vipy.video.Video.stream",
"url":39,
"doc":"Iterator to yield frames streaming from video  Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.  FFMPEG stdout pipe may screw up bash shell newlines, requiring issuing command \"reset\"",
"func":1
},
{
"ref":"vipy.video.Video.clear",
"url":39,
"doc":"no-op for Video()",
"func":1
},
{
"ref":"vipy.video.Video.bytes",
"url":39,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.video.Video.frames",
"url":39,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.video.Video.framelist",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.isdirty",
"url":39,
"doc":"Has the FFMPEG filter chain been modified from the default? If so, then ffplay() on the video file will be different from self.load().play()",
"func":1
},
{
"ref":"vipy.video.Video.probeshape",
"url":39,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe, this does not take into account any applied ffmpeg filters",
"func":1
},
{
"ref":"vipy.video.Video.duration_in_seconds_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache",
"func":1
},
{
"ref":"vipy.video.Video.duration_in_frames_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in frames, requires ffprobe",
"func":1
},
{
"ref":"vipy.video.Video.probe",
"url":39,
"doc":"Run ffprobe on the filename and return the result as a JSON file",
"func":1
},
{
"ref":"vipy.video.Video.print",
"url":39,
"doc":"Print the representation of the video - useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing",
"func":1
},
{
"ref":"vipy.video.Video.dict",
"url":39,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.video.Video.json",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.take",
"url":39,
"doc":"Return n frames from the clip uniformly spaced as numpy array",
"func":1
},
{
"ref":"vipy.video.Video.framerate",
"url":39,
"doc":"Change the input framerate for the video and update frame indexes for all annotations  NOTE: do not call framerate() after calling clip() as this introduces extra repeated final frames during load()",
"func":1
},
{
"ref":"vipy.video.Video.colorspace",
"url":39,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.video.Video.nourl",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.url",
"url":39,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.video.Video.isloaded",
"url":39,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.video.Video.canload",
"url":39,
"doc":"Return True if the video can be loaded successfully, useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version",
"func":1
},
{
"ref":"vipy.video.Video.iscolor",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.isgrayscale",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.hasfilename",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.isdownloaded",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.hasurl",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.array",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.fromarray",
"url":39,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.video.Video.fromframes",
"url":39,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.video.Video.tonumpy",
"url":39,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.video.Video.numpy",
"url":39,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.video.Video.zeros",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.reload",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.nofilename",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.filename",
"url":39,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.video.Video.abspath",
"url":39,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.video.Video.relpath",
"url":39,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.video.Video.rename",
"url":39,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.video.Video.filesize",
"url":39,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.video.Video.download",
"url":39,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.video.Video.fetch",
"url":39,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.video.Video.shape",
"url":39,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.video.Video.channels",
"url":39,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.video.Video.width",
"url":39,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Video.height",
"url":39,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Video.aspect_ratio",
"url":39,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.video.Video.preview",
"url":39,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.video.Video.thumbnail",
"url":39,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile",
"func":1
},
{
"ref":"vipy.video.Video.load",
"url":39,
"doc":"Load a video using ffmpeg, applying the requested filter chain. - If verbose=True. then ffmpeg console output will be displayed. - If ignoreErrors=True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. - shape tuple(height, width, channels): If provided, use this shape for reading and reshaping the byte stream from ffmpeg - knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size",
"func":1
},
{
"ref":"vipy.video.Video.speed",
"url":39,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.video.Video.clip",
"url":39,
"doc":"Load a video clip betweeen start and end frames",
"func":1
},
{
"ref":"vipy.video.Video.cliptime",
"url":39,
"doc":"Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly",
"func":1
},
{
"ref":"vipy.video.Video.rot90cw",
"url":39,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.rot90ccw",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.fliplr",
"url":39,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.video.Video.flipud",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.rescale",
"url":39,
"doc":"Rescale the video by factor s, such that the new dimensions are (s H, s W), can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Video.resize",
"url":39,
"doc":"Resize the video to be (rows=height, cols=width)",
"func":1
},
{
"ref":"vipy.video.Video.mindim",
"url":39,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Video.maxdim",
"url":39,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Video.randomcrop",
"url":39,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Video.centercrop",
"url":39,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Video.centersquare",
"url":39,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.video.Video.cropeven",
"url":39,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.video.Video.maxsquare",
"url":39,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.video.Video.maxmatte",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.zeropad",
"url":39,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after  NOTE: Older FFMPEG implementations can throw the error \"Input area  : : : not within the padded area  : : : or zero-sized, this is often caused by odd sized padding. Recommend calling self.cropeven().zeropad( .) to avoid this",
"func":1
},
{
"ref":"vipy.video.Video.crop",
"url":39,
"doc":"Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().",
"func":1
},
{
"ref":"vipy.video.Video.pkl",
"url":39,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Video.pklif",
"url":39,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Video.webp",
"url":39,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. -strict=[bool]: assert that the filename must have an .webp extension -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.video.Video.gif",
"url":39,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. WARNING: this will be very large for big videos, consider using webp instead. -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.video.Video.saveas",
"url":39,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file.  outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image)  If self.array() is loaded, then export the contents of self._array to the video file  If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video  If outfile None or outfile self.filename(), then overwrite the current filename  If ignoreErrors=True, then exit gracefully. Useful for chaining download().saveas() on parallel dataset downloads  Returns a new video object with this video filename, and a clean video filter chain  if flush=True, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel  framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video  pause: an integer in seconds to pause between loops of animated images",
"func":1
},
{
"ref":"vipy.video.Video.savetmp",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.savetemp",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.ffplay",
"url":39,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.video.Video.play",
"url":39,
"doc":"Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it, if the filter chain is dirty, dump to temp file first",
"func":1
},
{
"ref":"vipy.video.Video.quicklook",
"url":39,
"doc":"Generate a montage of n uniformly spaced frames. Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. Input: -n: Number of images in the quicklook -mindim: The minimum dimension of each of the elements in the montage -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.video.Video.torch",
"url":39,
"doc":"Convert the loaded video of shape N HxWxC frames to an MxCxHxW torch tensor, forces a load().  Order of arguments is (startframe, endframe) or (startframe, startframe+length) or (random_startframe, random_starframe+takelength), then stride or take.  Follows numpy slicing rules. Optionally return the slice used if withslice=True  Returns float tensor in the range [0,1] following torchvision.transforms.ToTensor()  order can be ['nchw', 'nhwc', 'cnhw'] for batchsize=n, channels=c, height=h, width=w  boundary can be ['repeat', 'strict', 'cyclic']  withlabel=True, returns tuple (t, labellist), where labellist is a list of tuples of activity labels occurring at the corresponding frame in the tensor  withslice=Trye, returnss tuple (t, (startframe, endframe, stride  nonelabel=True, returns tuple (t, None) if withlabel=False",
"func":1
},
{
"ref":"vipy.video.Video.clone",
"url":39,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.  flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos  rekey: Generate new unique track ID and activity ID keys for this scene  shallow: shallow copy everything (copy by reference), except for ffmpeg object  sharedarray: deep copy of everything, except for pixel buffer which is shared",
"func":1
},
{
"ref":"vipy.video.Video.flush",
"url":39,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.video.Video.flush_and_return",
"url":39,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.video.Video.map",
"url":39,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.video.Video.gain",
"url":39,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.video.Video.bias",
"url":39,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.video.Video.float",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.channel",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.normalize",
"url":39,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.video.Video.setattribute",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.hasattribute",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.delattribute",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Video.getattribute",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.VideoCategory",
"url":39,
"doc":"vipy.video.VideoCategory class A VideoCategory is a video with associated category, such as an activity class. This class includes all of the constructors of vipy.video.Video along with the ability to extract a clip based on frames or seconds."
},
{
"ref":"vipy.video.VideoCategory.from_json",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.VideoCategory.json",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.VideoCategory.category",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.VideoCategory.metadata",
"url":39,
"doc":"Return a dictionary of metadata about this video",
"func":1
},
{
"ref":"vipy.video.VideoCategory.videoid",
"url":39,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by hashing the filename() and url(). Notes: - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', $MY_ID)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.frame",
"url":39,
"doc":"Alias for self.__getitem__[k]",
"func":1
},
{
"ref":"vipy.video.VideoCategory.store",
"url":39,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.video.VideoCategory.unstore",
"url":39,
"doc":"Delete the currently stored video from store()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.restore",
"url":39,
"doc":"Save the currently stored video to filename, and set up filename",
"func":1
},
{
"ref":"vipy.video.VideoCategory.stream",
"url":39,
"doc":"Iterator to yield frames streaming from video  Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.  FFMPEG stdout pipe may screw up bash shell newlines, requiring issuing command \"reset\"",
"func":1
},
{
"ref":"vipy.video.VideoCategory.clear",
"url":39,
"doc":"no-op for Video()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.bytes",
"url":39,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.frames",
"url":39,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.isdirty",
"url":39,
"doc":"Has the FFMPEG filter chain been modified from the default? If so, then ffplay() on the video file will be different from self.load().play()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.probeshape",
"url":39,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe, this does not take into account any applied ffmpeg filters",
"func":1
},
{
"ref":"vipy.video.VideoCategory.duration_in_seconds_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache",
"func":1
},
{
"ref":"vipy.video.VideoCategory.duration_in_frames_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in frames, requires ffprobe",
"func":1
},
{
"ref":"vipy.video.VideoCategory.probe",
"url":39,
"doc":"Run ffprobe on the filename and return the result as a JSON file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.print",
"url":39,
"doc":"Print the representation of the video - useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing",
"func":1
},
{
"ref":"vipy.video.VideoCategory.dict",
"url":39,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.video.VideoCategory.take",
"url":39,
"doc":"Return n frames from the clip uniformly spaced as numpy array",
"func":1
},
{
"ref":"vipy.video.VideoCategory.framerate",
"url":39,
"doc":"Change the input framerate for the video and update frame indexes for all annotations  NOTE: do not call framerate() after calling clip() as this introduces extra repeated final frames during load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.colorspace",
"url":39,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.video.VideoCategory.url",
"url":39,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.video.VideoCategory.isloaded",
"url":39,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.video.VideoCategory.canload",
"url":39,
"doc":"Return True if the video can be loaded successfully, useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fromarray",
"url":39,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fromframes",
"url":39,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.video.VideoCategory.tonumpy",
"url":39,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.numpy",
"url":39,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.video.VideoCategory.filename",
"url":39,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.abspath",
"url":39,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.relpath",
"url":39,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rename",
"url":39,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.video.VideoCategory.filesize",
"url":39,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.video.VideoCategory.download",
"url":39,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fetch",
"url":39,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.video.VideoCategory.shape",
"url":39,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.video.VideoCategory.channels",
"url":39,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.video.VideoCategory.width",
"url":39,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.VideoCategory.height",
"url":39,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.VideoCategory.aspect_ratio",
"url":39,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.video.VideoCategory.preview",
"url":39,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.video.VideoCategory.thumbnail",
"url":39,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile",
"func":1
},
{
"ref":"vipy.video.VideoCategory.load",
"url":39,
"doc":"Load a video using ffmpeg, applying the requested filter chain. - If verbose=True. then ffmpeg console output will be displayed. - If ignoreErrors=True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. - shape tuple(height, width, channels): If provided, use this shape for reading and reshaping the byte stream from ffmpeg - knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size",
"func":1
},
{
"ref":"vipy.video.VideoCategory.speed",
"url":39,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.clip",
"url":39,
"doc":"Load a video clip betweeen start and end frames",
"func":1
},
{
"ref":"vipy.video.VideoCategory.cliptime",
"url":39,
"doc":"Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rot90cw",
"url":39,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rot90ccw",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.fliplr",
"url":39,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.video.VideoCategory.flipud",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.rescale",
"url":39,
"doc":"Rescale the video by factor s, such that the new dimensions are (s H, s W), can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.VideoCategory.resize",
"url":39,
"doc":"Resize the video to be (rows=height, cols=width)",
"func":1
},
{
"ref":"vipy.video.VideoCategory.mindim",
"url":39,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.VideoCategory.maxdim",
"url":39,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.VideoCategory.randomcrop",
"url":39,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.VideoCategory.centercrop",
"url":39,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.VideoCategory.centersquare",
"url":39,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.video.VideoCategory.cropeven",
"url":39,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.maxsquare",
"url":39,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.video.VideoCategory.zeropad",
"url":39,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after  NOTE: Older FFMPEG implementations can throw the error \"Input area  : : : not within the padded area  : : : or zero-sized, this is often caused by odd sized padding. Recommend calling self.cropeven().zeropad( .) to avoid this",
"func":1
},
{
"ref":"vipy.video.VideoCategory.crop",
"url":39,
"doc":"Spatially crop the video using the supplied vipy.geometry.BoundingBox, can only be applied prior to load().",
"func":1
},
{
"ref":"vipy.video.VideoCategory.pkl",
"url":39,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.VideoCategory.pklif",
"url":39,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.VideoCategory.webp",
"url":39,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. -strict=[bool]: assert that the filename must have an .webp extension -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.video.VideoCategory.gif",
"url":39,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. WARNING: this will be very large for big videos, consider using webp instead. -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.video.VideoCategory.saveas",
"url":39,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file.  outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image)  If self.array() is loaded, then export the contents of self._array to the video file  If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video  If outfile None or outfile self.filename(), then overwrite the current filename  If ignoreErrors=True, then exit gracefully. Useful for chaining download().saveas() on parallel dataset downloads  Returns a new video object with this video filename, and a clean video filter chain  if flush=True, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel  framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video  pause: an integer in seconds to pause between loops of animated images",
"func":1
},
{
"ref":"vipy.video.VideoCategory.ffplay",
"url":39,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.video.VideoCategory.play",
"url":39,
"doc":"Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it, if the filter chain is dirty, dump to temp file first",
"func":1
},
{
"ref":"vipy.video.VideoCategory.quicklook",
"url":39,
"doc":"Generate a montage of n uniformly spaced frames. Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. Input: -n: Number of images in the quicklook -mindim: The minimum dimension of each of the elements in the montage -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.video.VideoCategory.torch",
"url":39,
"doc":"Convert the loaded video of shape N HxWxC frames to an MxCxHxW torch tensor, forces a load().  Order of arguments is (startframe, endframe) or (startframe, startframe+length) or (random_startframe, random_starframe+takelength), then stride or take.  Follows numpy slicing rules. Optionally return the slice used if withslice=True  Returns float tensor in the range [0,1] following torchvision.transforms.ToTensor()  order can be ['nchw', 'nhwc', 'cnhw'] for batchsize=n, channels=c, height=h, width=w  boundary can be ['repeat', 'strict', 'cyclic']  withlabel=True, returns tuple (t, labellist), where labellist is a list of tuples of activity labels occurring at the corresponding frame in the tensor  withslice=Trye, returnss tuple (t, (startframe, endframe, stride  nonelabel=True, returns tuple (t, None) if withlabel=False",
"func":1
},
{
"ref":"vipy.video.VideoCategory.clone",
"url":39,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.  flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos  rekey: Generate new unique track ID and activity ID keys for this scene  shallow: shallow copy everything (copy by reference), except for ffmpeg object  sharedarray: deep copy of everything, except for pixel buffer which is shared",
"func":1
},
{
"ref":"vipy.video.VideoCategory.flush",
"url":39,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.video.VideoCategory.flush_and_return",
"url":39,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.video.VideoCategory.map",
"url":39,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.video.VideoCategory.gain",
"url":39,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.video.VideoCategory.bias",
"url":39,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.video.VideoCategory.normalize",
"url":39,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.video.Scene",
"url":39,
"doc":"vipy.video.Scene class The vipy.video.Scene class provides a fluent, lazy interface for representing, transforming and visualizing annotated videos. The following constructors are supported: >>> vid = vipy.video.Scene(filename='/path/to/video.ext') Valid video extensions are those that are supported by ffmpeg ['.avi','.mp4','.mov','.wmv','.mpg', 'mkv', 'webm']. >>> vid = vipy.video.Scene(url='https: www.youtube.com/watch?v=MrIN959JuV8') >>> vid = vipy.video.Scene(url='http: path/to/video.ext', filename='/path/to/video.ext') Youtube URLs are downloaded to a temporary filename, retrievable as vid.download().filename(). If the environment variable 'VIPY_CACHE' is defined, then videos are saved to this directory rather than the system temporary directory. If a filename is provided to the constructor, then that filename will be used instead of a temp or cached filename. URLs can be defined as an absolute URL to a video file, or to a site supported by 'youtube-dl' [https: ytdl-org.github.io/youtube-dl/supportedsites.html] >>> vid = vipy.video.Scene(array=frames, colorspace='rgb') The input 'frames' is an NxHxWx3 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video Note that the video transformations (clip, resize, rescale, rotate) are only available prior to load(), and the array() is assumed immutable after load(). >>> vid = vipy.video.Scene(array=greyframes, colorspace='lum') The input 'greyframes' is an NxHxWx1 numpy array corresponding to an N-length list of HxWx3 uint8 numpy array which is a single frame of pre-loaded video This corresponds to the luminance of an RGB colorspace >>> vid = vipy.video.Scene(array=greyframes, colorspace='lum', tracks=tracks, activities=activities)  tracks = [vipy.object.Track(),  .]  activities = [vipy.object.Activity(),  .] The inputs are lists of tracks and/or activities. An object is a spatial bounding box with a category label. A track is a spatiotemporal bounding box with a category label, such that the box contains the same instance of an object. An activity is one or more tracks with a start and end frame for an activity performed by the object instances. Track and activity timing must be relative to the start frame of the Scene() constructor."
},
{
"ref":"vipy.video.Scene.cast",
"url":39,
"doc":"Cast a conformal vipy object to this class. This is useful for downcast and upcast conversion of video objects.",
"func":1
},
{
"ref":"vipy.video.Scene.from_json",
"url":39,
"doc":"Restore an object serialized with self.json() Usage: >>> vs = vipy.video.Scene.from_json(v.json( ",
"func":1
},
{
"ref":"vipy.video.Scene.pack",
"url":39,
"doc":"Packing a scene returns the scene with the annotations JSON serialized. - This is useful for fast garbage collection when there are many objects in memory - This is useful for distributed processing prior to serializing from a scheduler to a client - This is useful for lazy deserialization of complex attributes when loading many videos into memory - Unpacking is transparent to the end user and is performed on the fly when annotations are accessed. There is no unpack() method. - See the notes in from_json() for why this helps with nested containers and reference cycle tracking with the python garbage collector",
"func":1
},
{
"ref":"vipy.video.Scene.frame",
"url":39,
"doc":"Return vipy.image.Scene object at frame k",
"func":1
},
{
"ref":"vipy.video.Scene.during",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.labeled_frames",
"url":39,
"doc":"Iterate over frames, yielding tuples (activity+object labelset in scene, vipy.image.Scene( ",
"func":1
},
{
"ref":"vipy.video.Scene.framecomposite",
"url":39,
"doc":"Generate a single composite image with minimum dimension mindim as the uniformly blended composite of n frames each separated by dt frames",
"func":1
},
{
"ref":"vipy.video.Scene.isdegenerate",
"url":39,
"doc":"Degenerate scene has empty or malformed tracks",
"func":1
},
{
"ref":"vipy.video.Scene.quicklook",
"url":39,
"doc":"Generate a montage of n uniformly spaced annotated frames centered on the union of the labeled boxes in the current frame to show the activity ocurring in this scene at a glance Montage increases rowwise for n uniformly spaced frames, starting from frame zero and ending on the last frame. This quicklook is most useful when len(self.activities() 1) for generating a quicklook from an activityclip(). Input: -n: Number of images in the quicklook -dilate: The dilation factor for the bounding box prior to crop for display -mindim: The minimum dimension of each of the elemnets in the montage -fontsize: The size of the font for the bounding box label -context: If true, replace the first and last frame in the montage with the full frame annotation, to help show the scale of the scene -animate: If true, return a video constructed by animating the quicklook into a video by showing dt consecutive frames -dt: The number of frames for animation -startframe: The initial frame index to start the n uniformly sampled frames for the quicklook",
"func":1
},
{
"ref":"vipy.video.Scene.tracks",
"url":39,
"doc":"Return mutable dictionary of tracks",
"func":1
},
{
"ref":"vipy.video.Scene.track",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.trackindex",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.trackidx",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activity",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.next_activity",
"url":39,
"doc":"Return the next activity just after the given activityid",
"func":1
},
{
"ref":"vipy.video.Scene.prev_activity",
"url":39,
"doc":"Return the previous activity just before the given activityid",
"func":1
},
{
"ref":"vipy.video.Scene.tracklist",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.actorid",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.actor",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.primary_activity",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activities",
"url":39,
"doc":"Return mutable dictionary of activities. All temporal alignment is relative to the current clip().",
"func":1
},
{
"ref":"vipy.video.Scene.activityindex",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activitylist",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.activityfilter",
"url":39,
"doc":"Apply boolean lambda function f to each activity and keep activity if function is true, remove activity if function is false Usage: Filter out all activities longer than 128 frames vid = vid.activityfilter(lambda a: len(a)<128) Usage: Filter out activities with category in set vid = vid.activityfilter(lambda a: a.category() in set(['category1', 'category2'] ",
"func":1
},
{
"ref":"vipy.video.Scene.trackfilter",
"url":39,
"doc":"Apply lambda function f to each object and keep if filter is True. -strict=True: remove track assignment from activities also, may result in activities with no tracks",
"func":1
},
{
"ref":"vipy.video.Scene.trackmap",
"url":39,
"doc":"Apply lambda function f to each activity -strict=True: enforce that lambda function must return non-degenerate Track() objects",
"func":1
},
{
"ref":"vipy.video.Scene.activitymap",
"url":39,
"doc":"Apply lambda function f to each activity",
"func":1
},
{
"ref":"vipy.video.Scene.rekey",
"url":39,
"doc":"Change the track and activity IDs to randomly assigned UUIDs. Useful for cloning unique scenes",
"func":1
},
{
"ref":"vipy.video.Scene.label",
"url":39,
"doc":"Return an iterator over labels in each frame",
"func":1
},
{
"ref":"vipy.video.Scene.labels",
"url":39,
"doc":"Return a set of all object and activity labels in this scene, or at frame int(k)",
"func":1
},
{
"ref":"vipy.video.Scene.activitylabel",
"url":39,
"doc":"Return an iterator over activity labels in each frame",
"func":1
},
{
"ref":"vipy.video.Scene.activitylabels",
"url":39,
"doc":"Return a set of all activity categories in this scene, or at startframe, or in range [startframe, endframe]",
"func":1
},
{
"ref":"vipy.video.Scene.objectlabels",
"url":39,
"doc":"Return a set of all activity categories in this scene, or at frame k",
"func":1
},
{
"ref":"vipy.video.Scene.categories",
"url":39,
"doc":"Alias for labels()",
"func":1
},
{
"ref":"vipy.video.Scene.activity_categories",
"url":39,
"doc":"Alias for activitylabels()",
"func":1
},
{
"ref":"vipy.video.Scene.hasactivities",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.hastracks",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.hastrack",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.add",
"url":39,
"doc":"Add the object obj to the scene, and return an index to this object for future updates This function is used to incrementally build up a scene frame by frame. Obj can be one of the following types:  obj = vipy.object.Detection(), this must be called from within a frame iterator (e.g. for im in video) to get the current frame index  obj = vipy.object.Track()  obj = vipy.activity.Activity()  obj = [xmin, ymin, width, height], with associated category kwarg, this must be called from within a frame iterator to get the current frame index It is recomended that the objects are added as follows. For a scene=vipy.video.Scene(): for im in scene:  Do some processing on frame im to detect objects (object_labels, xywh) = object_detection(im)  Add them to the scene, note that each object instance is independent in each frame, use tracks for object correspondence for (lbl,bb) in zip(object_labels, xywh): scene.add(bb, lbl)  Do some correspondences to track objects t2 = scene.add( vipy.object.Track( .) )  Update a previous track to add a keyframe scene.track(t2).add(  . ) This will keep track of the current frame in the video and add the objects in the appropriate place",
"func":1
},
{
"ref":"vipy.video.Scene.delete",
"url":39,
"doc":"Delete a given track or activity by id, if present",
"func":1
},
{
"ref":"vipy.video.Scene.addframe",
"url":39,
"doc":"Add im=vipy.image.Scene() into vipy.video.Scene() at given frame. The input image must have been generated using im=self[k] for this to be meaningful, so that trackid can be associated",
"func":1
},
{
"ref":"vipy.video.Scene.clear",
"url":39,
"doc":"Remove all activities and tracks from this object",
"func":1
},
{
"ref":"vipy.video.Scene.cleartracks",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.clearactivities",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.replace",
"url":39,
"doc":"Replace tracks and activities with other if activity/track is during frame",
"func":1
},
{
"ref":"vipy.video.Scene.json",
"url":39,
"doc":"Return JSON encoded string of this object. This may fail if attributes contain non-json encodeable object",
"func":1
},
{
"ref":"vipy.video.Scene.csv",
"url":39,
"doc":"Export scene to CSV file format with header. If there are no tracks, this will be empty.",
"func":1
},
{
"ref":"vipy.video.Scene.framerate",
"url":39,
"doc":"Change the input framerate for the video and update frame indexes for all annotations  NOTE: do not call framerate() after calling clip() as this introduces extra repeated final frames during load()",
"func":1
},
{
"ref":"vipy.video.Scene.activitysplit",
"url":39,
"doc":"Split the scene into k separate scenes, one for each activity. Do not include overlapping activities. This is useful for union()",
"func":1
},
{
"ref":"vipy.video.Scene.tracksplit",
"url":39,
"doc":"Split the scene into k separate scenes, one for each track. Each scene starts at frame 0 and is a shallow copy of self containing exactly one track. Use clone() to create a deep copy if needed.",
"func":1
},
{
"ref":"vipy.video.Scene.trackclip",
"url":39,
"doc":"Split the scene into k separate scenes, one for each track. Each scene starts and ends when the track starts and ends",
"func":1
},
{
"ref":"vipy.video.Scene.activityclip",
"url":39,
"doc":"Return a list of vipy.video.Scene() each clipped to be temporally centered on a single activity, with an optional padframes before and after.  The Scene() category is updated to be the activity, and only the objects participating in the activity are included.  Activities are returned ordered in the temporal order they appear in the video.  The returned vipy.video.Scene() objects for each activityclip are clones of the video, with the video buffer flushed.  Each activityclip() is associated with each activity in the scene, and includes all other secondary activities that the objects in the primary activity also perform (if multilabel=True). See activityclip().labels().  Calling activityclip() on activityclip(multilabel=True) can result in duplicate activities, due to the overlapping secondary activities being included in each clip. Be careful.  padframes=int for symmetric padding same before and after  padframes=(int, int) for asymmetric padding before and after  padframes=[(int, int),  .] for activity specific asymmetric padding",
"func":1
},
{
"ref":"vipy.video.Scene.noactivityclip",
"url":39,
"doc":"Return a list of vipy.video.Scene() each clipped on a track segment that has no associated activities.  Each clip will contain exactly one activity \"Background\" which is the interval for this track where no activities are occurring  Each clip will be at least one frame long  strict=True means that background can only occur in frames where no tracks are performing any activities. This is useful so that background is not constructed from secondary objects.  struct=False means that background can only occur in frames where a given track is not performing any activities.  label=str: The activity label to give the background activities. Defaults to the track category (lowercase)  padframes=0: The amount of temporal padding to apply to the clips before and after in frames",
"func":1
},
{
"ref":"vipy.video.Scene.trackbox",
"url":39,
"doc":"The trackbox is the union of all track bounding boxes in the video, or None if there are no tracks",
"func":1
},
{
"ref":"vipy.video.Scene.framebox",
"url":39,
"doc":"Return the bounding box for the image rectangle, requires preview() to get frame shape",
"func":1
},
{
"ref":"vipy.video.Scene.trackcrop",
"url":39,
"doc":"Return the trackcrop() of the scene which is the crop of the video using the trackbox().  If there are no tracks, return None.  if zeropad=True, the zero pad the crop if it is outside the image rectangle, otherwise return only valid pixels",
"func":1
},
{
"ref":"vipy.video.Scene.activitybox",
"url":39,
"doc":"The activitybox is the union of all activity bounding boxes in the video, which is the union of all tracks contributing to all activities. This is most useful after activityclip(). The activitybox is the smallest bounding box that contains all of the boxes from all of the tracks in all activities in this video.",
"func":1
},
{
"ref":"vipy.video.Scene.activitycuboid",
"url":39,
"doc":"The activitycuboid() is the fixed square spatial crop corresponding to the activitybox (or supplied bounding box), which contains all of the valid activities in the scene. This is most useful after activityclip(). The activitycuboid() is a spatial crop of the video corresponding to the supplied boundingbox or the square activitybox(). This crop must be resized such that the maximum dimension is provided since the crop can be tiny and will not be encodable by ffmpeg",
"func":1
},
{
"ref":"vipy.video.Scene.activitysquare",
"url":39,
"doc":"The activity square is the maxsquare activitybox that contains only valid (non-padded) pixels interior to the image",
"func":1
},
{
"ref":"vipy.video.Scene.activitytube",
"url":39,
"doc":"The activitytube() is a sequence of crops where the spatial box changes on every frame to track the activity. The box in each frame is the square activitybox() for this video which is the union of boxes contributing to this activity in each frame. This function does not perform any temporal clipping. Use activityclip() first to split into individual activities. Crops will be optionally dilated, with zeropadding if the box is outside the image rectangle. All crops will be resized so that the maximum dimension is maxdim (and square by default)",
"func":1
},
{
"ref":"vipy.video.Scene.actortube",
"url":39,
"doc":"The actortube() is a sequence of crops where the spatial box changes on every frame to track the primary actor performing an activity. The box in each frame is the square box centered on the primary actor performing the activity, dilated by a given factor (the original box around the actor is unchanged, this just increases the context, with zero padding) This function does not perform any temporal clipping. Use activityclip() first to split into individual activities. All crops will be resized so that the maximum dimension is maxdim (and square by default)",
"func":1
},
{
"ref":"vipy.video.Scene.speed",
"url":39,
"doc":"Change the speed by a multiplier s. If s=1, this will be the same speed, s=0.5 for half-speed (slower playback), s=2 for double-speed (faster playback)",
"func":1
},
{
"ref":"vipy.video.Scene.clip",
"url":39,
"doc":"Clip the video to between (startframe, endframe). This clip is relative to clip() shown by __repr__(). Return a clone of the video for idempotence",
"func":1
},
{
"ref":"vipy.video.Scene.cliptime",
"url":39,
"doc":"Load a video clip betweeen start seconds and end seconds, should be initialized by constructor, which will work but will not set __repr__ correctly",
"func":1
},
{
"ref":"vipy.video.Scene.crop",
"url":39,
"doc":"Crop the video using the supplied box, update tracks relative to crop, video is zeropadded if box is outside frame rectangle",
"func":1
},
{
"ref":"vipy.video.Scene.zeropad",
"url":39,
"doc":"Zero pad the video with padwidth columns before and after, and padheight rows before and after Update tracks accordingly.",
"func":1
},
{
"ref":"vipy.video.Scene.fliplr",
"url":39,
"doc":"Mirror the video left/right by flipping horizontally",
"func":1
},
{
"ref":"vipy.video.Scene.flipud",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Scene.rot90ccw",
"url":39,
"doc":"Rotate the video 90 degrees counter-clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Scene.rot90cw",
"url":39,
"doc":"Rotate the video 90 degrees clockwise, can only be applied prior to load()",
"func":1
},
{
"ref":"vipy.video.Scene.resize",
"url":39,
"doc":"Resize the video to (rows, cols), preserving the aspect ratio if only rows or cols is provided",
"func":1
},
{
"ref":"vipy.video.Scene.mindim",
"url":39,
"doc":"Resize the video so that the minimum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Scene.maxdim",
"url":39,
"doc":"Resize the video so that the maximum of (width,height)=dim, preserving aspect ratio",
"func":1
},
{
"ref":"vipy.video.Scene.rescale",
"url":39,
"doc":"Spatially rescale the scene by a constant scale factor",
"func":1
},
{
"ref":"vipy.video.Scene.startframe",
"url":39,
"doc":"",
"func":1
},
{
"ref":"vipy.video.Scene.extrapolate",
"url":39,
"doc":"Extrapolate the video to frame f and add the extrapolated tracks to the video",
"func":1
},
{
"ref":"vipy.video.Scene.dedupe",
"url":39,
"doc":"Find and delete duplicate tracks by track segmentiou() overlap. Algorithm - For each pair of tracks with the same category, find the larest temporal segment that contains both tracks. - For this segment, compute the IOU for each box interpolated at a stride of dt frames - Compute the mean IOU for this segment. This is the segment IOU. - If the segment IOU is greater than the threshold, merge the shorter of the two tracks with the current track.",
"func":1
},
{
"ref":"vipy.video.Scene.union",
"url":39,
"doc":"Compute the union two scenes as the set of unique activities and tracks. A pair of activities or tracks are non-unique if they overlap spatially and temporally by a given IoU threshold. Merge overlapping tracks. Tracks are merged by considering the mean IoU at the overlapping segment of two tracks with the same category greater than the provided spatial_iou_threshold threshold Activities are merged by considering the temporal IoU of the activities of the same class greater than the provided temporal_iou_threshold threshold Input: -Other: Scene or list of scenes for union. Other may be a clip of self at a different framerate, spatial isotropic scake, clip offset -spatial_iou_threshold: The intersection over union threshold for the mean of the two segments of an overlapping track, Disable by setting to 1.0 -temporal_iou_threshold: The intersection over union threshold for a temporal bounding box for a pair of activities to be declared duplicates. Disable by setting to 1.0 -strict: Require both scenes to share the same underlying video filename -overlap=['average', 'replace', 'keep'] -average: Merge two tracks by averaging the boxes (average=True) if overlapping -replace: merge two tracks by replacing overlapping boxes with other (discard self) -keep: merge two tracks by keeping overlapping boxes with other (discard other) -percentilecover [0,1]: When determining the assignment of two tracks, compute the percentilecover of two tracks by ranking the cover in the overlapping segment and computing the mean of the top-k assignments, where k=len(segment) percentilecover. -percentilesamples [>1]: the number of samples along the overlapping scemgne for computing percentile cover -activity [bool]: union() of activities only -track [bool]: union() of tracks only Output: -Updates this scene to include the non-overlapping activities from other. By default, it takes the strict union of all activities and tracks. Notes: -This is useful for merging scenes computed using a lower resolution/framerate/clipped object or activity detector without running the detector on the high-res scene -This function will preserve the invariance for v  v.clear().union(v.rescale(0.5).framerate(5).activityclip( , to within the quantization error of framerate() downsampling. -percentileiou is a robust method of track assignment when boxes for two tracks (e.g. ground truth and detections) where one track may deform due to occlusion.",
"func":1
},
{
"ref":"vipy.video.Scene.annotate",
"url":39,
"doc":"Generate a video visualization of all annotated objects and activities in the video, at the resolution and framerate of the underlying video, pixels in this video will now contain the overlay This function does not play the video, it only generates an annotation video frames. Use show() which is equivalent to annotate().saveas().play()  In general, this function should not be run on very long videos without the outfile kwarg, as it requires loading the video framewise into memory, try running on clips instead.  For long videos, a btter strategy given a video object vo with an output filename which will use a video stream for annotation",
"func":1
},
{
"ref":"vipy.video.Scene.show",
"url":39,
"doc":"Faster show using interative image show for annotated videos. This can visualize videos before video rendering is complete, but it cannot guarantee frame rates. Large videos with complex scenes will slow this down and will render at lower frame rates.",
"func":1
},
{
"ref":"vipy.video.Scene.thumbnail",
"url":39,
"doc":"Return annotated frame=k of video, save annotation visualization to provided outfile if provided, otherwise return vipy.image.Scene",
"func":1
},
{
"ref":"vipy.video.Scene.stabilize",
"url":39,
"doc":"Background stablization using flow based stabilization masking foreground region. This will output a video with all frames aligned to the first frame, such that the background is static.",
"func":1
},
{
"ref":"vipy.video.Scene.pixelmask",
"url":39,
"doc":"Replace all pixels in foreground boxes with pixelation",
"func":1
},
{
"ref":"vipy.video.Scene.binarymask",
"url":39,
"doc":"Replace all pixels in foreground boxes with white, zero in background",
"func":1
},
{
"ref":"vipy.video.Scene.asfloatmask",
"url":39,
"doc":"Replace all pixels in foreground boxes with fg, and bg in background, return a copy",
"func":1
},
{
"ref":"vipy.video.Scene.meanmask",
"url":39,
"doc":"Replace all pixels in foreground boxes with mean color",
"func":1
},
{
"ref":"vipy.video.Scene.fgmask",
"url":39,
"doc":"Replace all pixels in foreground boxes with zero",
"func":1
},
{
"ref":"vipy.video.Scene.zeromask",
"url":39,
"doc":"Alias for fgmask",
"func":1
},
{
"ref":"vipy.video.Scene.blurmask",
"url":39,
"doc":"Replace all pixels in foreground boxes with gaussian blurred foreground",
"func":1
},
{
"ref":"vipy.video.Scene.downcast",
"url":39,
"doc":"Cast the object to a vipy.video.Video class",
"func":1
},
{
"ref":"vipy.video.Scene.merge_tracks",
"url":39,
"doc":"Merge tracks if a track endpoint dilated by a fraction overlaps exactly one track startpoint, and the endpoint and startpoint are close enough together temporally.  This is useful for continuing tracking when the detection framerate was too low and the assignment falls outside the measurement gate.  This will not work for complex scenes, as it assumes that there is exactly one possible continuation for a track.",
"func":1
},
{
"ref":"vipy.video.Scene.assign",
"url":39,
"doc":"Assign a list of vipy.object.Detections at frame k to scene by greedy track association. In-place update.  miniou [float]: the minimum temporal IOU for activity assignment  minconf [float]: the minimum confidence for a detection to be considered as a new track  maxhistory [int]: the maximum propagation length of a track with no measurements, the frame history ised for velocity estimates  trackconfsamples [int]: the number of uniformly spaced samples along a track to compute a track confidence  gate [int]: the gating distance in pixels used for assignment of fast moving detections. Useful for low detection framerates if a detection does not overlap with the track.  trackcover [float]: the minimum cover necessary for assignment of a detection to a track  activitymerge [bool]: if true, then merge overlapping activity detections of the same track and category, otherwise each activity detection is added as a new detection",
"func":1
},
{
"ref":"vipy.video.Scene.metadata",
"url":39,
"doc":"Return a dictionary of metadata about this video",
"func":1
},
{
"ref":"vipy.video.Scene.videoid",
"url":39,
"doc":"Return a unique video identifier for this video, as specified in the 'video_id' attribute, or by hashing the filename() and url(). Notes: - If the video filename changes (e.g. from transformation), and video_id is not set in self.attributes, then the video ID will change. - If a video does not have a filename or URL or a video ID in the attributes, then this will return None - To preserve a video ID independent of transformations, set self.setattribute('video_id', $MY_ID)",
"func":1
},
{
"ref":"vipy.video.Scene.store",
"url":39,
"doc":"Store the current video file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored video using unstore() -Unpack this stored video and set up the video chains using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded video as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.video.Scene.unstore",
"url":39,
"doc":"Delete the currently stored video from store()",
"func":1
},
{
"ref":"vipy.video.Scene.restore",
"url":39,
"doc":"Save the currently stored video to filename, and set up filename",
"func":1
},
{
"ref":"vipy.video.Scene.stream",
"url":39,
"doc":"Iterator to yield frames streaming from video  Using this iterator may affect PDB debugging due to stdout/stdin redirection. Use ipdb instead.  FFMPEG stdout pipe may screw up bash shell newlines, requiring issuing command \"reset\"",
"func":1
},
{
"ref":"vipy.video.Scene.bytes",
"url":39,
"doc":"Return a bytes representation of the video file",
"func":1
},
{
"ref":"vipy.video.Scene.frames",
"url":39,
"doc":"Alias for __iter__()",
"func":1
},
{
"ref":"vipy.video.Scene.isdirty",
"url":39,
"doc":"Has the FFMPEG filter chain been modified from the default? If so, then ffplay() on the video file will be different from self.load().play()",
"func":1
},
{
"ref":"vipy.video.Scene.probeshape",
"url":39,
"doc":"Return the (height, width) of underlying video file as determined from ffprobe, this does not take into account any applied ffmpeg filters",
"func":1
},
{
"ref":"vipy.video.Scene.duration_in_seconds_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in seconds, requires ffprobe. Fetch once and cache",
"func":1
},
{
"ref":"vipy.video.Scene.duration_in_frames_of_videofile",
"url":39,
"doc":"Return video duration of the source filename (NOT the filter chain) in frames, requires ffprobe",
"func":1
},
{
"ref":"vipy.video.Scene.probe",
"url":39,
"doc":"Run ffprobe on the filename and return the result as a JSON file",
"func":1
},
{
"ref":"vipy.video.Scene.print",
"url":39,
"doc":"Print the representation of the video - useful for debugging in long fluent chains. Sleep is useful for adding in a delay for distributed processing",
"func":1
},
{
"ref":"vipy.video.Scene.dict",
"url":39,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.video.Scene.take",
"url":39,
"doc":"Return n frames from the clip uniformly spaced as numpy array",
"func":1
},
{
"ref":"vipy.video.Scene.colorspace",
"url":39,
"doc":"Return or set the colorspace as ['rgb', 'bgr', 'lum', 'float']",
"func":1
},
{
"ref":"vipy.video.Scene.url",
"url":39,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.video.Scene.isloaded",
"url":39,
"doc":"Return True if the video has been loaded",
"func":1
},
{
"ref":"vipy.video.Scene.canload",
"url":39,
"doc":"Return True if the video can be loaded successfully, useful for filtering bad videos or filtering videos that cannot be loaded using your current FFMPEG version",
"func":1
},
{
"ref":"vipy.video.Scene.fromarray",
"url":39,
"doc":"Alias for self.array( ., copy=True), which forces the new array to be a copy",
"func":1
},
{
"ref":"vipy.video.Scene.fromframes",
"url":39,
"doc":"Create a video from a list of frames",
"func":1
},
{
"ref":"vipy.video.Scene.tonumpy",
"url":39,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.video.Scene.numpy",
"url":39,
"doc":"Convert the video to a writeable numpy array, triggers a load() and copy() as needed",
"func":1
},
{
"ref":"vipy.video.Scene.filename",
"url":39,
"doc":"Update video Filename with optional copy from existing file to new file",
"func":1
},
{
"ref":"vipy.video.Scene.abspath",
"url":39,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.video.Scene.relpath",
"url":39,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.video.Scene.rename",
"url":39,
"doc":"Move the underlying video file preserving the absolute path, such that self.filename()  '/a/b/c.ext' and newname='d.ext', then self.filename() -> '/a/b/d.ext', and move the corresponding file",
"func":1
},
{
"ref":"vipy.video.Scene.filesize",
"url":39,
"doc":"Return the size in bytes of the filename(), None if the filename() is invalid",
"func":1
},
{
"ref":"vipy.video.Scene.download",
"url":39,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.video.Scene.fetch",
"url":39,
"doc":"Download only if hasfilename() is not found",
"func":1
},
{
"ref":"vipy.video.Scene.shape",
"url":39,
"doc":"Return (height, width) of the frames, requires loading a preview frame from the video if the video is not already loaded, or providing the shape=(height,width) by the user",
"func":1
},
{
"ref":"vipy.video.Scene.channels",
"url":39,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.video.Scene.width",
"url":39,
"doc":"Width (cols) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Scene.height",
"url":39,
"doc":"Height (rows) in pixels of the video for the current filter chain",
"func":1
},
{
"ref":"vipy.video.Scene.aspect_ratio",
"url":39,
"doc":"The width/height of the video expressed as a fraction",
"func":1
},
{
"ref":"vipy.video.Scene.preview",
"url":39,
"doc":"Return selected frame of filtered video, return vipy.image.Image object. This is useful for previewing the frame shape of a complex filter chain or the frame contents at a particular location without loading the whole video",
"func":1
},
{
"ref":"vipy.video.Scene.load",
"url":39,
"doc":"Load a video using ffmpeg, applying the requested filter chain. - If verbose=True. then ffmpeg console output will be displayed. - If ignoreErrors=True, then all load errors are warned and skipped. Be sure to call isloaded() to confirm loading was successful. - shape tuple(height, width, channels): If provided, use this shape for reading and reshaping the byte stream from ffmpeg - knowing the final output shape can speed up loads by avoiding a preview() of the filter chain to get the frame size",
"func":1
},
{
"ref":"vipy.video.Scene.randomcrop",
"url":39,
"doc":"Crop the video to shape=(H,W) with random position such that the crop contains only valid pixels, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Scene.centercrop",
"url":39,
"doc":"Crop the video to shape=(H,W) preserving the integer centroid position, and optionally return the box",
"func":1
},
{
"ref":"vipy.video.Scene.centersquare",
"url":39,
"doc":"Crop video of size (NxN) in the center, such that N=min(width,height), keeping the video centroid constant",
"func":1
},
{
"ref":"vipy.video.Scene.cropeven",
"url":39,
"doc":"Crop the video to the largest even (width,height) less than or equal to current (width,height). This is useful for some codecs or filters which require even shape.",
"func":1
},
{
"ref":"vipy.video.Scene.maxsquare",
"url":39,
"doc":"Pad the video to be square, preserving the upper left corner of the video",
"func":1
},
{
"ref":"vipy.video.Scene.pkl",
"url":39,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Scene.pklif",
"url":39,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.video.Scene.webp",
"url":39,
"doc":"Save a video to an animated WEBP file, with pause=N seconds on the last frame between loops. -strict=[bool]: assert that the filename must have an .webp extension -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.video.Scene.gif",
"url":39,
"doc":"Save a video to an animated GIF file, with pause=N seconds between loops. WARNING: this will be very large for big videos, consider using webp instead. -pause=[int]: seconds to pause between loops of the animation -smallest=[bool]: create the smallest possible file but takes much longer to run -smaller=[bool]: create a smaller file, which takes a little longer to run",
"func":1
},
{
"ref":"vipy.video.Scene.saveas",
"url":39,
"doc":"Save video to new output video file. This function does not draw boxes, it saves pixels to a new video file.  outfile: the absolute path to the output video file. This extension can be .mp4 (for video) or [\".webp\",\".gif\"] (for animated image)  If self.array() is loaded, then export the contents of self._array to the video file  If self.array() is not loaded, and there exists a valid video file, apply the filter chain directly to the input video  If outfile None or outfile self.filename(), then overwrite the current filename  If ignoreErrors=True, then exit gracefully. Useful for chaining download().saveas() on parallel dataset downloads  Returns a new video object with this video filename, and a clean video filter chain  if flush=True, then flush the buffer for this object right after saving the new video. This is useful for transcoding in parallel  framerate: input framerate of the frames in the buffer, or the output framerate of the transcoded video. If not provided, use framerate of source video  pause: an integer in seconds to pause between loops of animated images",
"func":1
},
{
"ref":"vipy.video.Scene.ffplay",
"url":39,
"doc":"Play the video file using ffplay",
"func":1
},
{
"ref":"vipy.video.Scene.play",
"url":39,
"doc":"Play the saved video filename in self.filename() using the system 'ffplay', if there is no filename, try to download it, if the filter chain is dirty, dump to temp file first",
"func":1
},
{
"ref":"vipy.video.Scene.torch",
"url":39,
"doc":"Convert the loaded video of shape N HxWxC frames to an MxCxHxW torch tensor, forces a load().  Order of arguments is (startframe, endframe) or (startframe, startframe+length) or (random_startframe, random_starframe+takelength), then stride or take.  Follows numpy slicing rules. Optionally return the slice used if withslice=True  Returns float tensor in the range [0,1] following torchvision.transforms.ToTensor()  order can be ['nchw', 'nhwc', 'cnhw'] for batchsize=n, channels=c, height=h, width=w  boundary can be ['repeat', 'strict', 'cyclic']  withlabel=True, returns tuple (t, labellist), where labellist is a list of tuples of activity labels occurring at the corresponding frame in the tensor  withslice=Trye, returnss tuple (t, (startframe, endframe, stride  nonelabel=True, returns tuple (t, None) if withlabel=False",
"func":1
},
{
"ref":"vipy.video.Scene.clone",
"url":39,
"doc":"Create deep copy of video object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.  flushfilter: Set the ffmpeg filter chain to the default in the new object, useful for saving new videos  rekey: Generate new unique track ID and activity ID keys for this scene  shallow: shallow copy everything (copy by reference), except for ffmpeg object  sharedarray: deep copy of everything, except for pixel buffer which is shared",
"func":1
},
{
"ref":"vipy.video.Scene.flush",
"url":39,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.video.Scene.flush_and_return",
"url":39,
"doc":"Flush the video and return the parameter supplied, useful for long fluent chains",
"func":1
},
{
"ref":"vipy.video.Scene.map",
"url":39,
"doc":"Apply lambda function to the loaded numpy array img, changes pixels not shape Lambda function must have the following signature:  newimg = func(img)  img: HxWxC numpy array for a single frame of video  newimg: HxWxC modified numpy array for this frame. Change only the pixels, not the shape The lambda function will be applied to every frame in the video in frame index order.",
"func":1
},
{
"ref":"vipy.video.Scene.gain",
"url":39,
"doc":"Pixelwise multiplicative gain, such that each pixel p_{ij} = g  p_{ij}",
"func":1
},
{
"ref":"vipy.video.Scene.bias",
"url":39,
"doc":"Pixelwise additive bias, such that each pixel p_{ij} = b + p_{ij}",
"func":1
},
{
"ref":"vipy.video.Scene.normalize",
"url":39,
"doc":"Pixelwise whitening, out =  scale in) - mean) / std); triggers load(). All computations float32",
"func":1
},
{
"ref":"vipy.video.RandomVideo",
"url":39,
"doc":"Return a random loaded vipy.video.video, useful for unit testing, minimum size (32x32x32)",
"func":1
},
{
"ref":"vipy.video.RandomScene",
"url":39,
"doc":"Return a random loaded vipy.video.Scene, useful for unit testing",
"func":1
},
{
"ref":"vipy.video.RandomSceneActivity",
"url":39,
"doc":"Return a random loaded vipy.video.Scene, useful for unit testing",
"func":1
},
{
"ref":"vipy.video.EmptyScene",
"url":39,
"doc":"Return an empty scene",
"func":1
},
{
"ref":"vipy.image",
"url":49,
"doc":""
},
{
"ref":"vipy.image.Image",
"url":49,
"doc":"vipy.image.Image class The vipy image class provides a fluent, lazy interface for representing, transforming and visualizing images. The following constructors are supported: >>> im = vipy.image.Image(filename=\"/path/to/image.ext\") All image file formats that are readable by PIL are supported here. >>> im = vipy.image.Image(url=\"http: domain.com/path/to/image.ext\") The image will be downloaded from the provided url and saved to a temporary filename. The environment variable VIPY_CACHE controls the location of the directory used for saving images, otherwise this will be saved to the system temp directory. >>> im = vipy.image.Image(url=\"http: domain.com/path/to/image.ext\", filename=\"/path/to/new/image.ext\") The image will be downloaded from the provided url and saved to the provided filename. The url() method provides optional basic authentication set for username and password >>> im = vipy.image.Image(array=img, colorspace='rgb') The image will be constructed from a provided numpy array 'img', with an associated colorspace. The numpy array and colorspace can be one of the following combinations: 'rgb': uint8, three channel (red, green, blue) 'rgba': uint8, four channel (rgb + alpha) 'bgr': uint8, three channel (blue, green, red), such as is returned from cv2.imread() 'bgra': uint8, four channel 'hsv': uint8, three channel (hue, saturation, value) 'lum;: uint8, one channel, luminance (8 bit grey level) 'grey': float32, one channel in range [0,1] (32 bit intensity) 'float': float32, any channel in range [-inf, +inf] The most general colorspace is 'float' which is used to manipulate images prior to network encoding, such as applying bias."
},
{
"ref":"vipy.image.Image.cast",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.from_json",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.print",
"url":49,
"doc":"Print the representation of the image and return self - useful for debugging in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Image.tile",
"url":49,
"doc":"Generate a list of tiled image",
"func":1
},
{
"ref":"vipy.image.Image.union",
"url":49,
"doc":"No-op for vipy.image.Image",
"func":1
},
{
"ref":"vipy.image.Image.untile",
"url":49,
"doc":"Undo tile",
"func":1
},
{
"ref":"vipy.image.Image.uncrop",
"url":49,
"doc":"Uncrop using provided bounding box and zeropad to shape=(Height, Width), NOT idempotent",
"func":1
},
{
"ref":"vipy.image.Image.splat",
"url":49,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.Image.store",
"url":49,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.Image.unstore",
"url":49,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.Image.restore",
"url":49,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.Image.abspath",
"url":49,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.Image.relpath",
"url":49,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.Image.canload",
"url":49,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.Image.dict",
"url":49,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.Image.json",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.loader",
"url":49,
"doc":"Lambda function to load an unsupported image filename to a numpy array",
"func":1
},
{
"ref":"vipy.image.Image.load",
"url":49,
"doc":"Load image to cached private '_array' attribute and return Image object",
"func":1
},
{
"ref":"vipy.image.Image.download",
"url":49,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.image.Image.reload",
"url":49,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.Image.isloaded",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.channels",
"url":49,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.Image.iscolor",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.Image.istransparent",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.Image.isgrey",
"url":49,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.Image.isluminance",
"url":49,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.Image.filesize",
"url":49,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.Image.width",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.height",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.shape",
"url":49,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image",
"func":1
},
{
"ref":"vipy.image.Image.aspectratio",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.area",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.centroid",
"url":49,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y)",
"func":1
},
{
"ref":"vipy.image.Image.centerpixel",
"url":49,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j)",
"func":1
},
{
"ref":"vipy.image.Image.array",
"url":49,
"doc":"Replace self._array with provided numpy array",
"func":1
},
{
"ref":"vipy.image.Image.channel",
"url":49,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None",
"func":1
},
{
"ref":"vipy.image.Image.red",
"url":49,
"doc":"Return red channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Image.green",
"url":49,
"doc":"Return green channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Image.blue",
"url":49,
"doc":"Return blue channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Image.alpha",
"url":49,
"doc":"Return alpha (transparency) channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Image.fromarray",
"url":49,
"doc":"Alias for array(data, copy=True), set new array() with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.Image.tonumpy",
"url":49,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.image.Image.numpy",
"url":49,
"doc":"Convert vipy.image.Image to numpy array, returns writeable array",
"func":1
},
{
"ref":"vipy.image.Image.zeros",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.pil",
"url":49,
"doc":"Convert vipy.image.Image to PIL Image, by reference",
"func":1
},
{
"ref":"vipy.image.Image.blur",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.torch",
"url":49,
"doc":"Convert the batch of 1 HxWxC images to a 1xCxHxW torch tensor, by reference",
"func":1
},
{
"ref":"vipy.image.Image.fromtorch",
"url":49,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.Image.nofilename",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.filename",
"url":49,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.Image.nourl",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.url",
"url":49,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.Image.colorspace",
"url":49,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.Image.uri",
"url":49,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.Image.setattribute",
"url":49,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.Image.setattributes",
"url":49,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.Image.getattribute",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.hasattribute",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.delattribute",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.hasurl",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.hasfilename",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.clone",
"url":49,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.Image.flush",
"url":49,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.Image.resize",
"url":49,
"doc":"Resize the image buffer to (rows x cols) with bilinear interpolation. If rows or cols is provided, rescale image maintaining aspect ratio",
"func":1
},
{
"ref":"vipy.image.Image.resize_like",
"url":49,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.Image.rescale",
"url":49,
"doc":"Scale the image buffer by the given factor - NOT idempotent",
"func":1
},
{
"ref":"vipy.image.Image.maxdim",
"url":49,
"doc":"Resize image preserving aspect ratio so that maximum dimension of image = dim, or return maxdim()",
"func":1
},
{
"ref":"vipy.image.Image.mindim",
"url":49,
"doc":"Resize image preserving aspect ratio so that minimum dimension of image = dim, or return mindim()",
"func":1
},
{
"ref":"vipy.image.Image.zeropad",
"url":49,
"doc":"Pad image using np.pad constant by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Image.zeropadlike",
"url":49,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.Image.meanpad",
"url":49,
"doc":"Pad image using np.pad constant=image mean by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Image.alphapad",
"url":49,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Image.minsquare",
"url":49,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Image.maxsquare",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Image.maxmatte",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.Image.centersquare",
"url":49,
"doc":"Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.Image.centercrop",
"url":49,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.Image.cornercrop",
"url":49,
"doc":"Crop image of size (height x width) from the upper left corner",
"func":1
},
{
"ref":"vipy.image.Image.crop",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.fliplr",
"url":49,
"doc":"Mirror the image buffer about the vertical axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.Image.flipud",
"url":49,
"doc":"Mirror the image buffer about the horizontal axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.Image.imagebox",
"url":49,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.Image.border_mask",
"url":49,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.Image.rgb",
"url":49,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.rgba",
"url":49,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.hsv",
"url":49,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.bgr",
"url":49,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.bgra",
"url":49,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Image.float",
"url":49,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.Image.greyscale",
"url":49,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.Image.grayscale",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Image.grey",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Image.gray",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Image.luminance",
"url":49,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.Image.lum",
"url":49,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.Image.jet",
"url":49,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.Image.rainbow",
"url":49,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Image.hot",
"url":49,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Image.bone",
"url":49,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Image.saturate",
"url":49,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.Image.intensity",
"url":49,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.Image.mat2gray",
"url":49,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.Image.gain",
"url":49,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.Image.bias",
"url":49,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.Image.stats",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.min",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.max",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.mean",
"url":49,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.Image.meanchannel",
"url":49,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.Image.sum",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Image.closeall",
"url":49,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.Image.close",
"url":49,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.Image.show",
"url":49,
"doc":"Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object",
"func":1
},
{
"ref":"vipy.image.Image.save",
"url":49,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.Image.pkl",
"url":49,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Image.pklif",
"url":49,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Image.saveas",
"url":49,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.Image.saveastmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.Image.savetmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.Image.base64",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.Image.html",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.Image.annotate",
"url":49,
"doc":"Change pixels of this image to include rendered annotation and return an image object",
"func":1
},
{
"ref":"vipy.image.Image.savefig",
"url":49,
"doc":"Save last figure output from self.show() with drawing overlays to provided filename and return filename",
"func":1
},
{
"ref":"vipy.image.Image.map",
"url":49,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.Image.downcast",
"url":49,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.ImageCategory",
"url":49,
"doc":"vipy ImageCategory class This class provides a representation of a vipy.image.Image with a category. Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label') >>> im = vipy.image.ImageCategory(filename='/path/to/dog_image.ext', category='dog') >>> im = vipy.image.ImageCategory(url='http: path/to/dog_image.ext', category='dog') >>> im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog')"
},
{
"ref":"vipy.image.ImageCategory.cast",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.is_",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.is_not",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.nocategory",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageCategory.category",
"url":49,
"doc":"Return or update the category",
"func":1
},
{
"ref":"vipy.image.ImageCategory.label",
"url":49,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.image.ImageCategory.score",
"url":49,
"doc":"Real valued score for categorization, larger is better",
"func":1
},
{
"ref":"vipy.image.ImageCategory.probability",
"url":49,
"doc":"Real valued probability for categorization, [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageCategory.print",
"url":49,
"doc":"Print the representation of the image and return self - useful for debugging in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageCategory.tile",
"url":49,
"doc":"Generate a list of tiled image",
"func":1
},
{
"ref":"vipy.image.ImageCategory.union",
"url":49,
"doc":"No-op for vipy.image.Image",
"func":1
},
{
"ref":"vipy.image.ImageCategory.untile",
"url":49,
"doc":"Undo tile",
"func":1
},
{
"ref":"vipy.image.ImageCategory.uncrop",
"url":49,
"doc":"Uncrop using provided bounding box and zeropad to shape=(Height, Width), NOT idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.splat",
"url":49,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.ImageCategory.store",
"url":49,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.ImageCategory.unstore",
"url":49,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.restore",
"url":49,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.abspath",
"url":49,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.relpath",
"url":49,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.canload",
"url":49,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.ImageCategory.dict",
"url":49,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.loader",
"url":49,
"doc":"Lambda function to load an unsupported image filename to a numpy array",
"func":1
},
{
"ref":"vipy.image.ImageCategory.load",
"url":49,
"doc":"Load image to cached private '_array' attribute and return Image object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.download",
"url":49,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.reload",
"url":49,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.ImageCategory.channels",
"url":49,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.iscolor",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.ImageCategory.istransparent",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.ImageCategory.isgrey",
"url":49,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.ImageCategory.isluminance",
"url":49,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.ImageCategory.filesize",
"url":49,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.ImageCategory.shape",
"url":49,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centroid",
"url":49,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centerpixel",
"url":49,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.array",
"url":49,
"doc":"Replace self._array with provided numpy array",
"func":1
},
{
"ref":"vipy.image.ImageCategory.channel",
"url":49,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None",
"func":1
},
{
"ref":"vipy.image.ImageCategory.red",
"url":49,
"doc":"Return red channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.green",
"url":49,
"doc":"Return green channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.blue",
"url":49,
"doc":"Return blue channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.alpha",
"url":49,
"doc":"Return alpha (transparency) channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.fromarray",
"url":49,
"doc":"Alias for array(data, copy=True), set new array() with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.ImageCategory.tonumpy",
"url":49,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.numpy",
"url":49,
"doc":"Convert vipy.image.Image to numpy array, returns writeable array",
"func":1
},
{
"ref":"vipy.image.ImageCategory.pil",
"url":49,
"doc":"Convert vipy.image.Image to PIL Image, by reference",
"func":1
},
{
"ref":"vipy.image.ImageCategory.torch",
"url":49,
"doc":"Convert the batch of 1 HxWxC images to a 1xCxHxW torch tensor, by reference",
"func":1
},
{
"ref":"vipy.image.ImageCategory.fromtorch",
"url":49,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.filename",
"url":49,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.url",
"url":49,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.ImageCategory.colorspace",
"url":49,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.ImageCategory.uri",
"url":49,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.ImageCategory.setattribute",
"url":49,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.ImageCategory.setattributes",
"url":49,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.ImageCategory.clone",
"url":49,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.ImageCategory.flush",
"url":49,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.ImageCategory.resize",
"url":49,
"doc":"Resize the image buffer to (rows x cols) with bilinear interpolation. If rows or cols is provided, rescale image maintaining aspect ratio",
"func":1
},
{
"ref":"vipy.image.ImageCategory.resize_like",
"url":49,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rescale",
"url":49,
"doc":"Scale the image buffer by the given factor - NOT idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.maxdim",
"url":49,
"doc":"Resize image preserving aspect ratio so that maximum dimension of image = dim, or return maxdim()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.mindim",
"url":49,
"doc":"Resize image preserving aspect ratio so that minimum dimension of image = dim, or return mindim()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.zeropad",
"url":49,
"doc":"Pad image using np.pad constant by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.zeropadlike",
"url":49,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.ImageCategory.meanpad",
"url":49,
"doc":"Pad image using np.pad constant=image mean by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.alphapad",
"url":49,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageCategory.minsquare",
"url":49,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.maxsquare",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.maxmatte",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centersquare",
"url":49,
"doc":"Crop image of size (NxN) in the center, such that N=min(width,height), keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.centercrop",
"url":49,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageCategory.cornercrop",
"url":49,
"doc":"Crop image of size (height x width) from the upper left corner",
"func":1
},
{
"ref":"vipy.image.ImageCategory.fliplr",
"url":49,
"doc":"Mirror the image buffer about the vertical axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.flipud",
"url":49,
"doc":"Mirror the image buffer about the horizontal axis - Not idempotent",
"func":1
},
{
"ref":"vipy.image.ImageCategory.imagebox",
"url":49,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.ImageCategory.border_mask",
"url":49,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rgb",
"url":49,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rgba",
"url":49,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.hsv",
"url":49,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bgr",
"url":49,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bgra",
"url":49,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageCategory.float",
"url":49,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.ImageCategory.greyscale",
"url":49,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageCategory.grayscale",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.grey",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.gray",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.luminance",
"url":49,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.ImageCategory.lum",
"url":49,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.jet",
"url":49,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.rainbow",
"url":49,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.hot",
"url":49,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bone",
"url":49,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageCategory.saturate",
"url":49,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.ImageCategory.intensity",
"url":49,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.mat2gray",
"url":49,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.gain",
"url":49,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageCategory.bias",
"url":49,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageCategory.mean",
"url":49,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.meanchannel",
"url":49,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageCategory.closeall",
"url":49,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.ImageCategory.close",
"url":49,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.ImageCategory.show",
"url":49,
"doc":"Display image on screen in provided figure number (clone and convert to RGB colorspace to show), return object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.save",
"url":49,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.pkl",
"url":49,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageCategory.pklif",
"url":49,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageCategory.saveas",
"url":49,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.saveastmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.savetmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.ImageCategory.base64",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.ImageCategory.html",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.ImageCategory.annotate",
"url":49,
"doc":"Change pixels of this image to include rendered annotation and return an image object",
"func":1
},
{
"ref":"vipy.image.ImageCategory.savefig",
"url":49,
"doc":"Save last figure output from self.show() with drawing overlays to provided filename and return filename",
"func":1
},
{
"ref":"vipy.image.ImageCategory.map",
"url":49,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageCategory.downcast",
"url":49,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.Scene",
"url":49,
"doc":"vipy.image.Scene class This class provides a representation of a vipy.image.ImageCategory with one or more vipy.object.Detections. The goal of this class is to provide a unified representation for all objects in a scene. Valid constructors include all provided by vipy.image.Image() and vipy.image.ImageCategory() with the additional kwarg 'objects', which is a list of vipy.object.Detections() >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='city', objects=[vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)]) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='city').objects([vipy.object.Detection(category='vehicle', xmin=0, ymin=0, width=100, height=100)]) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels='face', xywh=[0,0,100,100]) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels='face', xywh= 0,0,100,100], [100,100,200,200 ) >>> im = vipy.image.Scene(filename='/path/to/city_image.ext', category='office', boxlabels=['face', 'desk'] xywh= 0,0,100,100], [200,200,300,300 )"
},
{
"ref":"vipy.image.Scene.cast",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.from_json",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.json",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.append",
"url":49,
"doc":"Append the provided vipy.object.Detection object to the scene object list",
"func":1
},
{
"ref":"vipy.image.Scene.add",
"url":49,
"doc":"Alias for append",
"func":1
},
{
"ref":"vipy.image.Scene.objects",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.objectmap",
"url":49,
"doc":"Apply lambda function f to each object. If f is a list of lambda, apply one to one with the objects",
"func":1
},
{
"ref":"vipy.image.Scene.objectfilter",
"url":49,
"doc":"Apply lambda function f to each object and keep if filter is True",
"func":1
},
{
"ref":"vipy.image.Scene.nms",
"url":49,
"doc":"Non-maximum supporession of objects() by category based on confidence and spatial IoU and cover thresholds",
"func":1
},
{
"ref":"vipy.image.Scene.intersection",
"url":49,
"doc":"Return a Scene() containing the objects in both self and other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.Scene.difference",
"url":49,
"doc":"Return a Scene() containing the objects in self but not other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.Scene.union",
"url":49,
"doc":"Combine the objects of the scene with other and self with no duplicate checking unless miniou is not None",
"func":1
},
{
"ref":"vipy.image.Scene.uncrop",
"url":49,
"doc":"Uncrop a previous crop(bb) called with the supplied bb=BoundingBox(), and zeropad to shape=(H,W)",
"func":1
},
{
"ref":"vipy.image.Scene.clear",
"url":49,
"doc":"Remove all objects from this scene.",
"func":1
},
{
"ref":"vipy.image.Scene.boundingbox",
"url":49,
"doc":"The boundingbox of a scene is the union of all object bounding boxes, or None if there are no objects",
"func":1
},
{
"ref":"vipy.image.Scene.categories",
"url":49,
"doc":"Return list of unique object categories in scene",
"func":1
},
{
"ref":"vipy.image.Scene.imclip",
"url":49,
"doc":"Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image",
"func":1
},
{
"ref":"vipy.image.Scene.rescale",
"url":49,
"doc":"Rescale image buffer and all bounding boxes - Not idempotent",
"func":1
},
{
"ref":"vipy.image.Scene.resize",
"url":49,
"doc":"Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly. If cols or rows is None, then scale isotropically",
"func":1
},
{
"ref":"vipy.image.Scene.centersquare",
"url":49,
"doc":"Crop the image of size (H,W) to be centersquare (min(H,W), min(H,W preserving center, and update bounding boxes",
"func":1
},
{
"ref":"vipy.image.Scene.fliplr",
"url":49,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.Scene.flipud",
"url":49,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.Scene.dilate",
"url":49,
"doc":"Dilate all bounding boxes by scale factor, dilated boxes may be outside image rectangle",
"func":1
},
{
"ref":"vipy.image.Scene.zeropad",
"url":49,
"doc":"Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.Scene.meanpad",
"url":49,
"doc":"Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.Scene.rot90cw",
"url":49,
"doc":"Rotate the scene 90 degrees clockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.Scene.rot90ccw",
"url":49,
"doc":"Rotate the scene 90 degrees counterclockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.Scene.maxdim",
"url":49,
"doc":"Resize scene preserving aspect ratio so that maximum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.Scene.mindim",
"url":49,
"doc":"Resize scene preserving aspect ratio so that minimum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.Scene.crop",
"url":49,
"doc":"Crop the image buffer using the supplied bounding box object (or the only object if bbox=None), clipping the box to the image rectangle, update all scene objects",
"func":1
},
{
"ref":"vipy.image.Scene.centercrop",
"url":49,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.Scene.cornercrop",
"url":49,
"doc":"Crop image of size (height x width) from the upper left corner, returning valid pixels only",
"func":1
},
{
"ref":"vipy.image.Scene.padcrop",
"url":49,
"doc":"Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects",
"func":1
},
{
"ref":"vipy.image.Scene.cornerpadcrop",
"url":49,
"doc":"Crop image of size (height x width) from the upper left corner, returning zero padded result out to (height, width)",
"func":1
},
{
"ref":"vipy.image.Scene.rectangular_mask",
"url":49,
"doc":"Return a binary array of the same size as the image (or using the provided image width and height (W,H) size to avoid an image load), with ones inside the bounding box",
"func":1
},
{
"ref":"vipy.image.Scene.binarymask",
"url":49,
"doc":"Alias for rectangular_mask with in-place update",
"func":1
},
{
"ref":"vipy.image.Scene.bgmask",
"url":49,
"doc":"Set all pixels outside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.Scene.fgmask",
"url":49,
"doc":"Set all pixels inside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.Scene.setzero",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.Scene.pixelmask",
"url":49,
"doc":"Replace pixels within all foreground objects with a privacy preserving pixelated foreground with larger pixels",
"func":1
},
{
"ref":"vipy.image.Scene.blurmask",
"url":49,
"doc":"Replace pixels within all foreground objects with a privacy preserving blurred foreground",
"func":1
},
{
"ref":"vipy.image.Scene.replace",
"url":49,
"doc":"Set all image values within the bounding box equal to the provided img, triggers load() and imclip()",
"func":1
},
{
"ref":"vipy.image.Scene.meanmask",
"url":49,
"doc":"Replace pixels within the foreground objects with the mean pixel color",
"func":1
},
{
"ref":"vipy.image.Scene.fghash",
"url":49,
"doc":"Perceptual differential hash function, computed for each foreground region independently",
"func":1
},
{
"ref":"vipy.image.Scene.perceptualhash",
"url":49,
"doc":"Perceptual differentialhash function  bits [int]: longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes).  Algorithm: set foreground objects to mean color, convert to greyscale, resize with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs.  NOTE: Can be used for near duplicate detection of background scenes by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing.  NOTE: The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex( bghash()  which is equivalent to bghash(asbinary=True)  objmask [bool]: if trye, replace the foreground object masks with the mean color prior to computing",
"func":1
},
{
"ref":"vipy.image.Scene.bghash",
"url":49,
"doc":"Percetual differential hash function, masking out foreground regions",
"func":1
},
{
"ref":"vipy.image.Scene.isduplicate",
"url":49,
"doc":"Background hash near duplicate detection, returns true if self and im are near duplicate images using bghash",
"func":1
},
{
"ref":"vipy.image.Scene.show",
"url":49,
"doc":"Show scene detection  categories [list]: List of category (or shortlabel) names in the scene to show  fontsize (int, string): Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size  figure (int): Figure number, show the image in the provided figure=int numbered window  nocaption (bool): Show or do not show the text caption in the upper left of the box  nocaption_withstring (list): Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list  boxalpha (float, [0,1]): Set the text box background to be semi-transparent with an alpha  d_category2color (dict): Define a dictionary of required mapping of specific category() to box colors. Non-specified categories are assigned a random named color from vipy.show.colorlist()  caption_offset (int, int): The relative position of the caption to the upper right corner of the box.  nowindow (bool): Display or not display the image  textfacecolor (str): One of the named colors from vipy.show.colorlist() for the color of the textbox background  textfacealpha (float, [0,1]): The textbox background transparency  shortlabel (bool): Whether to show the shortlabel or the full category name in the caption  mutator (lambda): A lambda function with signature lambda im: f(im) which will modify this image prior to show. Useful for changing labels on the fly",
"func":1
},
{
"ref":"vipy.image.Scene.annotate",
"url":49,
"doc":"Alias for savefig",
"func":1
},
{
"ref":"vipy.image.Scene.savefig",
"url":49,
"doc":"Save show() output to given file or return buffer without popping up a window",
"func":1
},
{
"ref":"vipy.image.Scene.category",
"url":49,
"doc":"Return or update the category",
"func":1
},
{
"ref":"vipy.image.Scene.label",
"url":49,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.image.Scene.score",
"url":49,
"doc":"Real valued score for categorization, larger is better",
"func":1
},
{
"ref":"vipy.image.Scene.probability",
"url":49,
"doc":"Real valued probability for categorization, [0,1]",
"func":1
},
{
"ref":"vipy.image.Scene.print",
"url":49,
"doc":"Print the representation of the image and return self - useful for debugging in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Scene.tile",
"url":49,
"doc":"Generate a list of tiled image",
"func":1
},
{
"ref":"vipy.image.Scene.untile",
"url":49,
"doc":"Undo tile",
"func":1
},
{
"ref":"vipy.image.Scene.splat",
"url":49,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.Scene.store",
"url":49,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.Scene.unstore",
"url":49,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.Scene.restore",
"url":49,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.Scene.abspath",
"url":49,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.Scene.relpath",
"url":49,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.Scene.canload",
"url":49,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.Scene.dict",
"url":49,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.Scene.loader",
"url":49,
"doc":"Lambda function to load an unsupported image filename to a numpy array",
"func":1
},
{
"ref":"vipy.image.Scene.load",
"url":49,
"doc":"Load image to cached private '_array' attribute and return Image object",
"func":1
},
{
"ref":"vipy.image.Scene.download",
"url":49,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.image.Scene.reload",
"url":49,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.Scene.channels",
"url":49,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.Scene.iscolor",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.Scene.istransparent",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.Scene.isgrey",
"url":49,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.Scene.isluminance",
"url":49,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.Scene.filesize",
"url":49,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.Scene.shape",
"url":49,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image",
"func":1
},
{
"ref":"vipy.image.Scene.centroid",
"url":49,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y)",
"func":1
},
{
"ref":"vipy.image.Scene.centerpixel",
"url":49,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j)",
"func":1
},
{
"ref":"vipy.image.Scene.array",
"url":49,
"doc":"Replace self._array with provided numpy array",
"func":1
},
{
"ref":"vipy.image.Scene.channel",
"url":49,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None",
"func":1
},
{
"ref":"vipy.image.Scene.red",
"url":49,
"doc":"Return red channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Scene.green",
"url":49,
"doc":"Return green channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Scene.blue",
"url":49,
"doc":"Return blue channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Scene.alpha",
"url":49,
"doc":"Return alpha (transparency) channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.Scene.fromarray",
"url":49,
"doc":"Alias for array(data, copy=True), set new array() with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.Scene.tonumpy",
"url":49,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.image.Scene.numpy",
"url":49,
"doc":"Convert vipy.image.Image to numpy array, returns writeable array",
"func":1
},
{
"ref":"vipy.image.Scene.pil",
"url":49,
"doc":"Convert vipy.image.Image to PIL Image, by reference",
"func":1
},
{
"ref":"vipy.image.Scene.torch",
"url":49,
"doc":"Convert the batch of 1 HxWxC images to a 1xCxHxW torch tensor, by reference",
"func":1
},
{
"ref":"vipy.image.Scene.fromtorch",
"url":49,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.filename",
"url":49,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.Scene.url",
"url":49,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.Scene.colorspace",
"url":49,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.Scene.uri",
"url":49,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.Scene.setattribute",
"url":49,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.Scene.setattributes",
"url":49,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.Scene.clone",
"url":49,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.Scene.flush",
"url":49,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.Scene.resize_like",
"url":49,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.Scene.zeropadlike",
"url":49,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.Scene.alphapad",
"url":49,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.Scene.minsquare",
"url":49,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Scene.maxsquare",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.Scene.maxmatte",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.Scene.imagebox",
"url":49,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.Scene.border_mask",
"url":49,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.Scene.rgb",
"url":49,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.rgba",
"url":49,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.hsv",
"url":49,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.bgr",
"url":49,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.bgra",
"url":49,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.Scene.float",
"url":49,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.Scene.greyscale",
"url":49,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.Scene.grayscale",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Scene.grey",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Scene.gray",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.Scene.luminance",
"url":49,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.Scene.lum",
"url":49,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.Scene.jet",
"url":49,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.Scene.rainbow",
"url":49,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Scene.hot",
"url":49,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Scene.bone",
"url":49,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.Scene.saturate",
"url":49,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.Scene.intensity",
"url":49,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.Scene.mat2gray",
"url":49,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.Scene.gain",
"url":49,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.Scene.bias",
"url":49,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.Scene.mean",
"url":49,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.Scene.meanchannel",
"url":49,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.Scene.closeall",
"url":49,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.Scene.close",
"url":49,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.Scene.save",
"url":49,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.Scene.pkl",
"url":49,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Scene.pklif",
"url":49,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.Scene.saveas",
"url":49,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.Scene.saveastmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.Scene.savetmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.Scene.base64",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.Scene.html",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.Scene.map",
"url":49,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.Scene.downcast",
"url":49,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.ImageDetection",
"url":49,
"doc":"vipy.image.ImageDetection class This class provides a representation of a vipy.image.Image with a single object detection with a category and a vipy.geometry.BoundingBox This class inherits all methods of Scene and BoundingBox. Be careful with overloaded methods clone(), width() and height() which will correspond to these methods for Scene() and not BoundingBox(). Use bbclone(), bbwidth() or bbheight() to access the subclass. Valid constructors include all provided by vipy.image.Image with the additional kwarg 'category' (or alias 'label'), and BoundingBox coordinates >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, width=100, height=100) >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xmin=0, ymin=0, xmax=100, ymax=100) >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', xcentroid=50, ycentroid=50, width=100, height=100) >>> im = vipy.image.ImageDetection(filename='/path/to/dog_image.ext', category='dog', bbox=vipy.geometry.BoundingBox(xmin=0, ymin=0, width=100, height=100 >>> im = vipy.image.ImageCategory(url='http: path/to/dog_image.ext', category='dog').boundingbox(xmin=0, ymin=0, width=100, height=100) >>> im = vipy.image.ImageCategory(array=dog_img, colorspace='rgb', category='dog', xmin=0, ymin=0, width=100, height=100)"
},
{
"ref":"vipy.image.ImageDetection.cast",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.boundingbox",
"url":49,
"doc":"Modify the bounding box using the provided parameters, or return the box if no parameters provided",
"func":1
},
{
"ref":"vipy.image.ImageDetection.asimage",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.asbbox",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.boxmap",
"url":49,
"doc":"Apply the lambda function f to the bounding box, and return the imagedetection",
"func":1
},
{
"ref":"vipy.image.ImageDetection.crop",
"url":49,
"doc":"Crop the image using the bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.append",
"url":49,
"doc":"Append the provided vipy.object.Detection object to the scene object list",
"func":1
},
{
"ref":"vipy.image.ImageDetection.detection",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isinterior",
"url":49,
"doc":"Is the bounding box fully within the image rectangle? Use provided image width and height (W,H) to avoid lots of reloads in some conditions",
"func":1
},
{
"ref":"vipy.image.ImageDetection.add",
"url":49,
"doc":"Alias for append",
"func":1
},
{
"ref":"vipy.image.ImageDetection.objectmap",
"url":49,
"doc":"Apply lambda function f to each object. If f is a list of lambda, apply one to one with the objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.objectfilter",
"url":49,
"doc":"Apply lambda function f to each object and keep if filter is True",
"func":1
},
{
"ref":"vipy.image.ImageDetection.nms",
"url":49,
"doc":"Non-maximum supporession of objects() by category based on confidence and spatial IoU and cover thresholds",
"func":1
},
{
"ref":"vipy.image.ImageDetection.intersection",
"url":49,
"doc":"Return a Scene() containing the objects in both self and other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.ImageDetection.difference",
"url":49,
"doc":"Return a Scene() containing the objects in self but not other, that overlap by miniou with greedy assignment",
"func":1
},
{
"ref":"vipy.image.ImageDetection.union",
"url":49,
"doc":"Combine the objects of the scene with other and self with no duplicate checking unless miniou is not None",
"func":1
},
{
"ref":"vipy.image.ImageDetection.uncrop",
"url":49,
"doc":"Uncrop a previous crop(bb) called with the supplied bb=BoundingBox(), and zeropad to shape=(H,W)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.clear",
"url":49,
"doc":"Remove all objects from this scene.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.categories",
"url":49,
"doc":"Return list of unique object categories in scene",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imclip",
"url":49,
"doc":"Clip all bounding boxes to the image rectangle, silently rejecting those boxes that are degenerate or outside the image",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rescale",
"url":49,
"doc":"Rescale image buffer and all bounding boxes - Not idempotent",
"func":1
},
{
"ref":"vipy.image.ImageDetection.resize",
"url":49,
"doc":"Resize image buffer to (height=rows, width=cols) and transform all bounding boxes accordingly. If cols or rows is None, then scale isotropically",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centersquare",
"url":49,
"doc":"Crop the image of size (H,W) to be centersquare (min(H,W), min(H,W preserving center, and update bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fliplr",
"url":49,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.ImageDetection.flipud",
"url":49,
"doc":"Mirror buffer and all bounding box around vertical axis",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilate",
"url":49,
"doc":"Dilate all bounding boxes by scale factor, dilated boxes may be outside image rectangle",
"func":1
},
{
"ref":"vipy.image.ImageDetection.zeropad",
"url":49,
"doc":"Zero pad image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.ImageDetection.meanpad",
"url":49,
"doc":"Mean pad (image color mean) image with padwidth cols before and after and padheight rows before and after, then update bounding box offsets",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rot90cw",
"url":49,
"doc":"Rotate the scene 90 degrees clockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rot90ccw",
"url":49,
"doc":"Rotate the scene 90 degrees counterclockwise, and update objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxdim",
"url":49,
"doc":"Resize scene preserving aspect ratio so that maximum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mindim",
"url":49,
"doc":"Resize scene preserving aspect ratio so that minimum dimension of image = dim, update all objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centercrop",
"url":49,
"doc":"Crop image of size (height x width) in the center, keeping the image centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cornercrop",
"url":49,
"doc":"Crop image of size (height x width) from the upper left corner, returning valid pixels only",
"func":1
},
{
"ref":"vipy.image.ImageDetection.padcrop",
"url":49,
"doc":"Crop the image buffer using the supplied bounding box object, zero padding if box is outside image rectangle, update all scene objects",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cornerpadcrop",
"url":49,
"doc":"Crop image of size (height x width) from the upper left corner, returning zero padded result out to (height, width)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rectangular_mask",
"url":49,
"doc":"Return a binary array of the same size as the image (or using the provided image width and height (W,H) size to avoid an image load), with ones inside the bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.binarymask",
"url":49,
"doc":"Alias for rectangular_mask with in-place update",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bgmask",
"url":49,
"doc":"Set all pixels outside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fgmask",
"url":49,
"doc":"Set all pixels inside the bounding box to zero",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pixelmask",
"url":49,
"doc":"Replace pixels within all foreground objects with a privacy preserving pixelated foreground with larger pixels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blurmask",
"url":49,
"doc":"Replace pixels within all foreground objects with a privacy preserving blurred foreground",
"func":1
},
{
"ref":"vipy.image.ImageDetection.replace",
"url":49,
"doc":"Set all image values within the bounding box equal to the provided img, triggers load() and imclip()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.meanmask",
"url":49,
"doc":"Replace pixels within the foreground objects with the mean pixel color",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fghash",
"url":49,
"doc":"Perceptual differential hash function, computed for each foreground region independently",
"func":1
},
{
"ref":"vipy.image.ImageDetection.perceptualhash",
"url":49,
"doc":"Perceptual differentialhash function  bits [int]: longer hashes have lower TAR (true accept rate, some near dupes are missed), but lower FAR (false accept rate), shorter hashes have higher TAR (fewer near-dupes are missed) but higher FAR (more non-dupes are declared as dupes).  Algorithm: set foreground objects to mean color, convert to greyscale, resize with linear interpolation to small image based on desired bit encoding, compute vertical and horizontal gradient signs.  NOTE: Can be used for near duplicate detection of background scenes by unpacking the returned hex string to binary and computing hamming distance, or performing hamming based nearest neighbor indexing.  NOTE: The default packed hex output can be converted to binary as: np.unpackbits(bytearray().fromhex( bghash()  which is equivalent to bghash(asbinary=True)  objmask [bool]: if trye, replace the foreground object masks with the mean color prior to computing",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bghash",
"url":49,
"doc":"Percetual differential hash function, masking out foreground regions",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isduplicate",
"url":49,
"doc":"Background hash near duplicate detection, returns true if self and im are near duplicate images using bghash",
"func":1
},
{
"ref":"vipy.image.ImageDetection.show",
"url":49,
"doc":"Show scene detection  categories [list]: List of category (or shortlabel) names in the scene to show  fontsize (int, string): Size of the font, fontsize=int for points, fontsize='NN:scaled' to scale the font relative to the image size  figure (int): Figure number, show the image in the provided figure=int numbered window  nocaption (bool): Show or do not show the text caption in the upper left of the box  nocaption_withstring (list): Do not show captions for those detection categories (or shortlabels) containing any of the strings in the provided list  boxalpha (float, [0,1]): Set the text box background to be semi-transparent with an alpha  d_category2color (dict): Define a dictionary of required mapping of specific category() to box colors. Non-specified categories are assigned a random named color from vipy.show.colorlist()  caption_offset (int, int): The relative position of the caption to the upper right corner of the box.  nowindow (bool): Display or not display the image  textfacecolor (str): One of the named colors from vipy.show.colorlist() for the color of the textbox background  textfacealpha (float, [0,1]): The textbox background transparency  shortlabel (bool): Whether to show the shortlabel or the full category name in the caption  mutator (lambda): A lambda function with signature lambda im: f(im) which will modify this image prior to show. Useful for changing labels on the fly",
"func":1
},
{
"ref":"vipy.image.ImageDetection.annotate",
"url":49,
"doc":"Alias for savefig",
"func":1
},
{
"ref":"vipy.image.ImageDetection.savefig",
"url":49,
"doc":"Save show() output to given file or return buffer without popping up a window",
"func":1
},
{
"ref":"vipy.image.ImageDetection.category",
"url":49,
"doc":"Return or update the category",
"func":1
},
{
"ref":"vipy.image.ImageDetection.label",
"url":49,
"doc":"Alias for category",
"func":1
},
{
"ref":"vipy.image.ImageDetection.score",
"url":49,
"doc":"Real valued score for categorization, larger is better",
"func":1
},
{
"ref":"vipy.image.ImageDetection.probability",
"url":49,
"doc":"Real valued probability for categorization, [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageDetection.print",
"url":49,
"doc":"Print the representation of the image and return self - useful for debugging in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageDetection.tile",
"url":49,
"doc":"Generate a list of tiled image",
"func":1
},
{
"ref":"vipy.image.ImageDetection.untile",
"url":49,
"doc":"Undo tile",
"func":1
},
{
"ref":"vipy.image.ImageDetection.splat",
"url":49,
"doc":"Replace pixels within boundingbox in self with pixels in im",
"func":1
},
{
"ref":"vipy.image.ImageDetection.store",
"url":49,
"doc":"Store the current image file as an attribute of this object. Useful for archiving an object to be fully self contained without any external references. -Remove this stored image using unstore() -Unpack this stored image and set up the filename using restore() -This method is more efficient than load() followed by pkl(), as it stores the encoded image as a byte string. -Useful for creating a single self contained object for distributed processing. >>> v  v.store().restore(v.filename( ",
"func":1
},
{
"ref":"vipy.image.ImageDetection.unstore",
"url":49,
"doc":"Delete the currently stored image from store()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.restore",
"url":49,
"doc":"Save the currently stored image to filename, and set up filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.abspath",
"url":49,
"doc":"Change the path of the filename from a relative path to an absolute path (not relocatable)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.relpath",
"url":49,
"doc":"Replace the filename with a relative path to parent (or current working directory if none)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.canload",
"url":49,
"doc":"Return True if the image can be loaded successfully, useful for filtering bad links or corrupt images",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dict",
"url":49,
"doc":"Return a python dictionary containing the relevant serialized attributes suitable for JSON encoding",
"func":1
},
{
"ref":"vipy.image.ImageDetection.loader",
"url":49,
"doc":"Lambda function to load an unsupported image filename to a numpy array",
"func":1
},
{
"ref":"vipy.image.ImageDetection.load",
"url":49,
"doc":"Load image to cached private '_array' attribute and return Image object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.download",
"url":49,
"doc":"Download URL to filename provided by constructor, or to temp filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.reload",
"url":49,
"doc":"Flush the image buffer to force reloading from file or URL",
"func":1
},
{
"ref":"vipy.image.ImageDetection.channels",
"url":49,
"doc":"Return integer number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iscolor",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.ImageDetection.istransparent",
"url":49,
"doc":"Color images are three channel or four channel with transparency, float32 or uint8",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isgrey",
"url":49,
"doc":"Grey images are one channel, float32",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isluminance",
"url":49,
"doc":"Luninance images are one channel, uint8",
"func":1
},
{
"ref":"vipy.image.ImageDetection.filesize",
"url":49,
"doc":"Return size of underlying image file, requires fetching metadata from filesystem",
"func":1
},
{
"ref":"vipy.image.ImageDetection.shape",
"url":49,
"doc":"Return the (height, width) or equivalently (rows, cols) of the image",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centroid",
"url":49,
"doc":"Return the real valued center pixel coordinates of the image (col=x,row=y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centerpixel",
"url":49,
"doc":"Return the integer valued center pixel coordinates of the image (col=i,row=j)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.array",
"url":49,
"doc":"Replace self._array with provided numpy array",
"func":1
},
{
"ref":"vipy.image.ImageDetection.channel",
"url":49,
"doc":"Return a cloned Image() object for the kth channel, or return an iterator over channels if k=None",
"func":1
},
{
"ref":"vipy.image.ImageDetection.red",
"url":49,
"doc":"Return red channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.green",
"url":49,
"doc":"Return green channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blue",
"url":49,
"doc":"Return blue channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.alpha",
"url":49,
"doc":"Return alpha (transparency) channel as a cloned Image() object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fromarray",
"url":49,
"doc":"Alias for array(data, copy=True), set new array() with a numpy array copy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.tonumpy",
"url":49,
"doc":"Alias for numpy()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.numpy",
"url":49,
"doc":"Convert vipy.image.Image to numpy array, returns writeable array",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pil",
"url":49,
"doc":"Convert vipy.image.Image to PIL Image, by reference",
"func":1
},
{
"ref":"vipy.image.ImageDetection.torch",
"url":49,
"doc":"Convert the batch of 1 HxWxC images to a 1xCxHxW torch tensor, by reference",
"func":1
},
{
"ref":"vipy.image.ImageDetection.fromtorch",
"url":49,
"doc":"Convert a 1xCxHxW torch.FloatTensor to HxWxC np.float32 numpy array(), returns new Image() instance with selected colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.filename",
"url":49,
"doc":"Return or set image filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.url",
"url":49,
"doc":"Image URL and URL download properties",
"func":1
},
{
"ref":"vipy.image.ImageDetection.colorspace",
"url":49,
"doc":"Return or set the colorspace as ['rgb', 'rgba', 'bgr', 'bgra', 'hsv', 'float', 'grey', 'lum']",
"func":1
},
{
"ref":"vipy.image.ImageDetection.uri",
"url":49,
"doc":"Return the URI of the image object, either the URL or the filename, raise exception if neither defined",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setattribute",
"url":49,
"doc":"Set element self.attributes[key]=value",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setattributes",
"url":49,
"doc":"Set many attributes at once by providing a dictionary to be merged with current attributes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.clone",
"url":49,
"doc":"Create deep copy of object, flushing the original buffer if requested and returning the cloned object. Flushing is useful for distributed memory management to free the buffer from this object, and pass along a cloned object which can be used for encoding and will be garbage collected.  flushforward: copy the object, and set the cloned object array() to None. This flushes the video buffer for the clone, not the object  flushbackward: copy the object, and set the object array() to None. This flushes the video buffer for the object, not the clone.  flush: set the object array() to None and clone the object. This flushes the video buffer for both the clone and the object.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.flush",
"url":49,
"doc":"Alias for clone(flush=True), returns self not clone",
"func":1
},
{
"ref":"vipy.image.ImageDetection.resize_like",
"url":49,
"doc":"Resize image buffer to be the same size as the provided vipy.image.Image()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.zeropadlike",
"url":49,
"doc":"Zero pad the image balancing the border so that the resulting image size is (width, height)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.alphapad",
"url":49,
"doc":"Pad image using alpha transparency by adding padwidth on both left and right , or padwidth=(left,right) for different pre/postpadding and padheight on top and bottom or padheight=(top,bottom) for different pre/post padding",
"func":1
},
{
"ref":"vipy.image.ImageDetection.minsquare",
"url":49,
"doc":"Crop image of size (HxW) to (min(H,W), min(H,W , keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxsquare",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with zeropadding or (S,S) if provided, keeping upper left corner constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxmatte",
"url":49,
"doc":"Crop image of size (HxW) to (max(H,W), max(H,W with balanced zeropadding forming a letterbox with top/bottom matte or pillarbox with left/right matte",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imagebox",
"url":49,
"doc":"Return the bounding box for the image rectangle",
"func":1
},
{
"ref":"vipy.image.ImageDetection.border_mask",
"url":49,
"doc":"Return a binary uint8 image the same size as self, with a border of pad pixels in width or height around the edge",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rgb",
"url":49,
"doc":"Convert the image buffer to three channel RGB uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rgba",
"url":49,
"doc":"Convert the image buffer to four channel RGBA uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hsv",
"url":49,
"doc":"Convert the image buffer to three channel HSV uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bgr",
"url":49,
"doc":"Convert the image buffer to three channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bgra",
"url":49,
"doc":"Convert the image buffer to four channel BGR uint8 colorspace",
"func":1
},
{
"ref":"vipy.image.ImageDetection.float",
"url":49,
"doc":"Convert the image buffer to float32",
"func":1
},
{
"ref":"vipy.image.ImageDetection.greyscale",
"url":49,
"doc":"Convert the image buffer to single channel grayscale float32 in range [0,1]",
"func":1
},
{
"ref":"vipy.image.ImageDetection.grayscale",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.grey",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.gray",
"url":49,
"doc":"Alias for greyscale()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.luminance",
"url":49,
"doc":"Convert the image buffer to single channel uint8 in range [0,255] corresponding to the luminance component",
"func":1
},
{
"ref":"vipy.image.ImageDetection.lum",
"url":49,
"doc":"Alias for luminance()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.jet",
"url":49,
"doc":"Apply jet colormap to greyscale image and save as RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.rainbow",
"url":49,
"doc":"Apply rainbow colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hot",
"url":49,
"doc":"Apply hot colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bone",
"url":49,
"doc":"Apply bone colormap to greyscale image and convert to RGB",
"func":1
},
{
"ref":"vipy.image.ImageDetection.saturate",
"url":49,
"doc":"Saturate the image buffer to be clipped between [min,max], types of min/max are specified by _array type",
"func":1
},
{
"ref":"vipy.image.ImageDetection.intensity",
"url":49,
"doc":"Convert image to float32 with [min,max] to range [0,1], force colormap to be 'float'. Equivalent to self.mat2gray()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mat2gray",
"url":49,
"doc":"Convert the image buffer so that [min,max] -> [0,1], forces conversion to 'float' colorspace. This does not change the number of color channels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.gain",
"url":49,
"doc":"Elementwise multiply gain to image array, Gain should be broadcastable to array(). This forces the colospace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bias",
"url":49,
"doc":"Add a bias to the image array. Bias should be broadcastable to array(). This forces the colorspace to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mean",
"url":49,
"doc":"Mean over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.meanchannel",
"url":49,
"doc":"Mean per channel over all pixels",
"func":1
},
{
"ref":"vipy.image.ImageDetection.closeall",
"url":49,
"doc":"Close all open figure windows",
"func":1
},
{
"ref":"vipy.image.ImageDetection.close",
"url":49,
"doc":"Close the requested figure number, or close all of fignum=None",
"func":1
},
{
"ref":"vipy.image.ImageDetection.save",
"url":49,
"doc":"Save the current image to a new filename and return the image object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pkl",
"url":49,
"doc":"save the object to a pickle file and return the object, useful for intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pklif",
"url":49,
"doc":"Save the object to the provided pickle file only if b=True. Uuseful for conditional intermediate saving in long fluent chains",
"func":1
},
{
"ref":"vipy.image.ImageDetection.saveas",
"url":49,
"doc":"Save current buffer (not including drawing overlays) to new filename and return filename",
"func":1
},
{
"ref":"vipy.image.ImageDetection.saveastmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for savetmp()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.savetmp",
"url":49,
"doc":"Save current buffer to temp JPEG filename and return filename. Alias for saveastmp()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.base64",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page",
"func":1
},
{
"ref":"vipy.image.ImageDetection.html",
"url":49,
"doc":"Export a base64 encoding of the image suitable for embedding in an html page, enclosed in  tag Returns: -string:  containing base64 encoded JPEG and alt text with lazy loading",
"func":1
},
{
"ref":"vipy.image.ImageDetection.map",
"url":49,
"doc":"Apply lambda function to our numpy array img, such that newimg=f(img), then replace newimg -> self.array(). The output of this lambda function must be a numpy array and if the channels or dtype changes, the colorspace is set to 'float'",
"func":1
},
{
"ref":"vipy.image.ImageDetection.downcast",
"url":49,
"doc":"Cast the class to the base class (vipy.image.Image)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xmin",
"url":9,
"doc":"x coordinate of upper left corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ul",
"url":9,
"doc":"Upper left coordinate (x,y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ulx",
"url":9,
"doc":"Upper left coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.uly",
"url":9,
"doc":"Upper left coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ur",
"url":9,
"doc":"Upper right coordinate (x,y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.urx",
"url":9,
"doc":"Upper right coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ury",
"url":9,
"doc":"Upper right coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ll",
"url":9,
"doc":"Lower left coordinate (x,y), synonym for bl()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bl",
"url":9,
"doc":"Bottom left coordinate (x,y), synonym for ll()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.blx",
"url":9,
"doc":"Bottom left coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bly",
"url":9,
"doc":"Bottom left coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.lr",
"url":9,
"doc":"Lower right coordinate (x,y), synonym for br()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.br",
"url":9,
"doc":"Bottom right coordinate (x,y), synonym for lr()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.brx",
"url":9,
"doc":"Bottom right coordinate (x)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bry",
"url":9,
"doc":"Bottom right coordinate (y)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ymin",
"url":9,
"doc":"y coordinate of upper left corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xmax",
"url":9,
"doc":"x coordinate of lower right corner of box, x-axis is image column",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ymax",
"url":9,
"doc":"y coordinate of lower right corner of box, y-axis is image row",
"func":1
},
{
"ref":"vipy.image.ImageDetection.upperleft",
"url":9,
"doc":"Return the (x,y) upper left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bottomleft",
"url":9,
"doc":"Return the (x,y) lower left corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.upperright",
"url":9,
"doc":"Return the (x,y) upper right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bottomright",
"url":9,
"doc":"Return the (x,y) lower right corner coordinate of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.int",
"url":9,
"doc":"Convert corners to integer with rounding, in-place update",
"func":1
},
{
"ref":"vipy.image.ImageDetection.significant_digits",
"url":9,
"doc":"Convert corners to have at most n significant digits for efficient JSON storage",
"func":1
},
{
"ref":"vipy.image.ImageDetection.translate",
"url":9,
"doc":"Translate the bounding box by dx in x and dy in y",
"func":1
},
{
"ref":"vipy.image.ImageDetection.set_origin",
"url":9,
"doc":"Set the origin of the coordinates of this bounding box to be relative to the upper left of the other bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.offset",
"url":9,
"doc":"Alias for translate",
"func":1
},
{
"ref":"vipy.image.ImageDetection.invalid",
"url":9,
"doc":"Is the box a valid bounding box?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setwidth",
"url":9,
"doc":"Set new width keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.setheight",
"url":9,
"doc":"Set new height keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xcentroid",
"url":9,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centroid_x",
"url":9,
"doc":"Alias for x_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ycentroid",
"url":9,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.centroid_y",
"url":9,
"doc":"Alias for y_centroid()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.area",
"url":9,
"doc":"Return the area=width height of the bounding box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.to_xywh",
"url":9,
"doc":"Return bounding box corners as (x,y,width,height) tuple",
"func":1
},
{
"ref":"vipy.image.ImageDetection.xywh",
"url":9,
"doc":"Alias for to_xywh",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cxywh",
"url":9,
"doc":"Return or set bounding box corners as (centroidx,centroidy,width,height) tuple",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ulbr",
"url":9,
"doc":"Return bounding box corners as upper left, bottom right (xmin, ymin, xmax, ymax)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.to_ulbr",
"url":9,
"doc":"Alias for ulbr()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dx",
"url":9,
"doc":"Offset bounding box by same xmin as provided box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dy",
"url":9,
"doc":"Offset bounding box by ymin of provided box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.sqdist",
"url":9,
"doc":"Squared Euclidean distance between upper left corners of two bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dist",
"url":9,
"doc":"Distance between centroids of two bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.pdist",
"url":9,
"doc":"Normalized Gaussian distance in [0,1] between centroids of two bounding boxes, where 0 is far and 1 is same with sigma=maxdim() of this box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iou",
"url":9,
"doc":"area of intersection / area of union",
"func":1
},
{
"ref":"vipy.image.ImageDetection.intersection_over_union",
"url":9,
"doc":"Alias for iou",
"func":1
},
{
"ref":"vipy.image.ImageDetection.area_of_intersection",
"url":9,
"doc":"area of intersection",
"func":1
},
{
"ref":"vipy.image.ImageDetection.cover",
"url":9,
"doc":"Fraction of this bounding box intersected by other bbox (bb)",
"func":1
},
{
"ref":"vipy.image.ImageDetection.maxcover",
"url":9,
"doc":"The maximum cover of self to bb and bb to self",
"func":1
},
{
"ref":"vipy.image.ImageDetection.shapeiou",
"url":9,
"doc":"Shape IoU is the IoU with the upper left corners aligned. This measures the deformation of the two boxes by removing the effect of translation",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hasintersection",
"url":9,
"doc":"Return true if self and bb overlap by any amount, or by the cover threshold (if provided) or the iou threshold (if provided). This is a convenience function that allows for shared computation for fast non-maximum suppression.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.isinside",
"url":9,
"doc":"Is this boundingbox fully within the provided bounding box?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ispointinside",
"url":9,
"doc":"Is the 2D point p=(x,y) inside this boundingbox, or is the p=boundingbox() inside this bounding box?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilatepx",
"url":9,
"doc":"Dilate by a given pixel amount on all sides, keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilate_height",
"url":9,
"doc":"Change scale of bounding box in y direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.dilate_width",
"url":9,
"doc":"Change scale of bounding box in x direction keeping centroid constant",
"func":1
},
{
"ref":"vipy.image.ImageDetection.top",
"url":9,
"doc":"Make top of box taller (closer to top of image) by an offset dy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.bottom",
"url":9,
"doc":"Make bottom of box taller (closer to bottom of image) by an offset dy",
"func":1
},
{
"ref":"vipy.image.ImageDetection.left",
"url":9,
"doc":"Make left of box wider (closer to left side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.image.ImageDetection.right",
"url":9,
"doc":"Make right of box wider (closer to right side of image) by an offset dx",
"func":1
},
{
"ref":"vipy.image.ImageDetection.scalex",
"url":9,
"doc":"Multiply the box corners in the x dimension by a scale factor",
"func":1
},
{
"ref":"vipy.image.ImageDetection.scaley",
"url":9,
"doc":"Multiply the box corners in the y dimension by a scale factor",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imscale",
"url":9,
"doc":"Given a vipy.image object im, scale the box to be within [0,1], relative to height and width of image",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iseven",
"url":9,
"doc":"Are all corners even number integers?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.even",
"url":9,
"doc":"Force all corners to be even number integers. This is helpful for FFMPEG crop filters.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.hasoverlap",
"url":9,
"doc":"Does the bounding box intersect with the provided image rectangle?",
"func":1
},
{
"ref":"vipy.image.ImageDetection.iminterior",
"url":9,
"doc":"Transform bounding box to be interior to the image rectangle with shape (W,H). Transform is applyed by computing smallest (dx,dy) translation that it is interior to the image rectangle, then clip to the image rectangle if it is too big to fit",
"func":1
},
{
"ref":"vipy.image.ImageDetection.imclipshape",
"url":9,
"doc":"Clip bounding box to image rectangle [0,0,W-1,H-1], throw an exception on an invalid box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.convexhull",
"url":9,
"doc":"Given a set of points  x1,y1],[x2,xy], .], return the bounding rectangle, typecast to float",
"func":1
},
{
"ref":"vipy.image.ImageDetection.aspectratio",
"url":9,
"doc":"Return the aspect ratio (width/height) of the box",
"func":1
},
{
"ref":"vipy.image.ImageDetection.mindimension",
"url":9,
"doc":"Return min(width, height) typecast to float",
"func":1
},
{
"ref":"vipy.image.ImageDetection.ellipse",
"url":9,
"doc":"Convert the boundingbox to a vipy.geometry.Ellipse object",
"func":1
},
{
"ref":"vipy.image.ImageDetection.average",
"url":9,
"doc":"Compute the average bounding box between self and other, and set self to the average. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.averageshape",
"url":9,
"doc":"Compute the average bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.medianshape",
"url":9,
"doc":"Compute the median bounding box width and height between self and other. Other may be a singleton bounding box or a list of bounding boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.shapedist",
"url":9,
"doc":"L1 distance between (width,height) of two boxes",
"func":1
},
{
"ref":"vipy.image.ImageDetection.affine",
"url":9,
"doc":"Apply an 2x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.projective",
"url":9,
"doc":"Apply an 3x3 affine transformation to the box centroid. This operation preserves an axis aligned bounding box for an arbitrary affine transform.",
"func":1
},
{
"ref":"vipy.image.ImageDetection.confidence",
"url":9,
"doc":"Bounding boxes do not have confidences, use vipy.object.Detection()",
"func":1
},
{
"ref":"vipy.image.ImageDetection.grid",
"url":9,
"doc":"Split a bounding box into the smallest grid of non-overlapping bounding boxes such that the union is the original box",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackid",
"url":49,
"doc":"Mutate the image to show track ID with a fixed number of digits appended to the shortlabel as ( )",
"func":1
},
{
"ref":"vipy.image.mutator_show_jointlabel",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackindex",
"url":49,
"doc":"Mutate the image to show track index appended to the shortlabel as ( )",
"func":1
},
{
"ref":"vipy.image.mutator_show_userstring",
"url":49,
"doc":"Mutate the image to show user supplied strings in the shortlabel. The list be the same length oas the number of objects in the image. This is not checked. This is passed to show()",
"func":1
},
{
"ref":"vipy.image.mutator_show_noun_only",
"url":49,
"doc":"Mutate the image to show the noun only",
"func":1
},
{
"ref":"vipy.image.mutator_show_verb_only",
"url":49,
"doc":"Mutate the image to show the verb only",
"func":1
},
{
"ref":"vipy.image.mutator_show_noun_or_verb",
"url":49,
"doc":"Mutate the image to show the verb only if it is non-zero else noun",
"func":1
},
{
"ref":"vipy.image.mutator_capitalize",
"url":49,
"doc":"Mutate the image to show the shortlabel as 'Noun Verb1 Noun Verb2'",
"func":1
},
{
"ref":"vipy.image.mutator_show_activityonly",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackindex_activityonly",
"url":49,
"doc":"Mutate the image to show boxes colored by track index, and only show 'noun verb' captions",
"func":1
},
{
"ref":"vipy.image.mutator_show_trackindex_verbonly",
"url":49,
"doc":"Mutate the image to show boxes colored by track index, and only show 'verb' captions with activity confidence",
"func":1
},
{
"ref":"vipy.image.RandomImage",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.RandomImageDetection",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.RandomScene",
"url":49,
"doc":"",
"func":1
},
{
"ref":"vipy.image.owl",
"url":49,
"doc":"Return an owl image for testing",
"func":1
},
{
"ref":"vipy.image.vehicles",
"url":49,
"doc":"Return a highway scene with the four highest confidence vehicle detections for testing",
"func":1
},
{
"ref":"vipy.image.people",
"url":49,
"doc":"Return a crowd scene with the four highest confidence person detections for testing",
"func":1
},
{
"ref":"vipy.calibration",
"url":50,
"doc":""
},
{
"ref":"vipy.calibration.checkerboard",
"url":50,
"doc":"Create a 2D checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with black and white colors with black in upper left and bottom right, return np.array() float32 in [0,1]",
"func":1
},
{
"ref":"vipy.calibration.red_checkerboard_image",
"url":50,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with red colors, return Image()",
"func":1
},
{
"ref":"vipy.calibration.blue_checkerboard_image",
"url":50,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with red colors, return Image()",
"func":1
},
{
"ref":"vipy.calibration.color_checkerboard_image",
"url":50,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with random colors, return Image()",
"func":1
},
{
"ref":"vipy.calibration.color_checkerboard",
"url":50,
"doc":"Create a 2D color checkerboard pattern with squares of size (dx, dy) and image of size (dx ncols,dy nrows) with random colors\", return np.array",
"func":1
},
{
"ref":"vipy.calibration.testimage",
"url":50,
"doc":"Return an Image() object of a superb owl from wikipedia",
"func":1
},
{
"ref":"vipy.calibration.owl",
"url":50,
"doc":"Return an Image() object of a superb owl from wikipedia",
"func":1
},
{
"ref":"vipy.calibration.randomimage",
"url":50,
"doc":"",
"func":1
},
{
"ref":"vipy.calibration.testimg",
"url":50,
"doc":"Return a numpy array for a superb owl",
"func":1
},
{
"ref":"vipy.calibration.tile",
"url":50,
"doc":"Create a 2D tile pattern with texture T repeated (M,N) times",
"func":1
},
{
"ref":"vipy.calibration.greenblock",
"url":50,
"doc":"Return an (dx, dy, 3) float64 RGB channel image with green channel=1.0",
"func":1
},
{
"ref":"vipy.calibration.redblock",
"url":50,
"doc":"Return an (dx, dy, 3) float64 RGB channel image with red channel=1.0",
"func":1
},
{
"ref":"vipy.calibration.blueblock",
"url":50,
"doc":"Return an (dx, dy, 3) float64 RGB channel image with blue channel=1.0",
"func":1
},
{
"ref":"vipy.calibration.bayer",
"url":50,
"doc":"Return an (M,N) tiled texture pattern of [blue, green, blue, green; green red green red; blue green blue green, green red green red] such that each subblock element is (dx,dy) and the total repeated subblock size is (4 dx, 4 dy)",
"func":1
},
{
"ref":"vipy.calibration.bayer_image",
"url":50,
"doc":"Return bayer pattern as Image()",
"func":1
},
{
"ref":"vipy.calibration.dots",
"url":50,
"doc":"Create a sequence of dots (e.g. single pixels on black background) with strides (dx, dy) and image of size (dx ncols,dy nrows)",
"func":1
},
{
"ref":"vipy.calibration.vertical_gradient",
"url":50,
"doc":"Create a 2D linear ramp image",
"func":1
},
{
"ref":"vipy.calibration.centersquare",
"url":50,
"doc":"",
"func":1
},
{
"ref":"vipy.calibration.centersquare_image",
"url":50,
"doc":"",
"func":1
},
{
"ref":"vipy.calibration.circle",
"url":50,
"doc":"",
"func":1
},
{
"ref":"vipy.calibration.square",
"url":50,
"doc":"",
"func":1
}
]