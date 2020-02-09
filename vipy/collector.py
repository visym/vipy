import vipy.video
from vipy.util import isvideofile, isjsonfile, readjson
from vipy.downloader import download


def dataset():
    pass


def montage():
    pass


def consent(videofile):
    pass


def instance(videofile, jsonfile):
    assert isjsonfile(jsonfile) and isvideofile(videofile), "Invalid input for importer"
    json = readjson(download(jsonurl))

    pass


def export(vid):
    assert isinstance(vid, vipy.video.Scene), "Invalid input for exporter"


def dashboard():
    pass


# im.dict() export
# ffmpeg clip fix on youtube videos
# collector JSON and video URL import
# S3 URL support
# dynamoDB access (internal)
# gateway API (external)
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-creating-buckets.html
# pep8
# test coverage
# dashboard
# box filtering
# stabilization (client)
# video ffplay command
# dataset archival
# video transformations
# payments manager
# bulk email manager
# baseline training
# dataset prep
# consent montage with audio
# parallel batch
# vipy.video.Frames
# heatmaps: https://eatsleepdata.com/how-to-generate-a-geographical-heatmap-with-python/
# object detectors
# demographics statistics
# upload rates, collection rates per activity
# consent@visym.com
# consensus verification management
# consensus question management
# sorting by worst and best rated video
# new project configuration
# collector leaderboards
# local VIA integration for refinement, vs. matplotlib callbacks in existing GUI?
#    since we do not need to create the rectangles, only refine them maybe the matplotlib GUI is good enough?
#    but there may be a lot of rectangles.  how can I quickly fix it?


# client = boto3.client(
#   ...:     's3',
#   ...:     # Hard coded strings as credentials, not recommended.
#   ...:     aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'],
#   ...:     aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'])
