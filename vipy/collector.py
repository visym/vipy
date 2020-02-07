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
# boto S3 credentials
# S3 URL support
# dynamoDB access (internal)
# gateway API (external)
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-creating-buckets.html
# pep8
# test coverage
# dashboard
# box filtering
# stabilization
# video ffplay command
# dataset archival
# video transformations
# payments manager
# bulk email manager
# baseline training
# dataset prep
# consent montage with audio


# client = boto3.client( 
#   ...:     's3', 
#   ...:     # Hard coded strings as credentials, not recommended. 
#   ...:     aws_access_key_id=os.environ['VIPY_AWS_ACCESS_KEY_ID'], 
#   ...:     aws_secret_access_key=os.environ['VIPY_AWS_SECRET_ACCESS_KEY'])

 
