import vipy.video
from vipy.util import isvideofile, isjsonfile, readjson
from vipy.downloader import download

def dataset():
    pass

def montage():
    
def consent(videofile):
    pass
    
def collection(videofile, jsonfile):
    assert isjsonfile(jsonfile) and isvideofile(videofile), "Invalid input for importer"
    json = readjson(download(jsonurl))
    
    pass

def export(vid):
    assert isinstance(vid, vipy.video.Scene), "Invalid input for exporter"


def dashboard():
    pass
