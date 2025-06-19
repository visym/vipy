import os
import vipy


class CAP_classification_clip(vipy.dataset.Dataset):
    """https://visym.github.io/cap/"""

    URL = 'https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_clip.tar.gz'
    MD5 = '54315e2ce204f0dbbe298490a63b5b3b'
    def __init__(self, datadir):
        if not os.path.exists(os.path.join(datadir, 'cap_classification_clip', 'README.md')):
            vipy.downloader.download_and_unpack(CAP_classification_clip.URL, datadir)

        annotations = os.path.join(datadir, 'cap_classification_clip', 'annotations')
        super().__init__(vipy.dataset.Dataset.from_directory(annotations, filetype='json'), id='cap')


class CAP_classification_pad(vipy.dataset.Dataset):
    """https://visym.github.io/cap/"""    
    URL = 'https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_pad.tar.gz'
    MD5 = 'fbdc75e6ef10b874ddda20ee9765a710'
    def __init__(self, datadir):
        if not os.path.exists(os.path.join(datadir, 'cap_classification_pad', 'README.md')):
            vipy.downloader.download_and_unpack(CAP_classification_pad.URL, datadir)

        annotations = os.path.join(datadir, 'cap_classification_pad', 'annotations')
        super().__init__(vipy.dataset.Dataset.from_directory(annotations, filetype='json'), id='cap')

        
class CAP_detection(vipy.dataset.Dataset):
    """https://visym.github.io/cap/"""    
    URL = 'https://dl.dropboxusercontent.com/s/db11zv0gcycu384/cap_detection_handheld_val.tar.gz.72f58e69582c17dd366d3c7e85cf0da8'
    MD5 = '72f58e69582c17dd366d3c7e85cf0da8'
    def __init__(self, datadir):
        if not os.path.exists(os.path.join(datadir, 'cap_detection_handheld_val', 'README.md')):
            vipy.downloader.download(CAP_detection.URL, os.path.join(datadir, 'cap_detection_handheld_val.tar.gz'), md5=CAP_detection.MD5)
            vipy.downloader.unpack(os.path.join(datadir, 'cap_detection_handheld_val.tar.gz'), datadir)

        annotations = os.path.join(datadir, 'cap_detection_handheld_val', 'annotations')
        super().__init__(vipy.dataset.Dataset.from_directory(annotations, filetype='json'), id='cap_detection:val')
    
