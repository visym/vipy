from vipy.util import remkdir, filetail, videolist
import vipy.downloader
import vipy.video

URLS = ['http://www.nada.kth.se/cvap/actions/walking.zip',
        'http://www.nada.kth.se/cvap/actions/jogging.zip',
        'http://www.nada.kth.se/cvap/actions/running.zip',
        'http://www.nada.kth.se/cvap/actions/boxing.zip',
        'http://www.nada.kth.se/cvap/actions/handwaving.zip',
        'http://www.nada.kth.se/cvap/actions/handclapping.zip']
SHA1 = ['a3e81537271a0ab4576591774baa38c2d97b7e3a',
        '21943bdbcef9dad106db0d74661e49eaeaa15a25',
        'da83bb313edfd4455fffdda6263696fa10d43c6f',
        'adb36ed9c29c846d44d2ba15f348bd115c951bbd',
        '0792f7cd69f7f895a205c08ac212ddaa7177e370',
        'd3b81261aa822ef63d0d1523f945bfaff27814d2']
LABELS = ['walking','jogging','running','boxing','handwaving','handclapping']


class KTHActions(object):
    def __init__(self, datadir):
        """KTH ACtions dataset, provide a datadir='/path/to/store/kthactions' """
        self.datadir = remkdir(datadir)

    def __repr__(self):
        return str('<vipy.dataset.kthactions: %s>' % self.datadir)

    def split(self):
        trainPeople = ['person02','person03','person05','person06','person07','person08','person09','person10','person22']
        videos = self.dataset()
        trainset = [vid for vid in videos if filetail(vid.filename()).split('_')[0] in trainPeople]
        testset = [vid for vid in videos if filetail(vid.filename()).split('_')[0] not in trainPeople]
        return (trainset, testset)

    def download_and_unpack(self):
        print('[vipy.dataset.kthactions][WARNING]: downloads will not show percent progress since content length is unknown')
        for (url, label, sha1) in zip(URLS, LABELS, SHA1):
            vipy.downloader.download_and_unpack(url, self.datadir, sha1=sha1)

    def dataset(self):
        return [vipy.video.VideoCategory(filename=f, category=f.split('_')[1]) for f in videolist(self.datadir)]
