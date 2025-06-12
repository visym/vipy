import os
import vipy


URLS = ['https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip',
        'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip',
        'http://visualgenome.org/static/data/dataset/image_data.json.zip',
        'http://visualgenome.org/static/data/dataset/objects.json.zip',
        'http://visualgenome.org/static/data/dataset/relationships.json.zip']


class VisualGenome(vipy.dataset.Dataset):
    """Project: http://visualgenome.org/, version 1.4-objects"""
    def __init__(self, datadir=None, name='visualgenome', redownload=False):
        datadir = vipy.util.tocache('visualgenome') if datadir is None else datadir
        
        self._datadir = vipy.util.remkdir(datadir)
        for url in URLS:
            if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
                vipy.downloader.download_and_unpack(url, self._datadir)

        d_imageid_to_filename = {x['image_id']:os.path.join(self._datadir, vipy.util.filetail(vipy.util.filepath(x['url'])), vipy.util.filetail(x['url']))
                                 for x in vipy.util.readjson(os.path.join(self._datadir, 'image_data.json'))}

        imlist = []
        for obj in vipy.util.readjson(os.path.join(self._datadir, 'objects.json')):
            imlist.append( (d_imageid_to_filename[obj['image_id']], obj['objects']) )

        loader = lambda x: vipy.image.Scene(filename=x[0],
                                            objects=[vipy.object.Detection(tags=o['names'],
                                                                           xmin = o['x'], ymin=o['y'], width=o['w'], height=o['h'],
                                                                           id = o['object_id'],
                                                                           attributes={'synsets':o['synsets']}) for o in x[1]])
        super().__init__(imlist, id=name, loader=loader)
        

        open(os.path.join(self._datadir, '.complete'), 'a').close()
