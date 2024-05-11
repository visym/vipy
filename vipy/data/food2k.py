import os
import vipy


# Request from dataset organizers
URL = 'https://s3plus.meituan.net/v1/${KEY}/foodai-workshop-challenge/Food2k_complete.tar.gz'
INDEX = 'https://s3plus.meituan.net/v1/${KEY}/foodai-workshop-challenge/Food2k_label.tar.gz'
LABELS = 'http://123.57.42.89/Large-Scale_Food_Recognition_via_Deep_Progressive_Self-Transformer_Network/Supplementary%20tables.pdf'

class Food2k(vipy.dataset.Dataset):
    """Project: http://123.57.42.89/FoodProject.html"""
    def __init__(self, datadir=None, redownload=False):
        datadir = tocache('food2k') if datadir is None else datadir        

        # Download
        self._datadir = vipy.util.remkdir(datadir)        
        if redownload or not os.path.exists(os.path.join(self._datadir, '.complete')):
            vipy.downloader.download_and_unpack(URL, self._datadir)
            vipy.downloader.download_and_unpack(INDEX, self._datadir)

        d_idx_to_category = vipy.util.readjson(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'food2k.json'))  # scraped from PDF
        d_filename_to_idx = {vipy.util.filetail(f):str(v) for (f,v) in vipy.util.readcsv(os.path.join(self._datadir, 'Food2k_label', 'train_finetune.txt'), separator=' ')}
        loader = lambda f: vipy.image.ImageCategory(filename=f, category=d_idx_to_category[d_filename_to_idx[vipy.util.filetail(f)]])
        imlist = [f for f in vipy.util.findimages(os.path.join(datadir)) if vipy.util.filetail(f) in d_filename_to_idx]
        super().__init__(imlist, id='food2k', loader=loader)

        open(os.path.join(self._datadir, '.complete'), 'a').close()





