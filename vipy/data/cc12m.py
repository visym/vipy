import os
import vipy
from vipy.util import tocache, readcsv
from vipy.dataset import Dataset


class CC12M(Dataset):
    """https://github.com/google-research-datasets/conceptual-12m"""
    def __init__(self, datadir=None):
        outdir = tocache('cc12m') if datadir is None else datadir
        csvfile = os.path.join(outdir, 'cc12m.tsv')
        assert os.path.exists(csvfile), "download from https://github.com/google-research-datasets/conceptual-12m"
        
        csv = readcsv(csvfile, separator='\t')
        loader = lambda r: vipy.image.ImageCategory(url=r[0], category=r[1])
        super().__init__(csv, loader=loader)
        
        
