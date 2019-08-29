import numpy as np
import bobo.app
from bobo.util import load, save, csvlist, dirlist, imlist, readcsv, filebase, imwrite, remkdir, temppng, writecsv, filepath, fileext, filetail, islist
from bobo.image import ImageDetection, ImageCategory
import os
import copy


class ImageNet(object):
    def __init__(self, datadir='/proj/janus3/ILSVRC2015'):
        self.datadir = datadir 
        
    def __repr__(self):
        return str('<viset.imagenet: %s>' % self.datadir)

    def _parse_loc(self, imageset='train'):
        """ImageNet localization, imageset = {train, val}"""
        import xmltodict
        if imageset == 'train':
            imagesetfile = 'train_loc.txt'
        elif imageset == 'val':
            imagesetfile = 'val.txt'
        else:
            raise ValueError('unsupported imageset')

        csv = readcsv(os.path.join(self.datadir, 'ImageSets', 'CLS-LOC', imagesetfile), separator=' ')
        for (filepath, k) in csv:
            xmlfile = '%s.xml' % os.path.join(self.datadir, 'Annotations', 'CLS-LOC', imageset, filepath)
            d = xmltodict.parse(open(xmlfile, 'r').read())
            imfile = '%s.JPEG' % os.path.join(self.datadir, 'Data', 'CLS-LOC', imageset, filepath)
            objlist = d['annotation']['object'] if islist(d['annotation']['object']) else [d['annotation']['object']]
            for obj in objlist:
                yield ImageDetection(filename=imfile, category=obj['name'],
                                        xmin=int(obj['bndbox']['xmin']), ymin=int(obj['bndbox']['ymin']),
                                        xmax=int(obj['bndbox']['xmax']), ymax=int(obj['bndbox']['ymax']))

    def classes(self):
        return list(set([im.category() for im in self._parse_cls('val')]))

    def _parse_cls(self, imageset='train'):
        """ImageNet Classification, imageset = {train, val}"""
        import xmltodict            
        if imageset == 'train':
            imagesetfile = 'train_cls.txt'
        elif imageset == 'val':
            imagesetfile = 'val.txt'
        else:
            raise ValueError('unsupported imageset')
        csv = readcsv(os.path.join(self.datadir, 'ImageSets', 'CLS-LOC', imagesetfile), separator=' ')
        for (subpath, k) in csv:
            xmlfile = '%s.xml' % os.path.join(self.datadir, 'Annotations', 'CLS-LOC', imageset, subpath)
            imfile = '%s.JPEG' % os.path.join(self.datadir, 'Data', 'CLS-LOC', imageset, subpath)            
            if os.path.exists(xmlfile):
                d = xmltodict.parse(open(xmlfile, 'r').read())
                objlist = d['annotation']['object'] if islist(d['annotation']['object']) else [d['annotation']['object']]                
                yield ImageCategory(filename=imfile, category=objlist[0]['name'])
            else:
                yield ImageCategory(filename=imfile, category=filepath(subpath))
                

    
