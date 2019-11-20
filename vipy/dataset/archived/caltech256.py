import os
from bobo.viset import caltech101
from bobo.cache import Cache

caltech101.URL = ('http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar')
caltech101.SHA1 = '2195e9a478cf78bd23a1fe51f4dabe1c33744a1c'
caltech101.VISET = 'caltech256'
caltech101.SUBDIR = os.path.join('caltech256', '256_ObjectCategories')
caltech101.cache = Cache()

def stream():
    return caltech101.stream()

def export():
    return caltech101.export()    

    


