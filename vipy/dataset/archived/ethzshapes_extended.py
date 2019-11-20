from bobo.cache import Cache
from bobo.viset import ethzshapes

ethzshapes.URL = 'http://www.vision.ee.ethz.ch/datasets_extra/extended_ethz_shapes.tgz'
ethzshapes.SHA1 = None
ethzshapes.SUBDIR = 'extended_ethz_shapes'
ethzshapes.LABELS = ['apple','bottle','giraffe','hat','mug','starfish','swan']
ethzshapes.VISET = 'ethzshapes_extended'

cache = Cache()

def stream():
    return ethzshapes.stream()

def export():
    return ethzshapes.export()

