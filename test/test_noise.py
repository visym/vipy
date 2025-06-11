import vipy
import vipy.noise


def _test_noise(im=None):
    assert isinstance(im, vipy.image.Image) or im is None
    im = vipy.image.owl().centersquare().mindim(256) if im is None else im
    
    print('[test_noise]: %s' %  vipy.noise.Noise().montage(im).saveas('./test_noise.jpg'))
    print('[test_noise]: PASSED')

    
if __name__ == '__main__':
    _test_noise()
