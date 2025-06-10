import vipy
import vipy.noise


def test_noise(im=None):
    assert isinstance(im, vipy.image.Image) or im is None

    im = vipy.image.owl().centersquare().mindim(256) if im is None else im
    noise = vipy.noise.Noise()
    
    print(vipy.visualize.montage([noise.transform(im, k) for k in noise.distortions()], 256,256))

    print('[test_noise]: PASSED')

    
if __name__ == '__main__':
    test_noise()
