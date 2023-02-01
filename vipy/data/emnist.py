import os
import vipy
import vipy.data.mnist
import string

URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'


class EMNIST(vipy.data.mnist.MNIST):
    def __init__(self, datadir):
        self._datadir = vipy.util.remkdir(datadir)        
        if not os.path.exists(os.path.join(self._datadir, vipy.util.filetail(URL))):
            vipy.downloader.download_and_unpack(URL, self._datadir)        
        super().__init__(datadir)

    def _downloaded(self):
        return True

    def _wget(self):
        return self

    def letters_train(self):
        (imgfile, labelfile) = (os.path.join(self.outdir, 'gzip/emnist-letters-train-images-idx3-ubyte.gz'), os.path.join(self.outdir, 'gzip/emnist-letters-train-labels-idx1-ubyte.gz'))
        d_categoryidx_to_category = {str(k):x for (k,x) in enumerate(string.ascii_lowercase, start=1)}
        return vipy.dataset.Dataset([vipy.image.ImageCategory(array=img, category=d_categoryidx_to_category[str(y)], colorspace='lum') for (y,img) in zip(*self._dataset(imgfile, labelfile, 124800))], 'emnist_letters_train')

    def letters_test(self):
        (imgfile, labelfile) = (os.path.join(self.outdir, 'gzip/emnist-letters-test-images-idx3-ubyte.gz'), os.path.join(self.outdir, 'gzip/emnist-letters-test-labels-idx1-ubyte.gz'))
        d_categoryidx_to_category = {str(k):x for (k,x) in enumerate(string.ascii_lowercase, start=1)}        
        return vipy.dataset.Dataset([vipy.image.ImageCategory(array=img, category=d_categoryidx_to_category[str(y)], colorspace='lum') for (y,img) in zip(*self._dataset(imgfile, labelfile, 145600-124800))], 'emnist_letters_test')

    def letters(self):
        return (self.letters_train(), self.letters_test())

    def digits_train(self):
        (imgfile, labelfile) = (os.path.join(self.outdir, 'gzip/emnist-digits-train-images-idx3-ubyte.gz'), os.path.join(self.outdir, 'gzip/emnist-digits-train-labels-idx1-ubyte.gz'))
        return vipy.dataset.Dataset([vipy.image.ImageCategory(array=img, category=str(y), colorspace='lum') for (y,img) in zip(*self._dataset(imgfile, labelfile, 240000))], 'emnist_digits_train')

    def digits_test(self):
        (imgfile, labelfile) = (os.path.join(self.outdir, 'gzip/emnist-digits-test-images-idx3-ubyte.gz'), os.path.join(self.outdir, 'gzip/emnist-digits-test-labels-idx1-ubyte.gz'))
        return vipy.dataset.Dataset([vipy.image.ImageCategory(array=img, category=str(y), colorspace='lum') for (y,img) in zip(*self._dataset(imgfile, labelfile, 280000-240000))], 'emnist_digits_test')

    def digits(self):
        return (self.digits_train(), self.digits_test())
    
    def trainset(self):
        return self.letters_train().union(self.digits_train()).id('emnist_train')

    def testset(self):
        return self.letters_test().union(self.digits_test()).id('emnist_test')
    

