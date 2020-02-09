import os
import numpy as np
from vipy.util import remkdir
import gzip
import struct
from array import array


TRAIN_IMG_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
TRAIN_IMG_SHA1 = '6c95f4b05d2bf285e1bfb0e7960c31bd3b3f8a7d'
TRAIN_LBL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
TRAIN_LBL_SHA1 = '2a80914081dc54586dbdf242f9805a6b8d2a15fc'
TEST_IMG_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
TEST_IMG_SHA1 = 'c3a25af1f52dad7f726cce8cacb138654b760d48'
TEST_LBL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
TEST_LBL_SHA1 = '763e7fa3757d93b0cdec073cef058b2004252c17'


class MNIST(object):
    def __init__(self, outdir):
        """download URLS above to outdir, then run export()"""
        self.outdir = remkdir(outdir)
        if not self._downloaded():
            print('[vipy.dataset.mnist]: downloading MNIST to "%s"' % self.outdir)
            self._wget()

    def _downloaded(self):
        gzip_downloaded = (os.path.exists(os.path.join(self.outdir, 'train-images-idx3-ubyte.gz'))
                           and os.path.exists(os.path.join(self.outdir, 'train-labels-idx1-ubyte.gz'))
                           and os.path.exists(os.path.join(self.outdir, 't10k-images-idx3-ubyte.gz'))
                           and os.path.exists(os.path.join(self.outdir, 't10k-labels-idx1-ubyte.gz')))
        unpacked_downloaded = (os.path.exists(os.path.join(self.outdir, 'train-images-idx3-ubyte'))
                               and os.path.exists(os.path.join(self.outdir, 'train-labels-idx1-ubyte'))
                               and os.path.exists(os.path.join(self.outdir, 't10k-images-idx3-ubyte'))
                               and os.path.exists(os.path.join(self.outdir, 't10k-labels-idx1-ubyte')))
        return (unpacked_downloaded or gzip_downloaded)

    def _wget(self):
        os.system('wget --directory-prefix=%s %s' % (self.outdir, TRAIN_IMG_URL))
        os.system('wget --directory-prefix=%s %s' % (self.outdir, TRAIN_LBL_URL))
        os.system('wget --directory-prefix=%s %s' % (self.outdir, TEST_IMG_URL))
        os.system('wget --directory-prefix=%s %s' % (self.outdir, TEST_LBL_URL))

    def _labels(self, gzfile):
        with gzip.open(gzfile, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got %d' % magic)
            labels = array("B", file.read())
        return labels

    def _imread(self, dataset, index):
        """Read MNIST encoded images, adapted from: https://github.com/sorki/python-mnist/blob/master/mnist/loader.py"""
        gzfile = None

        with gzip.open(gzfile, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got %d' % magic)
            file.seek(index * rows * cols + 16)
            image = np.asarray(array("B", file.read(rows * cols)).tolist())
            return np.reshape(image, (rows,cols))

    def trainset(self):
        y_train = self._labels(os.path.join(self.outdir, 'train-labels-idx1-ubyte.gz')).tolist()
        x_train = []
        train_img_file = os.path.join(self.outdir, 'train-images-idx3-ubyte.gz')
        with gzip.open(train_img_file, 'rb') as gzfile:
            magic, size, rows, cols = struct.unpack(">IIII", gzfile.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got %d' % magic)

            for k in range(60000):
                img = np.asarray(array("B", gzfile.read(rows * cols)).tolist()).reshape((rows, cols)).astype(np.uint8)
                x_train.append(img)

        return (y_train, np.array(x_train))

    def testset(self):
        y_test = self._labels(os.path.join(self.outdir, 't10k-labels-idx1-ubyte.gz')).tolist()
        x_test = []
        test_img_file = os.path.join(self.outdir, 't10k-images-idx3-ubyte.gz')
        with gzip.open(test_img_file, 'rb') as gzfile:
            magic, size, rows, cols = struct.unpack(">IIII", gzfile.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got %d' % magic)

            for k in range(10000):
                img = np.asarray(array("B", gzfile.read(rows * cols)).tolist()).reshape((rows, cols)).astype(np.uint8)
                x_test.append(img)

        return (y_test, np.array(x_test))
