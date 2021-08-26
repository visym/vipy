import os
import numpy as np
import vipy
from vipy.data.mnist import MNIST
import pickle

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class CIFAR10(MNIST):
    """vipy.data.cifar.CIFAR10 class

    >>> D = vipy.data.cifar.CIFAR10('/path/to/outdir')
    >>> (x,y) = D.trainset()
    >>> im = D[0].mindim(512).show()

    """
    
    def __init__(self, outdir):        
        self.outdir = vipy.util.remkdir(outdir)

        if not os.path.exists(os.path.join(outdir, 'cifar-10-batches-py', 'data_batch_1')):
            print('[vipy.data.cifar10]: downloading CIFAR-10 to "%s"' % self.outdir)
            vipy.downloader.download_and_unpack(URL, self.outdir)

        self._archives = [os.path.join(outdir, 'cifar-10-batches-py', 'data_batch_%d' % k) for k in range(1,6)]

        f = os.path.join(self.outdir, 'cifar-10-batches-py', 'batches.meta')
        assert os.path.exists(f)
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        self._classes = [x.decode("utf-8") for x in d[b'label_names']]

        self._trainset()
        self._testset()

    def __len__(self):
        return len(self._trainset) + len(self._testset)
        
    def __getitem__(self, k):
        n = len(self._trainset)
        return (vipy.image.ImageCategory(category=self._classes[self._trainlabels[k]], array=self._trainset[k], colorspace='rgb') if k < n else
                vipy.image.ImageCategory(category=self._classes[self._testlabels[k-n]], array=self._testset[k-n], colorspace='rgb'))
    
    def classes(self):
        return self._classes

    def trainset(self):
        return [vipy.image.ImageCategory(category=self._classes[y], array=x, colorspace='rgb') for (x,y) in zip(self._trainset, self._trainlabels)]
    
    def testset(self):
        return [vipy.image.ImageCategory(category=self._classes[y], array=x, colorspace='rgb') for (x,y) in zip(self._testset, self._testlabels)]        
    
    def _trainset(self):
        (data, labels) = ([], [])
        for f in self._archives:
            assert os.path.exists(f)
            with open(f, 'rb') as fo:
                d = pickle.load(fo, encoding='bytes')
                data.append(d[b'data'])
                labels.append(d[b'labels'])

        self._trainset = np.vstack(data)
        self._trainset = [np.transpose(x.reshape(3, 32, 32), axes=(1,2,0)) for x in self._trainset]
        self._trainlabels = [l for lbl in labels for l in lbl]

    def _testset(self):
        (data, labels) = ([], [])
        f = os.path.join(self.outdir, 'cifar-10-batches-py', 'test_batch')        
        assert os.path.exists(f)
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
            data.append(d[b'data'])
            labels.append(d[b'labels'])

        self._testset = np.vstack(data)
        self._testset = [np.transpose(x.reshape(3, 32, 32), axes=(1,2,0)) for x in self._testset]
        self._testlabels = [l for lbl in labels for l in lbl]
        
        return self
            
            


