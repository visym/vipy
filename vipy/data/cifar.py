import os
import numpy as np
import vipy
import pickle


CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


class CIFAR10():
    """vipy.data.cifar.CIFAR10 class

    >>> D = vipy.data.cifar.CIFAR10('/path/to/outdir')
    >>> d = D.trainset()
    >>> im = d[0].mindim(512).show()

    """
    
    def __init__(self, outdir, name='cifar10', url=CIFAR10_URL, md5=CIFAR10_MD5):        
        self._datadir = vipy.util.remkdir(outdir)

        self._subdir = 'cifar-10-batches-py'
        if not os.path.exists(os.path.join(outdir, self._subdir, 'data_batch_1')):
            print('[vipy.data.cifar10]: downloading CIFAR-10 to "%s"' % self._datadir)
            vipy.downloader.download_and_unpack(url, self._datadir, md5=md5)

        self._train_archives = [os.path.join(outdir, self._subdir, 'data_batch_%d' % k) for k in range(1,6)]
        self._test_archives = [os.path.join(self._datadir, self._subdir, 'test_batch')]

        f = os.path.join(self._datadir, self._subdir, 'batches.meta')
        assert os.path.exists(f)
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        self._classes = [x.decode("utf-8") for x in d[b'label_names']]

        self._trainset()
        self._testset()

        self._name = name

    def __repr__(self):
        return '<vipy.data.%s: %s>' % (self._name, self._datadir)
    
    def classes(self):
        return self._classes

    def trainset(self):
        return vipy.dataset.Dataset([vipy.image.ImageCategory(category=self._classes[y], array=x, colorspace='rgb') for (x,y) in zip(self._trainset, self._trainlabels)], '%s_train' % self._name)
    
    def testset(self):
        return vipy.dataset.Dataset([vipy.image.ImageCategory(category=self._classes[y], array=x, colorspace='rgb') for (x,y) in zip(self._testset, self._testlabels)], '%s_test' % self._name)
    
    def _trainset(self, labelkey=b'labels'):
        (data, labels) = ([], [])
        for f in self._train_archives:
            assert os.path.exists(f)
            with open(f, 'rb') as fo:
                d = pickle.load(fo, encoding='bytes')
                data.append(d[b'data'])
                labels.append(d[labelkey])

        self._trainset = np.vstack(data)
        self._trainset = [np.transpose(x.reshape(3, 32, 32), axes=(1,2,0)) for x in self._trainset]
        self._trainlabels = [l for lbl in labels for l in lbl]
        return self
        
    def _testset(self, labelkey=b'labels'):
        (data, labels) = ([], [])
        for f in self._test_archives:
            assert os.path.exists(f)
            with open(f, 'rb') as fo:
                d = pickle.load(fo, encoding='bytes')
                data.append(d[b'data'])
                labels.append(d[labelkey])

        self._testset = np.vstack(data)
        self._testset = [np.transpose(x.reshape(3, 32, 32), axes=(1,2,0)) for x in self._testset]
        self._testlabels = [l for lbl in labels for l in lbl]        
        return self
            
            
class CIFAR100(CIFAR10):
    def __init__(self, datadir, name='cifar100', url=CIFAR100_URL, md5=CIFAR100_MD5):        

        self._name = name
        self._datadir = vipy.util.remkdir(datadir)
        self._subdir = 'cifar-100-python'
        if not os.path.exists(os.path.join(datadir, self._subdir, 'train')):
            print('[vipy.data.cifar10]: downloading CIFAR-100 to "%s"' % self._datadir)
            vipy.downloader.download_and_unpack(url, self._datadir, md5=md5)

        self._train_archives = [os.path.join(datadir, self._subdir, 'train')]
        self._test_archives = [os.path.join(datadir, self._subdir, 'test')]        

        f = os.path.join(self._datadir, self._subdir, 'meta')
        assert os.path.exists(f)
        with open(f, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        self._classes = [x.decode("utf-8") for x in d[b'fine_label_names']]
        self._coarse_classes = [x.decode("utf-8") for x in d[b'coarse_label_names']]        

        self._trainset()
        self._testset()
        
    def _trainset(self):
        return super()._trainset(labelkey=b'fine_labels')

    def _testset(self):
        return super()._testset(labelkey=b'fine_labels')
    
        
    
