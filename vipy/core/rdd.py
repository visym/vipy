import time
import numpy.random


def random_vector(sparkContext, n=1024, m=1024):
    numpy.random.seed(int(time.time()))    
    return sparkContext.parallelize([numpy.random.rand(m) for x in range(n)])

def random_binary_vector(sparkContext, n=1024, m=1024):
    numpy.random.seed(int(time.time()))    
    return sparkContext.parallelize([numpy.random.rand(m) > 0.5 for x in range(n)])


