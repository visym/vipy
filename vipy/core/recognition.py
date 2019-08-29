import os
from strpy.bobo.cache import Cache
import numpy as np
from strpy.bobo.matlab import randn

def _features(line=None):
    return np.array( np.reshape(randn(1,128), 128) )

def bagofwords(imtrain, imtest=None, features=_features, outdir=None):
    cache = Cache(cacheroot=outdir)

    # Unique labels
    labels = imtrain.map(lambda x: x.category()).distinct().collect()
    print labels

    # Features: each returns a row array of features 
    X = imtrain.map(features)  
    
    # Clustering: kmeans clustering to generate words
    # http://spark.apache.org/docs/0.9.0/mllib-guide.html
    from pyspark.mllib.clustering import KMeans
    model = KMeans.train(X, 2, maxIterations=10, runs=30, initializationMode='random')
    
    # construct bag of words representation
    print model.clusterCenters
    
    # One vs. rest linear svm
    #for lbl in labels:
    #    c = spark.mlib.svm()    
    
    # Return testing results
    #if imtest is not None:
    #    pass

    # Intermediate results are stored to cache



    

