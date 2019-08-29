import numpy as np


def linearsvm(X, Y, regType='l2', miniBatchFraction=1.0, iterations=100, regParam=1E-7):
    """Wrapper for spark linear svm: http://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-support-vector-machine-svm"""
    """X is an RDD of features, y is an RDD of binary labels"""

    from pyspark.mllib.classification import SVMWithSGD
    from pyspark.mllib.regression import LabeledPoint
    
    # Parse dataset into spark required "LabeledPoint" format
    #trainset = imageset.zip(features).map(lambda (im,x): LabeledPoint(float(im.iscategory(positiveclass)), np.array(x).astype(float).flatten().tolist()))
    trainset = X.zip(Y).map(lambda (x,y): LabeledPoint(float(y), np.array(x).astype(float).flatten().tolist()))        

    # Binary one-vs-rest classifier with regularization
    model = SVMWithSGD.train(trainset, intercept=True, regType=regType, miniBatchFraction=miniBatchFraction, iterations=iterations, regParam=regParam)

    # Model 
    return model

    
