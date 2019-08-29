def kmeans(data, clusters=16, maxIterations=20, runs=30, initialization_mode="random", seed=None):
    """Wrapper for Spark MLIB clustering: http://spark.apache.org/docs/latest/mllib-clustering.html"""
    
    # K-Means clustering!
    # assume that the data is an RDD of numpy 1D float arrays defining observations
    from pyspark.mllib.clustering import KMeans
    clusters = KMeans.train(data, clusters, maxIterations, runs, initializationMode=initialization_mode, seed=seed)

    # Export cluster centers as list of numpy arrays
    return clusters.centers
