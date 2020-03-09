import numpy as np

def featureNormalize(X):
    """
    %FEATURENORMALIZE Normalizes the features in X.
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when   
    working with learning algorithms.

    """
    X_norm = X
    mu = np.zeros((1,np.shape(X)[1]))
    sigma = np.zeros((1,np.shape(X)[1]))
    mu[0,0] = np.mean(X[:,0])
    mu[0,1] = np.mean(X[:,1])
    sigma[0,0] = np.std(X[:,0])
    sigma[0,1] = np.std(X[:,1])
    
    for i in np.arange(np.shape(X)[0]):
        X_norm[i,0] = (X[i,0] - mu[0,0])/sigma[0,0]
        X_norm[i,1] = (X[i,1] - mu[0,1])/sigma[0,1]
        
    return X_norm, mu, sigma