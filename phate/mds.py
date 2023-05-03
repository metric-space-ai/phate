# oiginal author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from sklearn import manifold
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import scipy.spatial
import numpy as np


def cmdscale_fast(D, ndim):
    return classic(D=D, n_components=ndim)

def classic(D, n_components=2, random_state=None):
    D = D**2
    D = D - D.mean(axis=0)[None, :]
    D = D - D.mean(axis=1)[:, None]
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    Y = pca.fit_transform(D)
    return Y

def sgd(D, n_components=2, random_state=None, init=None):
    mds = manifold.MDS(n_components=n_components, metric=True, dissimilarity='precomputed', 
                       random_state=random_state, n_init=1, max_iter=3000, eps=1e-6)
    Y = mds.fit_transform(D)
    return Y

def smacof(D, n_components=2, metric=True, init=None, random_state=None, verbose=0, max_iter=3000, eps=1e-6, n_jobs=1):
    Y, _ = manifold.smacof(D, n_components=n_components, metric=metric, max_iter=max_iter, 
                           eps=eps, random_state=random_state, n_jobs=n_jobs, n_init=1, init=init, verbose=verbose)
    return Y

def embed_MDS(X, ndim=2, how="metric", distance_metric="euclidean", solver="sgd", n_jobs=1, seed=None, verbose=0):
    X_dist = squareform(pdist(X, distance_metric))
    Y_classic = classic(X_dist, n_components=ndim, random_state=seed)
    if how == "classic":
        return Y_classic

    if solver == "sgd":
        Y = sgd(X_dist, n_components=ndim, random_state=seed, init=Y_classic)
    elif solver == "smacof":
        Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y_classic, metric=True)
    else:
        raise RuntimeError

    if how == "metric":
        _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
        return Y

    Y = smacof(X_dist, n_components=ndim, random_state=seed, init=Y, metric=False)
    _, Y, _ = scipy.spatial.procrustes(Y_classic, Y)
    return Y