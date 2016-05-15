'''
Created on May 13, 2016

@author: mgomrokchi
'''

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

class MyKmeans(object):
    '''
    classdocs
    '''
    def __init__( self, X, k=0, centres=None, nsample=0, **kwargs ):
        self.X = X
        if centres is None:
            self.centres, self.Xtocentre, self.distances = self.kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centres, self.Xtocentre, self.distances = self.kmeans(
                X, centres, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centres)):
            yield jc, (self.Xtocentre == jc)
        
#...............................................................................
    def kmeans(self, X, centres, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
        
        if not issparse(X):
            X = np.asanyarray(X)  # ?
        centres = centres.todense() if issparse(centres) \
            else centres.copy()
        N, dim = X.shape
        k, cdim = centres.shape
        if dim != cdim:
            raise ValueError( "kmeans: X %s and centres %s must have the same number of columns" % (
                X.shape, centres.shape ))
        if verbose:
            print("kmeans: X %s  centres %s  delta=%.2g  maxiter=%d  metric=%s",(
                X.shape, centres.shape, delta, maxiter, metric))
        allx = np.arange(N)
        prevdist = 0
        for jiter in range( 1, maxiter+1 ):
            D = self.cdist_sparse( X, centres, metric=metric, p=p )  # |X| x |centres|
            xtoc = D.argmin(axis=1)  # X -> nearest centre
            distances = D[allx,xtoc]
            avdist = distances.mean()  # median ?
            if verbose >= 2:
                print ("kmeans: av |X - nearest centre| = %.4g" % avdist)
            if (1 - delta) * prevdist <= avdist <= prevdist or jiter == maxiter:
                break
            prevdist = avdist
            for jc in range(k):  # (1 pass in C)
                c = np.where( xtoc == jc )[0]
                if len(c) > 0:
                    centres[jc] = X[c].mean( axis=0 )
        if verbose:
            print ("kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc))
        if verbose >= 2:
            r50 = np.zeros(k)
            r90 = np.zeros(k)
            for j in range(k):
                dist = distances[ xtoc == j ]
                if len(dist) > 0:
                    r50[j], r90[j] = np.percentile( dist, (50, 90) )
            print ("kmeans: cluster 50 % radius", r50.astype(int))
            print ("kmeans: cluster 90 % radius", r90.astype(int))
                # scale L1 / dim, L2 / sqrt(dim) ?
        return centres, xtoc, distances
    
    #...............................................................................
    def kmeanssample(self, X, k, nsample=0, **kwargs ):
        """ 2-pass kmeans, fast for large N:
            1) kmeans a random sample of nsample ~ sqrt(N) from X
            2) full kmeans, starting from those centres
        """
            # merge w kmeans ? mttiw
            # v large N: sample N^1/2, N^1/2 of that
            # seed like sklearn ?
        N, dim = X.shape
        if nsample == 0:
            nsample = max( 2*np.sqrt(N), 10*k )
        Xsample = random.sample(list(X), int(nsample))
        pass1centres = random.sample( list(X), int(k) )
        Xsample=np.reshape(Xsample,(len(Xsample),dim))
        pass1centres=np.reshape(pass1centres,(len(pass1centres),dim))
        samplecentres = self.kmeans( Xsample, pass1centres, **kwargs )[0]
        return self.kmeans( X, samplecentres, **kwargs )
    
    def cdist_sparse(self, X, Y, **kwargs ):
        """ -> |X| x |Y| cdist array, any cdist metric
            X or Y may be sparse -- best csr
        """
            # todense row at a time, v slow if both v sparse
        sxy = 2*issparse(X) + issparse(Y)
        if sxy == 0:
            return cdist( X, Y, **kwargs )
        d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
        if sxy == 2:
            for j, x in enumerate(X):
                d[j] = cdist( x.todense(), Y, **kwargs ) [0]
        elif sxy == 1:
            for k, y in enumerate(Y):
                d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
        else:
            for j, x in enumerate(X):
                for k, y in enumerate(Y):
                    d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
        return d
    
    def randomsample(self, X, n ):
        """ random.sample of the rows of X
            X may be sparse -- best csr
        """
        sampleix = random.sample(range( X.shape[0] ), int(n) )
        return X[sampleix]
    
    def nearestcentres(self, X, centres, metric="euclidean", p=2 ):
        """ each X -> nearest centre, any metric
                euclidean2 (~ withinss) is more sensitive to outliers,
                cityblock (manhattan, L1) less sensitive
        """
        D = cdist( X, centres, metric=metric, p=p )  # |X| x |centres|
        return D.argmin(axis=1)
    
    def Lqmetric(self, x, y=None, q=.5 ):
        # yes a metric, may increase weight of near matches; see ...
        return (np.abs(x - y) ** q) .mean() if y is not None \
            else (np.abs(x) ** q) .mean()
    
    def getAccuracy(self, testSet, predictions):
        correct=0
        for i in range(len(testSet)):
            for j in range(len(testSet)):
    
                if (testSet[i] == testSet[j]):
                    if predictions[i]==predictions[j]:
                        correct += 1
                else:
                    if predictions[i] != predictions[j]:
                        correct += 1
        Z= (len(testSet))**2
        return (correct/float(Z)) * 100.0
