import math
import numpy
import numpy as np
import operator
from collections import Counter
from statistics import mode



def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk

def dist(instance1,instance2,Matrix):
    q=numpy.subtract(instance1[0:7],instance2)
    X1=numpy.dot(q,Matrix)
    X2=numpy.dot(X1,q)
    return math.sqrt(X2)




def neigbors(X_train,instance1,k,M):
    Distance=[]
    for i in range(X_train.__len__()):
        distance=dist(X_train[i],instance1,M)
        Distance.append((X_train[i][7],distance))
    neighbors=[]
    Distance.sort(key=operator.itemgetter(1))
    for i in range(k):
        neighbors.append(Distance[i][0])
    data = Counter(neighbors)
    return data.most_common()[0][0]

def fit(X_train,Y_train):
    W=[]
    for i in range(X_train.__len__()):
        b=np.append(X_train[i],Y_train[i])
        W.append(b)
    return W

def prediction(X_train,X_test,k,M):
    Z=[]
    for i in range(X_test.__len__()):
        Z.append(neigbors(X_train,X_test[i],k,M))

    return Z
