'''
Created on Apr 16, 2016

@author: mgomrokchi
'''
import operator
import scipy
import KNNs
import numpy
from dataParser import Paeser
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as pyplot
from sklearn.metrics import pairwise_distances
# from metric_learn import ITML, LMNN, LSML, SDML
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import matplotlib.pyplot as plt
from Lmmn import *
from sklearn.neighbors import DistanceMetric
class Learner(object):
    '''
    classdocs
    '''


    def __init__(self, reigCoef,similarSet,dis_similarSet,featureDim):
        

        self.beta=reigCoef
        self.S=similarSet
        self.D=dis_similarSet
        self.A=self.compute_A()
        self.B=self.compute_B()
        self.featureDim=featureDim
        #self.Metric=0
    
    def compute_A_i(self,z_i):
        return np.mat(z_i).T*np.mat(z_i)
    
    def compute_B_j(self,z_j):
        return np.mat(z_j).T*np.mat(z_j)
    
    def compute_A(self):
        A=[]
        for i in range(len(self.S)):
            A.append(self.compute_A_i(self.S[i]))
        return A
    
    def compute_B(self):
        B=[]
        for i in range(len(self.D)):
            B.append(self.compute_B_j((self.D[i])))
        return B
    
    def I_v(self):
        return numpy.ones((len(self.D),1))
    
    def compute_A_star(self):
        A_star=[]
        for i in range(len(self.D)):
            
            temp=[]
            for j in range(len(self.D)):
                temp.append(np.mat(self.D[i])*np.mat(self.B[j])*np.mat(self.D[i]).T)
            A_star.append(np.ravel(temp))
        return A_star
    
    def compute_B_star(self):
        B_star=[]
        for j in range(len(self.D)):
            temp=numpy.zeros((self.featureDim,self.featureDim))
            for i in range(len(self.S)):
                temp+=np.mat(self.A[i])*np.mat(self.B[j])
            B_star.append(numpy.trace(temp))     
        return B_star
    
    def compute_a_star(self): 
        a_star=[]
        temp=0
        for j in range(len(self.D)):
            for i in range(len(self.S)):
                temp+=np.mat(self.S[i])*np.mat(self.B[j])*np.mat(self.S[i]).T
            a_star.append(temp)
        return a_star
        
    def compute_w_star(self):
        w_star=numpy.zeros((len(self.D),len(self.D)))
        for j in range(len(self.D)):
            for k in range(len(self.D)):
                w_star[j][k]= numpy.trace(np.mat(self.B[j])*np.mat(self.B[k]))
        return w_star
        
    def compute_M(self,lambdaVec):
        M=numpy.zeros((self.featureDim,self.featureDim))
        temp1= numpy.zeros((self.featureDim,self.featureDim))
        temp2= numpy.zeros((self.featureDim,self.featureDim))
        for i in range(len(self.S)):
            temp1+=self.A[i]
        for j in range(len(self.D)):
            temp2+=lambdaVec[j]*np.mat(self.B[j])
        M=(-1/self.beta)*(np.mat(temp1)-np.mat(temp2))
        #self.Metric=M
        return M
        
    def computeLambda(self,disSimilarDataSize):
        a_star=self.compute_a_star()
        a_star=np.reshape(a_star, (disSimilarDataSize,1))
        A_star=self.compute_A_star()
        A_star=np.reshape(A_star, (disSimilarDataSize,disSimilarDataSize))
        w_star=self.compute_w_star()
        w_star=np.reshape(w_star, (disSimilarDataSize,disSimilarDataSize))
        B_star=self.compute_B_star()
        B_star=np.reshape(B_star, (disSimilarDataSize,1))
        part1=(1/self.beta)*(np.mat(A_star)+np.mat(A_star).T)
        part2=(1/(2*self.beta))*(np.mat(w_star)+np.mat(w_star).T)
        part3=((-2/self.beta)*np.mat(a_star)-np.mat(self.I_v())+(1/self.beta)*np.mat(B_star))
        if self.is_invertible(part1+part2):
            lambdaVec=np.mat(numpy.linalg.inv(part1+part2))*np.mat(part3)
        else: 
            lambdaVec=np.mat(numpy.linalg.pinv(part1+part2))*np.mat(part3)
        for i in range(len(lambdaVec)):
            if lambdaVec[i]<0:
                lambdaVec[i]=0
        return lambdaVec
    def is_invertible(self,A):
        return A.shape[0] == A.shape[1] and numpy.linalg.matrix_rank(A) == A.shape[0]
    
    def is_pos_def(self,x):
        return np.all(np.linalg.eigvals(x) >= 0)



    def getAccuracy(self, testSet, predictions):
        correct = 0
        testSet=np.ravel(testSet)
        for x in range(len(testSet)):
            if testSet[x] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0
    def mydist(self,x,y):
        return np.math.sqrt(np.mat(x - y).T*self.Metric*np.mat(x - y))

    



def run_experiment(k):
    splitList=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    '''tempParser= Paeser(path,splitList[3])
    tempData=tempParser.DataGen()
    betaResults=computeBeta(reigCoef,tempData)
        
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    ax.plot(reigCoef,betaResults)
    ax.set_yscale('log')
    plt.ylabel('(Accuracy')
    plt.xlabel('(betaCoefs')
    #plt.legend(["L2"],loc=3)   
        #print(str(metricType)+' Accuracy: ' + repr(accuracy) + '%')
    #mLearner.myKNN(X_train, y,X_test,y_test,'myMetric')
    plt.show()'''
    
    
    i=0
    predictions=[]
    l2Accuray=[]
    LMNNAccuracy=[]
    myMetricAccuracy=[]
    
    while i < (len(splitList)):
        # reigCoef=[100,0.05,0.1,1,10,100,1000,10000]
        featureDim=7
    
        path="/Users/mgomrokchi/Documents/workspace/Distance Metric Learning/seeds_dataset.txt"
        myParser= Paeser(path,splitList[i])
        Data=myParser.DataGen()
        mLearner=Learner(1000,Data[0][0],Data[0][3],featureDim)
        lambdaVec=mLearner.computeLambda(len(Data[0][3]))
        M=mLearner.compute_M(np.ravel(lambdaVec))
        mLearner.Metric=M
    # mLearner.knnDemo(origData[1], origData[2])
        if not mLearner.is_pos_def(M):
            M=KNNs._getAplus(M)
        #print(M)
        X_train=Data[1][0]+Data[1][3]
        X_train=np.reshape(X_train, (len(X_train),8))
        y_train=X_train[:,7]
        X_train=scipy.delete(X_train, 7, 1)
        X_test=Data[1][9]
        X_test=np.reshape(X_test,(len(X_test),8)) 
        y_test=X_test[:,7] 
        X_test=scipy.delete(X_test, 7, 1)
        #mLearner.Metric=M
        # modelL2=neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None)
        # modelHM=neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='mahalanobis', metric_params={'V': M})
        #
        WTrain = KNNs.fit(X_train,y_train)
        for metricType in ['myMetric','l2','LMNN']:
            predictions=[]
            if metricType=='LMNN':
                z=python_LMNN(k=k)
                z.fit(X_train, y_train, False)
                L_Lmnn=z.metric()
                predictions=KNNs.prediction(WTrain,X_test,k,L_Lmnn
                                            )
                LMNNAccuracy.append(mLearner.getAccuracy(y_test, predictions))
            if metricType=='l2':

                predictions=KNNs.prediction(WTrain,X_test,k,numpy.identity(featureDim))
                l2Accuray.append(mLearner.getAccuracy(y_test, predictions))
            if metricType=='myMetric':

                predictions=predictions=KNNs.prediction(WTrain,X_test,k,M)
                myMetricAccuracy.append(mLearner.getAccuracy(y_test, predictions))
        i+=1
    return [l2Accuray,myMetricAccuracy,LMNNAccuracy,mLearner.beta]
def main():
    splitList=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    numRounds=20
    k=15
    l2Accuray=[]
    l2Accuraylb=[]
    l2Accurayub=[]
    lmnnAccuray=[]
    lmnnAccuraylb=[]
    lmnnAccurayub=[]
    myMetricAccuracy=[]
    myMetricAccuracylb=[]
    myMetricAccuracyub=[]
    
    expResultsL2=np.zeros((numRounds,len(splitList)))
    expResultsHM=np.zeros((numRounds,len(splitList)))
    expResultsLmnnn=np.zeros((numRounds,len(splitList)))
    for i in range(numRounds):
        temp=run_experiment(k)
        beta=temp[3]
        for j in range(len(splitList)):
            expResultsL2[i][j]=temp[0][j]
            expResultsHM[i][j]=temp[1][j]
            expResultsLmnnn[i][j]=temp[2][j]

    
    for i in range(len(splitList)):
        templ2=[]
        tempHM=[]
        tempLmnn=[]
        for j in range(numRounds):
            tempHM.append(expResultsHM[j,i])
            templ2.append(expResultsL2[j,i])
            tempLmnn.append(expResultsLmnnn[j,i])
        l2Accuray.append((np.mean(templ2)))
        l2Accuraylb.append(np.math.log(((np.mean(templ2))-(np.std(templ2)))))
        l2Accurayub.append(np.math.log(((np.mean(templ2)+np.std(templ2)))))        
        
        
        myMetricAccuracy.append(np.mean(tempHM))
        myMetricAccuracylb.append(np.math.log((np.mean(tempHM)-np.std(tempHM))))
        myMetricAccuracyub.append(np.math.log(np.mean(tempHM)+np.std(tempHM)))

        lmnnAccuray.append(np.mean(tempLmnn))
        lmnnAccuraylb.append(np.math.log((np.mean(tempLmnn)-np.std(tempLmnn))))
        lmnnAccurayub.append(np.math.log(np.mean(tempLmnn)+np.std(tempLmnn)))
        
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    ax.errorbar(splitList,l2Accuray,yerr=[l2Accurayub,l2Accuraylb])
    ax.errorbar(splitList,myMetricAccuracy,yerr=[myMetricAccuracyub,myMetricAccuracylb])
    ax.errorbar(splitList,lmnnAccuray,yerr=[lmnnAccurayub,lmnnAccuraylb])
    #ax.plot(splitList,l2Accuray)
    #ax.plot(splitList,myMetricAccuracy)
    ax.set_yscale('log')
    plt.ylabel('(log)Accuracy')
    plt.xlabel('percentage of the training data size')
    plt.legend(["L2", "H&M Metric","LMNN"],loc=3)
    plt.title("beta= "+str(beta)+"  KNN with k="+str(k)+"  Number of iteration= "+ str(numRounds))  
        #print(str(metricType)+' Accuracy: ' + repr(accuracy) + '%')
    #mLearner.myKNN(X, y,X_test,y_test,'myMetric')
    plt.show()

main()