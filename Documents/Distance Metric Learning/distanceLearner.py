'''
Created on Apr 16, 2016

@author: mgomrokchi
'''
import os
import operator
import scipy
from myKNN import KNNClassifier
import numpy
from Modified_Parser import Paeser
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
from numpy import math
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
            temp=self.compute_A_i(self.S[i])
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
            A_star.append(temp)
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
       
        for j in range(len(self.D)):
            temp=0
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
    def mydist(self,x,y):
        return np.math.sqrt(np.mat(x - y).T*self.Metric*np.mat(x - y))

    



def run_experiment(k,splitList):
    #splitList=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    Dimnsion=9
    featureDim=Dimnsion-1
    Density=0.5
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
    #mLearner.mKNN(X_train, y,X_test,y_test,'myMetric')
    plt.show()'''
    
    
    i=0
    predictions=[]
    l2Accuray=[]
    LMNNAccuracy=[]
    myMetricAccuracy=[]
    path="./abalone.txt"
    
    while i < (len(splitList)):
        # reigCoef=[100,0.05,0.1,1,10,100,1000,10000]
    
        #path="/Users/mgomrokchi/Documents/workspace/Distance Metric Learning/abalone.data.txt"
        
        Labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
        myParser= Paeser(similarDataFilePath=path,splitSize=splitList[i],Dimension=Dimnsion,Lables=Labels,startIndex=1)
        Data=myParser.DataGen(Density)
        beta=0.0001*((len(Data[0][0])+len(Data[0][1]))**0.2)
        mLearner=Learner(beta,Data[0][0],Data[0][1],featureDim-1)
        lambdaVec=mLearner.computeLambda(len(Data[0][1]))
        M=mLearner.compute_M(np.ravel(lambdaVec))
        mLearner.Metric=M
        mKNN=KNNClassifier(M,k)
    # mLearner.knnDemo(origData[1], origData[2])
        if not mLearner.is_pos_def(M):
            M=mKNN._getAplus(M)
            mKNN.setM(M)
        #print(M)
        X_train=Data[1][0]
        X_train=np.reshape(X_train, (len(X_train),8))
        y_train=X_train[:,7]
        X_train=scipy.delete(X_train, 7, 1)
        X_test=Data[1][1]
        X_test=np.reshape(X_test,(len(X_test),8))
        y_test=X_test[:,7]
        X_test=scipy.delete(X_test, 7, 1)
        #mLearner.Metric=M
        # modelL2=neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None)
        # modelHM=neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='mahalanobis', metric_params={'V': M})
        #
        mKNN.fit(X_train,y_train)
        for metricType in ['myMetric','l2','LMNN']:
            predictions=[]
            if metricType=='LMNN':
                z=python_LMNN(k=k)
                z.fit(X_train, y_train, False)
                L_Lmnn = z.metric()
                mKNN.setM(L_Lmnn)
                predictions=mKNN.predict(X_test)
                LMNNAccuracy.append(mLearner.getAccuracy(y_test, predictions))
                print("good")
            if metricType=='l2':
                mKNN.setM(np.identity(featureDim-1))
                predictions=mKNN.predict(X_test)
                l2Accuray.append(mLearner.getAccuracy(y_test, predictions))
                print("Excellent")
            if metricType=='myMetric':
                mKNN.setM(M)
                predictions=mKNN.predict(X_test)
                print("great")

                myMetricAccuracy.append(mLearner.getAccuracy(y_test, predictions))
        i+=1
    return [l2Accuray,myMetricAccuracy,LMNNAccuracy,mLearner.beta]

def betaExperiment(coefs,exps,splitSize,n_neighbour):
    Dimnsion=9
    featureDim=Dimnsion-1
    Density=200
    predictions=[]
    myMetricAccuracy_test=[]
    myMetricAccuracy_train=[]
    
    path="./abalone.txt"
    
    
    for c in coefs:
        for e in exps:
            
            
            Labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
            myParser= Paeser(similarDataFilePath=path,splitSize=splitSize,Flag=1,Dimension=Dimnsion,Labels=Labels)
            Data=myParser.DataGen(Density)
            beta= c*(len(Data[1][0])**e)
            mLearner=Learner(beta,Data[0][0],Data[0][1],featureDim-1)
            lambdaVec=mLearner.computeLambda(len(Data[0][1]))
            M=mLearner.compute_M(np.ravel(lambdaVec))
            mLearner.Metric=M
            mKNN=KNNClassifier(M,n_neighbour)
            
            if not mLearner.is_pos_def(M):
                M=mKNN._getAplus(M)
                mKNN.setM(M)
            #print(M)
            X_train=Data[1][0]
            X_train=np.reshape(X_train, (len(X_train),8))
            y_train=X_train[:,7]
            X_train=scipy.delete(X_train, 7, 1)
            X_test=Data[1][1]
            X_test=np.reshape(X_test,(len(X_test),8))
            y_test=X_test[:,7]
            X_test=scipy.delete(X_test, 7, 1)
            predictions=[]
            mKNN.fit(X_train,y_train)
            mKNN.setM(M)
            predictions=mKNN.predict(X_test)
            myMetricAccuracy_test.append([mLearner.getAccuracy(y_test, predictions),[c,e]])
            predictions=mKNN.predict(X_train)
            myMetricAccuracy_train.append(mLearner.getAccuracy(y_train, predictions))
            print("great")
    return sorted(myMetricAccuracy_test,key=getKey),sorted(myMetricAccuracy_train)

def getKey(item):
    return item[0]

def main():
    #splitList=[0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    splitList=[0.75,0.8,0.85,0.9]
    #coefs=[0.0001,0.001,0.01,0.1,1,10,100,1000]
    #exps=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    numRounds=5
    k=10
    #result_test,result_train=betaExperiment(coefs, exps, splitList[0], k)
    #minVal=numpy.abs(result_test[0][0]-result_train[0])
    #===========================================================================
    # minCoefs=result_test[0][1]
    # for i in range(len(exps)*len(coefs)):
    #     print(numpy.abs(result_test[i][0]-result_train[i]))
    #     print(result_test[i][1])
    #     if numpy.abs(result_test[i][0]-result_train[i])<minVal:
    #         minVal=numpy.abs(result_test[i][0]-result_train[i])
    #         
    #         minCoefs=result_test[i][1]
    #     
    # print(minCoefs)
    #===========================================================================
    #===========================================================================
    # ax = plt.gca()
    # ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    # ax.plot(result_test)
    # ax.plot(result_train)
    # ax.set_yscale('log')
    # plt.ylabel('(log)Accuracy')
    # #plt.xlabel('percentage of the training data size')
    # #plt.legend(["L2", "H&M Metric","LMNN"],loc=3)
    # #plt.title("beta= "+str(beta)+"  KNN with k="+str(k)+"  Number of iteration= "+ str(numRounds))  
    # #my_path = os.path.dirname(os.path.abspath(__file__))
    # plt.show()
    #===========================================================================

    l2Accuray = []
    l2Accuraylb = []
    l2Accurayub = []
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
        temp=run_experiment(k,splitList)
        beta=temp[3]
        for j in range(len(splitList)):
            expResultsL2[i][j]=temp[0][j]
            expResultsHM[i][j]=temp[1][j]
            expResultsLmnnn[i][j]=temp[2][j]

    for i in range(len(splitList)):
        templ2 = []
        tempHM = []
        tempLmnn = []
        for j in range(numRounds):
            tempHM.append(expResultsHM[j, i])
            templ2.append(expResultsL2[j, i])
            tempLmnn.append(expResultsLmnnn[j, i])
        l2Accuray.append((np.mean(templ2)))
        l2Accuraylb.append(np.math.log(abs((np.mean(templ2)) - (np.std(templ2)))))
        l2Accurayub.append(np.math.log(abs((np.mean(templ2) + np.std(templ2)))))

        myMetricAccuracy.append(np.mean(tempHM))
        myMetricAccuracylb.append(np.math.log(abs(np.mean(tempHM) - np.std(tempHM))))
        myMetricAccuracyub.append(np.math.log(abs(np.mean(tempHM) + np.std(tempHM))))

        lmnnAccuray.append(np.mean(tempLmnn))
        lmnnAccuraylb.append(np.math.log((abs(np.mean(tempLmnn) - np.std(tempLmnn)))))
        lmnnAccurayub.append(np.math.log(abs(np.mean(tempLmnn) + np.std(tempLmnn))))
        #
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    ax.errorbar(splitList,l2Accuray,yerr=[l2Accurayub,l2Accuraylb])
    ax.errorbar(splitList,myMetricAccuracy,yerr=[myMetricAccuracyub,myMetricAccuracylb])
    ax.errorbar(splitList,lmnnAccuray,yerr=[lmnnAccurayub,lmnnAccuraylb])

    ax.set_yscale('log')
    plt.ylabel('(log)Accuracy')
    plt.xlabel('percentage of the training data size')
    plt.legend(["L2", "H&M Metric","LMNN"],loc=3)
    plt.title("beta= "+str(beta)+"  KNN with k="+str(k)+"  Number of iteration= "+ str(numRounds))  
        #print(str(metricType)+' Accuracy: ' + repr(accuracy) + '%')
    #mLearner.myKNN(X, y,X_test,y_test,'myMetric')
    #my_path = os.path.dirname(os.path.abspath(__file__))
    #===========================================================================
    # plt.savefig(str(my_path)+'hello' + '.pdf')
    # plt.close()
    #===========================================================================
    plt.show()
main()
