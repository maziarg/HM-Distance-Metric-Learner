'''
Created on Apr 15, 2016

@author: mgomrokchi
'''
import numpy as np
import scipy
from sklearn.cross_validation import train_test_split
from _random import Random

class Paeser(object):
    '''
    classdocs
    '''


    def __init__(self,similarDataFilePath,splitSize):
        
        self.sFilePath=similarDataFilePath
        self.originalData=self.parseData()[0]
        self.class1lable=1
        self.class2lable=2
        self.class3lable=3
        self.classIndex=7
        self.splitSize=splitSize
        
    def parseData(self):
        X =np.genfromtxt(self.sFilePath)
        X_unlabled=scipy.delete(X, 7, 1)
        y=X[:,7]
        #y = np.loadtxt(self.sFilePath,usecols=range(1))
        #for i in range(len(x)):
        #    z.append(x[i]-y[i])
        return [X,X_unlabled,y]
    
    def twodTo1d(self,X):
        temp=[]
        for i in range(len(X)):
            for j in range(2):
                temp.append(X[i][j])
        return temp

    def DataGen(self):
        #testSet=[]
        S_class1=[]
        S_class2=[]
        S_class3=[]
        S1=[]
        SDif1_train=[]
        SDif1_test=[]
        S2=[]
        SDif2_train=[]
        SDif2_test=[]
        S3=[]
        SDif3_train=[]
        SDif3_test=[]
        D1=[]
        DDif1_train=[]
        DDif1_test=[]
        D2=[]
        DDif2_train=[]
        DDif2_test=[]
        D3=[]
        DDif3_train=[]
        DDif3_test=[]
        for i in range(len(self.originalData)):
           
            if self.originalData[i][self.classIndex]==self.class1lable:
                S_class1.append(self.originalData[i])
            if self.originalData[i][self.classIndex]==self.class2lable:
                S_class2.append(self.originalData[i])
            if self.originalData[i][self.classIndex]==self.class3lable:
                S_class3.append(self.originalData[i])
            
        tempMin=min([len(S_class1),len(S_class2),len(S_class3)])

        S_class1=np.reshape(S_class1, (tempMin,8))
        S_class2=np.reshape(S_class2, (tempMin,8))
        S_class3=np.reshape(S_class3, (tempMin,8))
        
        
        tempMin=np.arange(tempMin)
        tempMin=set(tempMin)
        while len(tempMin)>0:
            tempArray=list(tempMin)
            B=np.random.choice(tempArray,size=2,replace=False)
            tempMin=set(tempMin).difference(B)
            S1.append(S_class1[B,:])
            S2.append(S_class2[B,:])
            S3.append(S_class3[B,:])
            D1.append([S_class1[B[0],:],S_class2[B[1],:]])
            D2.append([S_class1[B[0],:],S_class3[B[1],:]])
            D3.append([S_class2[B[0],:],S_class3[B[1],:]])
        random_state=np.random.randint(1,200)
        print(random_state)   
        S1_train, S1_test = train_test_split(S1, train_size = self.splitSize,random_state=random_state)
        S2_train, S2_test = train_test_split(S2, train_size = self.splitSize,random_state=random_state)
        S3_train, S3_test = train_test_split(S3, train_size = self.splitSize,random_state=random_state)
        D1_train, D1_test = train_test_split(D1, train_size = self.splitSize,random_state=random_state)
        D2_train, D2_test = train_test_split(D2, train_size = self.splitSize,random_state=random_state)
        D3_train, D3_test = train_test_split(D3, train_size = self.splitSize,random_state=random_state)
            
        for i in range(len(S1_train)):
            SDif1_train.append(S1_train[i][0]-S1_train[i][1])
            SDif2_train.append(S2_train[i][0]-S2_train[i][1])
            SDif3_train.append(S3_train[i][0]-S3_train[i][1])
            DDif1_train.append(D1_train[i][0]-D1_train[i][1])
            DDif2_train.append(D2_train[i][0]-D2_train[i][1])
            DDif3_train.append(D3_train[i][0]-D3_train[i][1])
           
        for i in range(len(S1_test)):   
            SDif1_test.append(S1_test[i][0]-S1_test[i][1])
            SDif2_test.append(S2_test[i][0]-S2_test[i][1])
            SDif3_test.append(S3_test[i][0]-S3_test[i][1])
            DDif1_test.append(D1_test[i][0]-D1_test[i][1])
            DDif2_test.append(D2_test[i][0]-D2_test[i][1])
            DDif3_test.append(D3_test[i][0]-D3_test[i][1])
            
        
        SDif1_train=scipy.delete(SDif1_train, 7, 1)
        SDif2_train=scipy.delete(SDif2_train, 7, 1)
        SDif3_train=scipy.delete(SDif3_train, 7, 1)
        DDif1_train=scipy.delete(DDif1_train, 7, 1)
        DDif2_train=scipy.delete(DDif2_train, 7, 1)
        DDif3_train=scipy.delete(DDif3_train, 7, 1)
        SDif1_test=scipy.delete(SDif1_test, 7, 1)
        SDif2_test=scipy.delete(SDif2_test, 7, 1)
        SDif3_test=scipy.delete(SDif3_test, 7, 1)
        DDif1_test=scipy.delete(DDif1_test, 7, 1)
        DDif2_test=scipy.delete(DDif2_test, 7, 1)
        DDif3_test=scipy.delete(DDif3_test, 7, 1)
        
                            
        return [[SDif1_train,SDif2_train,SDif3_train,DDif1_train,DDif2_train,DDif3_train,SDif1_test,SDif2_test,SDif3_test,DDif1_test,DDif2_test,DDif3_test],[self.twodTo1d(S1_train),self.twodTo1d(S2_train),self.twodTo1d(S3_train),self.twodTo1d(D1_train),self.twodTo1d(D2_train),self.twodTo1d(D3_train),self.twodTo1d(S1_test),self.twodTo1d(S2_test),self.twodTo1d(S3_test),self.twodTo1d(D1_test),self.twodTo1d(D2_test),self.twodTo1d(D3_test)]]
        
    #def disSimilarDataGen(self):
        
        