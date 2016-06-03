'''
Created on Apr 15, 2016

@author: Hossein and Maziar

'''

import numpy as np
import random 
import scipy
from sklearn.cross_validation import train_test_split

class Paeser():


    def __init__(self,Lables,Dimension,similarDataFilePath,splitSize,startIndex=0):
        '''Dimension = index of the class label'''
        self.filePath=similarDataFilePath
        self.classIndex = Dimension-1
        self.splitSize = splitSize
        
        self.Classes=self.classConstructor(Lables)
        self.statrtIndex=startIndex
        self.originalData=self.parseData()
        
    def classConstructor(self,Lables): 
        temp=[]
        for i in Lables:
            temp.append([i,[]])
        return temp 
    
    def parseData(self):
        X=np.loadtxt(self.filePath, delimiter=',', usecols=range(self.statrtIndex,self.classIndex+self.statrtIndex))
        self.classIndex-=self.statrtIndex
        #X_unlabled=scipy.delete(X
        return X

    def DataGen(self,Density=0.5):
        '''
        This function extracts the relations among data-points and returns training and test data accordingly 
        Density= The percentqge o, len(X[0])-1, 1)
        #y=X[:,len(X[0])-1]f the total number of relations extracted from the input dataset
        
        '''
        numberOfClasses= len(self.Classes)-1
        random_state = np.random.randint(1, len(self.originalData))

        
        '''similar and disimilar difference of records'''
        SDif1_train=[]
        DDif1_train=[]
        
        
        ''' Random state chosen to have different train and test sets any time calling this piece of code'''  
        Train_Data, Test_Data = train_test_split(self.originalData, train_size=self.splitSize, random_state=random_state)
        
        '''Partition the training data into different classes'''
        for i in range(len(Train_Data)):
            for j in range(numberOfClasses):
                if(Train_Data[i][self.classIndex]==self.Classes[j][0]):
                    self.Classes[j][1].append(Train_Data[i])
                
        '''Filling up the training data with similar relations'''
        similarSample=[]
        disSimilarSample=[]
        for i in range(numberOfClasses):
            length = len(self.Classes[i][1]) 
            for j in range(int(0.5*length*(length-1)*Density)):
                tempPair=[]
                if len(self.Classes[i][1])==0:
                        break
                a=random.sample(self.Classes[i][1],2)
                tempPair=[np.array(a[0]).tolist(),np.array(a[1]).tolist()]
                if  all(tempPair) in disSimilarSample:    
                    j-=1
                else:
                    disSimilarSample.append(tempPair)
                    
            
        for i in range(numberOfClasses):
            for j in range(i+1,numberOfClasses):
                for k in range(int(Density*len(self.Classes[i][1])*len(self.Classes[j][1]))):
                    tempPair=[]
                    if len(self.Classes[j][1])==0 or len(self.Classes[i][1])==0:
                        break
                    a=random.sample(self.Classes[i][1],1)
                    b=random.sample(self.Classes[j][1],1)
                    a=(np.array(a).tolist())
                    b=(np.array(b).tolist())
                    tempPair=[a[0],b[0]]
                    if  all(tempPair) in disSimilarSample:    
                        k-=1
                    else:
                        disSimilarSample.append(tempPair)
                        
        
        for i in range(len(similarSample)):
            SDif1_train.append(np.subtract(similarSample[i][0], similarSample[i][1]))

        for i in range(len(disSimilarSample)):
            DDif1_train.append(np.subtract(disSimilarSample[i][0],disSimilarSample[i][1]))

        SDif1_train=scipy.delete(SDif1_train, self.classIndex, 1)
        DDif1_train=scipy.delete(DDif1_train, self.classIndex, 1)

        return [[SDif1_train,DDif1_train],[Train_Data,Test_Data]]
   
