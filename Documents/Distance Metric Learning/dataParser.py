'''
Created on Apr 15, 2016

@author: mgomrokchi
'''
import numpy as np
import scipy

class Paeser(object):
    '''
    classdocs
    '''


    def __init__(self,similarDataFilePath):
        
        self.sFilePath=similarDataFilePath
        self.originalData=self.parseData()
        self.class1lable=1
        self.class2lable=2
        self.class3lable=3
        self.classIndex=7
    def parseData(self):
        X =np.genfromtxt(self.sFilePath)
        #y = np.loadtxt(self.sFilePath,usecols=range(1))
        #for i in range(len(x)):
        #    z.append(x[i]-y[i])
        return X
    def DataGen(self):
        S_class1=[]
        S_class2=[]
        S_class3=[]
        S1=[]
        SDif1=[]
        S2=[]
        SDif2=[]
        S3=[]
        SDif3=[]
        D1=[]
        DDif1=[]
        D2=[]
        DDif2=[]
        D3=[]
        DDif3=[]
        for i in range(len(self.originalData)):
            temp=self.originalData[i][self.classIndex]
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
        for j in range(tempMin):
            B=np.random.randint(tempMin,size=2)
            S1Temp=S_class1[B,:]
            S1.append(S1Temp)
            S2.append(S_class2[B,:])
            S3.append(S_class3[B,:])
            D1.append([S_class1[B[0],:],S_class2[B[1],:]])
            D2.append([S_class1[B[0],:],S_class3[B[1],:]])
            D3.append([S_class2[B[0],:],S_class3[B[1],:]])
        for i in range(len(S1)):
            SDif1.append(S1[i][0]-S1[i][1])
            SDif2.append(S2[i][0]-S2[i][1])
            SDif3.append(S3[i][0]-S3[i][1])
            DDif1.append(D1[i][0]-D1[i][1])
            DDif2.append(D2[i][0]-D2[i][1])
            DDif3.append(D3[i][0]-D3[i][1])
        SDif1 = scipy.delete(SDif1, 7, 1)
        SDif2 = scipy.delete(SDif2, 7, 1)
        SDif3 = scipy.delete(SDif3, 7, 1)
        DDif1 = scipy.delete(DDif1, 7, 1)
        DDif2 = scipy.delete(DDif2, 7, 1)
        DDif3 = scipy.delete(DDif3, 7, 1)
                            
        return [SDif1,SDif2,SDif3,DDif1,DDif2,DDif3]
        
    #def disSimilarDataGen(self):
        
        