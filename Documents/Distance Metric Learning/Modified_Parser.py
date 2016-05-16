'''
Created on Apr 15, 2016



@author: mgomrokchi
'''

import numpy as np
from random import shuffle
import scipy

from sklearn.cross_validation import train_test_split

from _random import Random



class Paeser():

    '''

    classdocs

    '''

    def __init__(self,Labels,Dimension,similarDataFilePath,splitSize,Flag):

        self.sFilePath=similarDataFilePath

        self.classIndex = Dimension - 1
        self.splitSize = splitSize
        self.originalData=self.parseData()[0]
        A=[]
        Classes=[]

        for i in range(Flag, len(Labels)+Flag):
            Classes.append([i,[]])
        if(Flag==0):
            Classes.append(0)

        else:
            Classes.append(1)

        self.Classes__ = Classes



    def parseData(self):
        Class__index=self.classIndex-1
        X =np.genfromtxt(self.sFilePath)

        X_unlabled=scipy.delete(X, Class__index, 1)

        y=X[:,Class__index]

        #y = np.loadtxt(self.sFilePath,usecols=range(1))

        #for i in range(len(x)):

        #    z.append(x[i]-y[i])

        return [X,X_unlabled,y]



    def DataGen(self,Density):

        #testSet=[]
        Similars=[]
        Disimilars=[]



        SDif1_train=[]



        DDif1_train=[]

        EE= len(self.Classes__)-1
        random_state = np.random.randint(1, 200)
        q=self.splitSize/2
        Train_Data, Test_Data = train_test_split(self.originalData, train_size=q, random_state=random_state)
        Trian_Data1,Test_Data1= train_test_split(Test_Data, train_size=self.splitSize, random_state=random_state)
        for i in range(len(Train_Data)):

            for j in range(EE):
                if(Train_Data[i][self.classIndex]==self.Classes__[j][0]):
                    self.Classes__[j][1].append(Train_Data[i])
                else:
                    continue





        ss=len(self.Classes__)-1
        for i in range(ss):
            T=CombinSim(self.Classes__[i][1])
            q=i+1
            if(q<ss):
                for k in range(q,ss):
                    T2=CombinDiff(self.Classes__[i][1],self.Classes__[k][1])
                    for l in T2:
                        A = self.Classes__[i][1][l[0]]
                        B = self.Classes__[k][1][l[1]]
                        C = []
                        C.append(A)
                        C.append(B)
                        Disimilars.append(C)
            for j in T:
                A=self.Classes__[i][1][j[0]]
                B=self.Classes__[i][1][j[1]]
                C=[]
                C.append(A)
                C.append(B)
                Similars.append(C)


        shuffle(Similars)
        shuffle(Disimilars)
        i=0
        Similars=Similars[0:Density]
        Disimilars=Disimilars[0:Density]


        print(random_state)






        for i in range(len(Similars)):

            SDif1_train.append(np.subtract(Similars[i][0], Similars[i][1]))

        for i in range(len(Disimilars)):

            DDif1_train.append(np.subtract(Disimilars[i][0],Disimilars[i][1]))









        SDif1_train=scipy.delete(SDif1_train, 7, 1)



        DDif1_train=scipy.delete(DDif1_train, 7, 1)





        return [[SDif1_train,DDif1_train],[Trian_Data1,Test_Data1]]




def CombinSim(X1):
        Tupp=[]

        for i in range(0, len(X1)):
            q=i+1
            if(q==len(X1)):
                continue
            for j in range(q, len(X1)):
                if (i == j):
                    continue
                else:
                    Tupp.append((i, j))
        return Tupp


def CombinDiff(X1,X2):
    Tupp = []

    for i in range(0, len(X1)):
        for j in range(0, len(X2)):
            Tupp.append((i, j))
    return Tupp