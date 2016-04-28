'''
Created on Apr 16, 2016

@author: mgomrokchi
'''
from IPython.core.tests.test_formatters import numpy
from dataParser import Paeser
import numpy as np

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
        return (-1/self.beta)*(np.mat(temp1)-np.mat(temp2))
        
    def computeLambda(self):
        a_star=self.compute_a_star()
        a_star=np.reshape(a_star, (70,1))
        A_star=self.compute_A_star()
        A_star=np.reshape(A_star, (70,70))
        w_star=self.compute_w_star()
        w_star=np.reshape(w_star, (70,70))
        B_star=self.compute_B_star()
        B_star=np.reshape(B_star, (70,1))
        part1=(1/self.beta)*(np.mat(A_star)+np.mat(A_star).T)
        part2=(1/(2*self.beta))*(np.mat(w_star)+np.mat(w_star).T)
        part3=((-2/self.beta)*np.mat(a_star)-np.mat(self.I_v())+(1/self.beta)*np.mat(B_star))
        lambdaVec=np.mat(numpy.linalg.inv(part1+part2))*np.mat(part3)
        for i in range(len(lambdaVec)):
            if lambdaVec[i]<0:
                lambdaVec[i]=0
        return lambdaVec
    def is_pos_def(self,x):
        return np.all(np.linalg.eigvals(x) >= 0)
    
def main():
    reigCoef=[0.1,1,10,100,1000,10000]
    featureDim=7
    
    path="/Users/mgomrokchi/Documents/workspace/Distance Metric Learning/seeds_dataset.txt"
    myParser= Paeser(path)
    Data=myParser.DataGen()
    mLearner=Learner(reigCoef[2],Data[0],Data[3],featureDim)
    lambdaVec=mLearner.computeLambda()
    M=mLearner.compute_M(np.ravel(lambdaVec))
    print(mLearner.is_pos_def(M))

if __name__ == "__main__": main()  
        