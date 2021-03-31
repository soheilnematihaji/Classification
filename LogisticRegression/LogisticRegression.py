#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

from scipy import optimize
import NormalScalar

import math

import unittest


# In[2]:



class LogisticRegression():
    
        
    def __init__(self,X_train,Y_train):
        self.X_train=np.append([[1]]*len(X_train),X_train,1)
        self.Y_train=Y_train
        self.fittedLine=self.findTheta_using_gd()
        
    def fit(self,X_train,Y_train):
        self.X_train=np.append([[1]]*len(X_train), X_train,1)
        self.Y_train=Y_train
        self.fittedLine=self.findTheta_using_gd()
        
    def findTheta_using_gd(self,learning_rate=0.01):
        
        #Normalizing the inpurt before applying Gradient Descent
        normalScalar= NormalScalar.NormalScalar()
        normaled_val=normalScalar.fit_transform(self.X_train[:,1:])
        X_train=np.append([[1]]*len(normaled_val),normaled_val,1)
        Y_train=self.Y_train
        
        #Applying C Gradient Descent on cost funtion J
        (m,n)=X_train.shape
        
        def h(theta):
            return 1/(1 + np.exp(-X_train@theta))
        
        def J(theta,args):
            return -np.log(h(theta))@Y_train -np.log(1-h(theta))@(1-Y_train)
            
        def gradJ(theta, args):
            return ((h(theta)-Y_train.T)@X_train).reshape(theta.shape)
        
        theta=optimize.fmin_cg(J, np.array([1 for i in range(n)]), fprime=gradJ, args=[None])
        
        
        # Inverse Scaling the value of theta
        theta[0]=theta[0]-(sum(normalScalar.mean *(theta[1:]/normalScalar.std)))
        theta[1:]=(theta[1:]/normalScalar.std)
        return theta
        
    
    def predict(self,newPoint):
        point=np.append([[1]],newPoint,1)
        return np.sign(point@self.fittedLine)
    
    def predict_batch(self,batchPoint):
        batchPoint=np.append([[1]]*len(batchPoint),batchPoint,1)
        return np.sign(batchPoint@self.fittedLine) 
    
    def predict_proba(self,newPoint):
        point=np.append([[1]],newPoint,1)
        return 1/1+math.exp(point@self.fittedLine)


    


# In[3]:


class Test(unittest.TestCase):
    
    # This is a variable to generate normal train set with train_size size
    #increasing or decreasing it may effect the test
    train_size=5000
    
    # batchsize is used to test the batch methods
    batchsize=100
    
    # This is a variable to generate huge train set with train_size size
    #increasing this will effect the time of the tests
    efficiency_train_size=100000
    
    batchsize_eff=10000
    
    def generate_random_noraml_point(self,count,pointCount):
        point=[[(i+count)] for i in range(pointCount)]
        x_train=[[i] for i in range(2*count)]
        y_train=[[0] for i in range(count)]
        y_train.extend([[1] for i in range(count)])
        return point,x_train,y_train 
        
    def test_findTheta_using_gd(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LogisticRegression(np.array(x_train),np.array(y_train))
        bestLine=model.findTheta_using_gd()
        print('bestline is :',bestLine)
        assert LA.norm(bestLine-[-4.62451380e+04 , 9.24995262e+00]) < 100
        
    def test_predict(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LogisticRegression(np.array(x_train),np.array(y_train))
        y=model.predict(point)
        assert LA.norm(y-1) < 0.1

        
    def test_predictBatch(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,self.batchsize)
        model=LogisticRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch(np.array(point))
        for i in range(len(point)):
            assert LA.norm(y[i]-1) < 0.1
            
    def test_predictBatchNEG(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,self.batchsize)
        model=LogisticRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch([[(i)] for i in range(10)])
        for i in range(len(y)):
            assert LA.norm(y[i]+1) < 0.1
            
    def test_predictBatch_efficiency(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.efficiency_train_size,self.batchsize_eff)
        model=LogisticRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch(np.array(point))
        y=model.predict_batch([[(i)] for i in range(10)])
        for i in range(len(y)):
            assert LA.norm(y[i]+1) < 0.1
    
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

