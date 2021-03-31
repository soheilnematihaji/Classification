#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import linalg as LA

import heapq
from collections import defaultdict

import unittest


# In[ ]:



class knn():
    
    def __init__(self,k):
        self.k=k
        
    def __init__(self,X_train,Y_train,k):
        self.X_train=X_train
        self.Y_train=Y_train
        self.k=k
        
    def fit(self,X_train,Y_train):
        self.X_train=X_train
        self.Y_train=Y_train
        
    def set_k(self,k):
        self.k=k
        
    def knn_points(self,newPoint):
        X_train=self.X_train
        k=self.k
        result=[]
        for i,dataPoint in enumerate(X_train):
            dis=(dataPoint-newPoint)@((dataPoint-newPoint).T)
            update_top_k(k,result,dis,i)
        return result
    
    def predict_proba(self,newPoint):
        classCount=self.class_count(newPoint)
        for class_key in classCount:
            classCount[class_key]/=self.k
        return classCount
    
    def predict(self,newPoint):
        class_max=0
        value_max=0
        classCount=self.class_count(newPoint)
        for class_key in classCount:
            if classCount[class_key]>value_max:
                class_max=class_key
                value_max=classCount[class_key]
        return class_max
    
    def predict_batch(self,batchPoint):
        return [self.predict(point) for point in batchPoint]
    
    def predict_proba_batch(self,batchPoint):
        return [self.predict_proba(point) for point in batchPoint]
    
    def class_count(self,newPoint):
        knn_result=self.knn_points(newPoint)
        class_count=defaultdict(int)
        for (val,index) in knn_result:
            class_count[self.Y_train[index]]+=1
        return class_count
    
    def update_top_k(k,k_closet,index_dis,index):
        if len(k_closet)<k:
            heapq.heappush(k_closet,(-index_dis,index))
        else:
            heapq.heappushpop(k_closet, (-index_dis,index))
        


    


# In[ ]:


class Test(unittest.TestCase):
    
    # This is a variable to generate normal train set with train_size size
    #increasing or decreasing it may effect the test
    train_size=500
    
    # batchsize is used to test the batch methods
    batchsize=10
    
    # This is a variable to generate huge train set with train_size size
    #increasing this will effect the time of the tests
    efficiency_train_size=100000
    
    def generate_random_noraml_point(self,count,pointCount):
        point=1+np.random.randn(pointCount,2)
        x_train=1+np.random.randn(count, 2)
        y_train=[1]*count
        x_train=np.concatenate((x_train,5+np.random.randn(count, 2)))
        y_train=np.append(y_train,[2]*count)
        return point,x_train,y_train 
    
    def test_simple_predict_k_1(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        knn_model=knn(x_train,y_train,1)
        assert knn_model.predict(point[0])==1
        
    def test_simple_predict_k_3(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        knn_model=knn(x_train,y_train,3)
        assert knn_model.predict(point[0])==1
        
    def test_knn_efficiency(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.efficiency_train_size,1)
        knn_model=knn(x_train,y_train,15)
        assert knn_model.predict(point[0])==1
    
    def test_rand_normal_accuracy(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        knn_model=knn(x_train,y_train,15)
        assert knn_model.predict(point[0])==1
        
    def test_rand_normal_accuracy_batch(self):
        points,x_train,y_train=self.generate_random_noraml_point(self.train_size,self.batchsize)
        knn_model=knn(x_train,y_train,150)
        result=0
        for r in knn_model.predict_batch(points):
            if r :
                result+=1
        assert result>0.95*self.batchsize
        
    def test_knn_efficiency_batch(self):
        points,x_train,y_train=self.generate_random_noraml_point(int(self.efficiency_train_size/self.batchsize),self.batchsize)
        knn_model=knn(x_train,y_train,150)
        result=0
        for r in knn_model.predict_batch(points):
            if r :
                result+=1
        assert result>0.95*self.batchsize
        
    def test_rand_normal_accuracy_proba(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        knn_model=knn(x_train,y_train,15)
        assert knn_model.predict_proba(point[0])[1]>=14/15
        
    def test_rand_normal_efficiency_proba(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.efficiency_train_size,1)
        knn_model=knn(x_train,y_train,15)
        assert knn_model.predict_proba(point[0])[1]>=14/15

    def test_rand_normal_accuracy_predict_proba_batch(self):
        points,x_train,y_train=self.generate_random_noraml_point(self.train_size,self.batchsize)
        knn_model=knn(x_train,y_train,15)
        result=0
        for r in knn_model.predict_proba_batch(points):
            if r[1]>14/15 :
                result+=1
        assert result>0.95*self.batchsize
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:




