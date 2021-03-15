#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
from numpy import linalg as LA
import heapq
from collections import defaultdict


# In[87]:



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
        

def test_knn():
    
    point=np.array([1,1,1])
    x_train=np.array([[1,2,3] for i in range(100000)]) 
    y_train=np.array([i%3for i in range(100000)])
    knn0=knn(x_train,y_train,3)
    print(knn0.predict_proba_batch([point,point]))
test_knn()
    


# In[ ]:





# In[ ]:




