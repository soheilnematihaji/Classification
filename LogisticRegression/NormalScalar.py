#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

import unittest


# In[58]:



class NormalScalar:
    
    def fit(self,X):
        self.mean=X.mean(axis=0)
        self.std=np.std(X,axis=0)
        
    def transform(self,x):
        return (x-self.mean)/self.std
        
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
    def inverse(self,x):
        return x*self.std+self.mean
        


# In[59]:


class TestGD(unittest.TestCase): 
    
    dataSize=40
    eff_dataSize=1000000
    
    def getData(self,size):
        return np.array([[i,2*i] for i in range(size)])
    
    def test_fit(self):
        X_train=self.getData(self.dataSize)
        normalScalar=NormalScalar()
        normalScalar.fit(X_train)
        assert np.array_equal(normalScalar.mean, [19.5, 39.], equal_nan=True)
        assert np.array_equal(np.around(normalScalar.std,decimals=2), [11.54 ,23.09], equal_nan=True)
        
    def test_transform(self):
        X_train=self.getData(self.dataSize)
        normalScalar=NormalScalar()
        normalScalar.fit(X_train)
        transformed_x=normalScalar.transform(X_train)
        assert transformed_x.shape==(self.dataSize, 2)
        assert np.array_equal(np.round(transformed_x[:2,:2],decimals=2), [[-1.69 ,-1.69],[-1.6 , -1.6 ]], equal_nan=True)
        
    def test_inverse(self):
        X_train=self.getData(self.dataSize)
        normalScalar=NormalScalar()
        normalScalar.fit(X_train)
        transformed_x=normalScalar.transform(X_train)
        assert np.array_equal(np.rint(normalScalar.inverse(transformed_x)), X_train, equal_nan=True)
        
    def test_fit_transform(self):
        X_train=self.getData(self.dataSize)
        normalScalar=NormalScalar()
        transformed_x=normalScalar.fit_transform(X_train)
        assert np.array_equal(normalScalar.mean, [19.5, 39.], equal_nan=True)
        assert np.array_equal(np.around(normalScalar.std,decimals=2), [11.54 ,23.09], equal_nan=True)
        assert transformed_x.shape==(self.dataSize, 2)
        assert np.array_equal(np.round(transformed_x[:2,:2],decimals=2), [[-1.69 ,-1.69],[-1.6 , -1.6 ]], equal_nan=True)
        assert np.array_equal(np.rint(normalScalar.inverse(transformed_x)), X_train, equal_nan=True)
    
    def test_fit_transform_eff(self):
        X_train=self.getData(self.eff_dataSize)
        normalScalar=NormalScalar()
        transformed_x=normalScalar.fit_transform(X_train)
        assert transformed_x.shape==(self.eff_dataSize, 2)
        assert np.array_equal(np.round(transformed_x[:2,:2],decimals=2), [[-1.73 ,-1.73],[-1.73 , -1.73 ]], equal_nan=True)
        assert np.array_equal(np.rint(normalScalar.inverse(transformed_x)), X_train, equal_nan=True)
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:




