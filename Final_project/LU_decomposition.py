#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
def LU_decomposition(a):
    a=a.astype(float)                    # array a : type=float
    l=np.eye(len(a), dtype=float)        # an array of diagonal=1,type=float
    m=np.zeros((len(a),1))             
    x=np.zeros((1,len(a)))   
    
    for j in range(len(a)-1):
        for i in range(j+1,len(a)):   
            mul=a[i][j]/a[j][j]          # multiple of two rows
            a[i]=a[j]*(-mul)+a[i]        # upper triangular matrix
            l[i][j]=mul                  # lower triangular matrix
    return a,l
    
    

