# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:26:59 2016
Surrogate data.
@author: viveksagar
"""

import numpy as np
from sklearn import preprocessing

sz = (1000,42)
offset = 100
X1 = 0*offset/1000*np.ones(sz)+offset/1000*np.random.random(sz)-0.5
X2 = 9*offset/1000*np.ones(sz)+offset/1000*np.random.random(sz)-0.5
X1[0,:]=0.2
X2[0,:]=0.2
X1[1,:]=0.2
X2[1,:]=-0.2
X2[2,:]=-0.2
X1[2,:]=-0.2
X1[3,:]=-0.2
X2[3,:]=0.2
X3 = preprocessing.scale(np.append(X1,X2, axis=0))
X1 = X3[0:np.ceil(len(X3)/2),:]
X2 = X3[np.ceil(len(X3)/2):,:]                 
np.savez("Pre_processed2", X1=X1, X2=X2)