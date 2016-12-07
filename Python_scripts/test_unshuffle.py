# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 13:42:53 2016

@author: viveksagar
"""
import numpy as np
from scipy.spatial import distance

def unshuffler(shuffled, original):
    org_copy=original
    loc=np.empty([len(shuffled),1]).astype(np.int64)
    for ii in range(len(original)):        
        loc[ii,0] = distance.cdist(shuffled[ii,:], org_copy,'cityblock').argmin() 
    return loc

              
        
        
        
        
        
        
        
