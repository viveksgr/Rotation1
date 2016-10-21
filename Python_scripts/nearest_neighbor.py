# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:21:26 2016

@author: viveksagar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


# Input parameters, look for nbr number of neighbors around each patient.
nbr = 150;

x1 = pd.ExcelFile("entry_data.xlsx")
entry_data = x1.parse("Sheet1")
x2 = pd.ExcelFile("exit_data.xlsx")
exit_data = x2.parse("Sheet1")
entry_mat = entry_data.values
exit_mat = exit_data.values

# Calculates over distance matrix, only the common features are used for computing the distance.
def distmat_func(mat):
    sz = len(mat)
    s_width = mat.shape[1]
    distmat = np.zeros(shape=[sz,sz])
    for ii in range(len(mat)):
        for jj in range(len(mat)):
            temp1 = mat[ii,:]
            temp2 = mat[jj,:]
            temp3 = (temp1-temp2)**2
            temp4 = np.count_nonzero(np.isnan(temp3))
            if temp4 < s_width:
                distmat[ii,jj]=((np.nansum(temp3))**0.5)/(s_width-temp4)
            else:
                distmat[ii,jj]=-100000
    return distmat   

# Find the nbr number of nearest neighbors.
def sorted_mat(mat2,kk):
    sz = len(mat2)
    keys2 = np.zeros(shape=[sz,sz])
    for ii in range(sz):
        keys2[:,ii]=np.argsort(mat2[:,ii])
    keys = keys2[1:1+kk,:]     
    return keys
    
distmat_entry = distmat_func(entry_mat)
entry_nbr = sorted_mat(distmat_entry,nbr)
distmat_exit = distmat_func(exit_mat)
exit_nbr = sorted_mat(distmat_exit,nbr)

# Find the number of common elements for each patient with each of its set of neighbors.   
def common_elem(mat1, mat2):
    sz = mat1.shape[1]
    s_width = mat1.shape[0]
    num_common = np.zeros(shape=[sz,1])
    for ii in range(sz):               
        num_common[ii]= np.sum(np.in1d(mat1[:,ii],mat2[:,ii]))
#        num_common=num_common/s_width
    return num_common
    
score = common_elem(entry_nbr,exit_nbr)

output = open('dist_ext.pkl', 'wb')
pickle.dump(distmat_exit, output)
output.close()

# Histogram
n, bins, patches = plt.hist(score, 50, normed=1, facecolor='green', alpha=0.75)
    
    

#pkl_file = open('dist_mat.pkl', 'rb')
#distmat_entry = pickle.load(pkl_file)
#pkl_file.close()

