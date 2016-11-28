# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:43:53 2016

@author: viveksagar
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

def pre_process(df):
    batch_size = 7
    exclusion_days = 15;
    # This finds the row corresponding to the first day of each patient.
    df['dFIN'] = df['FIN']-df['FIN'].shift(1) 
    df.at[0,'dFIN']=1
    locus_in = df.loc[df['dFIN'] >0]
    num_in = locus_in.index.values

    # This finds the row corresponding to the last day of each patient.
    df['dFIN2'] = df['FIN'].shift(-1)-df['FIN']
    df.ix[len(df)-1,len(df.columns)-1]=1
    locus_out = df.loc[df['dFIN2'] >0]
    num_out = locus_out.index.values

    # Remove the data for those patients that attended less than <exclusion> days.
    num_diff = num_out-num_in
    rem_ind = np.where(num_diff<exclusion_days)[0]
    remove_these = num_in[rem_ind]
    remove_these_indx = num_diff[rem_ind]
    rem_list = []
    for ii in range(len(remove_these)):
        for jj in range(remove_these_indx[ii]+1):
            temp = remove_these[ii]+jj
            rem_list.append(temp)
    df = df.drop(rem_list)
    
    # Remove the weeks before the reference week   
    locus_in = df.loc[df['dFIN'] >0]
    num_in = locus_in.index.values
    locus_out = df.loc[df['dFIN2'] >0]
    num_out = locus_out.index.values
    rem_list2 =[]
    for ii in range(len(num_in)):
        for jj in range(batch_size):
            temp = num_in[ii]+jj
            rem_list2.append(temp)
    df = df.drop(rem_list2)
    
    # Remove the remainder days in the end that do not complete a week    
    num_in=num_in+batch_size
    duration = (num_out-num_in).astype(np.float64)
    num_weeks = np.remainder(duration,batch_size)+1
    rem_list3 = []
    for ii in range(len(num_out)):
        for jj in range(num_weeks[ii].astype(np.int64)):
            temp = num_out[ii]-num_weeks[ii]+jj+1
            rem_list3.append(temp)
    df = df.drop(rem_list3)    
    num_weeks = np.floor_divide(duration,batch_size)
    num_out = num_in+batch_size*num_weeks-1

    df=df.drop(['dFIN','assessmentDay','FIN','dFIN2'],1)

    df2 = pd.DataFrame()
    for ii in range(np.floor(len(df)/batch_size).astype(np.int64)):
        temp = df.ix[ii*batch_size:(ii+1)*batch_size-1,:]
        temp2 = temp.max(axis=0, skipna=True)
        df2=df2.append(temp2, ignore_index=True)  
        
    rem3 = []
    temp = 0
    for ii in range(len(num_weeks)):
        temp = temp+num_weeks[ii]
        rem3.append(temp)
    rem3 = [0]+rem3
    rem3.remove(rem3[len(rem3)-1])
    
    in_df = pd.DataFrame()
    out_df = pd.DataFrame()
    for ii in range(len(rem3)):
        for jj in range((num_weeks[ii]).astype(np.int64)-1):
            temp_in = df2.ix[rem3[ii]+jj,:]
            temp_out = df2.ix[rem3[ii]+jj+1,:]
            in_df=in_df.append(temp_in, ignore_index=True)
            out_df=out_df.append(temp_out, ignore_index=True)
            
    in_data = np.array(in_df)
    out_data = np.array(out_df)
    
    week_list = []
    for ii in range(len(num_weeks)):
        for jj in range((num_weeks[ii]-1).astype(np.int64)):
            week_list.append(jj+1)
    
    def add_nansigns(in_data):
        N = np.isnan(in_data)
        in_data[N]=0
        N = N.astype(float)
        New = np.append(in_data, N, axis =1)
        return New
    
    np_in = np.isnan(in_data)
    np_out = np.isnan(out_data)
    out_data[np_in]=np.nan
    in_data[np_out]=np.nan
    
    X1 = add_nansigns(in_data)
    X2 = add_nansigns(out_data)
    X3 = np.append(X1,X2, axis=0)
    X3 = preprocessing.scale(X3)
    X2 = np.delete(X3, np.s_[:len(X1)],axis=0)
    X1 = np.delete(X3, np.s_[len(X1):],axis=0)
    week_l = np.array(week_list)

    return X1, X2, week_l
 
x1 = pd.ExcelFile("SCD.xlsx")
df = x1.parse("SelfCare Deidentified")   
[X1, X2, week_l] = pre_process(df)
np.savez("Pre_processed", X1=X1, X2=X2, X3=week_l)