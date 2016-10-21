# -*- coding: utf-8 -*-
"""
The data pre-processing
"""

import pandas as pd
import numpy as np

x1 = pd.ExcelFile("SCD.xlsx")
df = x1.parse("SelfCare Deidentified")

# Input values, units are number of days.
batch_size = 7
batch_size_out = batch_size;
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

# Recompute the first and last rows because the data has been modified.
locus_in = df.loc[df['dFIN'] >0]
num_in = locus_in.index.values
locus_out = df.loc[df['dFIN2'] >0]
num_out = locus_out.index.values

# Data for the first <batch_size> days of the patient.
new_list_in = []
for ii in range(len(num_in)):
    for jj in range(batch_size):
        temp = num_in[ii]+jj
        new_list_in.append(temp)
                
# Data for the last <batch_size_out> days of the patient.
new_list_out = []
for ii in range(len(num_out)):
    for jj in range(batch_size_out):
        temp = num_out[ii]-batch_size_out+1+jj
        new_list_out.append(temp)

# The desired matrices containing the patient data.
in_df = df.ix[new_list_in]
out_df = df.ix[new_list_out]
in_df2 = pd.DataFrame()
out_df2 = pd.DataFrame()
for ii in range(len(num_in)):
    temp = in_df.ix[num_in[ii]:num_in[ii]+batch_size-1,:]
    temp2 = temp.max(0)
    in_df2=in_df2.append(temp2, ignore_index=True)    
for ii in range(len(num_out)):
    temp = out_df.ix[num_out[ii]:num_out[ii]+batch_size_out-1,:]
    temp2 = temp.min(0)
    out_df2=out_df2.append(temp2, ignore_index=True)
    
fin_data = in_df2[['FIN']]
in_df2=in_df2.drop(['dFIN','assessmentDay','FIN','dFIN2'],1)
out_df2=out_df2.drop(['dFIN','assessmentDay','FIN','dFIN2'],1)    
in_df2.to_excel("entry_data.xlsx")
out_df2.to_excel("exit_data.xlsx")
fin_data.to_excel("fin.xlsx")    


