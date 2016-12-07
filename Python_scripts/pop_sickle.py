# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 03:55:14 2016

@author: viveksagar

"""
import numpy as np
import matplotlib.pylab as plt
O3_t=np.empty([600,5])
num_col=35 

for ii in range(5):
    with np.load('Deep_data_simple'+str(ii+1)+'.npz') as data:
        O3_t[:,ii] = data['O3']
        
with np.load('Fim_data'+str(ii+1)+'.npz') as data2:
        X1_test = data2['X1']
        X2_test = data2['X2']
        fim[:,ii] = np.squeeze(np.asarray(np.sum(X2_test[:,:num_col]-X1_test[:,:num_col],axis=1)))
        duration[:,ii]=data2['duration']
        
O3_t=O3_t.reshape(3000)
fim = fim.reshape(3000)
duration = duration.reshape(3000)

def ghadha_ghoda(vec, week_l):
    week_l = week_l.astype(np.int64)
    ind = np.argsort(week_l)
    sorted_week = week_l[ind]
    sorted_diff = sorted_week-np.roll(sorted_week,1)
    start_ind = np.where(sorted_diff!=0)[0]
    sorted_diff2 = np.roll(sorted_week,-1)-sorted_week
    end_ind = np.where(sorted_diff2!=0)[0]
    x_axis = sorted_week[end_ind]
    sorted_data = vec[ind]
    mean_list = []
    stdv_list = []
    for ii in range(len(x_axis)):     
        if start_ind[ii]<end_ind[ii]:
            temp_dev = np.std(sorted_data[start_ind[ii]:end_ind[ii]])
            temp_mean=np.mean(sorted_data[start_ind[ii]:end_ind[ii]])
        else:
            temp_mean = sorted_data[start_ind[ii]]
            temp_dev=0
        mean_list.append(temp_mean)
        stdv_list.append(temp_dev)
    mean_list = np.asarray(mean_list)
    length = end_ind-start_ind
    stdv_list = np.asarray(stdv_list)/np.sqrt(length)

    return mean_list, stdv_list, x_axis, length

#duration = np.floor_divide(duration,7)
#all_duration = np.floor_divide(all_duration,7)

[mean_vec, stdv_vec, x_axis, length]=ghadha_ghoda(O3_t,duration)
[mean_vec2, stdv_vec2, x_axis2, length2]=ghadha_ghoda(fim,duration)
x_axis2=-x_axis2

plt.figure()
plt.errorbar(x_axis, mean_vec, yerr=stdv_vec)
#plt.plot(x_axis,np.log(length),'r')
plt.title("Deep-duration-score vs Duration")
plt.xlabel('Duration') 
plt.ylabel('Deep_score')
plt.xlim([0,15])
plt.ylim([-1,1])
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('deep_vs_dur.png')

plt.figure()
plt.errorbar(x_axis2, mean_vec2, yerr=stdv_vec2)
#plt.plot(x_axis2,np.log(length2),'r')
plt.title("DeltaFim-score vs Duration")
plt.xlabel('Duration')
plt.ylabel('Fim_score')
plt.xlim([0,-15])
fig2=plt.gcf()
fig2.set_size_inches(18.5,10.5)
fig2.savefig('Fim_vs_dur.png')

