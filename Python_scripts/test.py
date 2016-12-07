import numpy as np
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
    stdv_list = np.asarray(stdv_list)
    length = end_ind-start_ind
    return mean_list, stdv_list, x_axis, length


[mean_vec, stdv_vec, x_axis, length]=ghadha_ghoda(vec,week_l)

