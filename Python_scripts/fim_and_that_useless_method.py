# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:01:23 2016

Calculating FIM scores as well as clustering using K-means.

@author: viveksagar

"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

x1 = pd.ExcelFile("entry_data.xlsx")
entry_data = x1.parse("Sheet1")
x2 = pd.ExcelFile("exit_data.xlsx")
exit_data = x2.parse("Sheet1")

# Fim scores
entry_data['FIM']=entry_data.sum(1)
exit_data['FIM']=exit_data.sum(1)
diff = exit_data.FIM-entry_data.FIM
improved_number = len(diff.loc[diff>0])
percent_fim = improved_number*100/(len(entry_data))

# K-means
n_clust = 4
change = exit_data-entry_data
change = change.fillna(0)
X = np.array(change.drop(['FIM'], 1).astype(float))
Y = np.array(change['FIM'])
clf = KMeans(n_clusters=n_clust)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

# What does the label mean? This counts the number of negatives in the centroid.
centroid_label = []
for ii in range(n_clust):
    temp = centroids[ii,:]
    temp2 = len(np.where(temp<0)[0])
    centroid_label.append(temp2)

# A histogram of label values
predicted_labels = []
for ii in range(n_clust):
    temp = len(np.where(labels==ii)[0])
    predicted_labels.append(temp)

location_imp = np.where(np.array(centroid_label)<9)[0]    

imp = 0
for ii in range(len(location_imp)):
    imp=imp+predicted_labels[location_imp[ii]]
percent_km = imp*100/(len(entry_data))
