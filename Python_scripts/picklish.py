# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:56:39 2016

@author: viveksagar
"""
import pickle

# write python dict to a file
output = open('myfile.pkl', 'wb')
pickle.dump(v, output)
output.close()

pkl_file = open('myfile.pkl', 'rb')
v = pickle.load(pkl_file)
pkl_file.close()
