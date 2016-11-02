# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:26:59 2016
Surrogate data.
@author: viveksagar
"""

import numpy as np
sz = (1000,21)
offset = 100
X1 = 9*offset/10*np.ones(sz)+offset/10*np.random.random(sz)
X2 = offset/2*np.ones(sz)+offset/2*np.random.random(sz)
