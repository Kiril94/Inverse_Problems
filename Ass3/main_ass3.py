# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 08:34:04 2020

@author: kiril klein
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from IPython.core.display import Latex
import sympy 
from sympy import latex
from sympy.interactive.printing import init_printing
from scipy.stats import chi2
from scipy import constants as c
# In[Data]
G = c.G
#use SI
l_a,drho, delta = 3.42*1e3, -1.71e3, 1e-9 
#data, assigning a grav. anomaly dg to every position dist
dist = np.array([535,749,963,1177,1391,1605,1819,
                     2033,2247,2461,2675,2889])
dg = -np.array([15,24,31.2,36.8,40.8,42.7,42.4,40.9,
                37.3,31.5,21.8, 12.8])
# In[Functions]
def f_forward(h):
    """Given model parameters h, returns vector of grav. anomalies."""
    dg_pred = np.empty(Nd)
    for i in range(Nd):
        x_diff = x_arr-dist[i]
        dg_pred[i] = G*drho*np.sum(dx*np.log(((x_diff**2+h**2)/(x_diff**2+delta))))
    return dg_pred

def f_happrox(i):
    """Approximation for h given dg in a point x_i"""
    return dg[i]/(2*np.pi*G*drho)
# In[Initial guess h0]
Nd = len(dist) #data number
num_m = 6 #number model params
dx = l_a/(num_m-1) #discretize space
#create lengths and corresponding height vector
l_arr = np.linspace(0,l_a,num_m)
h0 = np.empty(num_m)
#separating the space in intervals of constant grav. anomaly
dist_diff = (np.roll(dist,-1)-dist)/2
dist_int = np.empty(len(dist)+1)
dist_int[0] = 0
dist_int[-1] = l_a
dist_int[1:-1] = dist[:-1]+dist_diff[:-1]
for i in range(len(dist)):
    mask = (l_arr>=dist_int[i]) & (l_arr<=dist_int[i+1])
    h0[mask] = f_happrox(i)
display(h0)
#display(f_happrox(np.arange(12)))
