# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:40:21 2020

@author: klein
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
# In[Load and inspect data]
    
data = np.loadtxt('mars_soil.txt')
z = data[:,0]
counts = data[:,1]
threshold = 12.8e3
fig, ax = plt.subplots()
ax.plot(z,counts)
plt.show

# In[define functions]
def Lorentz(m):
    A = m[0::3]
    f = m[1::3]
    c = m[2::3]
    g = np.empty(len(z))
    for i in range(len(g)):
        g[i] = np.sum(A*c**2/((z[i]-f)**2+c**2))
    return threshold-g

def Gauss(m):
    A = m[0::3]
    f = m[1::3]
    c = m[2::3]
    g = np.empty(len(z))
    for i in range(len(g)):
        g[i] = np.sum(A/(c*np.sqrt(2*np.pi))*np.exp(-(z[i]-f)**2/(2*c**2)))
    return threshold-g

def dG_dA(m):
    A = m[0::3]
    f = m[1::3]
    c = m[2::3]
    return 1/(np.sqrt(2*np.pi)*c)*np.exp(-())

# In[Try initial guess] 
m0_L = np.array([1.2e3,-10.4,.2,  4.3e3,-8.8,.1,  4.3e3,-7.5,.1,
               1e3,-7,.1,  2.5e3,-6.5,.1,  3e3,-5.5,.2,  3.7e3,-4.3,.1,
               1.2e3,-3.8,.1,  3.8e3,-3.2,.1,  1.7e3,-1.7,.1,  
               1.7e3,1.8,.1,  3.9e3,3.2,.1,  1.5e3,3.7,.1,  
               3.9e3,4.4,.1,  3.3e3,5.5,.1,  2.8e3,6.2,.1,  
               1.4e3,6.8,.2, 4.1e3,7.5,.1, 4.3e3,8.7,.1,  1.9e3,10.5,.15])
m0_G = np.array([.8e3,-10.4,.2,  1.3e3,-8.8,.1,  1.3e3,-7.5,.1,
               .4e3,-7,.1,  .8e3,-6.5,.1,  1.4e3,-5.5,.2,  1e3,-4.3,.1,
               .4e3,-3.8,.1,  1e3,-3.2,.1,  .4e3,-1.7,.1,  
               .5e3,1.8,.1,  1.1e3,3.2,.1,  .5e3,3.7,.1,  
               1.1e3,4.4,.1,  .9e3,5.5,.1,  .9e3,6.2,.1,  
               .8e3,6.8,.2, 1.2e3,7.5,.1, 1.3e3,8.7,.1,  .8e3,10.5,.15])


fig, ax = plt.subplots()
ax.plot(z,counts)
ax.plot(z, Lorentz(m0_L), label = 'Lorentz')
ax.plot(z, Gauss(m0_G), label = 'Gauss')
ax.legend()
plt.show()
