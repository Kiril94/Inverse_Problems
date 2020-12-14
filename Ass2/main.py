# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 22:40:21 2020

@author: klein
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['figure.dpi'] = 300
#sys.path.append('../Ass2')
basepath = os.path.abspath('')
A2_path = f"{basepath}/Ass2"
# In[Load and inspect data]
    
data = np.loadtxt(f'{A2_path}/mars_soil.txt')
z = data[:,0]
sigma = 0.03
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

def F_delta_m(G,m,dz,eps):
    """Given delta_d and matrix with derivatives returns delta_m"""
    dz = np.reshape(dz, (len(dz),1))
    return (la.inv(G.T@G+eps**2*np.eye(len(m)))@G.T@dz).flatten()  

# In[Try initial guess] 
#params stored as A1,f1,c1,A2,f2,c2...
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
#ax.plot(z, Gauss(m0_G), label = 'Gauss')
ax.legend()
plt.show()
# In[construct g matrix]
#Lorentzian case
index_array = np.arange(0,len(m0_L),3)
def G_mat_L(m,z):
    """Construct matrix G containing the derivatives given model parameters 
    and data for the Lorentzian model."""
    G_L = np.empty((len(z),len(m)))
    for i in index_array:
        A,f,c = m[i:i+3]
        dgdA = c**2/((z-f)**2+c**2)
        dgdf = 2*A*c**2*(z-f)/((z-f)**2+c**2)**2
        dgdc = 2*A*c*(z-f)**2/((z-f)**2+c**2)
        G_L[:,[i,i+1,i+2]] = np.array([dgdA, dgdf, dgdc]).T
    return G_L

def G_mat_G(m,z):
    """Construct matrix G containing the derivatives given model parameters 
    and data for the Gaussian model."""
    G_G = np.empty((len(z),len(m)))
    for i in index_array:
        A,f,c = m[i:i+3]
        c1 = A/np.sqrt(2*np.pi)
        c_exp = np.exp(-(z-f)**2/(2*c**2))
        dgdA = c1/(A*c)*c_exp
        dgdf = c1*c_exp*(z-f)/c**2
        dgdc = c1*c_exp/c**2*(-1+1/c**2*(z-f)**2)
        G_G[:,[i,i+1,i+2]] = np.array([dgdA, dgdf, dgdc]).T
    return G_G


# In[Run optimization]
#run search for m
alpha, eps, nmax, tol = 2e-8, 8e-5, 400, 1e-3
#alpha, eps, nmax, tol = 1e-8, 5e-4, 400, 1e-3
#dont change alpha and epsilon too much!!

def run_iterations(G_func,forward_func,m0,tol,eps,alpha):
    dm,n = 1,0
    misfit = []
    while (n<nmax) and (la.norm(dm)>tol):
        z0 = forward_func(m0)
        dz = z-z0
        misfit.append(np.sqrt(la.norm(dz)))
        G = G_func(m0,z)
        dm = F_delta_m(G, m0,dz,eps)
        m = m0+alpha*dm
        m0 = m
        n+=1
    return m, misfit

m0_L_f, misfit_L = run_iterations(G_mat_L,Lorentz,m0_L, tol,eps, alpha)
# In[Test result]
fig, ax = plt.subplots()
ax.plot(z,counts)
ax.plot(z, Lorentz(m0_L), label = 'Lorentz')
ax.plot(z, Lorentz(m0_L_f), label = 'Lorentz_fit')
ax.legend()
plt.show()
fig, ax = plt.subplots()
x = np.arange(len(misfit_L))
ax.plot(x,misfit_L)
# In[plot]
