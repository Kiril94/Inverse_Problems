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
from IPython.core.display import Latex
import sympy 
from sympy import latex
from sympy.interactive.printing import init_printing
mpl.rcParams['figure.dpi'] = 300
#sys.path.append('../Ass2')
basepath = os.path.abspath('')
A2_path = f"{basepath}/Ass2"
# In[Load and inspect data]
    
data = np.loadtxt(f'{A2_path}/mars_soil.txt')
v = data[:,0]
sigma = 0.03*1e4
counts= data[:,1]
threshold_l = 12.75e3
threshold_g = 12.65e3
fig, ax = plt.subplots()
ax.plot(v,counts)
plt.show
# In[Find derivatives]
funcl,Al,fl,cl,d = sympy.symbols("L,A_l,f_l,c_l,d")
funcg,Ag,fg,cg = sympy.symbols("G,A_g,f_g,c_g")
dldA,dldf,dldc = sympy.symbols("dLdA, dLdf, dLdc")
dgdA, dgdf,dgdc = sympy.symbols("dGdA,dGdf,dGdc")

# Define relation:
funcl = Al*cl**2/((d-fl)**2+cl**2)
funcg = Ag/(sympy.sqrt(2*sympy.pi)*cg)*sympy.exp(-(d-fg)**2/(2*cg**2))
#compute derivatives
dldA = funcl.diff(Al)
dldf = funcl.diff(fl)
dldc  = funcl.diff(cl)
dgdA = funcg.diff(Ag)
dgdf = funcg.diff(fg)
dgdc  = funcg.diff(cg)
#Printing
init_printing(forecolor = 'White')
display(sympy.Eq(sympy.symbols('dLdA'), dldA))
display(sympy.Eq(sympy.symbols('dLdf'), dldf))
display(sympy.Eq(sympy.symbols('dLdc'), dldc))

display(sympy.Eq(sympy.symbols('dGdA'), dgdA))
display(sympy.Eq(sympy.symbols('dGdf'), dgdf))
display(sympy.Eq(sympy.symbols('dGdc'), dgdc))

# In[define functions]
def Lorentz(m):
    A = m[0::3]
    f = m[1::3]
    c = m[2::3]
    g = np.empty(len(counts))
    for i in range(len(g)):
        g[i] = np.sum(A*c**2/((z[i]-f)**2+c**2))
    return threshold_l-g

def Gauss(m):
    A = m[0::3]
    f = m[1::3]
    c = m[2::3]
    g = np.empty(len(counts))
    for i in range(len(g)):
        g[i] = np.sum(A/(c*np.sqrt(2*np.pi))*np.exp(-(z[i]-f)**2/(2*c**2)))
    return threshold_g-g

def F_delta_m(G,m,dd):
    """Given delta_d and matrix with derivatives returns delta_m"""
    dd = np.reshape(dd, (len(dd),1))
    return (la.inv(G.T@G)@G.T@dd).flatten()  

# In[Try initial guess] 
#params stored as A1,f1,c1,A2,f2,c2...
m0_L = np.array([1.2e3,-10.4,.2,  4.3e3,-8.8,.1,  4.3e3,-7.55,.1,
               1e3,-6.95,.1,  2.5e3,-6.5,.1,  3e3,-5.5,.2,  3.7e3,-4.3,.1,
               1.2e3,-3.65,.1,  3.8e3,-3.2,.1,  1.7e3,-1.7,.1,  
               1.7e3,1.8,.1,  3.9e3,3.15,.1,  1.5e3,3.65,.1,  
               4.1e3,4.4,.1,  3.3e3,5.5,.1,  2.65e3,6.25,.1,  
               1.4e3,6.8,.2, 4.1e3,7.5,.1, 4.3e3,8.7,.1,  1.9e3,10.5,.15])
m0_G = np.array([.8e3,-10.4,.2,  2.1e3,-8.8,.2,  1.1e3,-7.55,.1,
               .35e3,-7,.1,  .8e3,-6.5,.1,  1.5e3,-5.5,.2,  1e3,-4.3,.1,
               .4e3,-3.8,.1,  1.8e3,-3.2,.2,  .7e3,-1.7,.18,  
               .35e3,1.8,.1,  .99e3,3.2,.1,  .45e3,3.7,.1,  
               1e3,4.4,.1,  .8e3,5.5,.1,  .65e3,6.2,.1,  
               .65e3,6.8,.2, 1.05e3,7.5,.1, 1.1e3,8.7,.1,  .8e3,10.5,.15])

fig, ax = plt.subplots()
ax.plot(v,counts)
#ax.plot(z, Lorentz(m0_L), label = 'Lorentz')
ax.plot(v, Gauss(m0_G), label = 'Gauss')
ax.axhline(threshold_g)
ax.legend()
plt.show()
# In[construct g matrix]
#Lorentzian case
index_array = np.arange(0,len(m0_L),3)
def G_mat_L(m,counts):
    """Construct matrix G containing the derivatives given model parameters 
    and data for the Lorentzian model."""
    G_L = np.empty((len(counts),len(m)))
    for i in index_array:
        A,f,c = m[i:i+3]
        dgdA = c**2/((counts-f)**2+c**2)
        dgdf = 2*A*c**2*(counts-f)/((counts-f)**2+c**2)**2
        dgdc = 2*A*c*(counts-f)**2/((counts-f)**2+c**2)**2
        G_L[:,i] = dgdA
        G_L[:,i+1] = dgdf
        G_L[:,i+2] =  dgdc
    return -G_L

def G_mat_G(m,counts):
    """Construct matrix G containing the derivatives given model parameters 
    and data for the Gaussian model."""
    G_G = np.empty((len(counts),len(m)))
    for i in index_array:
        A,f,c = m[i:i+3]
        c1 = A/np.sqrt(2*np.pi)
        c_exp = np.exp(-(counts-f)**2/(2*c**2))
        dgdA = c1/(A*c)*c_exp
        dgdf = c1*c_exp*(counts-f)/c**3
        dgdc = c1*c_exp/c**2*(-1+(counts-f)**2/c**2)
        G_G[:,i] = dgdA
        G_G[:,i+1]= dgdf
        G_G[:,i+2] = dgdc
    return -G_G
# In[Run optimization]

def run_iterations(G_func,forward_func,m0,tol,alpha):
    dm,n = 1,0
    misfit = []
    while (n<nmax) and (la.norm(dm)>tol):
        d0 = forward_func(m0)
        dd = counts-d0
        misfit.append(np.abs(np.mean(np.abs(dd))-sigma))
        G = G_func(m0,v)
        dm = F_delta_m(G, m0,dd)
        m = m0+alpha*dm
        m0 = m
        n+=1
    return m, misfit
# In[Optimize]
alpha, nmax, tol = 1e-3, 160, 1e-7
m0_L_f, misfit_L = run_iterations(G_mat_L,Lorentz,m0_L, tol,alpha)


fig, ax = plt.subplots()
ax.plot(z,counts, '.')
ax.plot(z, Lorentz(m0_L), label = 'Lorentz')
ax.plot(z, Lorentz(m0_L_f), label = 'Lorentz_fit')
ax.legend()
plt.show()
fig, ax = plt.subplots()
x = np.arange(len(misfit_L))
ax.plot(x,misfit_L)

# In[Gauss]
alpha, nmax, tol = 1e-2, 70, 1e-3
m_G_f, misfit_G = run_iterations(G_mat_G, Gauss, m0_G, tol, alpha)
fig, ax = plt.subplots()
ax.plot(z,counts, '.')

ax.plot(z, Gauss(m_G_f), label = 'Gauss_fit')
ax.legend()
plt.show()

fig, ax = plt.subplots()
x = np.arange(len(misfit_G))
ax.plot(x,misfit_G)