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
from scipy.stats import chi2
mpl.rcParams['figure.dpi'] = 300
#sys.path.append('../Ass2')
basepath = os.path.abspath('')
A2_path = f"{basepath}/Ass2"
# In[Load and inspect data]
    
data = np.loadtxt(f'{A2_path}/mars_soil.txt')
v, counts = data[:,0], data[:,1]
sigma = 0.03*1e4
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
#display(sympy.Eq(sympy.symbols('dLdA'), dldA))
#display(sympy.Eq(sympy.symbols('dLdf'), dldf))
#display(sympy.Eq(sympy.symbols('dLdc'), dldc))

#display(sympy.Eq(sympy.symbols('dGdA'), dgdA))
#display(sympy.Eq(sympy.symbols('dGdf'), dgdf))
#display(sympy.Eq(sympy.symbols('dGdc'), dgdc))
display(sympy.Eq(sympy.symbols('L'),funcl))
display(latex(funcg))
# In[define functions]
def Lorentz(m):
    A = m[0:-1:3]
    f = m[1:-1:3]
    c = m[2:-1:3]
    g = np.empty(len(counts))
    for i in range(len(g)):
        g[i] = np.sum(A*c**2/((v[i]-f)**2+c**2))
    return m[-1]-g

def Gauss(m):
    A = m[0:-1:3]
    f = m[1:-1:3]
    c = m[2:-1:3]
    g = np.empty(len(counts))
    for i in range(len(g)):
        g[i] = np.sum(A/(c*np.sqrt(2*np.pi))*np.exp(-(v[i]-f)**2/(2*c**2)))
    return m[-1]-g

def F_delta_m(G,m,dd):
    """Given delta_d and matrix with derivatives returns delta_m"""
    dd = np.reshape(dd, (len(dd),1))
    return (la.inv(G.T@G)@G.T@dd).flatten()  

# In[Try initial guess] 
#params stored as A1,f1,c1,A2,f2,c2...
m0_L = np.array([1.2e3,-10.4,.2,  4.3e3,-8.8,.1,  4.3e3,-7.55,.1,
               1e3,-6.95,.1,  2.5e3,-6.5,.1,  3e3,-5.5,.2,  3.7e3,-4.3,.1,
               1.2e3,-3.65,.1,  3.8e3,-3.2,.1,  1.7e3,-1.7,.1,  
               1.7e3,1.8,.12, .5e3,2.4,.05,  3.9e3,3.15,.12,  1.5e3,3.65,.1,  
               4.1e3,4.4,.1,  3.3e3,5.5,.1,  2.65e3,6.25,.1,  
               1.4e3,6.8,.2, 4.1e3,7.5,.1, 4.3e3,8.7,.1,  1.9e3,10.5,.15,
               12.75e3])
m0_G = np.array([.8e3,-10.4,.2,  2.1e3,-8.8,.2,  1.1e3,-7.55,.1,
               .35e3,-7,.1,  .8e3,-6.5,.1,  1.5e3,-5.5,.2,  1e3,-4.3,.1,
               .4e3,-3.8,.1,  1.8e3,-3.2,.2,  .7e3,-1.7,.18,  
               .35e3,1.8,.1,  .1e3,2.4,.05,  .99e3,3.2,.1,  .45e3,3.7,.1,  
               1e3,4.4,.1,  .8e3,5.5,.1,  .65e3,6.2,.1,  
               .65e3,6.8,.2, 1.05e3,7.5,.1, 1.1e3,8.7,.1,  .8e3,10.5,.15,
               12.65e3])

fig, ax = plt.subplots()
ax.plot(v,counts)
ax.plot(v, Gauss(m0_G), label = 'Gauss')
ax.legend()
plt.show()
# In[construct g matrix]
#Lorentzian case
index_array = np.arange(0,len(m0_L)-1,3)
def G_mat_L(m,counts):
    """Construct matrix G containing the derivatives given model parameters 
    and data for the Lorentzian model."""
    G_L = np.empty((len(counts),len(m)))
    for i in index_array:
        A,f,c = m[i:i+3]
        dgdA = c**2/((counts-f)**2+c**2)
        dgdf = 2*A*c**2*(counts-f)/((counts-f)**2+c**2)**2
        dgdc = 2*A*c*(counts-f)**2/((counts-f)**2+c**2)**2
        G_L[:,i] = -dgdA
        G_L[:,i+1] = -dgdf
        G_L[:,i+2] =  -dgdc
    G_L[:,-1] = 1
    return G_L

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
        G_G[:,i] = -dgdA
        G_G[:,i+1]= -dgdf
        G_G[:,i+2] = -dgdc
    G_G[:,-1] = 1
    return G_G

def f_misfit(dd): return np.mean(np.abs(dd))
def f_Chi2(dd,sigma): return 1/sigma**2*np.sum(dd**2)
# In[Run optimization]

def run_iterations(G_func,forward_func,m0,tol,alpha):
    dm,n =100 ,0
    misfit = []
    dm_mean = []
    while (n<nmax) and (np.mean(np.abs(dm))>tol):
        d0 = forward_func(m0)
        dd = counts-d0
        misfit.append(f_misfit(dd))
        G = G_func(m0,v)
        dm = F_delta_m(G, m0,dd)
        dm_mean.append(np.mean(np.abs(dm)))
        m = m0+alpha*dm
        m0 = m
        n+=1
    return m, misfit,dm_mean
# In[Optimize]
alpha_l, nmax, tol_l = 3e-1, 1200, 1
m_L_f, misfit_L,dm_mean_l = run_iterations(G_mat_L,Lorentz,m0_L, tol_l,alpha_l)
dd_L = counts-Lorentz(m_L_f)
fig, ax = plt.subplots()
ax.plot(v,counts, '.')
ax.plot(v, Lorentz(m0_L), label = 'Lorentz')
ax.plot(v, Lorentz(m_L_f), label = 'Lorentz_fit')
ax.legend()
plt.show()



# In[Gauss]
alpha_g, nmax, tol_g = 3e-1,1200, 1
m_G_f, misfit_G, dm_mean_g= run_iterations(G_mat_G, Gauss, m0_G, tol_g, alpha_g)
dd_G = counts-Gauss(m_G_f)
fig, ax = plt.subplots()
ax.plot(v,counts, '.')
ax.plot(v, Gauss(m0_G), label = 'Gauss')
ax.plot(v, Gauss(m_G_f), label = 'Gauss_fit')
ax.legend()
plt.show()

# In[]
import matplotlib.ticker as ticker
fig, ax = plt.subplots(1,2, figsize = (19,7))
x_g = np.arange(len(misfit_G))
x_l = np.arange(len(misfit_L))
ax2 = ax[0].twinx()
ax3 = ax[1].twinx()
ax2.plot(x_l, dm_mean_l, 'g-', linewidth = 4)
ax3.plot(x_g, dm_mean_g, 'g-',linewidth = 4)
text0 = 'mean(abs('
text0 += r'$\Delta \mathbf{m}$'
text0+= '))'
ax[0].set_xlabel('Iteration', fontsize = 35)
ax[1].set_xlabel('Iteration', fontsize = 35)
ax2.set_ylabel(text0, color='g', fontsize = 35,labelpad = 14)
ax3.set_ylabel(text0, color='g', fontsize = 35,labelpad = 14)

ax[1].yaxis.set_major_locator(ticker.MultipleLocator(50))
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))

ax[0].set_title('Lorentzian',fontsize = 35)
ax[1].set_title('Gaussian',fontsize = 35)
ax[1].plot(x_g,misfit_G,'-', color = 'r',linewidth = 4)
ax[0].plot(x_l,misfit_L,'-', color = 'r',linewidth = 4)
ax[0].set_ylabel('misfit', fontsize = 35, color = 'r',labelpad = 15)
ax[1].set_ylabel('misfit', fontsize = 35, color = 'r',labelpad = 15)
ax[0].tick_params(axis = 'both',labelsize  = 30)
ax[1].tick_params(axis = 'both',labelsize  = 30)
ax2.tick_params(axis = 'both',labelsize  = 30)
ax3.tick_params(axis = 'both',labelsize  = 30)
plt.subplots_adjust(wspace=.7)
#plt.tight_layout()
# In[Chi2]
chi2_l = f_Chi2(dd_L, sigma)
chi2_g = f_Chi2(dd_G, sigma)
prob_l = chi2.sf(chi2_l, len(counts)-len(m_L_f))
prob_g = chi2.sf(chi2_g, len(counts)-len(m_G_f))
print(chi2_l, prob_l)
print(chi2_g,prob_g)
print(np.sqrt(la.norm(dd_G)), np.sqrt(la.norm(dd_L)))
print(np.abs(np.sqrt(la.norm(dd_G)/len(counts))-sigma))
# In[Plot]
fig, ax = plt.subplots(4,figsize = (18,12),gridspec_kw={'height_ratios': [2.5, 1,2.5,1]})
res_l = counts-Lorentz(m_L_f)
res_g = counts-Gauss(m_G_f)

#text_l = r'$\alpha = $'
#text_l += f"{alpha_l}"
ax[1].plot(v, res_l, '.', color = 'r')

for a in ax[1:4:2]:
    a.axhline(y=sigma, color="b", zorder = 0)
    a.axhline(y=-sigma, color="b",zorder = 0)
    trans = mpl.transforms.blended_transform_factory(
        a.get_yticklabels()[0].get_transform(), a.transData)
    a.text(0,sigma, r"$\sigma$", color="b", transform=trans, 
        ha="right", va="center", fontsize = 20)
    a.text(0,-sigma, r"$-\sigma$", color="b", transform=trans, 
        ha="right", va="center", fontsize = 20)
ax[1].set_ylabel('residual', fontsize = 30,labelpad = 20)
ax[3].set_ylabel('residual', fontsize = 30,labelpad = 20)
#ax[1].legend(fontsize = 18, loc = 8)
ax[3].plot(v, res_g, '.', color = 'r')
ax[0].plot(v, counts, '.', label = 'data')
ax[0].plot(v, Lorentz(m0_L),'-.', label = 'initial guess', linewidth = 1.5)
ax[0].plot(v, Lorentz(m_L_f), label = 'solution', color = 'g',linewidth = 2)
#ax[0].text(-1.5,10000,text_l, fontsize = 25)
ax[0].text(-12,8500,r'$\mathbf{a)}$', fontsize = 30)
ax[2].text(-12,8500,r'$\mathbf{c)}$', fontsize = 30)
ax[2].plot(v, counts, '.', label = 'data')
ax[2].plot(v, Gauss(m0_G),'-.',label = 'initial guess', linewidth = 1.5)
ax[2].plot(v, Gauss(m_G_f), label = 'solution', color = 'g',linewidth = 2)
ax[1].text(-12,-500,r'$\mathbf{b)}$', fontsize = 25,zorder = 20)
ax[3].text(-12,-550,r'$\mathbf{d)}$', fontsize = 25,zorder = 20)

#ax[0].yaxis.set_major_locator(ticker.MultipleLocator(1000))
#ax[1].xaxis.set_major_locator(ticker.MultipleLocator(5))
#ax[2].yaxis.set_major_locator(ticker.MultipleLocator(2000))

ax[3].set_xlabel('v (mm/s)',fontsize = 30)
#ax[1,1].set_xlabel('v (mm/s)',fontsize = 20)
ax[0].set_ylabel('counts',fontsize = 30,labelpad = 12)
ax[2].set_ylabel('counts', fontsize = 30,labelpad = 12)
for a in ax:
    a.tick_params(axis = 'y',labelsize  = 25)
ax[3].tick_params(axis = 'both',labelsize  = 25)
#ax[1,0].tick_params(axis = 'both',labelsize  = 19)
ax[0].legend(fontsize = 23)
#ax[2].legend(fontsize = 18)
#ax[0,1].legend(fontsize = 15)
#ax[1,0].legend(fontsize = 15)
plt.subplots_adjust(hspace=0)