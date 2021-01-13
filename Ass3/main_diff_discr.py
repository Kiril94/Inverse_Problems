# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 08:34:04 2020

@author: kiril klein
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.random as rand
from scipy import stats
from scipy import constants as c
from scipy import signal
import numpy.linalg as la
import pandas as pd
from numpy import newaxis as na
# In[Data]
G = c.G
#use SI
l_a,drho, delta = 3.421e3, -1.7e3, 1e-12
#data, assigning a grav. anomaly dg to every position dist
dist = np.array([535,749,963,1177,1391,1605,1819,
                     2033,2247,2461,2675,2889])#*1e-3
dg = -np.array([15,24,31.2,36.8,40.8,42.7,42.4,40.9,
                37.3,31.5,21.8, 12.8])*1e-5
Nd = len(dist) #number data points
Nm =20#number model params
N_disc = 5*Nm
sigma_m = 300
Cov_mi = np.eye(N_disc)/sigma_m**2
sigma_d = 1e-5
Cov_di = np.eye(Nd)/sigma_d**2
frac = int(np.floor(N_disc/Nm))
#construct Covariance matrix
xi_arr = np.linspace(0,l_a,N_disc)
dx = l_a/N_disc #discretize space
#create lengths and corresponding height vector
l_arr = np.linspace(0,l_a,Nm)

# In[Functions]
def f_forward(h):
    """Given model parameters h, returns vector of grav. anomalies."""
    dg_pred = np.empty(Nd)
    k = np.arange(N_disc)
    diff0 = (k*dx-dist[:,na])
    diff1 = ((k+1)*dx-dist[:,na])
    dg_pred =G*drho*np.sum(  diff0*np.log(((diff1**2+h**2)*diff0**2)/(delta+diff1**2 * (diff0**2+h**2))) 
        +dx*np.log((diff1**2+h**2)/(diff1**2+delta))
        +2*h * (np.arctan(diff1/(h+delta)) - np.arctan(diff0/ (h+delta))) ,axis =1)       

    return dg_pred

def f_h_preferred(i):
    """Approximation for h given dg in a point x_i"""
    return dg[i]/(2*np.pi*G*drho)

def f_loglikelihood(h,Ci=Cov_di):
    g = f_forward(h)
    deltag = (dg-g)[na].T
    return -1/2*deltag.T@Ci@deltag

def f_rho(h, Ci = Cov_mi):
    deltah = (h-h0)[na].T
    return -1/2*deltah.T@Ci@deltah

def f_exponent(h, Cmi = Cov_mi, Cdi=Cov_di):
    return f_rho(h,Cmi)+f_loglikelihood(h,Cdi)    
 
# In[preferred model h0]
h0 = np.empty(N_disc)
x_h = np.linspace(0,l_a,N_disc)
for i in range(Nm):
    start_ind = frac*i
    stop_ind = frac*(i+1)
    #print(start_ind, stop_ind)
    mid_ind = int((stop_ind+start_ind)/2)
    idx_near = np.abs(dist - x_h[mid_ind]).argmin()#find idx of closest value
    if i!=(Nm-1):
        h0[start_ind:stop_ind] = f_h_preferred(idx_near)
    else:
        h0[start_ind:] = f_h_preferred(idx_near)
# In[MCMC]
num_it = int(3e4)
num_acc = 0
step_size = 50
H = []
Hred = []
hi = h0
r_list = []
llikelihood = np.empty(num_it)
for i in range(num_it):
    if i%2000==0:
        print('iteration: {}'.format(i))
    exp_i = f_exponent(hi, Cov_mi, Cov_di)
    rand_num = rand.uniform(-step_size,step_size, Nm)
    ones_arr = np.ones((Nm, frac))
    rand_arr = rand_num[:,na]*ones_arr
    rand_arr = np.ndarray.flatten(rand_arr)
    
    n_diff = int(len(hi)-len(rand_arr))
    if n_diff==0:
        pass
    else:
        arr_append = np.ones(n_diff)*rand_arr[-1]
        rand_arr = np.append(rand_arr,arr_append)
    hp = hi+rand_arr
    exp_p = f_exponent(hp, Cov_mi, Cov_di)
    p_acc = np.exp(exp_p-exp_i)
    prob = rand.uniform(0,1)
    if p_acc>prob:
        h = hp
        num_acc +=1
    else:
        h = hi
    llikelihood[i] = f_loglikelihood(h)
    hi = np.copy(h)
    Hred.append(h[0:N_disc:frac])
    H.append(h)
H = np.array(H)
Hred = np.array(Hred)
acc_rate = num_acc/num_it
print(acc_rate)
print(np.std(llikelihood[1000:])/np.sqrt(Nd/2))
print((np.mean(llikelihood[1000:])-(-Nd/2)))

# In[Plot log likelihood]

fig,ax = plt.subplots()
ax.plot(np.arange(num_it), llikelihood,'.')

ax.axhline(-Nd/2)
ax.axhline(-Nd/2-np.sqrt(Nd/2), color = 'r')
ax.axhline(-Nd/2+np.sqrt(Nd/2), color = 'r')
ax.set_xlim(0,num_it)
ax.set_ylim(-40,10)
plt.show()
print(np.std(llikelihood[1000:])/np.sqrt(Nd/2))
print(np.mean(llikelihood[1000:])-(-Nd/2))
# In[Find best prediction]
Res = []
for i in range(len(H[1000:])):
    Res.append(np.sqrt(np.sum( (dg-f_forward(H[1000+i]))**2)))
Res = np.array(Res)
index_min = np.argmin(Res)
#print(index_min)
h_best = H[index_min+1000]


# In[Plot with unc]
fig, ax = plt.subplots(2,figsize = (18,12),gridspec_kw={'height_ratios': [3,1]})
res = dg-f_forward(h)
sigma = np.std(res)
ax[1].plot(dist, res, '.', color = 'r')
ax[1].axhline(y=sigma, color="b", zorder = 0)
ax[1].axhline(y=-sigma, color="b",zorder = 0)
trans = mpl.transforms.blended_transform_factory(
        ax[1].get_yticklabels()[0].get_transform(), ax[1].transData)
ax[1].text(0,sigma, r"$\sigma$", color="b", transform=trans, 
        ha="right", va="center", fontsize = 20)
ax[1].text(0,-sigma, r"$-\sigma$", color="b", transform=trans, 
        ha="right", va="center", fontsize = 20)
ax[1].set_ylabel('residual', fontsize = 30,labelpad = 20)

ax[0].errorbar(x = dist, y = dg, yerr = sigma_d,capsize = 2,linestyle = 'dotted', label = 'data')
ax[0].plot(dist, f_forward(h0),label = 'preferred')
ax[0].plot(dist, f_forward(h), label = 'final solution', color = 'g',linewidth = 2)
ax[0].plot(dist, f_forward(np.mean(H[1000:,:], axis = 0)), label = 'mean solution', color = 'blue',linewidth = 2)
ax[0].plot(dist, f_forward(h_best), label = 'best solution', color = 'red',linewidth = 2)
ax[0].set_ylabel('gravity anomaly [mGal]',fontsize = 30,labelpad = 12)
ax[1].set_xlabel('distance [m]',fontsize = 30,labelpad = 12)

ax[0].tick_params(axis = 'y',labelsize  = 25)
ax[1].tick_params(axis = 'both',labelsize  = 19)
ax[0].legend(fontsize = 23)
plt.subplots_adjust(hspace=0)
# In[Histogram]
#Take models for n>1600
n_rows = 5
if Nm<30:
    fig, ax = plt.subplots(n_rows, int(Nm/n_rows))
    for i, ax in enumerate(fig.axes):
        ax.hist(Hred[1600:,i],50)
else:
    fig, ax = plt.subplots(n_rows, 6)
    for i, ax in enumerate(fig.axes):
        ax.hist(Hred[1600:,i],30)
plt.show()
fig.savefig('prior_Nm20_Ndisc100.png')

# In[2d distribution]
num_rows = 5
if Nm<=30:
    fig, ax = plt.subplots(num_rows,int(Nm/num_rows), figsize = (18,10))
    for i in range(Nm-1):
        axis = ax.flatten()[i]
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set(xlabel=f"{i}", ylabel = f"{i+1}")
        axis.hist2d(Hred[1000:,i],Hred[1000:,i+1], 50)
else:
    fig, ax = plt.subplots(num_rows,6, figsize = (18,10))
    for i in range(30):
        axis = ax.flatten()[i]
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set(xlabel=f"{i}", ylabel = f"{i+1}")
        axis.hist2d(Hred[1000:,i],Hred[1000:,i+1], 50)
plt.subplots_adjust(wspace=.3, hspace=.3)
plt.show()
# In[h0]
fig, ax = plt.subplots()
if Nm<24:
    ax.plot(np.linspace(0,l_a,len(h0)),h0, label = 'h0')
    #ax.plot(np.linspace(0,l_a,len(h0)),H[-1,:], label = 'h_final')
    ax.plot(np.linspace(0,l_a,len(h0)),h_best, label = 'h_best')
    ax.plot(np.linspace(0,l_a,len(h0)),np.mean(H[1500:,:], axis =0 ), label = 'h_mean')
else:
    ax.scatter(np.linspace(0,l_a,len(h0)),h0,s = 3, label = 'h0')
    #ax.scatter(np.linspace(0,l_a,len(h0)),H[-1,:], label = 'h_final')
    ax.scatter(np.linspace(0,l_a,len(h0)),h_best, label = 'h_best', s=15)
    ax.scatter(np.linspace(0,l_a,len(h0)),np.mean(H[1500:,:], axis =0 ),s = 5, label = 'h_mean')
ax.legend()
# In[Autocorrelation]
fig, ax = plt.subplots(4,5)
for i, ax in enumerate(fig.axes):
    x = pd.Series(Hred[:,i])
    ax = pd.plotting.autocorrelation_plot(x, ax= ax)
    ax.set_xlabel(f"m_{i}")
    ax.set_ylabel("autocorr")    
    #ax.set_yscale('log')
    #ax.set_xlim(0,10000)
plt.subplots_adjust(wspace=.5, hspace=.8)
# In[Correlation between adj h]
"""
Corr_adj = []
for i in range(Nm-1):
    Corr_adj.append(np.corrcoef(Hred[1000:,i].T,Hred[1000:,i+1].T)[1,0])
Corr_adj = np.array(Corr_adj)
fig, ax = plt.subplots()
ax.plot(np.arange(Nm-1),Corr_adj)
#ax.plot(np.arange(Nm-1), Corr_adj)
"""
# In[MCMC]
"""
num_it = 5000
num_acc = 0
step_size = 50
H = []
hi = h0
r_list = []
llikelihood = np.empty(num_it)
for i in range(num_it):
    if i%500==0:
        print('iteration: {}'.format(i))
    #if i>500:
    #    r_list.append(dg-f_forward(hi))
        #print(r_list[i-501])
    #if i>1000:
    #    print(np.array(r_list).shape)
    #    Cov_di = la.inv(np.cov(np.array(r_list)))
    exp_i = f_exponent(hi, Cov_mi, Cov_di)
    rand_num = rand.uniform(-step_size,step_size)
    hp = np.copy(hi)
    new_ind = int(i%Nm)
    start_ind = frac*new_ind
    stop_ind = frac*(new_ind+1)
    #print(start_ind, stop_ind)
    if new_ind!=(Nm-1):
        hp[start_ind:stop_ind] = hi[start_ind:stop_ind]+rand_num
    else:
        hp[start_ind:] = hi[start_ind:]+rand_num
    exp_p = f_exponent(hp, Cov_mi, Cov_di)
    if exp_p>=exp_i:
        h = hp
        num_acc+=1
    else:
        p_acc = np.exp(exp_p-exp_i)
        prob = rand.uniform(0,1)
        if p_acc>prob:
            h = hp
            num_acc +=1
        else:
            h = hi
            
    llikelihood[i] = f_loglikelihood(h)
    hi = np.copy(h)
    H.append(h)
H = np.array(H)
acc_rate = num_acc/num_it
print(acc_rate)
"""
# In[]
"""num_it = 50000
num_acc = 0
step_size = 40
H = []
hi = h0
r_list = []
llikelihood = np.empty(num_it)
for i in range(num_it):
    if i%500==0:
        print('iteration: {}'.format(i))
    exp_i = f_exponent(hi, Cov_mi, Cov_di)
    rand_num = rand.uniform(-step_size,step_size, Nm)
    ones_arr = np.ones((Nm, frac))
    rand_arr = rand_num[:,np.newaxis]*ones_arr
    rand_arr = np.ndarray.flatten(rand_arr)
    
    n_diff = int(len(hi)-len(rand_arr))
    if n_diff==0:
        pass
    else:
        arr_append = np.ones(n_diff)*rand_arr[-1]
        rand_arr = np.append(rand_arr,arr_append)
    hp = hi+rand_arr
    exp_p = f_exponent(hp, Cov_mi, Cov_di)
    p_acc = np.exp(exp_p-exp_i)
    prob = rand.uniform(0,1)
    if p_acc>prob:
        h = hp
        num_acc +=1
    else:
        h = hi
    llikelihood[i] = f_loglikelihood(h)
    hi = np.copy(h)
    H.append(h)
H = np.array(H)
acc_rate = num_acc/num_it
print(acc_rate)
"""