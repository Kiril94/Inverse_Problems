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
Nm =20 #number model params
N_disc = 120
frac = int(np.floor(N_disc/Nm))
#construct Covariance matrix
sigma_m = 300
Cov_mi = np.eye(N_disc)/sigma_m**2
sigma_d = 1e-5
Cov_di = np.eye(Nd)/sigma_d**2
xi_arr = np.linspace(0,l_a,N_disc)
dx = l_a/N_disc #discretize space
#create lengths and corresponding height vector
l_arr = np.linspace(0,l_a,Nm)

# In[Functions]
def f_forward(h):
    """Given model parameters h, returns vector of grav. anomalies."""
    dg_pred = np.empty(Nd)
    x_diff = xi_arr-dist[:,np.newaxis]
    dg_pred =G*drho*dx*np.sum((
        np.log((x_diff**2+h**2)/(x_diff**2+delta))),axis =1)       
    return dg_pred

def f_h_preferred(i):
    """Approximation for h given dg in a point x_i"""
    return dg[i]/(2*np.pi*G*drho)

def f_loglikelihood(h):
    g = f_forward(h)
    deltag = (dg-g)[np.newaxis].T
    return -1/2*deltag.T@Cov_di@deltag

def f_rho(h):
    deltah = (h-h0)[np.newaxis].T
    return -1/2*deltah.T@Cov_mi@deltah

def f_exponent(h): return f_rho(h)+f_loglikelihood(h)     
# In[preferred model h0]
h0 = np.empty(N_disc)
#separating the space in intervals of constant grav. anomaly
#dist_diff = (np.roll(dist,-1)-dist)/2
#dist_int = np.empty(len(dist)+1)
#dist_int[0] = 0
#dist_int[-1] = l_a
#dist_int[1:-1] = dist[:-1]+dist_diff[:-1]
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
num_it = 10000
num_acc = 0
step_size = 50
H = []
hi = h0
llikelihood = np.empty(num_it)
for i in range(num_it):
    if i%500==0:
        print('iteration: {}'.format(i))
    exp_i = f_exponent(hi)
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
    exp_p = f_exponent(hp)
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

# In[Plot log likelihood]

fig,ax = plt.subplots()
ax.plot(np.arange(num_it), llikelihood,'.')

ax.axhline(-Nd/2)
ax.axhline(-Nd/2-np.sqrt(Nd/2), color = 'r')
ax.axhline(-Nd/2+np.sqrt(Nd/2), color = 'r')
ax.set_xlim(0,num_it)
ax.set_ylim(-40,0)
plt.show()
print(np.std(llikelihood[1000:])/np.sqrt(Nd/2))
print(np.mean(llikelihood[1000:])-(-Nd/2))
# In[Plot prediction]
fig, ax= plt.subplots()
ax.plot(dist,dg, label = 'data')
ax.plot(dist,f_forward(h0),label =  'pred. h0')
ax.plot(dist, f_forward(H[-1]), label = 'pred final')
ax.legend()
plt.show()
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
ax[0].plot(dist, f_forward(h), label = 'solution', color = 'g',linewidth = 2)

ax[0].set_ylabel('gravity anomaly [mGal]',fontsize = 30,labelpad = 12)
ax[1].set_xlabel('distance [m]',fontsize = 30,labelpad = 12)

ax[0].tick_params(axis = 'y',labelsize  = 25)
ax[1].tick_params(axis = 'both',labelsize  = 19)
ax[0].legend(fontsize = 23)
plt.subplots_adjust(hspace=0)
# In[Histogram]
#Take models for n>1600
H = np.array(H)
fig, ax = plt.subplots(5, 3)
for i, ax in enumerate(fig.axes):
    ax.hist(H[1600:,i],30)
plt.show()

# In[2d distribution]
fig, ax = plt.subplots(2,int(Nm/2))
for i in range(Nm):
    axis = ax.flatten()[i]
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set(xlabel=f"{i}", ylabel = f"{i+1}")
    axis.hist2d(H[1000:,i],H[1000:,i+1], 20)
plt.subplots_adjust(wspace=.5, hspace=.3)

plt.show()
# In[h0]
fig, ax = plt.subplots()
ax.plot(np.linspace(0,l_a,len(h0)),h0, label = 'h0')
ax.plot(np.linspace(0,l_a,len(h0)),H[-1,:], label = 'h_final')
ax.plot(np.linspace(0,l_a,len(h0)),np.mean(H[1500:,:], axis =0 ), label = 'h_mean')

ax.legend()
# In[avg model]
fig, ax = plt.subplots()
ax.plot(np.linspace(0,l_a,len(h0)),h0, label = 'h0')
ax.plot(np.linspace(0,l_a,len(h0)),np.mean(H[1000:,:], axis =0 ), label = 'h_mean')

ax.legend()