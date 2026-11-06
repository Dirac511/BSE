import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math
import os
from scipy.stats import norm,gaussian_kde
from scipy import integrate


c = ['C0','C1','C2', 'C3']

pi = math.pi

Nc = 8
U = 3

ds = [2.7,3.1]
Ts = [0.125]

fig, axes = plt.subplots(2, 1)

for iid in range(0,len(ds)):
    d = ds[iid]
    for iT in range(0,len(Ts)):
        T = Ts[iT]
        Y = []
        dataname = '../../d'+str(d)+'/bse/lambda_vs_K_T'+str(T)+'.txt'

        a = np.loadtxt(dataname,unpack=True)
        x = a[0,:]   ;y = a[1,:]
        x1 = x[:8150];y1 = y[:8150] 
    
        #mu = np.mean(y1);sigma = np.std(y1,ddof=1) 
        s = np.argsort(y1);y2 = y1[s] ## sort
        kde = gaussian_kde(y2,bw_method='silverman')
        test_points = np.linspace(y2.min(), y2.max(), 1000)
        kde_values = kde(test_points)
        dx = test_points[1] - test_points[0]
        integral = np.trapz(kde_values, test_points)
        print(integral)  ###check normalization

        Y = kde(y2)
        axes[0].plot(y2,Y,color=c[iid],label='$n=$'+str(d))
        axes[0].set_xlabel(r'$\lambda$',fontsize=22)
        axes[0].set_ylabel(r'$\rho(\lambda)$',fontsize=22)
        axes[0].tick_params(axis='both',labelsize=14) 
        axes[0].legend(loc='best', fontsize=16)
        axes[1].plot(x1,y1,color=c[iid])
        axes[1].set_xlabel(r'$n$',fontsize=22)
        axes[1].set_xlim(0,20)
        axes[1].set_ylabel(r'$\lambda_n$',fontsize=22)
        axes[1].tick_params(axis='both',labelsize=14)

    plt.tight_layout()    
    plt.savefig('compress_Nc' +'_T' + str(T) + '.pdf')

