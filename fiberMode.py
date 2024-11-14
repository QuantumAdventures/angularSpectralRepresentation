# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:11:29 2024

@author: tandeitnik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kn

NA = 0.1
#a = 12.5e-6 # [m] core radius

n1 = 1.45636 # core index
n2 = 1.44399 # cladding index
wl = 1550e-9 # [m] wavelength
k = 2*np.pi/wl # [m^-1]
a = 10/(k*NA)
l = 3

beta_array = np.linspace(n2*k,n1*k,10000)

kt = np.sqrt(n1**2*k**2-beta_array**2)
gamma = np.sqrt(beta_array**2-n2**2*k**2)

X = kt*a
V = np.ones(len(X))*10
Y = np.sqrt(V**2-X**2)
#Y = gamma*a
#V = np.sqrt(X**2+Y**2)

plt.plot(X,X*jv(l+1,X)/jv(l,X))
plt.plot(X,Y*kn(l+1,Y)/kn(l,Y))

LHS = X*jv(l+1,X)/jv(l,X)
RHS = Y*kn(l+1,Y)/kn(l,Y)
plt.plot(np.abs(LHS-RHS)/np.max(np.abs(LHS-RHS)))

diff = np.abs(X*jv(1,X)/jv(0,X) - Y*kn(l+1,Y)/kn(l,Y))
beta = np.sqrt(n1**2*k**2-X[np.argmin(diff)]**2/a**2)

res = 300

x = np.linspace(-20*wl,20*wl,res) # [m] x range in the focus plane of the collimating lens
y = np.linspace(-20*wl,20*wl,res) # [m] y range in the focus plane of the collimating lens

u = np.zeros([res,res]).astype(np.complex_)

kt = np.sqrt(n1**2*k**2-beta**2)
gamma = np.sqrt(beta**2-n2**2*k**2)

for i in range(res): #rows

    for j in range(res): #columns
        
        rho = np.sqrt(x[j]**2+y[i]**2)
        phi = np.arctan2(y[i],x[j])
        A = (jv(l,kt*a)*np.exp(-1j*l*phi)+jv(-l,kt*a)*np.exp(1j*l*phi))/(kn(l,gamma*a)*np.exp(-1j*l*phi) + kn(-l,gamma*a)*np.exp(1j*l*phi))
        
        if rho <= a:
            u[i,j] = jv(l,kt*rho)*np.exp(-1j*l*phi)+jv(-l,kt*rho)*np.exp(1j*l*phi)
        else:
            u[i,j] = A*(kn(l,gamma*rho)*np.exp(-1j*l*phi) + kn(-l,gamma*rho)*np.exp(1j*l*phi))
            
I = np.abs(u)**2
plt.imshow(I,cmap = 'inferno')
plt.plot(I[:,150])

l = 1
a = 3
plt.plot(np.linspace(0,a,200),(jv(l,np.linspace(0,a,200)))**2)
plt.plot(np.linspace(a,2*a,200),(jv(l,a)*kn(l,np.linspace(a,2*a,200))/kn(l,a))**2)