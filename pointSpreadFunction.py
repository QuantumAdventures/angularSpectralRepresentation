# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:22:53 2024

@author: tandeitnik
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simps

NA_tw = 0.8 # numerical aperture of the tweezing lens
f_tw = 1e-3 # [m] focal length of the tweezing lens
f_cl = 5e-3 # [m] focal length of the collimating lens
wl = 1550e-9 # [m] tweezer wavelength
k = 2*np.pi/wl # [m^-1]
res = 100 # resolution of the arrays used in the script

x = np.linspace(-10*wl,10*wl,res) # [m] x range in the focus plane of the collimating lens
y = np.linspace(-10*wl,10*wl,res) # [m] y range in the focus plane of the collimating lens

pol_vector = [1,0,0] # polarization vector of the dipole at the focus of the tweezing lens

phi   = np.linspace(0,2*np.pi,res)
theta_tw = np.linspace(0,np.arcsin(NA_tw),res)
theta_cl = np.linspace(0,np.arcsin(NA_tw*f_tw/f_cl),res)

PHI, THETA_TW = np.meshgrid(phi,theta_tw)
PHI, THETA_CL = np.meshgrid(phi,theta_cl)

G = [[],[],[]] # dyadic Green function of a dipole located at the origin in the far-field
a = np.exp(1j*k*f_tw)/(4*np.pi*f_tw) 

G[0].append(a*(1-np.cos(PHI)**2*np.sin(THETA_TW)**2))
G[0].append(a*(-np.sin(PHI)*np.cos(PHI)*np.sin(THETA_TW)**2))
G[0].append(a*(-np.cos(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[1].append(a*(-np.sin(PHI)*np.cos(PHI)*np.sin(THETA_TW)**2))
G[1].append(a*(1-np.sin(PHI)**2*np.sin(THETA_TW)**2))
G[1].append(a*(-np.sin(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[2].append(a*(-np.cos(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[2].append(a*(-np.sin(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[2].append(a*(np.sin(THETA_TW)**2))

Einf_tw = [] # the far-field of the dipole before being diffracted by the tweezing lens
Einf_tw.append(G[0][0]*pol_vector[0]+G[0][1]*pol_vector[1]+G[0][2]*pol_vector[2])
Einf_tw.append(G[1][0]*pol_vector[0]+G[1][1]*pol_vector[1]+G[1][2]*pol_vector[2])
Einf_tw.append(G[2][0]*pol_vector[0]+G[2][1]*pol_vector[1]+G[2][2]*pol_vector[2])

nph = [-np.sin(PHI),np.cos(PHI),0] # unit vector phi
nth_tw = [np.cos(THETA_TW)*np.cos(PHI),np.cos(THETA_TW)*np.sin(PHI),-np.sin(THETA_TW)] # unit vector theta for the tweezing lens reference sphere
nth_cl = [np.cos(THETA_CL)*np.cos(PHI),np.cos(THETA_CL)*np.sin(PHI),-np.sin(THETA_CL)] # unit vector theta for the tweezer reference sphere

Einf_tw_nph_tw = Einf_tw[0]*nph[0]+Einf_tw[1]*nph[1]+Einf_tw[2]*nph[2]
Einf_tw_nth_tw = Einf_tw[0]*nth_tw[0]+Einf_tw[1]*nth_tw[1]+Einf_tw[2]*nth_tw[2]

Einf_cl = []
b = np.sqrt(np.cos(THETA_CL)/np.cos(THETA_TW))

Einf_cl.append(b*(Einf_tw_nph_tw*nph[0]+Einf_tw_nth_tw*nth_cl[0]))
Einf_cl.append(b*(Einf_tw_nph_tw*nph[1]+Einf_tw_nth_tw*nth_cl[1]))
Einf_cl.append(b*(Einf_tw_nph_tw*nph[2]+Einf_tw_nth_tw*nth_cl[2]))

E_xy = np.zeros([3,res,res]).astype(np.complex_)

for i in tqdm(range(res)):
    
    for j in range(res):

        propagator_xy = np.exp(1j*k*(np.sqrt(x[i]**2+y[j]**2)*np.sin(THETA_CL)*np.cos(PHI-np.arctan2(y[j],x[i]))))
        
        E_xy[0,j,i] = simps(simps(Einf_cl[0]*propagator_xy*np.sin(THETA_CL),theta_cl),phi)
        E_xy[1,j,i] = simps(simps(Einf_cl[1]*propagator_xy*np.sin(THETA_CL),theta_cl),phi)
        E_xy[2,j,i] = simps(simps(Einf_cl[2]*propagator_xy*np.sin(THETA_CL),theta_cl),phi)
       

I_xy = np.abs(E_xy[0,:,:])**2 + np.abs(E_xy[1,:,:])**2 + np.abs(E_xy[2,:,:])**2

plt.imshow(I_xy, cmap = 'jet')