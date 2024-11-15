# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:11:29 2024

@author: tandeitnik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kn
from scipy.integrate import simps
from tqdm import tqdm

# tweezer parameters

wl = 1550e-9 # [m] tweezer wavelength
k0 = 2*np.pi/wl # [m^-1]
NA_tw = 0.8 # numerical aperture of the tweezing lens
f_tw = 1e-3 # [m] focal length of the tweezing lens
f_cls = np.linspace(1e-3,20e-3,5) # [m] focal lengths of the collimating lens to be tested
res = 100 # resolution of the arrays used in the script

# optical fiber parameters

n1 = 1.45636 # core index
n2 = 1.44399 # cladding index
NA_of = 0.13
a = 5e-6 # [m] core radius
l = 0 # fiber mode index l
m = 1 # fiber mode index m
V = k0*a*NA_of

'''
optical fiber mode calculation
'''

def solutionFinder(x, y1, y2): # this function finds the graphical solution to determine X
    # Calculate the difference between the functions
    diff = y1 - y2

    # Find indices where the sign changes from negative to positive
    indices_crossing = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0]

    # Linear interpolation to find exact points
    solutions = []
    for i in indices_crossing:
        x1, x2 = x[i], x[i+1]
        y1_diff, y2_diff = diff[i], diff[i+1]
        
        # Linear interpolation for the crossing point
        x_crossing = x1 - y1_diff * (x2 - x1) / (y2_diff - y1_diff)
        solutions.append((x_crossing))

    return np.array(solutions)

X = np.linspace(0+1e-6,V-1e-6,1000) # Tentative X values. I avoid the X = 0 and X = V points since they should be treated as a limit
Y = np.sqrt(V**2-X**2)

LHS = X*jv(l+1,X)/jv(l,X)
RHS = Y*kn(l+1,Y)/kn(l,Y)

solutions = solutionFinder(X, LHS, RHS)

# LHS_plot = [LHS[0]] # this array is just for plotting purposes
# X_plot = [X[0]] # this array is just for plotting purposes
# for i in range(len(LHS)-1):
#     if (LHS[i+1] - LHS[i]) < 0:
#         LHS_plot.append(np.nan)
#         X_plot.append(np.nan)
#     else:
#         LHS_plot.append(LHS[i+1])
#         X_plot.append(X[i+1])

# plt.plot(X_plot,LHS_plot,label = 'LHS')
# plt.plot(X,RHS,label = 'RHS')
# plt.xlim([0,V*1.1])
# plt.ylim([0,RHS[0]*1.5])
# plt.legend()

assert len(solutions) >= m, "there is no propagating mode for the selected m value"

Y = np.sqrt(V**2-solutions[m-1]**2)
kt = solutions[m-1]/a
gamma = Y/a

x = np.linspace(-2*a,2*a,res) # [m] x range in the focus plane of the collimating lens
y = np.linspace(-2*a,2*a,res) # [m] y range in the focus plane of the collimating lens

X, Y = np.meshgrid(x, y)

# Calculate the variables rho and phi
rho = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

# Initialize 'u' as a zero matrix
u_fiber = np.zeros_like(rho, dtype=complex)

# Conditions for the values of u
mask_inside = rho <= a  # Mask for values inside the radius 'a'
mask_outside = rho > a  # Mask for values outside the radius 'a'

# For values inside the radius 'a'
u_fiber[mask_inside] = (jv(l, kt * rho[mask_inside]) * np.exp(-1j * l * phi[mask_inside]) / jv(l, kt * a) +
                  jv(-l, kt * rho[mask_inside]) * np.exp(1j * l * phi[mask_inside]) / jv(-l, kt * a))

# For values outside the radius 'a'
u_fiber[mask_outside] = (kn(l, gamma * rho[mask_outside]) * np.exp(-1j * l * phi[mask_outside]) / kn(l, gamma * a) +
                   kn(-l, gamma * rho[mask_outside]) * np.exp(1j * l * phi[mask_outside]) / kn(-l, gamma * a))

u_fiber = u_fiber/np.sqrt(simps(simps(np.abs(u_fiber)**2,x),y))

# mod_u = np.abs(u_fiber)**2

# plt.figure(figsize=(6, 6))
# plt.imshow(mod_u, extent=[x[0]/a, x[-1]/a, y[0]/a, y[-1]/a], origin='lower', cmap='inferno', aspect='equal')
# plt.colorbar(label='normalized intensity')

# # Plotting the dashed circle
# circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', linewidth=2)
# plt.gca().add_artist(circle)

# # Adjusting the plot
# plt.xlabel('x [a]')
# plt.ylabel('y [a]')
# plt.show()

'''
point-spread function and overlap evaluation
'''

phi   = np.linspace(0,2*np.pi,res)
theta_tw = np.linspace(0,np.arcsin(NA_tw),res)

PHI, THETA_TW = np.meshgrid(phi,theta_tw)

G = [[],[],[]] # dyadic Green function of a dipole located at the origin in the far-field
a = np.exp(1j*k0*f_tw)/(4*np.pi*f_tw) 

G[0].append(a*(1-np.cos(PHI)**2*np.sin(THETA_TW)**2))
G[0].append(a*(-np.sin(PHI)*np.cos(PHI)*np.sin(THETA_TW)**2))
G[0].append(a*(-np.cos(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[1].append(a*(-np.sin(PHI)*np.cos(PHI)*np.sin(THETA_TW)**2))
G[1].append(a*(1-np.sin(PHI)**2*np.sin(THETA_TW)**2))
G[1].append(a*(-np.sin(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[2].append(a*(-np.cos(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[2].append(a*(-np.sin(PHI)*np.sin(THETA_TW)*np.cos(THETA_TW)))
G[2].append(a*(np.sin(THETA_TW)**2))

pol_vector = [1,0,0] # polarization vector of the dipole at the focus of the tweezing lens

Einf_tw = [] # the far-field of the dipole before being diffracted by the tweezing lens
Einf_tw.append(G[0][0]*pol_vector[0]+G[0][1]*pol_vector[1]+G[0][2]*pol_vector[2])
Einf_tw.append(G[1][0]*pol_vector[0]+G[1][1]*pol_vector[1]+G[1][2]*pol_vector[2])
Einf_tw.append(G[2][0]*pol_vector[0]+G[2][1]*pol_vector[1]+G[2][2]*pol_vector[2])

nph = [-np.sin(PHI),np.cos(PHI),0] # unit vector phi
nth_tw = [np.cos(THETA_TW)*np.cos(PHI),np.cos(THETA_TW)*np.sin(PHI),-np.sin(THETA_TW)] # unit vector theta for the tweezing lens reference sphere

Einf_tw_nph_tw = Einf_tw[0]*nph[0]+Einf_tw[1]*nph[1]+Einf_tw[2]*nph[2]
Einf_tw_nth_tw = Einf_tw[0]*nth_tw[0]+Einf_tw[1]*nth_tw[1]+Einf_tw[2]*nth_tw[2]

overlap = np.zeros(len(f_cls))

for k in tqdm(range(len(f_cls))):

    f_cl = f_cls[k]

    theta_cl = np.linspace(0,np.arcsin(NA_tw*f_tw/f_cl),res)
    
    PHI, THETA_CL = np.meshgrid(phi,theta_cl)
    
    nth_cl = [np.cos(THETA_CL)*np.cos(PHI),np.cos(THETA_CL)*np.sin(PHI),-np.sin(THETA_CL)] # unit vector theta for the tweezer reference sphere
    
    Einf_cl = []
    b = np.sqrt(np.cos(THETA_CL)/np.cos(THETA_TW))
    
    Einf_cl.append(b*(Einf_tw_nph_tw*nph[0]+Einf_tw_nth_tw*nth_cl[0]))
    Einf_cl.append(b*(Einf_tw_nph_tw*nph[1]+Einf_tw_nth_tw*nth_cl[1]))
    Einf_cl.append(b*(Einf_tw_nph_tw*nph[2]+Einf_tw_nth_tw*nth_cl[2]))
    
    E_xy = np.zeros([3,res,res]).astype(np.complex_)
    
    for i in range(res):
        
        for j in range(res):
    
            propagator_xy = np.exp(1j*k0*(np.sqrt(x[i]**2+y[j]**2)*np.sin(THETA_CL)*np.cos(PHI-np.arctan2(y[j],x[i]))))
            
            E_xy[0,j,i] = simps(simps(Einf_cl[0]*propagator_xy*np.sin(THETA_CL),theta_cl),phi)
            E_xy[1,j,i] = simps(simps(Einf_cl[1]*propagator_xy*np.sin(THETA_CL),theta_cl),phi)
            E_xy[2,j,i] = simps(simps(Einf_cl[2]*propagator_xy*np.sin(THETA_CL),theta_cl),phi)
       
    E_xy = E_xy/np.sqrt(simps(simps(np.abs(E_xy[0,:,:])**2 + np.abs(E_xy[1,:,:])**2 + np.abs(E_xy[2,:,:])**2,x),y))
    
    overlap[k] = np.abs(simps(simps(E_xy[0,:,:]*np.conj(u_fiber),x),y))

plt.plot(f_cls*1000,overlap)
