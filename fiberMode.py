# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:11:29 2024

@author: tandeitnik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kn
from scipy.integrate import simps

n1 = 1.45636 # core index
n2 = 1.44399 # cladding index
wl = 1550e-9 # [m] wavelength
k = 2*np.pi/wl # [m^-1]
NA = 0.13
a = 5e-6 # [m] core radius
l = 0
m = 1
V = k*a*NA

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

LHS_plot = [LHS[0]] # this array is just for plotting purposes
X_plot = [X[0]] # this array is just for plotting purposes
for i in range(len(LHS)-1):
    if (LHS[i+1] - LHS[i]) < 0:
        LHS_plot.append(np.nan)
        X_plot.append(np.nan)
    else:
        LHS_plot.append(LHS[i+1])
        X_plot.append(X[i+1])

plt.plot(X_plot,LHS_plot,label = 'LHS')
plt.plot(X,RHS,label = 'RHS')
plt.xlim([0,V*1.1])
plt.ylim([0,RHS[0]*1.5])
plt.legend()

solutions = solutionFinder(X, LHS, RHS)

assert len(solutions) >= m, "there is no propagating mode for the selected m value"

Y = np.sqrt(V**2-solutions[m-1]**2)
kt = solutions[m-1]/a
gamma = Y/a

res = 500

x = np.linspace(-2*a,2*a,res) # [m] x range in the focus plane of the collimating lens
y = np.linspace(-2*a,2*a,res) # [m] y range in the focus plane of the collimating lens

X, Y = np.meshgrid(x, y)

# Calculate the variables rho and phi
rho = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)

# Initialize 'u' as a zero matrix
u = np.zeros_like(rho, dtype=complex)

# Conditions for the values of u
mask_inside = rho <= a  # Mask for values inside the radius 'a'
mask_outside = rho > a  # Mask for values outside the radius 'a'

# For values inside the radius 'a'
u[mask_inside] = (jv(l, kt * rho[mask_inside]) * np.exp(-1j * l * phi[mask_inside]) / jv(l, kt * a) +
                  jv(-l, kt * rho[mask_inside]) * np.exp(1j * l * phi[mask_inside]) / jv(-l, kt * a))

# For values outside the radius 'a'
u[mask_outside] = (kn(l, gamma * rho[mask_outside]) * np.exp(-1j * l * phi[mask_outside]) / kn(l, gamma * a) +
                   kn(-l, gamma * rho[mask_outside]) * np.exp(1j * l * phi[mask_outside]) / kn(-l, gamma * a))

u = u/np.sqrt(simps(simps(np.abs(u)**2,x),y))

mod_u = np.abs(u)**2

plt.figure(figsize=(6, 6))
plt.imshow(mod_u, extent=[x[0]/a, x[-1]/a, y[0]/a, y[-1]/a], origin='lower', cmap='inferno', aspect='equal')
plt.colorbar(label='normalized intensity')

# Plotting the dashed circle
circle = plt.Circle((0, 0), 1, color='white', fill=False, linestyle='--', linewidth=2)
plt.gca().add_artist(circle)

# Adjusting the plot
plt.xlabel('x [a]')
plt.ylabel('y [a]')
plt.show()
