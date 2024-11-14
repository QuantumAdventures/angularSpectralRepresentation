import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simps
from scipy.special import hermite
from scipy.special import eval_genlaguerre

NA = 0.6 #lens numerical aperture
f0 =  0.6 #filling factor
wl = 1550e-9
n_medium = 1
k = 2*np.pi/wl
res = 200
intRes = 200

basis = 'laguerre' #choose 'hermite' or 'laguerre' for the basis
indices = [[0,2],[0,-2]] #list the indices of the modes present in the beam
coef = np.array([1,1]) #Amplitude of the modes. Relative phases can be used and the vector don't need to be normalized.

unit_vector = np.array([1,0,0])

x = np.linspace(-2*wl,2*wl,intRes)
y = np.linspace(-2*wl,2*wl,intRes)
z = np.linspace(-2*wl,2*wl,intRes)

phi = np.linspace(0,2*np.pi,res) #integration limit
theta = np.linspace(0,np.arcsin(NA/n_medium),res) #integration limit

PHI, THETA = np.meshgrid(phi,theta)

def hermite_inf(n,m,theta,phi,f0,NA,n_medium):
    
    H_n = hermite(n, monic = False)
    H_m = hermite(m, monic = False)
        
    amplitude = np.exp(-(np.sin(theta)/f0/NA*n_medium)**2)*H_m(np.sqrt(2)*np.cos(PHI)*np.sin(THETA)/f0/NA*n_medium)*H_n(np.sqrt(2)*np.sin(PHI)*np.sin(THETA)/f0/NA*n_medium)

    return amplitude

def laguerre_inf(p,l,theta,phi,f0,NA,n_medium):

    laguerre = eval_genlaguerre(p, abs(l), 2*(np.sin(theta)/f0/NA*n_medium)**2)
    
    amplitude = np.exp(-(np.sin(theta)/f0/NA*n_medium)**2)*(np.sqrt(2)*(np.sin(theta)/f0/NA*n_medium))**abs(l)*laguerre

    phase =  l*phi
        
    return amplitude*np.exp(1j*phase)

def beamConstructor(basis,indices,coef,theta,phi,f0,NA,n_medium):

    if basis == 'hermite':
            
        E = coef[0]*hermite_inf(indices[0][0],indices[0][1],theta,phi,f0,NA,n_medium)
        
        for mode in range(1,len(indices)):
            
            E += coef[mode]*hermite_inf(indices[mode][0],indices[mode][1],theta,phi,f0,NA,n_medium)
                
    elif basis == 'laguerre':
        
        E = coef[0]*laguerre_inf(indices[0][0],indices[0][1],theta,phi,f0,NA,n_medium)
        
        for mode in range(1,len(indices)):
            
            E += coef[mode]*laguerre_inf(indices[mode][0],indices[mode][1],theta,phi,f0,NA,n_medium)
            
    return E


E = beamConstructor(basis,indices,coef,THETA,PHI,f0,NA,n_medium)

E_inc = [unit_vector[0]*E,unit_vector[1]*E,unit_vector[2]*E]
E_inc_dot_nphi = -E_inc[0]*np.sin(PHI) + E_inc[1]*np.cos(PHI)
E_inc_dot_nrho = E_inc[0]*np.cos(PHI) + E_inc[1]*np.sin(PHI)

E_inf = [(E_inc_dot_nphi*-np.sin(PHI) + E_inc_dot_nrho*np.cos(PHI)*np.cos(THETA))*np.sqrt(np.cos(THETA)),
         (E_inc_dot_nphi*np.cos(PHI) + E_inc_dot_nrho*np.sin(PHI)*np.cos(THETA))*np.sqrt(np.cos(THETA)),
         -E_inc_dot_nrho*np.sin(THETA)*np.sqrt(np.cos(THETA))]

E_xy = np.zeros([3,intRes,intRes]).astype(np.complex_)
E_xz = np.zeros([3,intRes,intRes]).astype(np.complex_)
E_yz = np.zeros([3,intRes,intRes]).astype(np.complex_)

for i in tqdm(range(intRes)):
    
    for j in range(intRes):

        propagator_xy = np.exp(1j*k*(0*np.cos(THETA)+np.sqrt(x[i]**2+y[j]**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(y[j],x[i]))))
        propagator_xz = np.exp(1j*k*(z[j]*np.cos(THETA)+np.sqrt(x[i]**2+0**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(0,x[i]))))
        propagator_yz = np.exp(1j*k*(z[j]*np.cos(THETA)+np.sqrt(0**2+y[i]**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(y[i],0))))
        
        E_xy[0,j,i] = simps(simps(E_inf[0]*propagator_xy*np.sin(THETA),theta),phi)
        E_xy[1,j,i] = simps(simps(E_inf[1]*propagator_xy*np.sin(THETA),theta),phi)
        E_xy[2,j,i] = simps(simps(E_inf[2]*propagator_xy*np.sin(THETA),theta),phi)
        
        E_xz[0,j,i] = simps(simps(E_inf[0]*propagator_xz*np.sin(THETA),theta),phi)
        E_xz[1,j,i] = simps(simps(E_inf[1]*propagator_xz*np.sin(THETA),theta),phi)
        E_xz[2,j,i] = simps(simps(E_inf[2]*propagator_xz*np.sin(THETA),theta),phi)
        
        E_yz[0,j,i] = simps(simps(E_inf[0]*propagator_yz*np.sin(THETA),theta),phi)
        E_yz[1,j,i] = simps(simps(E_inf[1]*propagator_yz*np.sin(THETA),theta),phi)
        E_yz[2,j,i] = simps(simps(E_inf[2]*propagator_yz*np.sin(THETA),theta),phi)

I_xy = np.abs(E_xy[0,:,:])**2 + np.abs(E_xy[1,:,:])**2 + np.abs(E_xy[2,:,:])**2
I_xy = I_xy/simps(simps(I_xy,x),y)

I_xz = np.abs(E_xz[0,:,:])**2 + np.abs(E_xz[1,:,:])**2 + np.abs(E_xz[2,:,:])**2
I_xz = I_xz/simps(simps(I_xz,x),z)

I_yz = np.abs(E_yz[0,:,:])**2 + np.abs(E_yz[1,:,:])**2 + np.abs(E_yz[2,:,:])**2
I_yz = I_yz/simps(simps(I_yz,y),z)

plt.imshow(I_xy, cmap = 'jet')

I_x = I_xy[int(intRes/2),:]
I_x = I_x/np.max(I_x)

I_y = I_xy[:,int(intRes/2)]
I_y = I_y/np.max(I_y)

I_z = I_xz[:,int(intRes/2)]
I_z = I_z/np.max(I_z)

plt.plot(I_x)

window = 20

coeffs = np.polyfit(x[int(intRes/2)-window:int(intRes/2)+window], I_x[int(intRes/2)-window:int(intRes/2)+window], 4)
quartic_func = np.poly1d(coeffs)

# Plot
plt.plot(x,I_x)
plt.plot(x, quartic_func(x), color='red')
plt.xlabel('x')
plt.ylim([0,1.1])
plt.legend()
plt.show()


# X, Z = np.meshgrid(x,z)

# levels = np.sort( I_xz[int(intRes/2),int(intRes/2)::8] )

# # Plot do heatmap com as curvas de nível
# plt.figure(figsize=(8, 6))
# plt.contourf(X, Z, I_xz, cmap='gray_r', levels= levels)  # Heatmap com preenchimento

# # Adicionar as linhas das curvas de nível
# contours = plt.contour(X, Z, I_xz, colors='k', levels= levels)

# plt.xlabel("X")
# plt.ylabel("Z")
# plt.show()