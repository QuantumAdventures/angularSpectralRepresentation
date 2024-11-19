import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simps
from scipy.special import hermite
from scipy.special import eval_genlaguerre

NA = 0.6 # lens numerical aperture
f0 =  1 # filling factor
wl = 1550e-9  # [m] wavelength
n_medium = 1
k = 2*np.pi/wl
res = 100
intRes = 100
P = 1 # [W] incident power

c = 3e8 # [m/s] speed of light in vacuum
e0 = 8.85e-12

basis = 'laguerre' #choose 'hermite' or 'laguerre' for the basis
indices = [[0,0]] #list the indices of the modes present in the beam
coef = np.array([1]) #Amplitude of the modes. Relative phases can be used and the vector don't need to be normalized.

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

# evaluating the power that got through the lens
u = np.linspace(0,1/f0,100000)
Pf = 4*P*simps(np.exp(-2*u)*u,u) #[W] the total power of the beam that got through the tweezing lens

# proper normalization constant
norm_xy = np.sqrt(2*Pf/c/e0)/np.sqrt(simps( simps( np.abs(E_xy[0,:,:])**2 + np.abs(E_xy[1,:,:])**2 + np.abs(E_xy[2,:,:])**2 ,x),y))

# properly normalized electric field in the XY plane for z = 0
E_xy[0,:,:] *= norm_xy
E_xy[1,:,:] *= norm_xy
E_xy[2,:,:] *= norm_xy

# properly normalized electric field in the XZ plane for y = 0
E_xz[0,:,:] *= norm_xy
E_xz[1,:,:] *= norm_xy
E_xz[2,:,:] *= norm_xy

# properly normalized electric field in the YZ plane for x = 0
E_yz[0,:,:] *= norm_xy
E_yz[1,:,:] *= norm_xy
E_yz[2,:,:] *= norm_xy

# intensities of the field at the XY, XZ, and YZ planes
I_xy = c*e0*(np.abs(E_xy[0,:,:])**2 + np.abs(E_xy[1,:,:])**2 + np.abs(E_xy[2,:,:])**2)/2
I_xz = c*e0*(np.abs(E_xz[0,:,:])**2 + np.abs(E_xz[1,:,:])**2 + np.abs(E_xz[2,:,:])**2)/2
I_yz = c*e0*(np.abs(E_yz[0,:,:])**2 + np.abs(E_yz[1,:,:])**2 + np.abs(E_yz[2,:,:])**2)/2

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    'font.size': 10
})

# plotting the intensity at the XY plane
plt.figure(figsize=(4, 3),constrained_layout=True)
plt.imshow(I_xy, extent=[x[0]/wl, x[-1]/wl, y[0]/wl, y[-1]/wl], origin='lower', cmap='inferno', aspect='equal')
plt.xlabel('x [$\lambda$]')
plt.ylabel('y [$\lambda$]')
plt.show()

# plotting the intensity at the XY plane along the x and y axis
plt.figure(figsize=(4, 3),constrained_layout=True)
plt.plot(x/wl,I_xy[int(res/2),:]/np.max(I_xy), label = 'x axis')
plt.plot(y/wl,I_xy[:,int(res/2)]/np.max(I_xy),'--' ,label = 'y axis')
plt.xlabel('distance [$\lambda$]')
plt.ylabel('normalized intensity [A.U.]')
plt.legend()

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
