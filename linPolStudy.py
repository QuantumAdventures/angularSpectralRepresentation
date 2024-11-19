import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simps
from scipy.special import hermite
from scipy.special import eval_genlaguerre

NA = 0.75 #lens numerical aperture
f0 =  0.9
wl = 1550e-9 # [m] wavelength
n_medium = 1
k0 = 2*np.pi/wl
res = 100
intRes = 100
P = 1 # [W] incident power

n_particle = 1.42 #https://microparticles.de/eigenschaften
radius = 156e-9/2 # [m]
rho_SiO2 = 1850 # [kg/m^3] https://microparticles.de/eigenschaften

kb = 1.38e-23 # [J/K] Boltzmann constant
T0 = 293 # [K] room temperature
c = 3e8 # [m/s] speed of light in vacuum
e0 = 8.85e-12
e_r = (n_particle/n_medium)**2
V = 4*np.pi*radius**3/3
alpha_cm = 3*V*e0*(e_r-1)/(e_r+2)
alpha_rad = alpha_cm/(1-((e_r-1)/(e_r+2))*((k0*radius)**2 + 2j/3*(k0*radius)**3))
mass_particle = (4/3)*np.pi*radius**3*rho_SiO2

basis = 'laguerre' #choose 'hermite' or 'laguerre' for the basis
indices = [[0,0]] #list the indices of the modes present in the beam
coef = np.array([1]) #Amplitude of the modes. Relative phases can be used and the vector don't need to be normalized.

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

f_x = np.zeros(51)
f_y = np.zeros(51)
f_z = np.zeros(51)
potDepth = np.zeros(51)

for k in tqdm(range(51)):

    unit_vector = np.array([1-0.02*k,0+0.02*k,0])
    unit_vector = unit_vector/np.sqrt(np.matmul(unit_vector,unit_vector))
    
    # incident beam electrical field before tweezing lens
    E = beamConstructor(basis,indices,coef,THETA,PHI,f0,NA,n_medium)
    E_inc = [unit_vector[0]*E,unit_vector[1]*E,unit_vector[2]*E]
    
    E_inc_dot_nphi = -E_inc[0]*np.sin(PHI) + E_inc[1]*np.cos(PHI) # incident field dot unit vector along phi
    E_inc_dot_nrho = E_inc[0]*np.cos(PHI) + E_inc[1]*np.sin(PHI)  # incident field dot unit vector along rho
    
    # incident beam electrical field after the tweezing lens
    E_inf = [(E_inc_dot_nphi*-np.sin(PHI) + E_inc_dot_nrho*np.cos(PHI)*np.cos(THETA))*np.sqrt(np.cos(THETA)),
             (E_inc_dot_nphi*np.cos(PHI) + E_inc_dot_nrho*np.sin(PHI)*np.cos(THETA))*np.sqrt(np.cos(THETA)),
             -E_inc_dot_nrho*np.sin(THETA)*np.sqrt(np.cos(THETA))]
    
    E_xy = np.zeros([3,intRes,intRes]).astype(np.complex_) # electric field in the XY plane for z = 0
    E_xz = np.zeros([3,intRes,intRes]).astype(np.complex_) # electric field in the XZ plane for y = 0
    E_yz = np.zeros([3,intRes,intRes]).astype(np.complex_) # electric field in the YZ plane for x = 0
    
    # evaluation of the propagation through the tweezing lens
    for i in range(intRes):
        
        for j in range(intRes):
    
            propagator_xy = np.exp(1j*k0*(0*np.cos(THETA)+np.sqrt(x[i]**2+y[j]**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(y[j],x[i]))))
            propagator_xz = np.exp(1j*k0*(z[j]*np.cos(THETA)+np.sqrt(x[i]**2+0**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(0,x[i]))))
            propagator_yz = np.exp(1j*k0*(z[j]*np.cos(THETA)+np.sqrt(0**2+y[i]**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(y[i],0))))
            
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
    
    # [J] potential energies for the gradient force
    U_x = np.real(alpha_rad)*(np.abs(E_xy[0,int(intRes/2),:])**2 + np.abs(E_xy[1,int(intRes/2),:])**2 + np.abs(E_xy[2,int(intRes/2),:])**2)/4
    U_y = np.real(alpha_rad)*(np.abs(E_xy[0,:,int(intRes/2)])**2 + np.abs(E_xy[1,:,int(intRes/2)])**2 + np.abs(E_xy[2,:,int(intRes/2)])**2)/4
    U_z = np.real(alpha_rad)*(np.abs(E_xz[0,:,int(intRes/2)])**2 + np.abs(E_xz[1,:,int(intRes/2)])**2 + np.abs(E_xz[2,:,int(intRes/2)])**2)/4
    
    # potential depth in units of kb*T0
    potDepth[k] = np.max(U_x/kb/T0)
    
    # [N] gradient force for each direction
    F_x = np.gradient(U_x,x)
    F_y = np.gradient(U_y,y)
    F_z = np.gradient(U_z,z)
    
    # [N/m] spring constant for each direction
    k_x = np.polyfit(x,F_x,20)[-2] 
    k_y = np.polyfit(y,F_y,20)[-2]
    k_z = np.polyfit(z,F_z,20)[-2]
    
    # [Hz] natural oscillating frequency for each direction
    f_x[k] = np.sqrt(-k_x/mass_particle)/(2*np.pi)
    f_y[k] = np.sqrt(-k_y/mass_particle)/(2*np.pi)
    f_z[k] = np.sqrt(-k_z/mass_particle)/(2*np.pi)
    
    
angle = np.linspace(0,1,51)    
    
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    'font.size': 10
})
    
# Create the figure and the primary axis
fig, ax1 = plt.subplots(figsize=(5, 4),constrained_layout=True)

# Configure the first axis (left)
ax1.set_xlabel('linear polarization angle $[\pi/2]$')  # Label for the X-axis
ax1.set_ylabel('frequency [kHz]', color='k')  # Label for the Y1 axis
ax1.plot(angle, f_x/1000, label='freq. x')  # Line for the first axis
ax1.plot(angle, f_y/1000, label='freq. y')  # Line for the first axis
ax1.tick_params(axis='y', labelcolor='k')  # Color of the Y1 ticks
plt.grid(alpha=0.4)
plt.legend()
