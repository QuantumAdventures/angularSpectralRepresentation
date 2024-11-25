import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import simps
from scipy.special import hermite
from scipy.special import eval_genlaguerre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

NA = 0.75 # lens numerical aperture
f = 0.53e-3 # [m] lens focus
f0 = 0.9 # filling factor
wl = 1550e-9  # [m] wavelength
n1 = 1 # refractive index of the medium before the lens
n2 = 1 # refractive index of the medium after the lens
w0 = NA*f*f0/n2 # [m] waist of the incident beam
k = 2*np.pi/wl # modulus of the wavevector
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
z = np.linspace(-4*wl,4*wl,intRes)

n_particle = 1.42 #https://microparticles.de/eigenschaften
radius = 156e-9/2 # [m]
rho_SiO2 = 1850 # [kg/m^3] https://microparticles.de/eigenschaften

kb = 1.38e-23 # [J/K] Boltzmann constant
T0 = 293 # [K] room temperature
c = 3e8 # [m/s] speed of light in vacuum
e0 = 8.85e-12
e_r = (n_particle/n2)**2
V = 4*np.pi*radius**3/3
alpha_cm = 3*V*e0*(e_r-1)/(e_r+2)
alpha_rad = alpha_cm/(1-((e_r-1)/(e_r+2))*((k*radius)**2 + 2j/3*(k*radius)**3))
mass_particle = (4/3)*np.pi*radius**3*rho_SiO2

phi = np.linspace(0,2*np.pi,res) #integration limit
theta = np.linspace(0,np.arcsin(NA/n2),res) #integration limit

PHI, THETA = np.meshgrid(phi,theta)

def hermite_inf(n,m,theta,phi):
    
    H_n = hermite(n, monic = False)
    H_m = hermite(m, monic = False)
    x = f*np.cos(phi)*np.sin(theta)
    y = f*np.sin(phi)*np.sin(theta)
    z = f*np.cos(theta)
    r = np.sqrt(x**2+y**2)
    zr = np.pi*w0**2/wl
    w = w0*np.sqrt(1+(z/zr)**2)
    
    E = (1/w)*np.exp(-r**2/w**2)*H_m(np.sqrt(2)*x/w)*H_n(np.sqrt(2)*y/w)
    
    return E

def laguerre_inf(p,l,theta,phi):

    x = f*np.cos(phi)*np.sin(theta)
    y = f*np.sin(phi)*np.sin(theta)
    z = f*np.cos(theta)
    r = np.sqrt(x**2+y**2)
    zr = np.pi*w0**2/wl
    w = w0*np.sqrt(1+(z/zr)**2)
    
    laguerre = eval_genlaguerre(p, abs(l), 2*r**2/w**2)
    C = np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))/w
    
    E = C*(np.sqrt(2)*r/w)**np.abs(l)*np.exp(-r**2/w**2)*laguerre*np.exp(1j*l*phi)
  
    return E

def beamConstructor(basis,indices,coef,theta,phi):
    
    if basis == 'hermite':
            
        E = coef[0]*hermite_inf(indices[0][0],indices[0][1],theta,phi)
        
        for mode in range(1,len(indices)):
            
            E += coef[mode]*hermite_inf(indices[mode][0],indices[mode][1],theta,phi)
            
    elif basis == 'laguerre':
        
        E = coef[0]*laguerre_inf(indices[0][0],indices[0][1],theta,phi)
        
        for mode in range(1,len(indices)):
            
            E += coef[mode]*laguerre_inf(indices[mode][0],indices[mode][1],theta,phi)
            
    return E


E = beamConstructor(basis,indices,coef,THETA,PHI)

E_inc = [unit_vector[0]*E,unit_vector[1]*E,unit_vector[2]*E]
E_inc_dot_nphi = -E_inc[0]*np.sin(PHI) + E_inc[1]*np.cos(PHI)
E_inc_dot_nrho = E_inc[0]*np.cos(PHI) + E_inc[1]*np.sin(PHI)

E_inf = [(E_inc_dot_nphi*-np.sin(PHI) + E_inc_dot_nrho*np.cos(PHI)*np.cos(THETA))*np.sqrt(np.cos(THETA)),
         (E_inc_dot_nphi*np.cos(PHI) + E_inc_dot_nrho*np.sin(PHI)*np.cos(THETA))*np.sqrt(np.cos(THETA)),
         -E_inc_dot_nrho*np.sin(THETA)*np.sqrt(np.cos(THETA))]

E_xy = np.zeros([3,intRes,intRes]).astype(np.complex_)
E_xz = np.zeros([3,intRes,intRes]).astype(np.complex_)

for i in tqdm(range(intRes)):
    
    for j in range(intRes):

        propagator_xy = np.exp(1j*k*(0*np.cos(THETA)+np.sqrt(x[i]**2+y[j]**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(y[j],x[i]))))
        propagator_xz = np.exp(1j*k*(z[j]*np.cos(THETA)+np.sqrt(x[i]**2+0**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(0,x[i]))))
       
        E_xy[0,j,i] = simps(simps(E_inf[0]*propagator_xy*np.sin(THETA),theta),phi)
        E_xy[1,j,i] = simps(simps(E_inf[1]*propagator_xy*np.sin(THETA),theta),phi)
        E_xy[2,j,i] = simps(simps(E_inf[2]*propagator_xy*np.sin(THETA),theta),phi)
        
        E_xz[0,j,i] = simps(simps(E_inf[0]*propagator_xz*np.sin(THETA),theta),phi)
        E_xz[1,j,i] = simps(simps(E_inf[1]*propagator_xz*np.sin(THETA),theta),phi)
        E_xz[2,j,i] = simps(simps(E_inf[2]*propagator_xz*np.sin(THETA),theta),phi)


# evaluating the power that got through the lens
u = np.linspace(0,1/f0,100000)
Pf = 4*P*simps(np.exp(-2*u)*u,u) #[W] the total power of the beam that got through the tweezing lens

# proper normalization constant
norm = np.sqrt(2*Pf/c/e0)/np.sqrt(simps( simps( np.abs(E_xy[0,:,:])**2 + np.abs(E_xy[1,:,:])**2 + np.abs(E_xy[2,:,:])**2 ,x),y))

# properly normalized electric field in the XY plane for z = 0
E_xy *= norm
E_xz *= norm

# [J] potential energies for the gradient force
U_x = np.real(alpha_rad)*(np.abs(E_xy[0,int(intRes/2),:])**2 + np.abs(E_xy[1,int(intRes/2),:])**2 + np.abs(E_xy[2,int(intRes/2),:])**2)/4
U_y = np.real(alpha_rad)*(np.abs(E_xy[0,:,int(intRes/2)])**2 + np.abs(E_xy[1,:,int(intRes/2)])**2 + np.abs(E_xy[2,:,int(intRes/2)])**2)/4
U_z = np.real(alpha_rad)*(np.abs(E_xz[0,:,int(intRes/2)])**2 + np.abs(E_xz[1,:,int(intRes/2)])**2 + np.abs(E_xz[2,:,int(intRes/2)])**2)/4

# [N] gradient force for each direction
F_x = np.gradient(U_x,x)
F_y = np.gradient(U_y,y)
F_z = np.gradient(U_z,z)

# [N/m] spring constant for each direction
k_x = np.polyfit(x,F_x,20)[-2] 
k_y = np.polyfit(y,F_y,20)[-2]
k_z = np.polyfit(z,F_z,20)[-2]

# [Hz] natural oscillating frequency for each direction
w_x = np.sqrt(-k_x/mass_particle)
w_y = np.sqrt(-k_y/mass_particle)
w_z = np.sqrt(-k_z/mass_particle)

# [m] particle's motion standard deviation
x_std = np.sqrt(kb*T0/mass_particle/w_x**2)
y_std = np.sqrt(kb*T0/mass_particle/w_y**2)
z_std = np.sqrt(kb*T0/mass_particle/w_z**2)

# new limits for integration
x = np.linspace(-4*x_std,4*x_std,intRes)
y = np.linspace(-4*y_std,4*y_std,intRes)
z = np.linspace(-4*z_std,4*z_std,intRes)

X, Y, Z = np.meshgrid(x, y, z)

E_3D = np.zeros([3,intRes,intRes,intRes]).astype(np.complex_)

for o in tqdm(range(intRes)):

    for i in range(intRes):
        
        for j in range(intRes):
    
            propagator = np.exp(1j*k*(z[o]*np.cos(THETA)+np.sqrt(x[i]**2+y[j]**2)*np.sin(THETA)*np.cos(PHI-np.arctan2(y[j],x[i]))))
           
            E_3D[0,j,i,o] = simps(simps(E_inf[0]*propagator*np.sin(THETA),theta),phi)
            E_3D[1,j,i,o] = simps(simps(E_inf[1]*propagator*np.sin(THETA),theta),phi)
            E_3D[2,j,i,o] = simps(simps(E_inf[2]*propagator*np.sin(THETA),theta),phi)
            
E_3D *= norm

E_3D_sqrd = (np.abs(E_3D[0])**2 + np.abs(E_3D[1])**2 + np.abs(E_3D[2])**2)

points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
E_3D_sqrd_flat = E_3D_sqrd.ravel()
poly = PolynomialFeatures(degree=10, include_bias=False)
points_poly = poly.fit_transform(points)

model = LinearRegression()
model.fit(points_poly, E_3D_sqrd_flat)

# Coeficientes do polinômio
coefficients = model.coef_
intercept = model.intercept_

# Exibir os termos do polinômio
feature_names = poly.get_feature_names_out(['x', 'y', 'z'])
for name, coef in zip(feature_names, coefficients):
    print(f"{coef:.4f} * {name}")

print(f"Intercept: {intercept:.4f}") 
