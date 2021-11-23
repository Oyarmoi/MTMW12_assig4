""""Code code to solve the one dimensional diffusion equation representing the 
variation of wind speed, u(z, t), with height, z, and time, t, in the neutrally
stable atmospheric boundary"""

# Importing key Python modules
import numpy as np
import matplotlib.pyplot as plt

# Dictionary of physical parameters
phyPara = {'k' : 0.4,  # von-Karmen constant 
           'K' : 5.,  # Diffusion coefficient in m2/s
           'Kmin' : 1e-6, # MinDiffusionCoeff for turbulent simulation (m2/s) 
           'z0' : 0.1,  # surface roughness height in m 
           'zmin' : 0.,  # Start of z domain at ground surface in m
           'zmax' : 100.,  # End of z domain in m
           'up' : -5e-3}  # uniform pressure gradient (1/ρ * ∂p/∂x)
#print(phyPara)

# Dictionary of intitial conditions
initialCond = {'u0' : 10., # initial wind speed through entire domain (m/s) 
               'T' : 3600,  # Total time of simulation (s) 
               'dt' : 0.5,  # Simulation time step (s)
               'dz' : 5.}  # Simulation grid spacing (m)
#print(initialCond)

def FTCS_diffuse(phi, d, nt, dtForcing):
    '''Solve the one dimensional diffusion coeffiecient with the FTCS finite
    difference scheme starting from initial conditions in array phi.
    d is the non-dimensional diffusion coefficient: d = dt K/dx^2
    dtForcing is the time step multiplied by an additional term on the right
    hand side of the equation.
    The start boundary condition is zero and the end boundary condition is
    zero gradient.
    phi after nt time steps is returned.'''

    # Create an array for phi at the next time step
    phiNew = phi.copy()
    
    # Short cut to the length of the array phi
    nx = len(phi)
    
    # Loop through all time steps. phiNew is always phi at time step n+1
    # and phi is at time step n. 
    for it in range(nt):
        # Loop over space away from the boundaries
        for i in range(1,nx-1):
            phiNew[i] = phi[i] + d*(phi[i+1] - 2*phi[i] + \
                                       phi[i-1]) + dtForcing

        # Update the boundary conditions
        phiNew[0] = 0 
        phiNew[nx-1] = phiNew[nx-2] # u is constant for zero slope
        
        # Update phi for the next time step
        phi = phiNew.copy()
    
    return phi

def FTCS_turbulentDiffuse(u, z, Kmin, nt, dt, dz, forcing):
    '''Solve the one dimensional diffusion coeffiecient with the FTCS finite
    difference scheme starting from initial conditions in array u (wind speed).
    z is height above the ground.
    Kmin is the minimum diffusion coefficient.
    dt is the time step.
    forcing is the additional term on the right hand side of the equation.
    The start boundary condition is zero and the end boundary condition is
    zero gradient.
    u after nt time steps is returned.
    The diffusion coefficient is calculated as
    K = L^2 |dudz|
    where the length scale L = 0.4z'''

    # Declare an array for the height of the K locations
    zK = np.arange(dt * dz, z[-1], dz)
    
    # Declare array for the diffusion coefficients, K and the wind shear
    K = np.zeros_like(zK)
    dudz = np.zeros_like(zK)
    
    # Loop over all time steps
    for it in range(nt):
        # Calculate dudz and K for each level
        for k in range(len(zK)):
            dudz[k] = (u[k+1] - u[k]) / dz
            length = phyPara['k'] * zK[k]
            K[k] = length**2 * abs(dudz[k]) + Kmin
        #print(length, dudz, Kmin)
        # Update u for each internal level based on dudz and K
       #print(K)
        for k in range(1,len(u)-1):
            #u[k] = u[k] + (K[k] * dt / dz**2)*(u[k+1] - 2*u[k] +u[k-1]) + forcing*dt
            u[k] = u[k] + dt/dz * (K[k]*dudz[k] - K[k-1]*dudz[k-1]) +  forcing*dt     
        # Update the boundary conditions
        u[0] = 0 
        u[k-1] = u[k-2] # u is constant for zero slope
    print(u)    
    return u

def windDiffusion():
    """This function setsup the coding for simulation of the variation of wind 
    speed, u, with height, z, and time, t, in the atmospheric boundary layer,
    including plotting of simulation results"""
    
    # Setting out the grid domain in z direction
    z = np.arange(0, phyPara['zmax'] + 1, initialCond['dz']) #  grid doimain
    
    # Setting out the time domain
    tDomain = int(np.round(initialCond['T'] / initialCond['dt']))

    # Uniform pressure gradient
    pConst = phyPara['up']

    # Non-dimensional diffusion coefficient (d = dt K/dx^2)
    d = phyPara['K'] * initialCond['dt'] / (initialCond['dz'])**2
    
    print('Solving diffusion equation for ', tDomain, \
          ' time steps with non-dimensional diffusion coefficient ', d)

    # Setting out intial wind condition at all grid domains
    u = initialCond['u0']*np.ones_like(z)

    # Solution
    u_FTCS = FTCS_diffuse(u.copy(), d, tDomain, -initialCond['dt']*pConst)
    u_turb = FTCS_turbulentDiffuse(u, z, phyPara['Kmin'], tDomain, \
                                   initialCond['dt'], initialCond['dz'],\
                                       -pConst)
    
    # Plot with height
    plt.plot(u_FTCS, z, 'b-', label='u fixed K')
    plt.plot(u_turb, z, 'r-', label='u turbulent')
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Height [m]')
    plt.xlim([0,10])
    plt.ylim([0,100])
    plt.legend(loc='best')
    plt.show()
    
if __name__ == "__main__":
    windDiffusion()
