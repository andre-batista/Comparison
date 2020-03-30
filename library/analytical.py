""""           SCATTERING BY A LOSSLESS DIELECTRIC CYLINDER

This script implements the analytical solution for the scattering of a
lossless dielectric sphere in the presence of a incident plane wave [1].

REFERENCES
[1] - Roger F. Harrington. (1961). Time-harmonic electromagnetic 
      fields. Mcgraw-Hill.
"""

# Base libraries
import numpy as np
import numpy.random as rnd
import numpy.linalg as lag
import scipy.constants as ct
import scipy.special as spc
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import pickle
from numba import jit
import model as md

# Main function
def get_data(magnitude=1.,proportion=.5,frequency=1e9,Nsources=2,
             Nsamples=4,epsilon_rb=1.,mu_rb=1.,epsilon_rd=2.,mu_rd=1.,
             Nx=None,Ny=None,file_name=None,file_path='',delta=None):
    """ GET_DATA
    Compute the fields for the scattering of a dielectric cylinder. 
    Inputs:

    - magnitude: magnitude of incident wave [V/m]
    - proportion: ratio radius/wavelength
    - frequency: linear frequency for calculation [Hz]
    - Nsources: number of incidences
    - Nsamples: number of scattered field samples
    - epsilon_rb: relative permittivity of background
    - mu_rb: relative permeability of background
    - epsilon_rd: relative permittivity of cylinder
    - mu_rd: relative permeability of cylinder
    """

    # Main parameters
    E0      = magnitude     # Amplitude of the plane wave [V/m]
    A       = proportion    # Proportion radius/wavelength
    f       = frequency     # Linear frequency [Hz]
    omega   = 2*np.pi*f     # Angular frequency [rad/s]
    M       = Nsamples      # Number of scatered field samples
    L       = Nsources      # Number of sources of incident waves
    N       = 50            # Number of terms in the summing serie

    # Main constants
    epsd    = epsilon_rd*ct.epsilon_0           # Cylinder's permittivity [F/m]         
    epsb    = epsilon_rb*ct.epsilon_0           # Background permittivity [F/m]
    mub     = mu_rb*ct.mu_0                     # Background permeability [H/m]
    mud     = mu_rd*ct.mu_0                     # Cylinder's permeability [H/m]
    c       = 1/np.sqrt(ct.epsilon_0*ct.mu_0)   # Speed of light [m/s]
    kb      = omega*np.sqrt(mub*epsb)           # Wavenumber of background [rad/m] 
    kd      = omega*np.sqrt(mud*epsd)           # Wavenumber of cylinder [rad/m]
    lambdab = 2*np.pi/kb                        # Wavelength of background [m]
    lambdad = 2*np.pi/kd                        # Wavelength of cylinder [m]
    a       = A*lambdab                         # Sphere's radius [m]
    thetal  = np.linspace(0,2*np.pi,L,endpoint=False)
    thetam  = np.linspace(0,2*np.pi,M,endpoint=False)

    # Summing coefficients
    an, cn = get_coefficients(N,kb,kd,a,epsd,epsb)

    # Mesh parameters
    Lx, Ly, dx, dy, x, y, Nx, Ny = get_mesh(a,lambdab,Nx,Ny)

    # Incident field
    ei = get_incident_field(E0,kb,x,y,thetal)

    # Total field array
    et = compute_total_field(x,y,a,an,cn,N,kb,kd,E0,thetal)
            
    # Map of parameters
    epsilon_r, sigma = get_map(x,y,a,epsilon_rb,epsilon_rd)
    
    # Scatered field
    rho = 1/2*np.sqrt(Lx**2+Ly**2)*1.05
    xm, ym = rho*np.cos(thetam), rho*np.sin(thetam)
    es = compute_scattered_field(xm,ym,an,kb,thetal,E0)

    # Green function
    gs = md.get_greenfunction(xm,ym,x,y,kb)

    if delta is not None:
        es = md.add_noise(es,delta)

    if file_name is not None:
        
        config = {
            'model_name':file_name,
            'Lx':Lx, 'Ly':Ly,
            'radius_observation':rho,
            'number_measurements':M,
            'number_sources':L,
            'dx':dx,
            'dy':dy,
            'x':x, 'y':y,
            'incident_field':ei,
            'green_function_s':gs,
            'wavelength':lambdab,
            'wavenumber':kb,
            'angular_frequency':omega,
            'relative_permittivity_background':epsilon_rb,
            'conductivity_background':.0,
            'measurement_coordinates_x':xm,
            'measurement_coordinates_y':ym,
            'frequency':f,
            'Nx':Nx, 'Ny':Ny,
            'incident_field_magnitude':E0
        }
        
        data = {
            'scattered_field':es,
            'total_field':et,
            'relative_permittivity_map':epsilon_r,
            'conductivity_map':sigma,
            'maximum_number_iterations':0,
            'error_tolerance':0
        }

        with open(file_path + file_name + '_config','wb') as configfile:
            pickle.dump(config,configfile)
        
        with open(file_path + file_name,'wb') as datafile:
            pickle.dump(data,datafile)

# Auxiliar functions
def cart2pol(x,y):
    rho = np.sqrt(x**2+y**2)
    phi = np.arctan2(y,x)
    phi[phi<0] = 2*np.pi+phi[phi<0]
    return rho, phi

def get_incident_field(magnitude,wavenumber,x,y,theta=None):
    
    if theta is None:
        rho, phi = cart2pol(x.reshape(-1),y.reshape(-1))
        return magnitude*np.exp(-1j*wavenumber*rho*np.cos(phi)) # [V/m]
    
    else:
        L = theta.size
        ei = np.zeros((x.size,L),dtype=complex)
        for l in range(L):
            xp, yp = rotate_axis(theta[l],x.reshape(-1),y.reshape(-1))
            rho, phi = cart2pol(xp,yp)
            ei[:,l] = magnitude*np.exp(-1j*wavenumber*rho*np.cos(phi)) # [V/m]
    
    return ei
        
def get_coefficients(Nterms,wavenumber_b,wavenumber_d,radius,epsilon_d,
                     epsilon_b):
    
    n = np.arange(-Nterms,Nterms+1)
    kb, kd = wavenumber_b, wavenumber_d
    a = radius
    
    an = (-spc.jv(n,kb*a)/spc.hankel2(n,kb*a)*(
        (epsilon_d*spc.jvp(n,kd*a)/(epsilon_b*kd*a*spc.jv(n,kd*a)) 
         - spc.jvp(n,kb*a)/(kb*a*spc.jv(n,kb*a)))
        /(epsilon_d*spc.jvp(n,kd*a)/(epsilon_b*kd*a*spc.jv(n,kd*a)) 
          - spc.h2vp(n,kb*a)/(kb*a*spc.hankel2(n,kb*a)))
    ))
    
    cn = 1/spc.jv(n,kd*a)*(spc.jv(n,kb*a)+an*spc.hankel2(n,kb*a))
    
    return an, cn

def rotate_axis(theta,x,y):
    T = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    r = np.vstack((x.reshape(-1),y.reshape(-1)))
    rp = T@r
    xp, yp = np.vsplit(rp,2)
    xp = np.reshape(np.squeeze(xp),x.shape)
    yp = np.reshape(np.squeeze(yp),y.shape)
    return xp, yp

def compute_total_field(x,y,radius,an,cn,N,wavenumber_b,wavenumber_d,
                        magnitude,theta=None):
    E0 = magnitude
    kb, kd = wavenumber_b, wavenumber_d
    a = radius
    
    if theta is None:
        rho, phi = cart2pol(x,y)
        et = np.zeros(rho.shape,dtype=complex)
        i = 0
        for n in range(-N,N+1):
            
            et[rho>a] = et[rho>a] + (
                E0*1j**(-n)*(spc.jv(n,kb*rho[rho>a])
                             +an[i]*spc.hankel2(n,kb*rho[rho>a]))
                * np.exp(1j*n*phi[rho>a])
            )
            
            et[rho<=a] = et[rho<=a] + (
                E0*1j**(-n)*cn[i]*spc.jv(n,kd*rho[rho<=a])
                * np.exp(1j*n*phi[rho<=a])
            )
            
            i+=1
        
    else:
        L = theta.size
        et = np.zeros((x.size,L),dtype=complex) # [V/m]
        for l in range(L):
            xp, yp = rotate_axis(theta[l],x.reshape(-1),y.reshape(-1))
            rho, phi = cart2pol(xp,yp)
            i=0
            for n in range(-N,N+1):
                
                et[rho>a,l] = et[rho>a,l] + (
                    E0*1j**(-n)*(spc.jv(n,kb*rho[rho>a])
                                 + an[i]*spc.hankel2(n,kb*rho[rho>a]))
                    * np.exp(1j*n*phi[rho>a])
                )
                
                et[rho<=a,l] = et[rho<=a,l] + (
                    E0*1j**(-n)*cn[i]*spc.jv(n,kd*rho[rho<=a])
                    *np.exp(1j*n*phi[rho<=a])
                )
                
                i+=1
        
    return et

def get_map(x,y,radius,epsilon_rb,epsilon_rd):
    epsilon_r = epsilon_rb*np.ones(x.shape)
    sigma = np.zeros(x.shape)
    epsilon_r[x**2+y**2<=radius**2] = epsilon_rd
    return epsilon_r, sigma

def get_mesh(radius,lambda_b,Nx=None,Ny=None):
    
    Lx, Ly = 4*radius, 4*radius # Image size [m], [m]
        
    if Nx is None:
        Nx = int(np.ceil(Lx/(lambda_b/25))) # Domain size x-axis [number of cells]
    dx = Lx/Nx # Cell size in x-axis [m]
        
    if Ny is None:
        Ny = int(np.ceil(Ly/(lambda_b/25))) # Domain size y-axis [number of cells]
    dy = Ly/Ny # Cell size in y-axis [m]
        
    # Mesh arrays
    x, y = np.meshgrid(np.arange(-Lx/2+dx/2,Lx/2,dx),
                       np.arange(-Ly/2+dy/2,Ly/2,dy))
    
    return Lx, Ly, dx, dy, x, y, Nx, Ny

def compute_integral(et,gs,X):
    L = et.shape[1]
    J = np.tile(X.reshape((-1,1)),L)*et
    return gs@J

def compute_scattered_field(xm,ym,an,kb,theta,magnitude):
    M, L, N = xm.size, theta.size, round((an.size-1)/2)
    E0 = magnitude
    es = np.zeros((M,L),dtype=complex)
    for l in range(L):
        i = 0
        for n in range(-N,N+1):
            xp, yp = rotate_axis(theta[l],xm,ym)
            rho, phi = cart2pol(xp,yp)
            es[:,l] = es[:,l] + (E0*1j**(-n)*an[i]*spc.hankel2(n,kb*rho)
                                 * np.exp(1j*n*phi))
            i += 1
        
    return es