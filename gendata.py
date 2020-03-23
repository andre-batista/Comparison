import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,'./library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
exp_name = 'basic_triangle'
R_obs, Lx, Ly = .25 ,.36, .36
E0 = 1
L = 18 # Points measured from phi=0
M = 18
f = .8e9
# f = np.array([300e6,400e6,500e6,600e6])
Nx, Ny = 60, 60 # (DOI size is NxxNy)
eps_obj = 2.0 # Relative permittivity of the object
epsb    = 1 # Relative permittivity of background
sig_obj = 0 # Conductivity of the object
sigb    = 0 # Conductivity of background

# Constructor
experiment = md.Model(dm.Domain(Lx,Ly,R_obs,L,M),frequencies=f,
                      incident_field_magnitude=E0,epsilon_r_background=epsb,
                      sigma_background=sigb,model_name=exp_name)

# Get map
epsilon_r, sigma = maps.build_triangle(Nx,Ny,Lx/Nx,Ly/Ny,epsb,sigb,eps_obj,
                                     sig_obj,21e-2)

# Solve fields
es = experiment.solve(epsilon_r=epsilon_r,sigma=sigma,COMPUTE_INTERN_FIELD=True,
                      maximum_iterations=1000,save=True)

# Save configuration
experiment.save_model_configuration(exp_name+'_config')