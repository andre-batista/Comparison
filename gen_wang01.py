import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,'./library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
exp_name = 'wang01'
epsilon_rpeak, sigma_peak = 11, 0
epsilon_rb, sigma_b = 1., .0
f = 10e6
lambda_b = md.get_wavelength(frequencies=f,epsilon_r=epsilon_rb)
R_obs, Lx, Ly = lambda_b/10, lambda_b/10, lambda_b/10
E0 = 1
L, M = 4, 36
Nx, Ny = 12, 12

# Constructor
experiment = md.Model(dm.Domain(Lx,Ly,R_obs,L,M),frequencies=f,
                      incident_field_magnitude=E0,epsilon_r_background=epsilon_rb,
                      sigma_background=sigma_b,model_name=exp_name)

# Get map
epsilon_r, sigma = maps.build_sine(Nx,Ny,lambda_b/10,epsilon_rb,sigma_b,
                                   epsilon_rpeak,sigma_peak)

# # Draw setup
# experiment.draw_setup(epsr=epsilon_r,sig=sigma)

# Solve fields
es = experiment.solve(epsilon_r=epsilon_r,sigma=sigma,COMPUTE_INTERN_FIELD=True,
                      maximum_iterations=1000,save=True)

# Save configuration
experiment.save_model_configuration(exp_name+'_config')