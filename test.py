import sys
import numpy as np

sys.path.insert(1,'./library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
R_obs, Lx, Ly = 3., 2., 2.
E0 = 1
L = 7 # Points measured from phi=0
M = 16
f = 300e6
# f = np.array([300e6,400e6,500e6,600e6])
Nx, Ny = 60, 60 # (DOI size is NxxNy)
eps_obj = 3.2 # Relative permittivity of the object
epsb    = 1 # Relative permittivity of background
sig_obj = 0 # Conductivity of the object
sigb    = 0 # Conductivity of background

experiment = md.Model(dm.Domain(Lx,Ly,R_obs,L,M),frequencies=f,
                      incident_field_magnitude=E0,epsilon_r_background=epsb,
                      sigma_background=sigb)

epsilon_r, sigma = maps.build_square(Nx,Ny,Lx/Nx,Ly/Ny,epsb,sigb,eps_obj,
                                     sig_obj,.3*Lx)


# experiment.draw_setup(epsr=eps_r,sig=sig)
es = experiment.solve(epsilon_r=epsilon_r,sigma=sigma,COMPUTE_INTERN_FIELD=False)
# experiment.plot_total_field(frequency_index=3)

method = slv.BIM_Tikhonov(experiment)
method.solve(es,number_iterations=1,plot_results=True,experiment_name='test',
             file_format='png')
