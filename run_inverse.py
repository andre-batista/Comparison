import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,'./library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
datafilename = 'basic_triangle'
alpha = 1e-4
number_iterations = 10

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config')

# Load scattered field
es = md.load_scattered_field(datafilename)

# Load total field
et = md.load_total_field(datafilename)

# # Run initial guess with exact data
# sim.model.Et = np.copy(et)
# N = et.shape[0]
# epsilon_r, sigma = slv.initialsolution1(es,et,sim.model.GS,sim.model.domain.M,
#                                         sim.model.domain.L,N,sim.model.epsilon_rb,
#                                         sim.model.sigma_b,sim.model.omega)
# epsilon_r = epsilon_r.reshape((sim.model.Nx,sim.model.Ny))
# sigma = sigma.reshape((sim.model.Nx,sim.model.Ny))
# sim.plot_results(epsilon_r=epsilon_r,sigma=sigma,file_name=datafilename,file_format='png')

# # Run initial guess with Born approximation
# ei = np.copy(sim.model.Ei)
# N = et.shape[0]
# epsilon_r, sigma = slv.initialsolution1(es,ei,sim.model.GS,sim.model.domain.M,
#                                         sim.model.domain.L,N,sim.model.epsilon_rb,
#                                         sim.model.sigma_b,sim.model.omega)
# epsilon_r = epsilon_r.reshape((sim.model.Nx,sim.model.Ny))
# sigma = sigma.reshape((sim.model.Nx,sim.model.Ny))
# sim.plot_results(epsilon_r=epsilon_r,sigma=sigma,file_name=datafilename,file_format='png')

# # Run Born approximation
# sim.model.Et = np.copy(sim.model.Ei)
# sim.alpha = alpha
# epsilon_r, sigma = sim.tikhonov_regularization(es)
# sim.plot_results(epsilon_r=epsilon_r,sigma=sigma,file_name=datafilename,file_format='png')

# # Run linear problem
# sim.model.Et = np.copy(et)
# sim.alpha = alpha
# epsilon_r, sigma = sim.tikhonov_regularization(es)
# sim.plot_results(epsilon_r=epsilon_r,sigma=sigma,file_name=datafilename,file_format='png')

# Run nonlinear problem
sim.solve(es,number_iterations=number_iterations,
          plot_results=True,experiment_name=datafilename,
          file_format='png',save_results=True,alpha=alpha)