import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,'..\..\library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
datafilename = 'ana_1GHz_e2_d0'
datafilepath = '.\data\\'
alpha = 10**(np.linspace(0,-18,50))
# alpha = 10**(np.linspace(2,-5,20))

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config',model_path=datafilepath)

# Load scattered field
es = md.load_scattered_field(datafilename,file_path=datafilepath)

# Load total field
et = md.load_total_field(datafilename,file_path=datafilepath)

# Load original profiles
epsilon_r_original, _ = md.load_maps(datafilename,file_path=datafilepath)

# # Run inverse solver
# error = np.zeros(alpha.size)
# maperror = np.zeros(alpha.size)
# for i in range(alpha.size):
#     epsilon_r, sigma = sim.solve_linear(es,et=et,alpha=alpha[i])
#     error[i] = sim.compute_norm_residual(et,es,
#                                          epsilon_r=epsilon_r,
#                                          sigma=sigma)
#     maperror[i] = sim.compute_map_error(epsilon_original=epsilon_r_original,
#                                         epsilon_recovered=epsilon_r)
#     print('alpha: %.1e - ' %alpha[i] + 'res_error: %.3e -' %error[i] 
#           + 'map_error: %.3e' %maperror[i])

# # Plot residual results
# plt.loglog(alpha,error,'--*')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$||y-Kx||$')
# plt.title('Residual of equations')
# plt.grid()
# plt.show()

# # Plot amp results
# plt.loglog(alpha,maperror,'--*')
# plt.xlabel(r'$\alpha$')
# plt.ylabel(r'$\zeta_\epsilon$')
# plt.title('Map error')
# plt.grid()
# plt.show()

# Plot best solution
# i = np.argmin(maperror)
# sim.alpha = 1e-2 # alpha[i]
epsilon_r, sigma = sim.solve_linear(es,et,1e-2)
sim.plot_results(epsilon_r=epsilon_r)