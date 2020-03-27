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
resultspath = '.\\results\\'
alpha = 10**(np.linspace(0,-18,100))

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config',model_path=datafilepath)

# Load scattered field
es = md.load_scattered_field(datafilename,file_path=datafilepath)

# Load total field
et = md.load_total_field(datafilename,file_path=datafilepath)

# Load original profiles
epsilon_r_original, _ = md.load_maps(datafilename,file_path=datafilepath)

# Run inverse solver
error = np.zeros(alpha.size)
maperror = np.zeros(alpha.size)
for i in range(alpha.size):
    epsilon_r, sigma = sim.solve_linear(es,et=et,alpha=alpha[i])
    error[i] = sim.compute_norm_residual(et,es,
                                         epsilon_r=epsilon_r,
                                         sigma=sigma)
    maperror[i] = sim.compute_map_error(epsilon_original=epsilon_r_original,
                                        epsilon_recovered=epsilon_r)
    print('alpha: %.1e - ' %alpha[i] + 'res_error: %.3e -' %error[i] 
          + 'map_error: %.3e' %maperror[i])

# Plot residual results
plt.loglog(alpha,error,'--*')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$||y-Kx||$')
plt.title('Residual of equations')
plt.grid()
plt.savefig(resultspath + 'residualconvergence.eps',format='eps')
plt.close()

# Plot amp results
plt.loglog(alpha,maperror,'--*')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\zeta_\epsilon$')
plt.title('Map error')
plt.grid()
plt.savefig(resultspath + 'mapconvergence.eps',format='eps')
plt.close()

# Plot best solution - error
i = np.argmin(error)
epsilon_r,_ = sim.solve_linear(es,et,alpha[i])
sim.plot_results(epsilon_r=epsilon_r,title='Best solution - residual',
                 file_path=resultspath,file_name='best_residual')

# Plot best solution - map
i = np.argmin(maperror)
epsilon_r,_ = sim.solve_linear(es,et,alpha[i])
sim.plot_results(epsilon_r=epsilon_r,title='Best solution - map',
                 file_path=resultspath,file_name='best_map')

# Save data
np.savez_compressed(resultspath+'results',alpha=alpha,
                    residual_error=error,map_error=maperror)