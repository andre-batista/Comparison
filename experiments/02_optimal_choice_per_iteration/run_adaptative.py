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
resultsfilepath = '.\\results\\'
filename = 'adaptativealpha'
alpha = 10**(np.linspace(0,-18,100))
number_iterations = 10

# Convergence variables
Ja = np.zeros(number_iterations)
residual = np.zeros(number_iterations)
zeta_e = np.zeros(number_iterations)
zeta_s = np.zeros(number_iterations)
alphaerror = np.zeros(alpha.size)
alphachoice = np.zeros(number_iterations)

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config',model_path=datafilepath)

# Load scattered field
es = md.load_scattered_field(datafilename,file_path=datafilepath)

# Load incident field
ei = md.load_incident_field(datafilename,file_path=datafilepath)

# Load testbench
epsilon_r_goal, sigma_goal = md.load_maps(datafilename,
                                          file_path=datafilepath)

for it in range(number_iterations):
    
    print('Iteration %d - ' %(it+1), end='')
    
    if it == 0:
        et = ei
    else:
        et = sim.get_intern_field()
    
    for ia in range(alpha.size):
        epsilon_r, sigma = sim.solve_linear(es,et=et,alpha=alpha[ia])
        alphaerror[ia] = sim.compute_norm_residual(et,es,
                                                   epsilon_r=epsilon_r,
                                                   sigma=sigma)
    
    ia = np.argmin(alphaerror)
    alphachoice[it] = alpha[ia]
    print('alpha = %.3e - ' %alpha[ia], end='')
    
    if it == 0:
        initial_field = None
    else:
        initial_field = sim.get_intern_field()
    
    epilon_r, sigma, Ja[it], residual[it], zeta_e[it], zeta_s[it] = sim.solve(
        es,number_iterations=1,experiment_name=filename,alpha=alpha[ia],
        epsilon_r_goal=epsilon_r_goal,sigma_goal=sigma_goal,
        initial_field=initial_field,print_info=False
    )
    
    print('Ja(X) = %.3e - ' %Ja[it]
          + 'Residual = %.3e - ' %residual[it]
          + 'zeta_e = %.2f - ' %zeta_e[it]
          + 'zeta_s = %.3e' %zeta_s[it])

# Plot alpha choice
plt.plot(np.arange(1,number_iterations+1),alphachoice,'--*')
plt.xlabel('Iterations')
plt.ylabel(r'$\alpha$')
plt.title(r'\alpha Choice')
plt.grid()
plt.savefig(resultsfilepath + filename + '.eps',format='eps')
plt.close()

# Save data
np.savez_compressed(resultsfilepath + filename + '_results',
                    alpha=alpha,
                    alphachoice=alphachoice,
                    Ja=Ja,
                    residual=residual,
                    zeta_e=zeta_e,
                    zeta_s=zeta_s)