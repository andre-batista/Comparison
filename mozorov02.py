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
alpha = 10**(np.linspace(-13,-15))

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config')

# Load scattered field
es = md.load_scattered_field(datafilename)

# Load total field
et = md.load_total_field(datafilename)

# Run linear problem
sim.model.Et = np.copy(et)

error = np.zeros(alpha.size)
for i in range(alpha.size):
    sim.alpha = alpha[i]
    epsilon_r, sigma = sim.tikhonov_regularization(es)
    error[i] = sim.compute_norm_residual(et,es,
                                         epsilon_r=epsilon_r,
                                         sigma=sigma)
    print('alpha: %.1e - ' %alpha[i] + 'error: %.3e' %error[i])
    
plt.semilogx(alpha,error,'--*')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$||y-Kx||$')
plt.title('The Discrepancy Principle of Mozorov')
plt.grid()
plt.show()