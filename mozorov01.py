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
alpha = 10**(np.arange(-1,-18,-1,dtype=float))
# alpha = np.array([1e-2,1e-3,1e-4,1e-5,
#                   1e-6,1e-7,1e-8,1e-9,1e-10,
#                   1e-11,1e-12,1e-13])

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
    
plt.loglog(alpha,error,'--*')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$||y-Kx||$')
plt.title('The Discrepancy Principle of Mozorov')
plt.grid()
plt.show()