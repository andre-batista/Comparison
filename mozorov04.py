import sys
import numpy as np
import numpy.linalg as lag
import matplotlib.pyplot as plt

sys.path.insert(1,'./library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
datafilename = 'basic_triangle'
delta = 1e-8

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config')

# Load scattered field
es = md.load_scattered_field(datafilename)

# Load total field
et = md.load_total_field(datafilename)

# Run linear problem
sim.model.Et = np.copy(et)
sim.alpha = delta*lag.norm()
epsilon_r, sigma = sim.tikhonov_regularization(es)
error = sim.compute_norm_residual(et,es,epsilon_r=epsilon_r,sigma=sigma)
print('alpha: %.1e - ' %alpha + 'error: %.3e' %error)
sim.plot_results(epsilon_r=epsilon_r,sigma=sigma)