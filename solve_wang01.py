import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1,'./library')
import domain as dm
import model as md
import syntheticmaps as maps
import inversesolvers as slv

## DEFINE PARAMETERS
datafilename = 'wang01'
alpha = 1e-4
number_iterations = 50

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config')

# Load scattered field
es = md.load_scattered_field(datafilename)

# Load total field
et = md.load_total_field(datafilename)

# Run nonlinear problem
sim.solve(es,number_iterations=number_iterations,
          plot_results=True,experiment_name=datafilename,
          file_format='png',save_results=True,alpha=alpha)