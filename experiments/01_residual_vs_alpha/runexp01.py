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
alpha = 1e-2
number_iterations = 6

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config',model_path=datafilepath)

# Load scattered field
es = md.load_scattered_field(datafilename,file_path=datafilepath)

# Run nonlinear problem
sim.solve(es,number_iterations=number_iterations,
          plot_results=True,experiment_name=datafilename,
          file_format='png',save_results=True,alpha=alpha)