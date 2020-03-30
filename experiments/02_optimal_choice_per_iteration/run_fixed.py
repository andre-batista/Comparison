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
filename = 'fixedalpha'
alpha = 4.329e-03
number_iterations = 10

# Load data
sim = slv.BIM_Tikhonov(datafilename+'_config',model_path=datafilepath)

# Load scattered field
es = md.load_scattered_field(datafilename,file_path=datafilepath)

# Load testbench
epsilon_r_goal, sigma_goal = md.load_maps(datafilename,
                                          file_path=datafilepath)

# Run nonlinear problem
sim.solve(es,number_iterations=number_iterations,
          plot_results=True,experiment_name=filename,
          file_format='eps',save_results=True,alpha=alpha,
          epsilon_r_goal=epsilon_r_goal,sigma_goal=sigma_goal,
          file_path=resultsfilepath)