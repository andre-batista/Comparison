import numpy as np
import copy as cp
import model as md
import domain as dm
import syntheticmaps as maps

DATAFILENAME = '_DATA'

class Experiment:
    
    experiment_name = ''
    model = md.Model(dm.Domain(0,0,0,0,0))
    epsilon_r = np.ones((0,0))
    sigma = np.zeros((0,0))
    
    def __init__(self,experiment_name,model,epsilon_r,sigma):
        
        self.experiment_name = experiment_name
        self.model = cp.deepcopy(model)
        self.epsilon_r = np.copy(epsilon_r)
        self.sigma = np.copy(sigma)
        
    def generate_data(self,frequency=None,file_path=''):
        
        self.model.solve(epsilon_r=self.epsilon_r,
                         sigma=self.sigma,
                         frequencies=frequency,
                         file_name=self.experiment_name+DATAFILENAME,
                         file_path=file_path)
    
    def solve_inverse(self,datafile_path=''):
        pass
        